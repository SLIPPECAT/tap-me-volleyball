# 출처: 실험 레포 baegugu-highlight tests/test_anonymize.py (커밋 4989e48, 대회 기간 작성). import만 바꿈
from __future__ import annotations

import numpy as np

from tapme import anonymize


def _gradient_frame(h: int = 240, w: int = 320) -> np.ndarray:
    # 모자이크가 실제로 픽셀을 바꾸도록 위치마다 값이 다른 프레임
    ys, xs = np.mgrid[0:h, 0:w]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = (xs % 256).astype(np.uint8)
    frame[..., 1] = (ys % 256).astype(np.uint8)
    frame[..., 2] = ((xs + ys) % 256).astype(np.uint8)
    return frame


def test_pixelate_changes_only_inside_box() -> None:
    frame = _gradient_frame()
    box = (100, 80, 60, 60)
    out = anonymize.pixelate_boxes(frame, [box])

    x, y, w, h = box
    pad_x, pad_y = int(w * anonymize.BOX_PAD), int(h * anonymize.BOX_PAD)
    # 박스 바깥은 그대로
    assert np.array_equal(out[: y - pad_y - 1], frame[: y - pad_y - 1])
    assert np.array_equal(out[:, : x - pad_x - 1], frame[:, : x - pad_x - 1])
    # 박스 안은 바뀌었다
    assert not np.array_equal(out[y : y + h, x : x + w], frame[y : y + h, x : x + w])


def test_pixelate_small_face_is_coarse() -> None:
    # 18~23px짜리 얼굴도 알아볼 수 없을 만큼 굵게 가려야 한다(블록 2px이면 실패)
    frame = _gradient_frame()
    box = (100, 80, 20, 20)
    out = anonymize.pixelate_boxes(frame, [box])
    x, y, w, h = box
    pad = int(w * anonymize.BOX_PAD)
    region = out[y - pad : y + h + pad, x - pad : x + w + pad].reshape(-1, 3)
    assert len({tuple(p) for p in region}) <= 9


def test_pixelate_does_not_modify_input() -> None:
    frame = _gradient_frame()
    before = frame.copy()
    anonymize.pixelate_boxes(frame, [(10, 10, 50, 50)])
    assert np.array_equal(frame, before)


def test_pixelate_handles_box_at_frame_edge() -> None:
    frame = _gradient_frame()
    out = anonymize.pixelate_boxes(frame, [(-5, -5, 40, 40), (300, 220, 40, 40)])
    assert out.shape == frame.shape


def test_detector_loads() -> None:
    assert anonymize.load_detector() is not None


def test_detection_settings_cover_small_and_profile_faces() -> None:
    # 10/28 H-21: 0.6·원래 크기·정면 검출기만으로는 옆얼굴 3개를 놓쳤다 (H-22 보강값)
    assert anonymize.SCORE_THRESHOLD == 0.5
    assert anonymize.DETECT_UPSCALE == 2.0
    assert anonymize.load_profile_detector() is not None


def test_merged_detection_maps_boxes_back_to_frame_size(monkeypatch) -> None:
    # 2배로 키워 찾은 박스와 뒤집어 찾은 옆얼굴 박스를 원래 프레임 좌표로 되돌린다
    frame = _gradient_frame(h=100, w=200)

    class FakeYuNet:
        def setInputSize(self, size):
            assert size == (400, 200)  # 2배

        def detect(self, img):
            return None, np.array([[40, 20, 20, 30] + [0] * 11], dtype=np.float32)

    class FakeProfile:
        calls = 0

        def detectMultiScale(self, gray, **kw):
            FakeProfile.calls += 1
            # 첫 호출(원래 방향)은 없음, 두 번째(뒤집은 프레임)는 오른쪽 끝 근처
            return np.empty((0, 4)) if FakeProfile.calls == 1 else np.array([[10, 50, 20, 20]])

    boxes = anonymize.detect_faces(frame, FakeYuNet(), FakeProfile())
    assert (20, 10, 10, 15) in boxes  # YuNet 박스 /2
    assert (170, 50, 20, 20) in boxes  # 뒤집은 x=10 → 200-10-20


def test_no_faces_on_blank_frame() -> None:
    detector = anonymize.load_detector()
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert anonymize.detect_faces(blank, detector) == []


def test_clips_for_meta_skip_the_profile_detector(tmp_path, monkeypatch) -> None:
    # H-51(11/1 사용자 결정): Meta로 보내는 흐림본은 옆얼굴 검출 없이 YuNet만. 결과 영상 모자이크는 그대로
    import subprocess

    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=5:duration=0.4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    seen: list = []
    monkeypatch.setattr(anonymize, "detect_faces", lambda f, d, p=None, s=anonymize.PROFILE_SCALE_FACTOR: seen.append(p) or [])
    anonymize.anonymize_video(src, tmp_path / "out.mp4")
    assert seen and set(seen) == {None}


def test_parallel_blur_matches_single_process(tmp_path, monkeypatch) -> None:
    # 10/31 클라우드: 얼굴 찾기를 여러 프로세스로 나눠도 프레임 순서·결과가 같아야 한다
    import hashlib
    import subprocess

    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)

    def frames(p):
        return hashlib.md5(subprocess.check_output(["ffmpeg", "-loglevel", "error", "-i", str(p), "-f", "rawvideo", "-"])).hexdigest()

    one = tmp_path / "one.mp4"
    anonymize.anonymize_video(src, one)
    monkeypatch.setattr(anonymize, "BLUR_WORKERS", 2)
    monkeypatch.setattr(anonymize, "_pool", None)
    two = tmp_path / "two.mp4"
    stats = anonymize.anonymize_video(src, two)
    anonymize._pool.shutdown()
    assert stats["frames"] == 20 and frames(one) == frames(two)


def test_blur_hides_detail_only_inside_box_without_blocks() -> None:
    # 결과 영상 모자이크(11/2 흐림): 안쪽 잔무늬는 지우고, 바깥은 그대로, 네모 블록(같은 색 조각)은 없다
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    box = (100, 80, 60, 60)
    out = anonymize.blur_boxes(frame, [box])
    x, y, w, h = box
    pad = int(w * anonymize.BOX_PAD)
    assert np.array_equal(out[: y - pad], frame[: y - pad])
    inner = out[y : y + h, x : x + w].astype(float)
    assert inner.std() < frame[y : y + h, x : x + w].astype(float).std() / 4
    assert len({tuple(p) for p in out[y : y + h, x : x + w].reshape(-1, 3)}) > 100


def test_small_far_faces_are_blurred_hard_enough() -> None:
    # 11/3 사용자: 멀리 있는 심판 얼굴(13×17px 안팎)이 살짝 드러났다 — 작은 얼굴 칸은 한 덩어리로 뭉개야 한다
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    box = (150, 100, 13, 17)
    out = anonymize.blur_boxes(frame, [box], pad=0.4)
    x, y, w, h = box
    face = out[y : y + h, x : x + w].astype(float)
    assert face.std() < frame[y : y + h, x : x + w].astype(float).std() / 6  # 예전(9px 커널)은 약 1/3에 그쳤다


def test_blur_has_a_soft_round_edge_not_a_hard_square() -> None:
    # 11/3 사용자: 큰 흐림이 네모 덩어리로 보였다 → 칸 귀퉁이는 거의 원래대로, 가운데는 흐림
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)
    out = anonymize.blur_boxes(frame, [(150, 150, 100, 100)], pad=0.4)  # 칸 110~290
    corner = np.abs(out[112:118, 112:118].astype(int) - frame[112:118, 112:118].astype(int)).mean()
    center = np.abs(out[195:205, 195:205].astype(int) - frame[195:205, 195:205].astype(int)).mean()
    assert corner < center / 4, (corner, center)
