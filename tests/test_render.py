from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from tapme import render
from tapme.track import Box


def _checker(w: int = 320, h: int = 180) -> Image.Image:
    img = Image.new("RGB", (w, h))
    img.putdata([(255,) * 3 if (x // 4 + y // 4) % 2 == 0 else (0,) * 3 for y in range(h) for x in range(w)])
    return img


def test_interpolate_midpoint_and_holds() -> None:
    by = {0: [Box("9", 0, 100, 10, 200), Box("3", 50, 50, 60, 60)], 1: [Box("9", 60, 160, 70, 260)]}
    out = render.interpolate(by, sam_fps=5, out_fps=30, n_out=12)
    me = [next(b for b in f if b.object_id == "9") for f in out]
    assert (me[3].left, me[3].top, me[3].right, me[3].bottom) == (30, 130, 40, 230)
    assert me[11].left == 60  # 마지막 SAM 프레임 뒤에도 남는다
    assert all(any(b.object_id == "3" for b in f) for f in out[:6])  # 한쪽에만 있어도 가리기가 안 끊긴다


def test_zoom_crops_stay_inside_with_9_16() -> None:
    centers = [(800.0, 1500.0)] * 5 + [None] * 3 + [(100.0, 100.0)] * 5
    crops = render.zoom_crops(centers, 1080, 1920, zoom=2.0, smooth=1)
    assert all((w, h) == (540, 960) and 0 <= x <= 540 and 0 <= y <= 960 for x, y, w, h in crops)
    assert crops[0] == (530, 960, 540, 960)
    assert crops[-1] == (0, 0, 540, 960)


def test_zoom_follows_a_moving_me_at_the_first_and_last_frame() -> None:
    # S5: 오른쪽으로 달리는 '나'를 처음 프레임에서 앞질러 링이 화면 가장자리에 걸렸다
    centers = [(400.0 + 20 * i, 960.0) for i in range(30)]
    crops = render.zoom_crops(centers, 1920, 1920, zoom=2.0, smooth=10)
    for i in (0, len(centers) - 1):
        x, _, w, _ = crops[i]
        assert abs(x + w / 2 - centers[i][0]) <= 1


@pytest.mark.parametrize(("privacy", "other_blurred", "outside_blurred"), [
    ("face", False, False),
    ("silhouette", True, False),
    ("background", True, True),
])
def test_privacy_modes(privacy: str, other_blurred: bool, outside_blurred: bool) -> None:
    img = _checker()
    boxes = [Box("me", 140, 40, 180, 140), Box("other", 40, 40, 90, 140)]
    out = render.compose(img, boxes, {}, "me", (0, 0, 320, 180), (320, 180), privacy=privacy)

    def blurred(xy: tuple[int, int]) -> bool:
        return 30 < out.getpixel(xy)[0] < 225  # 흑백 바둑판이 섞여 회색이 됨

    assert out.getpixel((161, 91)) == img.getpixel((161, 91))  # '나'는 늘 선명
    assert blurred((61, 91)) is other_blurred
    assert blurred((261, 21)) is outside_blurred  # 아무도 없는 곳(SAM이 놓친 사람 자리)


def test_unknown_privacy_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        render.compose(_checker(), [], {}, "me", (0, 0, 320, 180), (320, 180), privacy="none")


@pytest.mark.skipif(not any(Path(p).exists() for p in render.FONT_CANDIDATES), reason="CJK 글꼴 없음")
def test_font_has_korean_and_japanese() -> None:
    text = "검정팀 득점!\n黒チーム得点！"
    assert render._covers(render.load_font(40, text), text)


def test_render_video_keeps_fps_and_makes_9_16(tmp_path: Path) -> None:
    src = tmp_path / "s.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    by = {k: [Box("me", 140 + k, 40, 180 + k, 140)] for k in range(5)}
    out = render.render_video(src, tmp_path / "o.mp4", by, {}, "me", sam_fps=5,
                              subtitles=[(0.5, 1.0, "블로킹! 득점!")], zoom=1.0)
    w, h, fps, n = render.probe(out)
    assert (w, h, round(fps), n) == (100, 180, 30, 30)

    # 자막은 0.5초부터만: 앞 프레임엔 배너가 없고 뒤 프레임엔 있다(배너는 어두운 띠)
    def bottom_dark(t: float) -> int:
        raw = subprocess.check_output(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1",
                                       "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        row = raw[w * (h - 20): w * (h - 19)]
        return sum(1 for v in row if v < 60)

    assert bottom_dark(0.1) < bottom_dark(0.8)


def test_background_mode_blurs_lighter_than_silhouette() -> None:
    # 10/29 사용자 테스트: '나 빼고 전부'가 반지름 14면 경기가 안 보였다
    assert render.BACKGROUND_BLUR_RADIUS < render.SILHOUETTE_BLUR_RADIUS
    assert render.BACKGROUND_BLUR_RADIUS == 6


def test_my_face_stays_clear_and_others_are_mosaicked() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    me = Box("me", 180, 20, 260, 190)
    faces = [(200, 30, 30, 30), (40, 30, 30, 30)]  # 첫째는 '나'의 박스 안, 둘째는 다른 회원
    out = render.mosaic_faces_except_me(frame, faces, me)
    assert np.array_equal(out[30:60, 200:230], frame[30:60, 200:230])
    assert not np.array_equal(out[30:60, 40:70], frame[30:60, 40:70])
    # '나'를 못 잡은 프레임은 모든 얼굴을 가린다
    assert not np.array_equal(render.mosaic_faces_except_me(frame, faces, None)[30:60, 200:230], frame[30:60, 200:230])


def test_face_mosaic_off_skips_face_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tapme import anonymize

    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=0.5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    calls: list = []
    monkeypatch.setattr(anonymize, "detect_faces", lambda *a: calls.append(1) or [])
    render.render_video(src, tmp_path / "a.mp4", {}, {}, "me", sam_fps=5, zoom=1.0, face_mosaic=False)
    assert calls == []
    render.render_video(src, tmp_path / "b.mp4", {}, {}, "me", sam_fps=5, zoom=1.0, face_mosaic=True)
    assert len(calls) == 5  # 0.5초 × 10fps
