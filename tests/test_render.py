from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tapme import render, track
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
    monkeypatch.setattr(anonymize, "detect_faces_split", lambda *a, **k: calls.append(1) or ([], []))
    render.render_video(src, tmp_path / "a.mp4", {}, {}, "me", sam_fps=5, zoom=1.0, face_mosaic=False)
    assert calls == []
    render.render_video(src, tmp_path / "b.mp4", {}, {}, "me", sam_fps=5, zoom=1.0, face_mosaic=True)
    assert len(calls) == 5  # 0.5초 × 10fps


def test_other_face_overlapping_my_body_is_mosaicked() -> None:
    # H-40: 몸 박스 전체로 '나'를 빼면 내 앞뒤로 겹친 다른 사람 얼굴이 드러났다. 머리 쪽 얼굴 하나만 뺀다
    import numpy as np

    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    me = Box("me", 180, 20, 260, 190)
    mine, other = (205, 25, 30, 30), (200, 120, 30, 30)  # 내 머리, 내 몸 앞을 지나는 다른 사람
    out = render.mosaic_faces_except_me(frame, [mine, other, (207, 27, 27, 27)], me)  # 마지막은 다른 검출기가 찾은 같은 내 얼굴
    assert np.array_equal(out[25:55, 205:235], frame[25:55, 205:235])
    assert not np.array_equal(out[120:150, 200:230], frame[120:150, 200:230])


def test_face_found_once_stays_mosaicked_for_a_moment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # H-40: 검출기가 한두 프레임 놓치면 그 순간만 모자이크가 빠졌다. 찾은 자리를 FACE_HOLD_SEC 동안 유지한다
    import cv2
    import numpy as np

    from tapme import anonymize

    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=0.5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    calls = iter(range(1000))
    monkeypatch.setattr(anonymize, "detect_faces_split", lambda *a, **k: ([(100, 40, 60, 60)] if next(calls) == 0 else [], []))
    monkeypatch.setattr(anonymize, "detect_faces_plain", lambda *a, **k: [])
    plain_out, mos_out = tmp_path / "plain.mp4", tmp_path / "m.mp4"
    render.render_video(src, plain_out, {}, {}, "me", sam_fps=5, zoom=1.0, face_mosaic=False)
    render.render_video(src, mos_out, {}, {}, "me", sam_fps=5, zoom=1.0, face_mosaic=True)

    def frames(p: Path) -> list[np.ndarray]:
        cap, out = cv2.VideoCapture(str(p)), []
        while (r := cap.read())[0]:
            out.append(r[1].astype(np.int16))
        return out

    a, b = frames(plain_out), frames(mos_out)
    hold = round(render.FACE_HOLD_SEC * 30)
    # 출력은 9:16으로 잘린 화면이라 얼굴 박스가 들어간 가운데 줄을 본다
    diff = [float(np.abs(x - y)[40:100].mean()) for x, y in zip(a, b)]
    assert all(d > 5 for d in diff[:hold]) and all(d < 5 for d in diff[hold + 2:])


def test_profile_faces_are_searched_every_other_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # H-52(11/1 사용자 결정): 옆얼굴 Haar는 2프레임마다. 빠진 프레임은 FACE_HOLD_SEC 유지로 덮인다
    from tapme import anonymize

    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=0.3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    used: list[bool] = []
    monkeypatch.setattr(anonymize, "detect_faces_split", lambda f, d, p=None, s=None: used.append(p is not None) or ([], []))
    monkeypatch.setattr(anonymize, "detect_faces_plain", lambda *a, **k: [])
    render.render_video(src, tmp_path / "m.mp4", {}, {}, "me", sam_fps=5, zoom=1.0, face_mosaic=True)
    assert len(used) == 9 and used == [i % render.PROFILE_EVERY == 0 for i in range(9)]
    assert render.PROFILE_EVERY == 2
    # 서버(여러 프로세스) 경로: 일꾼에게 옆얼굴을 찾을지 함께 넘긴다
    import numpy as np

    used.clear()
    monkeypatch.setitem(render._W, "detector", object())
    monkeypatch.setitem(render._W, "profile", object())
    monkeypatch.setitem(render._W, "plain", None)
    buf = np.zeros((4, 4, 3), np.uint8).tobytes()
    for side in (True, False):
        render._worker_faces((buf, 4, 4, (0, 0, 4, 4), (4, 4), side))
    assert used == [True, False]


def test_me_marker_glows_at_the_feet_and_points_from_above() -> None:
    # 10/30 사용자 C안: 발밑 스포트라이트 + 머리 위 삼각형
    import numpy as np
    from PIL import Image

    frame = Image.new("RGB", (400, 400), (90, 90, 90))
    me = Box("me", 170, 120, 230, 320)
    out = np.asarray(render.compose(frame, [me], {}, "me", (0, 0, 400, 400), (400, 400))).astype(int)
    base = 90
    assert np.abs(out[318:323, 150:250] - base).max() > 60  # 발밑
    above = out[60:118, 185:215]
    assert (above[..., 0] > 200).any() and (above[..., 2] < 60).any()  # 머리 위 노란 삼각형


def test_my_moving_face_never_gets_a_held_mosaic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/30 사용자: 탭한 나에게 중간중간 모자이크가 붙었다. 0.2초 유지한 얼굴에 앞 프레임의 내 얼굴이 섞여 있었다
    import cv2
    import numpy as np

    from tapme import anonymize

    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x568:rate=30:duration=0.4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    # 내가 프레임마다 오른쪽으로 8px씩 움직인다(몸 박스와 얼굴이 같이)
    by = {i: [Box(track.TAPPED_ID, 60 + 8 * i, 100, 140 + 8 * i, 400)] for i in range(12)}
    calls = iter(range(1000))
    monkeypatch.setattr(anonymize, "detect_faces_split", lambda *a, **k: ([(85 + 8 * next(calls), 105, 30, 30)], []))
    monkeypatch.setattr(anonymize, "detect_faces_plain", lambda *a, **k: [])
    plain_out, mos_out = tmp_path / "plain.mp4", tmp_path / "m.mp4"
    render.render_video(src, plain_out, by, {}, track.TAPPED_ID, sam_fps=30, zoom=1.0, face_mosaic=False)
    render.render_video(src, mos_out, by, {}, track.TAPPED_ID, sam_fps=30, zoom=1.0, face_mosaic=True)

    def frames(p: Path) -> list[np.ndarray]:
        cap, out = cv2.VideoCapture(str(p)), []
        while (r := cap.read())[0]:
            out.append(r[1].astype(np.int16))
        return out

    a, b = frames(plain_out), frames(mos_out)
    # 출력은 원본과 같은 크기(1배). 내 얼굴 둘레(앞 프레임 얼굴 자리 포함)가 모자이크 없이 같아야 한다
    for i in (4, 8):
        x = 85 + 8 * i
        assert float(np.abs(a[i][100:140, x - 30:x + 34] - b[i][100:140, x - 30:x + 34]).mean()) < 6


def test_ground_marker_goes_under_my_feet_not_over_them() -> None:
    # 10/30 사용자: 발밑 표시가 발을 덮는 느낌 → '나'의 몸(마스크) 위에는 표시를 그리지 않는다
    import numpy as np
    from PIL import Image

    me = Box(track.TAPPED_ID, 170, 120, 230, 320)
    frame = Image.new("RGB", (400, 400), (90, 90, 90))
    frame.paste((200, 30, 30), (180, 290, 220, 320))  # 빨간 신발
    rec = SimpleNamespace(mask=SimpleNamespace(width=60, height=200))
    body = [1 if 290 - 120 <= y < 320 - 120 and 10 <= x < 50 else 0 for y in range(200) for x in range(60)]
    import tapme.render as r

    real = r.decode_mask_to_raster
    r.decode_mask_to_raster = lambda m: body
    try:
        out = np.asarray(render.compose(frame, [me], {track.TAPPED_ID: rec}, track.TAPPED_ID, (0, 0, 400, 400), (400, 400)))
    finally:
        r.decode_mask_to_raster = real
    assert (out[295:318, 185:215] == (200, 30, 30)).all()  # 신발은 그대로
    assert np.abs(out[300:330, 120:170].astype(int) - 90).max() > 40  # 신발 옆 바닥에는 표시


def test_sixty_fps_source_can_be_rendered_at_thirty(tmp_path: Path) -> None:
    # 10/30: 60fps 원본은 렌더가 두 배 걸렸다. 사용자가 30fps를 고르면 프레임을 절반만 만든다
    import cv2

    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x568:rate=60:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    out = render.render_video(src, tmp_path / "o.mp4", {}, {}, "me", sam_fps=10, zoom=1.0, face_mosaic=False, max_fps=30)
    cap = cv2.VideoCapture(str(out))
    assert round(cap.get(cv2.CAP_PROP_FPS)) == 30 and 28 <= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 31
    same = render.render_video(src, tmp_path / "p.mp4", {}, {}, "me", sam_fps=10, zoom=1.0, face_mosaic=False)
    assert round(cv2.VideoCapture(str(same)).get(cv2.CAP_PROP_FPS)) == 60  # 기본은 원본 그대로


def test_side_faces_on_bodies_are_dropped_but_front_faces_kept() -> None:
    # H-48: 옆얼굴 Haar가 유니폼 등·다리를 얼굴로 잡아 모자이크가 몸에 생겼다. 사람 상자 위 35% 아래에 잡힌 옆얼굴만 뺀다
    person = Box("o1", 100, 100, 200, 400)  # 키 300 → 205부터 몸
    head, back, leg, outside = (130, 110, 30, 30), (140, 230, 30, 30), (140, 350, 30, 30), (400, 300, 30, 30)
    assert render.off_bodies([head, back, leg, outside], [person]) == [head, outside]
    assert render.off_bodies([back], []) == [back]  # 선수 위치를 모르면 그대로 가린다


def test_team_mosaic_clears_only_heads_of_that_team() -> None:
    # 11/1 사용자: 팀별 모자이크. 풀어 줄 팀 선수의 머리 자리 얼굴만 빼고, 겹치거나 모르는 얼굴은 그대로 가린다
    ours, theirs = Box("a", 100, 100, 200, 400), Box("b", 300, 100, 400, 400)
    our_head, their_head, our_back, nobody = (130, 110, 30, 30), (330, 110, 30, 30), (140, 300, 30, 30), (600, 100, 30, 30)
    faces = [our_head, their_head, our_back, nobody]
    assert render.off_team(faces, [ours, theirs], {"a"}) == [their_head, our_back, nobody]
    overlap = Box("c", 120, 90, 220, 400)  # 우리 팀 머리 자리에 다른 사람 머리도 겹치면 가린다
    assert render.off_team([our_head], [ours, overlap], {"a"}) == [our_head]
