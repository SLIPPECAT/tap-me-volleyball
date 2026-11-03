from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tapme import auto, jobs, track


def test_tap_between_overlapping_players_picks_the_nearest() -> None:
    # 상자가 겹치면 탭한 곳(90, 90)과 상자 가운데가 가장 가까운 사람
    near_box = track.Box("near", 60, 60, 160, 160)
    far_box = track.Box("far", 0, 0, 100, 100)
    pick, why = auto.pick_tapped([far_box, near_box], 90, 90, 500)
    assert pick == near_box and why == "ok"


def test_tap_just_outside_the_box_is_forgiven_but_far_is_a_miss() -> None:
    # 상자(100~160, 300~500) 너비의 10%(6px)까지는 그 사람으로 본다
    box = track.Box("0", 100, 300, 160, 500)
    assert auto.pick_tapped([box], 166, 400, 1080)[1] == "ok"
    assert auto.pick_tapped([box], 167, 400, 1080) == (None, "miss")
    assert auto.pick_tapped([], 130, 400, 1080) == (None, "miss")


def test_reference_height_boundary_is_settled() -> None:
    # 상자 높이가 화면의 12%면 기준 사진으로 쓰고, 그보다 작으면 다시 탭하게 한다
    ok_box = track.Box("ok", 0, 0, 10, 120)
    small_box = track.Box("small", 0, 0, 10, 119)
    assert auto.pick_tapped([ok_box], 5, 60, 1000)[1] == "ok"
    assert auto.pick_tapped([small_box], 5, 60, 1000)[1] == "small"


def test_not_found_is_reviewed_even_on_the_far_side() -> None:
    # far는 찾은 선수만: 못 찾았으면(-1) 먼 쪽이어도 확인 묶음으로
    assert auto.decide({"card": -1, "confidence": 0.99, "me_side": "far"}) == "review"


def _scene(tmp_path: Path, kinds: dict[str, int]) -> Path:
    from PIL import Image

    scene = tmp_path / "scene00"
    scene.mkdir(exist_ok=True)
    for kind in kinds:
        Image.new("RGB", (400, 400), (200, 200, 200)).save(scene / f"tap_{kind}.jpg")
    ref = tmp_path / "ref.jpg"
    Image.new("RGB", (160, 240), (10, 20, 30)).save(ref)
    return ref


def _usage_client(text: str, input_tokens: int = 1000, output_tokens: int = 100) -> SimpleNamespace:
    def create(**kw):
        create.sent = kw
        return SimpleNamespace(output_text=text, usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))

    return SimpleNamespace(responses=SimpleNamespace(create=create))


def test_ask_frame_parses_the_answer_and_counts_the_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tapme import muse

    ref = _scene(tmp_path, {"score": 2})
    monkeypatch.setattr(jobs, "player_cards", lambda job, i, kind, c: [[10, 10, 100, 200], [150, 150, 250, 350]])
    client = _usage_client('{"index": 1, "confidence": 0.85, "reason": "초록 10번"}')
    job = SimpleNamespace(dir=tmp_path, spent_usd=0.0, charge=lambda usd: None)
    ans, usd = auto._ask_frame(job, 0, "score", ref, "10번", client)
    assert (ans["card"], ans["confidence"], ans["reason"]) == (1, 0.85, "초록 10번")
    assert usd == pytest.approx(1000 * muse.INPUT_USD_PER_TOKEN + 100 * muse.OUTPUT_USD_PER_TOKEN)
    assert "기준 선수 정보: 10번" in client.responses.create.sent["input"][0]["content"][0]["text"]
    # 긴 이유는 200자로 자른다
    long_client = _usage_client('{"index": 0, "confidence": 0.9, "reason": "' + "가" * 250 + '"}')
    ans, _ = auto._ask_frame(job, 0, "score", ref, "", long_client)
    assert len(ans["reason"]) == 200


def test_ask_frame_without_players_or_answer_marks_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = _scene(tmp_path, {"score": 2})
    job = SimpleNamespace(dir=tmp_path, spent_usd=0.0, charge=lambda usd: None)
    monkeypatch.setattr(jobs, "player_cards", lambda job, i, kind, c: [])
    client = _usage_client('{"index": 0}')
    ans, usd = auto._ask_frame(job, 0, "score", ref, "", client)
    assert (ans, usd) == ({"card": -1, "confidence": 0.0, "reason": "선수를 찾지 못함"}, 0.0)
    assert not hasattr(client.responses.create, "sent")  # 물을 선수가 없으면 API를 부르지 않는다
    # 답을 읽지 못하거나 없는 번호를 답하면 못 찾음으로. 낸 돈은 센다
    monkeypatch.setattr(jobs, "player_cards", lambda job, i, kind, c: [[10, 10, 100, 200]] * 2)
    ans, usd = auto._ask_frame(job, 0, "score", ref, "", _usage_client("no json here"))
    assert (ans["card"], ans["reason"], usd > 0) == (-1, "답을 읽지 못함", True)
    ans, _ = auto._ask_frame(job, 0, "score", ref, "", _usage_client('{"index": 9, "confidence": 0.9}'))
    assert ans["card"] == -1
    ans, _ = auto._ask_frame(job, 0, "score", ref, "", _usage_client('{"index": 0}'))
    assert (ans["card"], ans["confidence"], ans["reason"]) == (0, 0.0, "")


def _find_me_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: dict[int, str]
) -> tuple[SimpleNamespace, SimpleNamespace, list[float]]:
    sizes = {"score": 2, "b2": 3, "start": 4}  # 순간마다 후보 수를 달리해 어느 질문인지 안다
    _scene(tmp_path, sizes)
    kinds: list[str] = []
    monkeypatch.setattr(jobs, "player_cards",
                        lambda job, i, kind, c: kinds.append(kind) or [[10, 10, 100, 200]] * sizes[kind])

    def create(**kw):
        n = sum(c["type"] == "input_image" for c in kw["input"][0]["content"]) - 1  # 기준 사진 빼고
        return SimpleNamespace(output_text=replies[n], usage=SimpleNamespace(input_tokens=1000, output_tokens=100))

    charged: list[float] = []
    job = SimpleNamespace(dir=tmp_path, spent_usd=0.0, charge=charged.append, kinds=kinds)
    return job, SimpleNamespace(responses=SimpleNamespace(create=create)), charged


def test_find_me_prefers_the_earlier_moment_on_a_tie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 득점 순간과 구간 시작이 같은 확신이면 앞 순간(득점 순간). 못 찾은 순간(-1)은 확신이 높아도 지지 않는다
    job, client, charged = _find_me_setup(
        tmp_path, monkeypatch, {2: '{"index": 1, "confidence": 0.9}', 3: '{"index": -1, "confidence": 0.99}',
                                 4: '{"index": 1, "confidence": 0.9}'})
    found = auto.find_me(job, 0, tmp_path / "ref.jpg", "10번", client)
    assert (found["frame"], found["card"], found["confidence"]) == ("score", 1, 0.9)
    assert job.kinds[:3] == ["score", "b2", "start"]  # 선수 찾기는 먼저 차례로
    assert charged == pytest.approx([auto.USD_PER_QUESTION * len(auto.FRAMES)])
    assert job.spent_usd == pytest.approx(3 * (1000 * 1.25e-6 + 100 * 4.25e-6))


def test_find_me_reports_not_found_when_every_moment_misses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job, client, _ = _find_me_setup(
        tmp_path, monkeypatch, {2: '{"index": -1, "confidence": 0.5}', 3: '{"index": -1, "confidence": 0.9}',
                                 4: '{"index": -1, "confidence": 0.7}'})
    found = auto.find_me(job, 0, tmp_path / "ref.jpg", "", client)
    assert (found["frame"], found["card"], found["confidence"]) == ("b2", -1, 0.9)
    assert found["frames"] == {"score": [-1, 0.5], "b2": [-1, 0.9], "start": [-1, 0.7]}


def test_tap_check_is_refused_when_the_service_budget_is_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    import numpy as np

    monkeypatch.setattr(jobs, "total_spent", lambda: jobs.MAX_USD_TOTAL)
    jpg = cv2.imencode(".jpg", np.zeros((100, 100, 3), np.uint8))[1].tobytes()
    with pytest.raises(jobs.BudgetExceeded):
        auto.check_tap(jpg, 50, 50, client=None)


def _reference_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boxes: list[track.Box]
) -> tuple[SimpleNamespace, dict]:
    from PIL import Image

    job_dir = tmp_path / "job"
    job_dir.mkdir(exist_ok=True)
    seen: dict = {}

    def fake_cut(video, start, duration, out, fps):
        seen.update(video=video, start=start, duration=duration, out=out, fps=fps)
        Path(out).write_bytes(b"clip")
        return out

    def fake_run(cmd, **kw):
        Image.new("RGB", (1920, 1080), (200, 200, 200)).save(cmd[-1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jobs, "_cut_blurred", fake_cut)
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(track, "detect_players", lambda c, frame: boxes)
    job = SimpleNamespace(dir=job_dir, video=str(tmp_path / "src.mp4"), spent_usd=0.0,
                          charge=lambda usd: seen.setdefault("charged", []).append(usd))
    return job, seen


def test_make_reference_reuses_the_saved_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job, seen = _reference_setup(tmp_path, monkeypatch, [])
    ref = job.dir / "auto" / "ref.jpg"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(b"card")
    assert auto.make_reference(job, 3.0, 130, 400, None) == ref
    assert ref.read_bytes() == b"card" and seen == {}  # 다시 자르지도 묻지도 않는다


def test_make_reference_cuts_the_tapped_player(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    job, seen = _reference_setup(tmp_path, monkeypatch, [track.Box("0", 100, 300, 160, 500)])
    ref = auto.make_reference(job, 3.0, 130, 400, None)
    assert (seen["start"], seen["duration"], seen["fps"]) == (3.0, 1.0, 10)  # 탭한 순간 1초를 10fps로
    assert seen["charged"] == [track.USD_PER_IMAGE] and job.spent_usd == track.USD_PER_IMAGE
    assert Image.open(ref).size == (78, 232)  # 상자 + 여유 자리를 카드로
    assert not (job.dir / "auto" / "ref_frame.jpg").exists() and not (job.dir / "auto" / "ref_blur.mp4").exists()


def test_make_reference_refuses_a_missed_or_too_small_tap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job, _ = _reference_setup(tmp_path, monkeypatch, [])
    with pytest.raises(jobs.TapMissed, match="선수를 찾지 못했어요"):
        auto.make_reference(job, 3.0, 130, 400, None)
    job, _ = _reference_setup(tmp_path, monkeypatch, [track.Box("0", 100, 300, 160, 500)])
    with pytest.raises(jobs.TapMissed) as e:
        auto.make_reference(job, 3.0, 1500, 800, None)
    assert str(e.value) == auto.TAP_MESSAGES["miss"]
    job, _ = _reference_setup(tmp_path, monkeypatch, [track.Box("1", 900, 400, 930, 480)])
    with pytest.raises(jobs.TapMissed) as e:
        auto.make_reference(job, 3.0, 915, 440, None)
    assert str(e.value) == auto.TAP_MESSAGES["small"]
    assert not (job.dir / "auto" / "ref.jpg").exists()  # 못 고르면 기준 사진을 남기지 않는다


def test_tap_size_uses_the_short_side_so_portrait_videos_are_not_penalized() -> None:
    # 11/3: 화면 높이로 재면 세로 영상(1080×1920)은 같은 크기의 선수도 '너무 작아요'였다 → 짧은 변 기준(가로는 그대로)
    box = track.Box("p", 500, 800, 560, 950)  # 키 150px
    assert auto.pick_tapped([box], 530, 900, min(1080, 1920))[1] == "ok"  # 세로 1080×1920: 150 ≥ 0.12×1080
    assert auto.pick_tapped([box], 530, 900, 1920)[1] == "small"  # 예전처럼 높이(1920)로 재면 작다고 했다
    assert auto.pick_tapped([box], 530, 900, min(1920, 1080))[1] == "ok"  # 가로 1080p는 전과 같다


def test_tap_check_returns_every_player_box_so_the_screen_can_show_them(monkeypatch: pytest.MonkeyPatch) -> None:
    # 11/3 사용자: 처음 쓸 때 탭해도 인식이 안 된다는 느낌(오늘 시작 전 탭 확인 6~22번). 장면을 고르면 사람을 먼저 찾아
    # 테두리로 보여 주고, 탭은 화면이 그 테두리로 바로 맞춘다 — 탭할 때마다 서버에 다시 묻지 않는다
    import cv2
    import numpy as np

    monkeypatch.setattr(jobs, "total_spent", lambda: 0.0)
    monkeypatch.setattr(jobs, "_add_spend", lambda usd: None)
    monkeypatch.setattr(auto.anonymize, "load_detector", lambda: None)
    monkeypatch.setattr(auto.anonymize, "detect_faces", lambda frame, det: [])
    monkeypatch.setattr(track, "detect_players", lambda c, p: [track.Box("a", 100, 300, 160, 500), track.Box("b", 900, 400, 930, 480)])
    jpg = cv2.imencode(".jpg", np.zeros((1080, 1920, 3), np.uint8))[1].tobytes()
    seen = auto.check_tap(jpg, None, None, client=None)
    assert seen["boxes"] == [{"l": 100, "t": 300, "r": 160, "b": 500, "ok": True},
                             {"l": 900, "t": 400, "r": 930, "b": 480, "ok": False}]  # 80px < 0.12×1080 → 너무 작음
    assert "ok" not in seen  # 탭 없이 물으면 판정은 없다
    assert auto.check_tap(jpg, 130, 400, client=None)["ok"] is True  # 예전처럼 탭과 같이 물어도 된다


def test_reference_keeps_the_chosen_player_when_the_tap_point_misses_on_the_server(tmp_path: Path,
                                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    # 11/3 21:30: 화면 확인은 통과했는데 서버가 다시 찾은 상자가 조금 어긋나 '탭한 곳에서 선수를 찾지 못했어요'로 끝났다.
    # 화면에서 고른 테두리를 같이 받아, 탭한 점이 빗나가도 그 테두리와 가장 많이 겹치는 사람을 고른다
    shifted = track.Box("0", 120, 300, 180, 500)  # 화면 테두리(100~160)보다 20px 옆 — 탭한 점(110)은 빗나간다
    job, _ = _reference_setup(tmp_path, monkeypatch, [shifted, track.Box("1", 600, 300, 660, 500)])
    assert auto.make_reference(job, 3.0, 110, 400, None, box=[100, 300, 160, 500]).exists()
    (tmp_path / "far").mkdir()
    job, _ = _reference_setup(tmp_path / "far", monkeypatch, [track.Box("1", 600, 300, 660, 500)])
    with pytest.raises(jobs.TapMissed):  # 겹치는 사람이 없으면 엉뚱한 사람을 고르지 않는다
        auto.make_reference(job, 3.0, 110, 400, None, box=[100, 300, 160, 500])
