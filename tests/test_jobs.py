from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tapme import jobs, muse, score, track
from tests.test_track import _MASK_A, WIRE  # 320×334 화면, 선수 0·1


def _video(path: Path, fps: int, seconds: float = 1.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x334:rate={fps}:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> jobs.Job:
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    src5, src30 = _video(tmp_path / "w5.mp4", 5), _video(tmp_path / "w30.mp4", 30)

    def fake_cut(video: Path, start: float, duration: float, out: Path, fps: int | None) -> Path:
        shutil.copy(src5 if fps else src30, out)
        return out

    monkeypatch.setattr(jobs, "_cut_blurred", fake_cut)
    # 결과 영상용 클립은 흐리지 않고 원본에서 자른다(얼굴은 렌더할 때 '나'만 빼고 가림)
    monkeypatch.setattr(jobs, "_cut", fake_cut)
    # 탭할 장면의 선수 찾기(SAM 사진): 오른쪽·왼쪽 선수 둘
    monkeypatch.setattr(track, "detect_players", lambda c, img: [track.Box("7", 211, 228, 271, 255),
                                                                 track.Box("2", 155, 228, 203, 255)])
    j = jobs.create_job("m.mp4", b"x")
    j.rallies = [{"index": 0, "start_sec": 5.0, "end_sec": 13.0, "clip": str(src5), "clip_start_sec": 4.0}]
    return j


def _reply(usd: float = 0.01) -> muse.MuseReply:
    return muse.MuseReply(text="", input_tokens=0, output_tokens=0, usd=usd, seconds=0)


def test_budget_cap_stops_paid_step(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "MAX_USD_PER_JOB", 0.01)
    with pytest.raises(jobs.BudgetExceeded):
        jobs.prepare_scene(job, 0, client=None)


def test_prepare_scene_window_is_in_source_time(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(score, "ask_score_time", lambda c, clip: ({"score_time_sec": 8.5}, _reply()))
    scene = jobs.prepare_scene(job, 0, client=None)
    # 클립은 원본 4.0초에서 시작 → 득점 12.5초 → 7초 전~5초 뒤
    assert scene["score_sec"] == 12.5
    assert scene["window"] == [3.5, 17.5]  # 서브(5.0) 1.5초 전 ~ 득점(12.5) 5초 뒤 — 랠리 전체(11/2)
    assert scene["status"] == "awaiting_tap"
    # 탭할 장면 5장: 득점 순간·1.5/3/4.5초 전·구간 시작. 가짜 SAM 클립이 5프레임뿐이라 앞 셋은 마지막 프레임
    assert scene["tap_frames"] == {"score": 4, "b1": 4, "b2": 4, "b3": 4, "start": 0}
    assert all((Path(scene["sam_clip"]).parent / f"tap_{k}.jpg").exists() for k in jobs.TAP_KINDS)
    # 선수 카드는 아직 안 만든다(사용자가 장면을 열 때). 득점 시점(Muse $0.01)만 썼다
    assert scene["players"] == {}
    assert jobs.load_job(job.id).spent_usd == pytest.approx(0.01)


def test_player_cards_are_made_once_per_opened_scene(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(score, "ask_score_time", lambda c, clip: ({"score_time_sec": 8.5}, _reply()))
    jobs.prepare_scene(job, 0, client=None)
    calls: list = []
    real = track.detect_players
    monkeypatch.setattr(track, "detect_players", lambda c, img: calls.append(img.name) or real(c, img))
    assert jobs.player_cards(job, 0, "b2", client=None) == [[155, 228, 203, 255], [211, 228, 271, 255]]
    assert jobs.player_cards(job, 0, "b2", client=None) == [[155, 228, 203, 255], [211, 228, 271, 255]]
    assert calls == ["tap_b2.jpg"]  # 두 번째는 다시 찾지 않는다
    assert (Path(job.scenes["0"]["sam_clip"]).parent / "player_b2_01.jpg").exists()
    assert jobs.load_job(job.id).spent_usd == pytest.approx(0.01 + track.USD_PER_IMAGE)


# '나'(점 프롬프트 결과, id 0)가 프레임 0·1에 있다. WIRE의 1번은 다른 선수
_ME = "\n".join(line.split(",1<")[0] for line in WIRE.split("\n"))


def _prepared(job: jobs.Job, monkeypatch: pytest.MonkeyPatch, sam: list) -> None:
    monkeypatch.setattr(score, "ask_score_time", lambda c, clip: ({"score_time_sec": 8.5}, _reply()))
    jobs.prepare_scene(job, 0, client=None)

    def fake_track(c, clip, prompt="volleyball player"):
        sam.append(prompt)
        return (WIRE if prompt == "volleyball player" else _ME), {"video_frames_processed": 50}, 0.0

    monkeypatch.setattr(track, "track_players", fake_track)

    def fake_commentary(c, clip, team, t, role, sec, winner="mine"):
        sam.append(f"muse:{role}")
        return [{"t": 1.0, "ko": "서브!", "ja": "サーブ!", "en": "Serve!"}, {"t": t, "ko": "득점!", "ja": "得点!", "en": "Point!"}], _reply()

    monkeypatch.setattr(score, "ask_commentary", fake_commentary)


def test_tap_sends_points_from_both_scenes_and_keeps_visible_span(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    sam: list = []
    _prepared(job, monkeypatch, sam)
    scene = jobs.make_highlight(job, 0, [("score", 240, 240), ("start", 175, 240)], "검정", [], "", "face", 2.0,
                                client=None)
    # 두 장면의 탭이 한 선수의 점으로 간다(득점 순간 = SAM 프레임 4, 시작 = 0)
    assert "<4f>0<|point;x=240;y=240" in sam[0] and "<0f>0<|point;x=175;y=240" in sam[0]
    # '나' 추적 + 다른 선수 추적(몸 위에 잡힌 옆얼굴 모자이크를 거르려고, H-48). 자막을 끄면 해설(Muse)은 없다
    assert len(sam) == 2 and sam[1] == "volleyball player"
    # '나'는 SAM 프레임 0·1(5.5~5.7초)에만 있다. 영상은 그 구간과 최소 득점 4초 전~4초 뒤(8.5~16.5초)를 모두 담는다
    # (득점 시각이 ±4초 어긋나도 실제 득점이 들어가게, H-43)
    assert scene["span_frames"] == [0, 1] and scene["clip"] == [3.5, 16.5]
    assert scene["status"] == "done" and Path(scene["video"]).exists()
    assert scene["clip_is_original"] and scene["options"]["face_mosaic"] is True


def test_any_scene_can_be_tapped_and_at_least_one_is_needed(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    sam: list = []
    _prepared(job, monkeypatch, sam)
    with pytest.raises(jobs.TapMissed):
        jobs.make_highlight(job, 0, [], "검정", ["ko"], "", "face", 2.0, client=None)
    # 득점 순간에 내가 안 보여도 구간 시작 장면만으로 만든다(10/30 사용자 피드백)
    scene = jobs.make_highlight(job, 0, [("start", 175, 240)], "검정", [], "", "face", 2.0, client=None)
    assert "<0f>0<|point;x=175;y=240" in sam[0] and scene["status"] == "done"


def test_tap_on_nobody_asks_to_tap_again(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepared(job, monkeypatch, [])
    monkeypatch.setattr(track, "track_players", lambda c, clip, prompt="": ("", {}, 0.0))
    with pytest.raises(jobs.TapMissed):
        jobs.make_highlight(job, 0, [("score", 5, 5)], "검정", ["ko"], "", "face", 2.0, client=None)
    assert job.scenes["0"]["status"] == "awaiting_tap"


def test_rerender_pays_only_when_something_new_is_needed(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    _prepared(job, monkeypatch, calls)
    jobs.make_highlight(job, 0, [("score", 240, 240)], "검정", [], "", "face", 2.0, client=None)
    assert calls == [calls[0], "volleyball player"]  # SAM '나' + 다른 선수(H-48) 한 번씩, 자막을 껐으니 Muse 없음
    spent = jobs.load_job(job.id).spent_usd

    jobs.rerender(job, 0, [], "", "background", 1.0)  # 비용 없음
    assert len(calls) == 2 and jobs.load_job(job.id).spent_usd == spent
    jobs.rerender(job, 0, ["ko"], "assist", "face", 2.0)  # 자막을 처음 켬 → 해설 한 번
    jobs.rerender(job, 0, ["ja", "en"], "assist", "face", 2.0)  # 언어만 바꿈 → 다시 묻지 않음
    jobs.rerender(job, 0, ["ko"], "defense", "face", 2.0)  # 역할을 바꿈 → 다시 묻는다
    scene = jobs.rerender(job, 0, ["ko"], "defense", "silhouette", 2.0)  # 다른 선수 추적은 처음 만들 때 저장해 둔 것
    jobs.rerender(job, 0, ["en"], "defense", "silhouette", 1.5)
    assert calls[2:] == ["muse:assist", "muse:defense"]
    assert scene["options"] == {"languages": ["en"], "role": "defense", "privacy": "silhouette", "zoom": 1.5,
                                "face_mosaic": True, "fps": 0, "mosaic_team": "all", "face_scale": 2.0}
    scene = jobs.rerender(job, 0, ["en"], "defense", "face", 1.5, face_mosaic=False)  # 모자이크 끄기도 비용 없음
    assert scene["options"]["face_mosaic"] is False and len(calls) == 4
    # 얼굴 찾기 배율(11/2)도 비용 없이 바꿔 다시 렌더하고, 렌더에 그대로 넘긴다
    from tapme import render

    seen: list = []
    real = render.render_video
    monkeypatch.setattr(render, "render_video", lambda *a, **k: (seen.append(k.get("face_scale")), real(*a, **k))[1])
    scene = jobs.rerender(job, 0, ["en"], "defense", "face", 1.5, face_scale=1.0)
    assert seen == [1.0] and scene["options"]["face_scale"] == 1.0 and len(calls) == 4


def test_team_mosaic_asks_two_moments_and_clears_only_agreed_team(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    # 11/1 사용자: 상대 팀만(또는 우리 팀만) 모자이크. 다른 선수마다 떨어진 두 순간을 따로 물어 같은 답일 때만 팀을 정한다
    # (한 번만 물으면 겹친 카드에서 약 5% 틀렸다, H-49). 틀리면 얼굴이 드러나니 다르면 모름 = 가린다
    from tapme import render

    _prepared(job, monkeypatch, [])
    jobs.make_highlight(job, 0, [("score", 240, 240)], "초록", [], "", "face", 2.0, client=None)
    scene_dir = job.dir / "scene00"
    box = "<|box;x1={};y1=228;x2={};y2=254;w=320;h=334|>"
    # 두 선수 모두 0프레임과 4프레임에 잡힘(시험 영상은 5프레임이라 두 순간 간격을 줄인다) — 5번이 더 크다
    monkeypatch.setattr(jobs, "TEAM_GAP_SEC", 0.3)
    (scene_dir / "others_wire.txt").write_text("\n".join(
        f"<{k}f>5{box.format(150, 280)}{_MASK_A},6{box.format(10, 60)}{_MASK_A}" for k in (0, 4)))
    asked: list = []
    replies = iter(['{"teams": {"0": "same", "1": "other"}}', '{"teams": {"0": "same", "1": "other"}}',
                    '{"teams": {"0": "same", "1": "other"}}', '{"teams": {"0": "same", "1": "same"}}'])

    def create(**kw):
        asked.append(kw["input"][0]["content"])
        return SimpleNamespace(output_text=next(replies), usage=SimpleNamespace(input_tokens=1000, output_tokens=100))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    cleared: list = []
    real = render.render_video
    monkeypatch.setattr(render, "render_video", lambda *a, **k: (cleared.append(k.get("clear_ids")), real(*a, **k))[1])
    spent = jobs.load_job(job.id).spent_usd

    scene = jobs.rerender(job, 0, [], "", "face", 2.0, client, mosaic_team="opponent")
    assert scene["teams"] == {"5": "mine", "6": "opponent"} and len(asked) == 2
    assert "초록" in asked[0][0]["text"] and asked[0][1]["text"] == "나"  # 우리 팀 이름과 '나' 카드가 먼저
    assert cleared[-1] == {"5"} and scene["options"]["mosaic_team"] == "opponent"
    assert jobs.load_job(job.id).spent_usd > spent
    jobs.rerender(job, 0, [], "", "face", 2.0, client, mosaic_team="mine")  # 팀은 저장해 둔 것(다시 묻지 않음)
    assert len(asked) == 2 and cleared[-1] == {"6"}
    jobs.rerender(job, 0, [], "", "face", 2.0, client, mosaic_team="all")
    assert cleared[-1] is None

    scene.pop("teams")  # 두 순간 답이 다르면(한 선수: other ↔ same) 모름 → 늘 가린다
    scene = jobs.rerender(job, 0, [], "", "face", 2.0, client, mosaic_team="mine")
    assert scene["teams"] == {"5": "mine", "6": "unknown"} and cleared[-1] == set()


def test_finding_me_needs_only_three_blurred_stills_and_the_scene_reuses_them(
        job: jobs.Job, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # H-49 뒤: 나 찾기에 장면 전체(12초) 흐리기를 먼저 해 랠리당 약 2분 → 탭할 순간 사진만 흐려 묻고, 만들 때 장면 전체
    job.video = str(_video(tmp_path / "source.mp4", 30, seconds=20.0))
    monkeypatch.setattr(score, "ask_score_time", lambda c, clip: ({"score_time_sec": 8.5}, _reply()))
    blurred: list[int] = []
    real_pixelate = jobs.anonymize.pixelate_boxes
    monkeypatch.setattr(jobs.anonymize, "pixelate_boxes", lambda f, b, **k: (blurred.append(len(b)), real_pixelate(f, b, **k))[1])
    scene = jobs.prepare_frames(job, 0, client=None)
    assert scene["status"] == "frames_ready" and len(blurred) == len(jobs.TAP_KINDS)  # 탭할 순간 사진 모두 흐리기를 거쳤다
    assert all((job.dir / "scene00" / f"tap_{k}.jpg").exists() for k in jobs.TAP_KINDS)
    assert not Path(scene["sam_clip"]).exists()  # 장면 전체는 아직
    assert scene["tap_frames"]["score"] == round((scene["score_sec"] - scene["window"][0]) * jobs.SAM_FPS)
    jobs.player_cards(job, 0, "score", client=None)  # 나 찾기에서 고른 선수 카드
    job.scenes["0"]["status"] = "queued"  # 한 번에 맡기기가 만들기로 예약
    again = jobs.prepare_scene(job, 0, client=None)
    assert Path(again["sam_clip"]).exists() and again["status"] == "queued"
    assert again["tap_frames"] == scene["tap_frames"] and again["players"]["score"]  # 같은 사진·카드(추적 프롬프트가 맞게)
    assert again["window"] == scene["window"] and again["score_sec"] == scene["score_sec"]


def test_load_job_rejects_path_tricks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    with pytest.raises(KeyError):
        jobs.load_job("../etc")


def test_find_rallies_keeps_rejected_candidates_for_the_user(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    from tapme import rally, rally_filter, scan

    monkeypatch.setattr(jobs, "RALLY_MODE", "whistle")
    monkeypatch.setattr(scan, "duration_sec", lambda v: 30.0)
    monkeypatch.setattr(rally, "load_audio", lambda v: __import__("numpy").zeros(16000))
    monkeypatch.setattr(rally, "find_whistles", lambda a: [])
    monkeypatch.setattr(rally, "segments", lambda w, d=None: [rally.RallyCandidate(1, 9), rally.RallyCandidate(20, 28)])
    seen: list = []

    def fake_check(c, v, cands, d, progress):
        progress(1, 2)  # 화면에 "랠리 1/2 확인 중" 같은 진행을 알린다
        seen.append(jobs.load_job(job.id).progress)
        return [
            {"start_sec": 1, "end_sec": 9, "is_rally": True, "usd": 0.006, "clip": "a", "input_tokens": 1, "output_tokens": 1},
            {"start_sec": 20, "end_sec": 28, "is_rally": False, "usd": 0.006, "clip": "b", "input_tokens": 1, "output_tokens": 1}]

    monkeypatch.setattr(rally_filter, "check_candidates", fake_check)
    made: list = []
    monkeypatch.setattr(jobs, "make_preview", lambda j, i: made.append(i))
    jobs.find_rallies(job, SimpleNamespace())
    assert made == [0, 1]  # 후보마다 장면 사진·미리보기
    assert seen == [{"step": "rallies", "done": 1, "total": 2}]
    assert jobs.load_job(job.id).progress == {}  # 끝나면 비운다
    saved = jobs.load_job(job.id)
    assert saved.status == "rallies_ready"
    assert [r["is_rally"] for r in saved.rallies] == [True, False]
    assert saved.spent_usd == pytest.approx(0.012)


def _scene_video(path: Path, seconds: int, color: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:size=90x160:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_reel_joins_scenes_around_each_score(job: jobs.Job, tmp_path: Path) -> None:
    # 장면 0: 창 5.5~15.5초(영상 10초), 득점 12.5초 → 창 안 7.0초 → 2.0~10.0초(뒤 4초가 영상 끝에서 잘림)
    # 장면 1: 창 0~6초(득점이 영상 앞쪽이라 앞이 잘림), 득점 3.0초 → 0~6.0초를 씀
    job.scenes = {
        "0": {"status": "done", "score_sec": 12.5, "window": [5.5, 15.5], "video": str(_scene_video(tmp_path / "a.mp4", 10, "red"))},
        "1": {"status": "done", "score_sec": 3.0, "window": [0.0, 6.0], "video": str(_scene_video(tmp_path / "b.mp4", 6, "blue"))},
    }
    probe = __import__("tapme.render", fromlist=["probe"]).probe
    out = jobs.make_reel(job, [1, 0], "short")
    w, h, fps, n = probe(Path(out["video"]))
    assert (w, h, round(fps)) == (90, 160, 30)
    assert abs(n / fps - 14.0) < 0.2  # 6초 + 8초
    assert out["order"] == [1, 0] and out["style"] == "short"  # 사용자가 고른 순서
    assert jobs.load_job(job.id).reel["status"] == "done"
    # 기본은 랠리 전체(11/2 사용자: 하이라이트와 같은 길이): 장면 영상을 자르지 않고 잇는다 → 6초 + 10초
    out = jobs.make_reel(job, [1, 0])
    w, h, fps, n = probe(Path(out["video"]))
    assert abs(n / fps - 16.0) < 0.2 and out["style"] == "full"


def test_reel_needs_finished_scenes(job: jobs.Job) -> None:
    job.scenes = {"0": {"status": "awaiting_tap"}}
    with pytest.raises(ValueError):
        jobs.make_reel(job, [0])


def test_reel_without_audio(job: jobs.Job, tmp_path: Path) -> None:
    silent = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:size=90x160:rate=30:duration=4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)], check=True)
    job.scenes = {"0": {"status": "done", "score_sec": 2.0, "window": [0.0, 4.0], "video": str(silent)},
                  "1": {"status": "done", "score_sec": 2.0, "window": [0.0, 4.0], "video": str(silent)}}
    assert Path(jobs.make_reel(job, [0, 1])["video"]).exists()


def test_preview_is_a_small_clip_and_middle_thumbnail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    src = _video(tmp_path / "src.mp4", 30, seconds=6)  # 가로 320×334
    with src.open("rb") as f:
        j = jobs.create_job("m.mp4", f)
    j.rallies = [{"index": 0, "start_sec": 2.0, "end_sec": 4.0}]
    thumb, preview = jobs.make_preview(j, 0)
    from tapme import render

    w, h, _, n = render.probe(preview)
    assert min(w, h) <= 360  # 짧은 변 360 이하(작은 영상은 키우지 않음)
    assert abs(n / 30 - 4.0) < 0.2  # 앞뒤 1초를 붙인 1~5초
    assert thumb.exists() and thumb.suffix == ".jpg"
    # 두 번째 부를 때는 다시 만들지 않는다
    mtime = preview.stat().st_mtime
    assert jobs.make_preview(j, 0)[1].stat().st_mtime == mtime


def test_find_rallies_merges_a_rally_split_by_a_middle_whistle(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    from tapme import rally, rally_filter, scan

    monkeypatch.setattr(jobs, "RALLY_MODE", "whistle")
    monkeypatch.setattr(scan, "duration_sec", lambda v: 30.0)
    monkeypatch.setattr(rally, "load_audio", lambda v: __import__("numpy").zeros(16000))
    monkeypatch.setattr(rally, "find_whistles", lambda a: [])
    monkeypatch.setattr(rally, "segments", lambda w, d=None: [rally.RallyCandidate(3.0, 16.75), rally.RallyCandidate(16.75, 26.5)])
    monkeypatch.setattr(rally_filter, "check_candidates", lambda c, v, cands, d, progress: [
        {"start_sec": 3.0, "end_sec": 16.75, "is_rally": True, "confidence": 0.95, "usd": 0.006, "clip": "a",
         "input_tokens": 1, "output_tokens": 1},
        {"start_sec": 16.75, "end_sec": 26.5, "is_rally": True, "confidence": 0.9, "usd": 0.006, "clip": "b",
         "input_tokens": 1, "output_tokens": 1}])
    cut: list = []
    monkeypatch.setattr(rally_filter, "blurred_clip", lambda v, s, e, out: cut.append((s, e)) or out)
    monkeypatch.setattr(jobs, "make_preview", lambda j, i: None)
    jobs.find_rallies(job, SimpleNamespace())
    saved = jobs.load_job(job.id)
    assert [(r["start_sec"], r["end_sec"], r["parts"]) for r in saved.rallies] == [(3.0, 26.5, [0, 1])]
    assert cut == [(3.0, 26.5)] and saved.rallies[0]["clip_start_sec"] == 2.0  # 득점 시점은 랠리 전체에서 찾는다


def test_unreadable_commentary_is_retried_then_skipped(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    _prepared(job, monkeypatch, calls)
    answers = iter([None, None])
    monkeypatch.setattr(score, "ask_commentary", lambda *a: (next(answers), _reply(0.01)))
    spent = jobs.load_job(job.id).spent_usd
    scene = jobs.make_highlight(job, 0, [("score", 240, 240)], "검정", ["ko"], "", "face", 2.0, client=None)
    assert scene["status"] == "done" and scene["commentary"] == [] and scene["commentary_failed"]
    # 두 번 다 실패해도 낸 돈(SAM '나' 0.01 + 다른 선수 추적 + Muse 2회)은 센다
    others = track.estimate_usd(50)
    assert jobs.load_job(job.id).spent_usd == pytest.approx(spent + 0.01 + others + 0.02)


def test_picking_a_player_card_sends_that_players_box(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    sam: list = []
    _prepared(job, monkeypatch, sam)
    jobs.player_cards(job, 0, "score", client=None)
    jobs.make_highlight(job, 0, [("score", 1)], "검정", [], "", "face", 2.0, client=None)
    # 득점 순간(SAM 프레임 4)의 1번 카드 = 오른쪽 선수 박스
    assert "<4f>0<|box;x1=211;y1=228" in sam[0]
    with pytest.raises(jobs.TapMissed):
        jobs.make_highlight(job, 0, [("score", 9)], "검정", [], "", "face", 2.0, client=None)


def test_sam_clip_frames_line_up_with_render_clip(tmp_path: Path) -> None:
    # S1: SAM 클립(10fps)의 프레임 k는 원본 클립(30fps)의 프레임 3k여야 링·확대가 '나'를 제때 따라간다
    import cv2
    import numpy as np

    def gray(path: Path) -> list[np.ndarray]:
        cap, out = cv2.VideoCapture(str(path)), []
        while (r := cap.read())[0]:
            out.append(cv2.cvtColor(r[1], cv2.COLOR_BGR2GRAY).astype(np.float32))
        return out

    src = _video(tmp_path / "src.mp4", 30, seconds=4.0)  # testsrc2는 프레임마다 그림이 움직인다
    sam = gray(jobs._cut(src, 1.0, 2.0, tmp_path / "sam.mp4", 10))
    clip = gray(jobs._cut(src, 1.0, 2.0, tmp_path / "clip.mp4", None))

    def err(shift: int) -> float:
        return float(np.mean([np.abs(s - clip[3 * k + shift]).mean()
                              for k, s in enumerate(sam) if 0 <= 3 * k + shift < len(clip)]))

    assert min(range(-6, 7), key=err) == 0


def test_find_rallies_scans_the_video_and_prepare_uses_the_found_point(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    from tapme import scan

    monkeypatch.setattr(scan, "duration_sec", lambda v: 3600.0)
    seen: list = []

    def fake_scan(c, v, duration, d, progress, offset):
        assert (duration, offset) == (1800.0, 0.0)  # 구간 없이 온 1시간 영상은 앞 30분만 훑는다
        progress(1, 72, 0.02, [{"start": 0.0, "answer": {"rallies": [{"start_sec": 5.5, "point_sec": 12.5,
                                                                       "confidence": 0.9}]}}])
        now = jobs.load_job(job.id)
        seen.append((now.scan, now.spent_usd, [r["point_sec"] for r in now.rallies]))
        return [{"start_sec": 5.5, "point_sec": 12.5, "end_sec": 13.5, "is_rally": True, "confidence": 0.9,
                 "reason": None, "uncertain": False}], [
            {"usd": 0.02, "chunk": 0, "start": 0.0, "seconds": 60.0, "answer": {"rallies": []}},
            {"usd": 0.01, "chunk": 1, "start": 50.0, "seconds": 60.0, "answer": {"rallies": [], "unreadable": True}}]

    monkeypatch.setattr(scan, "scan_video", fake_scan)
    monkeypatch.setattr(jobs, "make_preview", lambda j, i: None)
    jobs.find_rallies(job, SimpleNamespace())
    saved = jobs.load_job(job.id)
    # 훑는 30분에 비례한 상한($0.05/분 → $1.5)이 예상 비용(약 $1.1)을 막지 않는다
    assert saved.scan_range == [0.0, 1800.0]
    assert saved.budget_usd == pytest.approx(1.5) and 1.0 < saved.estimate_usd < 1.5
    # 찾는 동안 찾은 랠리·진행·쓴 돈을 조각마다 저장한다(찾은 랠리부터 고를 수 있게)
    assert seen == [({"done": 1, "total": 72}, 0.02, [12.5])]
    assert saved.scan == {} and len(saved.rallies) == 1  # 끝난 뒤 같은 득점이 두 번 붙지 않는다
    assert saved.rallies[0]["point_sec"] == 12.5 and saved.spent_usd == pytest.approx(0.03)
    assert saved.unread == [[50.0, 110.0]]  # 못 읽은 구간은 화면에 알린다

    def no_muse(*a):
        raise AssertionError("훑어보기가 찾은 득점 순간을 다시 묻지 않는다")

    monkeypatch.setattr(score, "ask_score_time", no_muse)
    scene = jobs.prepare_scene(saved, 0, client=None)
    assert scene["score_sec"] == 12.5 and scene["window"] == [4.0, 17.5]
    assert jobs.load_job(job.id).spent_usd == pytest.approx(0.03)


def test_short_video_keeps_the_minimum_budget(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    from tapme import scan

    monkeypatch.setattr(scan, "duration_sec", lambda v: 60.0)
    monkeypatch.setattr(scan, "scan_video", lambda *a, **k: ([], []))
    jobs.find_rallies(job, SimpleNamespace())
    assert jobs.load_job(job.id).budget_usd == jobs.MAX_USD_PER_JOB


def test_scan_range_stays_inside_the_video_and_under_the_limit() -> None:
    assert jobs.scan_range(600.0, 0, 0) == [0.0, 600.0]
    assert jobs.scan_range(4000.0, 1200, 0) == [1200.0, 3000.0]  # 넘치면 끝을 당긴다
    assert jobs.scan_range(4000.0, 3500, 9999) == [3500.0, 4000.0]
    with pytest.raises(ValueError):
        jobs.scan_range(600.0, 700, 0)


def test_find_rallies_scans_only_the_picked_range(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    from tapme import scan

    monkeypatch.setattr(scan, "duration_sec", lambda v: 4000.0)
    calls: list = []
    monkeypatch.setattr(scan, "scan_video", lambda c, v, dur, d, p, offset: calls.append((dur, offset)) or ([], []))
    job.scan_range = [1500.0, 2700.0]
    jobs.find_rallies(job, SimpleNamespace())
    saved = jobs.load_job(job.id)
    assert calls == [(1200.0, 1500.0)] and saved.scan_range == [1500.0, 2700.0]
    assert saved.budget_usd == pytest.approx(max(jobs.MAX_USD_PER_JOB, 1.0))  # 20분 × $0.05


def test_highlights_are_numbered_in_the_order_they_were_made(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    # 저장 파일 번호(_01, _02): 만든 순서. 같은 장면을 다시 만들어도 번호는 그대로
    calls: list = []
    _prepared(job, monkeypatch, calls)
    job.rallies.append({**job.rallies[0], "index": 1})
    jobs.prepare_scene(job, 1, client=None)
    jobs.make_highlight(job, 1, [("score", 240, 240)], "검정", [], "", "face", 1.0, client=None)
    jobs.make_highlight(job, 0, [("score", 240, 240)], "검정", [], "", "face", 1.0, client=None)
    jobs.make_highlight(job, 1, [("score", 240, 240)], "검정", [], "", "face", 1.0, client=None)
    saved = jobs.load_job(job.id)
    assert (saved.scenes["1"]["number"], saved.scenes["0"]["number"]) == (1, 2)


def test_scan_and_scene_steps_saving_at_once_keep_each_others_changes(job: jobs.Job) -> None:
    # 10/30: 1시간 영상은 랠리를 찾는 동안 장면을 만든다. 같은 job.json을 두 객체가 저장해도 서로 지우지 않는다
    job.save()
    scanner, scener = jobs.load_job(job.id), jobs.load_job(job.id)
    scanner.rallies.append({"index": 1, "start_sec": 30.0, "end_sec": 40.0, "point_sec": 39.0})
    scanner.spent_usd += 0.03
    scanner.save()
    scener.scenes["0"] = {"index": 0, "status": "awaiting_tap"}
    scener.spent_usd += 0.01
    scener.save()
    scanner.scan = {"done": 2, "total": 72}
    scanner.save()
    saved = jobs.load_job(job.id)
    assert len(saved.rallies) == 2 and saved.scenes["0"]["status"] == "awaiting_tap"
    assert saved.scan == {"done": 2, "total": 72} and saved.spent_usd == pytest.approx(0.04)


def test_rallies_found_later_never_renumber_a_picked_rally(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "make_preview", lambda j, i: None)
    job.rallies = [{"index": 0, "start_sec": 5.0, "end_sec": 13.5, "point_sec": 12.5, "confidence": 0.6}]
    job.scenes = {"0": {"index": 0, "status": "awaiting_tap"}}
    jobs._add_rallies(job, [{"start_sec": 1.0, "end_sec": 3.0, "point_sec": 2.0, "confidence": 0.9},
                            {"start_sec": 6.0, "end_sec": 13.0, "point_sec": 12.0, "confidence": 0.95}])
    assert [(r["index"], r["point_sec"]) for r in job.rallies] == [(0, 12.5), (1, 2.0)]  # 고른 랠리는 그대로, 새것은 뒤에


def test_readers_never_see_a_half_written_job_file(job: jobs.Job) -> None:
    # 10/30 성능 테스트 T3: 저장하는 순간 읽으면 빈 파일이라 오류가 났고, 장면 준비가 멈췄다
    import threading

    job.save()
    errors: list = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                jobs.load_job(job.id)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    t = threading.Thread(target=reader)
    t.start()
    for i in range(300):
        job.rallies = [{"index": k, "point_sec": float(k)} for k in range(i % 50)]
        job.save()
    stop.set()
    t.join()
    assert errors == []


def test_renders_run_one_at_a_time_while_tracking_overlaps(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/30 일괄 만들기: 렌더(CPU)는 하나씩, 그 사이 다른 장면의 추적(네트워크)은 겹친다
    import threading
    import time

    from tapme import render

    running, peak = [0], [0]
    lock = threading.Lock()

    def fake_render(*a, **k):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return Path(a[1])

    monkeypatch.setattr(render, "render_video", fake_render)
    monkeypatch.setattr(jobs, "scene_render_inputs", lambda *a: (Path("src.mp4"), {}, {}, []))
    monkeypatch.setattr(jobs, "scene_people", lambda *a: {})
    scene = {"sam_clip": str(job.dir / "scene00" / "sam_blur.mp4"), "sam_fps": 10}
    threads = [threading.Thread(target=jobs._render_scene, args=(job, scene, [], "face", 1.0, True, None)) for _ in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert peak[0] == 1


def test_at_most_two_sam_trackings_run_at_once(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    running, peak = [0], [0]
    lock = threading.Lock()

    def fake_track(*a):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1

    threads = []
    for _ in range(4):
        def go():
            with jobs.TRACK_SLOTS:
                fake_track()
        threads.append(threading.Thread(target=go))
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert peak[0] == 2


def test_turns_go_in_arrival_order_and_report_positions() -> None:
    import threading

    turns, order = jobs.Turns(1), []
    gate = threading.Event()

    def first() -> None:
        with turns.turn("a"):
            gate.wait(2)
            order.append("a")

    def later(key: str) -> None:
        with turns.turn(key):
            order.append(key)

    threads = [threading.Thread(target=first)]
    threads[0].start()
    time.sleep(0.05)
    for key in ("b", "c", "d"):
        threads.append(threading.Thread(target=later, args=(key,)))
        threads[-1].start()
        time.sleep(0.05)  # 온 순서를 정한다
    assert (turns.position("a"), turns.position("b"), turns.position("d"), turns.waiting()) == (None, 1, 3, 3)
    gate.set()
    for th in threads:
        th.join()
    assert order == ["a", "b", "c", "d"]


def test_spending_of_all_jobs_is_capped(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "MAX_USD_TOTAL", 1.0)
    job.spent_usd = 0.6
    job.save()
    other = jobs.create_job("b.mp4", b"x")
    other.spent_usd = 0.3
    other.save()
    other.spent_usd = 0.35  # 바뀐 만큼만 더한다
    other.save()
    assert jobs.total_spent() == pytest.approx(0.95)
    with pytest.raises(jobs.BudgetExceeded, match="전체"):
        other.charge(0.1)
    shutil.rmtree(job.dir)  # 작업을 지워도 장부는 남는다
    assert jobs.total_spent() == pytest.approx(0.95)


KST_4AM_NOV1 = 1793473200.0  # 2026-11-01 04:00 한국 시각


def test_the_day_starts_at_4am_korean_time() -> None:
    assert jobs.day_start(KST_4AM_NOV1 + 60) == KST_4AM_NOV1
    assert jobs.day_start(KST_4AM_NOV1 - 60) == KST_4AM_NOV1 - 86400  # 새벽 3:59는 아직 전날
    assert (jobs.today(KST_4AM_NOV1 - 60), jobs.today(KST_4AM_NOV1 + 60)) == ("2026-10-31", "2026-11-01")


def test_previous_days_are_wiped_but_busy_jobs_and_ledgers_stay(job: jobs.Job) -> None:
    import os

    busy, fresh = jobs.create_job("b.mp4", b"x"), jobs.create_job("c.mp4", b"x")
    for j, t in ((job, KST_4AM_NOV1 - 3 * 3600), (busy, KST_4AM_NOV1 - 600), (fresh, KST_4AM_NOV1 + 600)):
        os.utime(j.dir / "job.json", (t, t))
    (jobs.JOBS_DIR / "_access.json").write_text("{}")
    now = KST_4AM_NOV1 + 1800  # 새벽 4:30
    assert jobs.delete_previous_days(now) == [job.id]  # 3:50에 저장한 작업은 아직 처리 중일 수 있다
    assert jobs.delete_previous_days(now + 3600) == [busy.id]  # 한 시간 조용하면 지운다
    assert fresh.dir.exists() and (jobs.JOBS_DIR / "_access.json").exists()


def test_uploads_are_counted_per_day() -> None:
    jobs.count_upload(KST_4AM_NOV1 - 60)
    jobs.count_upload(KST_4AM_NOV1 + 60)
    jobs.count_upload(KST_4AM_NOV1 + 120)
    assert (jobs.uploads_today(KST_4AM_NOV1 + 180), jobs.uploads_today(KST_4AM_NOV1 + 86400)) == (2, 0)


def test_browser_segment_keeps_original_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 2단계: 기기가 다시 압축하지 않고 잘라 올린 랠리 구간(원본 시각 그대로, 영상·소리 따로)을 합쳐
    # 장면 준비가 원본 시각으로 자를 수 있게 한다
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=30", "-c:v", "libx264", "-g", "30", "-c:a", "aac",
                    "-shortest", str(src)], check=True)

    def piece(stream: str, out: Path) -> Path:  # 기기가 잘라 올리는 조각처럼: 키프레임(9초)부터 다시 압축하지 않고
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "9", "-i", str(src), "-t", "14", "-map", f"0:{stream}",
                        "-c", "copy", "-movflags", "frag_keyframe+empty_moov+default_base_moof", str(out)], check=True)
        return out

    def first_pts(path: Path) -> float:  # 원본에서 9초 키프레임의 표시 시각(기기는 mp4의 cts로 읽는다)
        out = subprocess.check_output(["ffprobe", "-v", "error", "-read_intervals", "9%+#1", "-select_streams", "v:0",
                                       "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)])
        return float(out.decode().split()[0])

    job, _ = jobs.create_browser_job("a.mp4", 30.0, 0, 0)
    job.rallies = [{"index": 0, "start_sec": 12.0, "point_sec": 17.0, "end_sec": 18.0}]
    first = first_pts(src)
    jobs.save_segment(job, 0, piece("v", tmp_path / "v.mp4"), piece("a", tmp_path / "a.mp4"), first)
    saved = jobs.load_job(job.id)
    assert 8.9 <= saved.segments["0"] <= 9.1
    # 구간 파일에서 원본 12초 자리의 그림이 원본 12초와 같다(SAM 박스가 원본 클립과 어긋나지 않게)
    seg, at = jobs._source(saved, 0, 12.0)
    grab = lambda p, t: subprocess.check_output(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(p),
                                                 "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    assert grab(seg, at) == grab(src, 12.0)
    path, at = jobs._source(saved, 0, 12.5)
    assert path.name == "segment00.mp4" and at == pytest.approx(12.5 - saved.segments["0"])
    streams = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)])
    assert sorted(streams.decode().split()) == ["audio", "video"]
    job.rallies[0]["point_sec"] = 5.0  # 구간(9초~)이 장면(득점 7초 전~)을 못 덮으면 거절
    with pytest.raises(ValueError, match="덮지"):
        jobs.save_segment(job, 0, piece("v", tmp_path / "v2.mp4"), None, first)


def test_highlight_numbers_do_not_repeat_when_made_at_once(job: jobs.Job) -> None:
    # 10/31 한 번에 맡기기: 동시에 만든 장면이 각자 처음 읽은 job으로 번호를 매겨 _02가 둘이었다
    job.scenes = {"0": {"index": 0}, "1": {"index": 1}, "2": {"index": 2}}
    job.save()
    first, second, third = jobs.load_job(job.id), jobs.load_job(job.id), jobs.load_job(job.id)  # 따로 돈 세 작업
    numbers = [jobs.assign_number(first, first.scenes["0"]), jobs.assign_number(second, second.scenes["1"]),
               jobs.assign_number(third, third.scenes["2"])]
    assert numbers == [1, 2, 3]
    assert jobs.assign_number(second, second.scenes["1"]) == 2  # 다시 만들어도 그대로


def test_request_log_keeps_who_when_what_but_no_names_and_drops_old_lines(tmp_path, monkeypatch) -> None:
    # 11/1 사용자: 누가 언제 어떤 영상을 요청했는지 남긴다. 새벽 4시 삭제 뒤에도 남고 30일 지나면 지운다
    import json
    import time

    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    job = jobs.Job(id="abc123def456", video="", match_name="우리팀 정모", code="ABCD1234")
    old = {"at": "2026-09-01 10:00:00", "t": time.time() - 31 * 86400, "job": "old", "event": "upload"}
    (tmp_path / "_requests.jsonl").write_text(json.dumps(old) + "\n")
    jobs.log_request(job, "upload", "배너", size_mb=89.1)
    rows = jobs.requests_log()
    assert [r["job"] for r in rows] == ["abc123def456"]  # 30일 지난 줄은 지웠다
    r = rows[0]
    assert r["code"] == "ABCD" and r["kind"] == "배너" and r["event"] == "upload" and r["size_mb"] == 89.1
    assert "우리팀" not in json.dumps(r, ensure_ascii=False)  # 경기 이름·파일 이름은 남기지 않는다


def test_render_turns_alternate_between_jobs() -> None:
    # 11/1 사용자: 한 사람이 10분 영상 랠리를 다 맡겨도(장면 여러 개 예약) 뒤에 온 사람은 지금 렌더 하나만 기다린다
    import threading
    import time

    turns = jobs.Turns(1, group=lambda k: k.split("/")[0])
    ran: list[str] = []
    go = threading.Event()

    def work(key: str, hold: threading.Event | None = None) -> None:
        with turns.turn(key):
            ran.append(key)
            if hold:
                hold.wait(2)

    first = threading.Thread(target=work, args=("A/0", go))
    first.start()
    time.sleep(0.05)
    rest = []
    for key in ("A/1", "A/2", "A/3", "B/0", "B/1"):  # A가 먼저 줄을 많이 세웠다
        rest.append(threading.Thread(target=work, args=(key,)))
        rest[-1].start()
        time.sleep(0.02)
    assert turns.position("B/0") == 1 and turns.position("A/1") == 2  # 화면에 보이는 순번도 번갈아
    go.set()
    for th in [first, *rest]:
        th.join(2)
    assert ran == ["A/0", "B/0", "A/1", "B/1", "A/2", "A/3"]
    fifo = jobs.Turns(1)  # 묶음이 없으면 온 순서대로(랠리 찾기)
    fifo._waiting = ["A/1", "B/0"]
    assert fifo.position("B/0") == 2


def test_render_slots_let_several_renders_run_together_on_big_machines(monkeypatch: pytest.MonkeyPatch) -> None:
    # 11/2 H-57: 코어가 많은 기종에서는 렌더를 동시에 여러 개(TAPME_RENDER_SLOTS). 기본은 지금처럼 하나씩
    import importlib
    import threading
    import time

    assert jobs.RENDER_SLOTS == 1 and jobs.RENDER_TURNS.slots == 1
    turns = jobs.Turns(2, group=lambda k: k.split("/")[0])
    running, peak, lock = [0], [0], threading.Lock()

    def work(key: str) -> None:
        with turns.turn(key):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.1)
            with lock:
                running[0] -= 1

    threads = [threading.Thread(target=work, args=(f"A/{i}",)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(3)
    assert peak[0] == 2
    monkeypatch.setenv("TAPME_RENDER_SLOTS", "3")
    try:
        assert importlib.reload(jobs).RENDER_TURNS.slots == 3
    finally:
        monkeypatch.delenv("TAPME_RENDER_SLOTS")
        importlib.reload(jobs)


def test_point_winner_is_the_team_that_serves_the_next_rally_on_our_side_per_set() -> None:
    # 11/1 H-55: 배구는 이긴 팀이 다음 서브. 우리 팀 쪽은 세트마다 '나'의 쪽 다수결(하나 틀려도), 세트 사이(90초 넘게 빔)는 모름
    job = jobs.Job(id="w0000000000a", video="")
    starts = [0, 25, 50, 75, 200, 225, 250]  # 75 → 200: 세트가 바뀜(코트를 바꿔 우리 팀이 먼 쪽)
    me = ["near", "near", "far", "near", "far", "near", "far"]  # 2번·5번은 틀린 값(다수결로 바로잡힘)
    serve = ["far", "near", "near", "far", "near", "far", "near"]
    job.rallies = [{"index": i, "start_sec": float(s), "is_rally": True} for i, s in enumerate(starts)]
    job.scenes = {str(i): {"index": i, "me_side": me[i], "serve": {"side": serve[i]}} for i in range(len(starts))}
    assert jobs.point_winners(job) == {"0": "mine", "1": "mine", "2": "opponent", "4": "mine", "5": "opponent"}
    job.scenes["1"]["serve"]["side"] = "unsure"
    assert "0" not in jobs.point_winners(job)  # 다음 서브를 모르면 모름
    boxes = [[0, 100, 10, 300], [0, 100, 10, 500], [0, 100, 10, 420]]
    assert jobs.side_of(boxes, 1) == "near" and jobs.side_of(boxes, 0) == "far" and jobs.side_of(boxes, 5) == "unknown"


def test_scene_is_the_whole_rally_from_the_serve_but_never_too_long() -> None:
    # 11/2 사용자: 고정 12초(득점 7초 전~5초 뒤)는 짧았다 — 10.6초 랠리는 서브가 잘리고, 3.5초 랠리는 빈 시간이 들어갔다
    assert jobs.scene_window({"start_sec": 52.2}, 62.8) == (50.7, 67.8)  # 서브 1.5초 전부터
    assert jobs.scene_window({"start_sec": 125.0}, 128.5) == (123.5, 133.5)  # 짧은 랠리: 서브 전 빈 시간 없음
    assert jobs.scene_window({"start_sec": 127.0}, 128.5) == (124.5, 133.5)  # 득점 앞은 적어도 4초
    assert jobs.scene_window({"start_sec": 100.0}, 140.0) == (120.0, 145.0)  # 긴 랠리는 25초까지
    assert jobs.scene_window({}, 12.5) == (5.5, 17.5)  # 서브 시각을 모르면 예전처럼 7초 전부터
    assert jobs.scene_window({"start_sec": 0.0}, 3.0) == (0.0, 8.0)  # 영상 시작 앞으로는 가지 않는다
