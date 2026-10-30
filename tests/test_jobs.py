from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tapme import jobs, muse, score, track
from tests.test_track import WIRE  # 320×334 화면, 선수 0·1


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
    # 클립은 원본 4.0초에서 시작 → 득점 12.5초 → 7초 전~3초 뒤
    assert scene["score_sec"] == 12.5
    assert scene["window"] == [5.5, 15.5]
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

    def fake_commentary(c, clip, team, t, role, sec):
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
    assert len(sam) == 1  # 얼굴만 가리기는 다른 선수 추적이, 자막을 끄면 해설(Muse)이 필요 없다
    # '나'는 SAM 프레임 0·1(5.5~5.7초)에만 있다. 영상은 그 구간과 최소 득점 4초 전~2초 뒤(8.5~14.5초)를 모두 담는다
    assert scene["span_frames"] == [0, 1] and scene["clip"] == [5.5, 14.5]
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
    assert calls == [calls[0]]  # SAM('나') 한 번뿐, 자막을 껐으니 Muse 없음
    spent = jobs.load_job(job.id).spent_usd

    jobs.rerender(job, 0, [], "", "background", 1.0)  # 비용 없음
    assert len(calls) == 1 and jobs.load_job(job.id).spent_usd == spent
    jobs.rerender(job, 0, ["ko"], "assist", "face", 2.0)  # 자막을 처음 켬 → 해설 한 번
    jobs.rerender(job, 0, ["ja", "en"], "assist", "face", 2.0)  # 언어만 바꿈 → 다시 묻지 않음
    jobs.rerender(job, 0, ["ko"], "defense", "face", 2.0)  # 역할을 바꿈 → 다시 묻는다
    scene = jobs.rerender(job, 0, ["ko"], "defense", "silhouette", 2.0)  # 다른 선수 추적을 처음 한 번
    jobs.rerender(job, 0, ["en"], "defense", "silhouette", 1.5)  # 저장해 둔 것을 다시 씀
    assert calls[1:] == ["muse:assist", "muse:defense", "volleyball player"]
    assert scene["options"] == {"languages": ["en"], "role": "defense", "privacy": "silhouette", "zoom": 1.5,
                                "face_mosaic": True}
    scene = jobs.rerender(job, 0, ["en"], "defense", "face", 1.5, face_mosaic=False)  # 모자이크 끄기도 비용 없음
    assert scene["options"]["face_mosaic"] is False and len(calls) == 4


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
    # 장면 0: 창 5.5~15.5초, 득점 12.5초 → 창 안 7.0초 → 2.0~9.0초를 씀
    # 장면 1: 창 0~6초(득점이 영상 앞쪽이라 앞이 잘림), 득점 3.0초 → 0~5.0초를 씀
    job.scenes = {
        "0": {"status": "done", "score_sec": 12.5, "window": [5.5, 15.5], "video": str(_scene_video(tmp_path / "a.mp4", 10, "red"))},
        "1": {"status": "done", "score_sec": 3.0, "window": [0.0, 6.0], "video": str(_scene_video(tmp_path / "b.mp4", 6, "blue"))},
    }
    out = jobs.make_reel(job, [1, 0])
    w, h, fps, n = __import__("tapme.render", fromlist=["probe"]).probe(Path(out["video"]))
    assert (w, h, round(fps)) == (90, 160, 30)
    assert abs(n / fps - 12.0) < 0.2  # 5초 + 7초
    assert out["order"] == [1, 0]  # 사용자가 고른 순서
    assert jobs.load_job(job.id).reel["status"] == "done"


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
    # 두 번 다 실패해도 낸 돈(SAM 0.01 + Muse 2회)은 센다
    assert jobs.load_job(job.id).spent_usd == pytest.approx(spent + 0.01 + 0.02)


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

    def fake_scan(c, v, duration, d, progress):
        progress(1, 72)
        seen.append(jobs.load_job(job.id).progress)
        return [{"start_sec": 5.5, "point_sec": 12.5, "end_sec": 13.5, "is_rally": True, "confidence": 0.9,
                 "reason": None, "uncertain": False}], [
            {"usd": 0.02, "chunk": 0, "start": 0.0, "seconds": 60.0, "answer": {"rallies": []}},
            {"usd": 0.01, "chunk": 1, "start": 50.0, "seconds": 60.0, "answer": {"rallies": [], "unreadable": True}}]

    monkeypatch.setattr(scan, "scan_video", fake_scan)
    monkeypatch.setattr(jobs, "make_preview", lambda j, i: None)
    jobs.find_rallies(job, SimpleNamespace())
    saved = jobs.load_job(job.id)
    # 1시간 영상: 상한이 길이에 비례($0.05/분 → $3)해 예상 비용(약 $2.3)으로 막히지 않는다
    assert saved.budget_usd == pytest.approx(3.0) and 2.0 < saved.estimate_usd < 3.0
    assert seen == [{"step": "rallies", "done": 1, "total": 72}]
    assert saved.rallies[0]["point_sec"] == 12.5 and saved.spent_usd == pytest.approx(0.03)
    assert saved.unread == [[50.0, 110.0]]  # 못 읽은 구간은 화면에 알린다

    def no_muse(*a):
        raise AssertionError("훑어보기가 찾은 득점 순간을 다시 묻지 않는다")

    monkeypatch.setattr(score, "ask_score_time", no_muse)
    scene = jobs.prepare_scene(saved, 0, client=None)
    assert scene["score_sec"] == 12.5 and scene["window"] == [5.5, 15.5]
    assert jobs.load_job(job.id).spent_usd == pytest.approx(0.03)


def test_short_video_keeps_the_minimum_budget(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    from tapme import scan

    monkeypatch.setattr(scan, "duration_sec", lambda v: 60.0)
    monkeypatch.setattr(scan, "scan_video", lambda *a: ([], []))
    jobs.find_rallies(job, SimpleNamespace())
    assert jobs.load_job(job.id).budget_usd == jobs.MAX_USD_PER_JOB
