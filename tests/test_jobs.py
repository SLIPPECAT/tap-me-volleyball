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
    assert scene["status"] == "awaiting_tap" and Path(scene["tap_frame"]).exists()
    assert jobs.load_job(job.id).spent_usd == pytest.approx(0.01)


def test_tap_then_rerender_without_paying_again(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(score, "ask_score_time", lambda c, clip: ({"score_time_sec": 8.5}, _reply()))
    jobs.prepare_scene(job, 0, client=None)
    monkeypatch.setattr(track, "track_players", lambda c, clip: (WIRE, {"video_frames_processed": 50}, 0.0))
    monkeypatch.setattr(score, "ask_caption",
                        lambda c, clip, team, t: ({"ko": "검정 득점!", "ja": "黒得点!", "en": "Black scores!"}, _reply()))

    with pytest.raises(jobs.TapMissed):
        jobs.make_highlight(job, 0, 5, 5, "검정", ["ko"], "face", 2.0, client=None)
    assert job.scenes["0"]["status"] == "awaiting_tap"

    scene = jobs.make_highlight(job, 0, 175, 240, "검정", ["en", "ko"], "face", 2.0, client=None)
    assert scene["me"] == "1" and scene["status"] == "done" and Path(scene["video"]).exists()
    spent = jobs.load_job(job.id).spent_usd

    # 옵션만 바꾼 렌더는 API를 부르지 않는다
    monkeypatch.setattr(track, "track_players", lambda *a: pytest.fail("SAM을 다시 부름"))
    monkeypatch.setattr(score, "ask_caption", lambda *a: pytest.fail("Muse를 다시 부름"))
    scene = jobs.rerender(job, 0, ["ja"], "background", 1.0)
    assert scene["options"] == {"languages": ["ja"], "privacy": "background", "zoom": 1.0}
    assert jobs.load_job(job.id).spent_usd == spent


def test_load_job_rejects_path_tricks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    with pytest.raises(KeyError):
        jobs.load_job("../etc")


def test_find_rallies_keeps_rejected_candidates_for_the_user(job: jobs.Job, monkeypatch: pytest.MonkeyPatch) -> None:
    from tapme import rally, rally_filter

    monkeypatch.setattr(rally, "load_audio", lambda v: None)
    monkeypatch.setattr(rally, "find_whistles", lambda a: [])
    monkeypatch.setattr(rally, "pair_rallies", lambda w: [rally.RallyCandidate(1, 9), rally.RallyCandidate(20, 28)])
    monkeypatch.setattr(rally_filter, "check_candidates", lambda c, v, cands, d: [
        {"start_sec": 1, "end_sec": 9, "is_rally": True, "usd": 0.006},
        {"start_sec": 20, "end_sec": 28, "is_rally": False, "usd": 0.006}])
    jobs.find_rallies(job, SimpleNamespace())
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
