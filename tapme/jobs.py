"""하이라이트 작업의 단계와 상태. 서버(server.py)가 부른다.

작업 폴더 data/jobs/<id>/ (git 제외)에 원본·흐림본·결과와 job.json을 둔다.
단계:
  1. find_rallies    휘슬 후보 → 흐림본을 Muse로 거르기 (유료, 후보당 약 $0.006)
  2. prepare_scene   사용자가 고른 랠리의 득점 시점(Muse) → 득점 7초 전~3초 뒤 5fps 흐림본 + 탭할 첫 장면
  3. make_highlight  탭 → SAM 추적 → 자막(Muse) → 30fps 흐림본 렌더
외부로 나가는 영상은 흐림본뿐이다. 얼굴이 그대로인 잘린 사본은 흐림 직후 지운다.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tapme import anonymize, muse, rally, rally_filter, render, score, track

JOBS_DIR = Path(os.environ.get("TAPME_JOBS_DIR", "data/jobs"))
MAX_USD_PER_JOB = float(os.environ.get("TAPME_MAX_USD_PER_JOB", "1.0"))
WINDOW_BEFORE_SEC = 7.0  # SAM은 5fps 50프레임(10초)보다 짧아도 50으로 과금된다(비용 원장 R-10·R-17)
WINDOW_AFTER_SEC = 3.0
SAM_FPS = 5
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


class TapMissed(ValueError):
    pass


@dataclass
class Job:
    id: str
    video: str
    status: str = "uploaded"
    rallies: list[dict[str, Any]] = field(default_factory=list)
    scenes: dict[str, dict[str, Any]] = field(default_factory=dict)
    spent_usd: float = 0.0
    error: str | None = None

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def save(self) -> None:
        with _lock:
            (self.dir / "job.json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))

    def charge(self, estimate_usd: float) -> None:
        if self.spent_usd + estimate_usd > MAX_USD_PER_JOB:
            raise BudgetExceeded(f"작업당 상한 ${MAX_USD_PER_JOB:.2f}을 넘는다 (쓴 돈 ${self.spent_usd:.3f} + 예상 ${estimate_usd:.3f})")


def create_job(filename: str, data: bytes) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], video="")
    job.dir.mkdir(parents=True, exist_ok=True)
    src = job.dir / f"source{Path(filename).suffix.lower() or '.mp4'}"
    src.write_bytes(data)
    job.video = str(src)
    job.save()
    return job


def load_job(job_id: str) -> Job:
    path = JOBS_DIR / job_id / "job.json"
    if not job_id.isalnum() or not path.exists():
        raise KeyError(job_id)
    return Job(**json.loads(path.read_text()))


def _cut_blurred(video: Path, start: float, duration: float, out: Path, fps: int | None) -> Path:
    raw = out.with_name(out.stem + "_raw.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(video), "-t", f"{duration:.2f}"]
    cmd += ["-r", str(fps)] if fps else []
    subprocess.run([*cmd, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(raw)], check=True)
    anonymize.anonymize_video(raw, out)
    raw.unlink()
    return out


def find_rallies(job: Job, client: Any) -> None:
    job.status = "finding_rallies"
    job.save()
    candidates = rally.pair_rallies(rally.find_whistles(rally.load_audio(Path(job.video))))
    job.charge(sum(muse.estimate_usd(c.end_sec - c.start_sec + 2 * rally_filter.PAD_SEC) for c in candidates))
    results = rally_filter.check_candidates(client, Path(job.video), candidates, job.dir / "rallies")
    job.spent_usd += sum(r["usd"] for r in results)
    # 사용자가 Muse 판정을 뒤집을 수 있게 랠리 아님으로 본 후보도 남긴다
    job.rallies = [{"index": i, **r} for i, r in enumerate(results)]
    job.status = "rallies_ready"
    job.save()


def prepare_scene(job: Job, index: int, client: Any) -> dict[str, Any]:
    r = job.rallies[index]
    job.scenes[str(index)] = {"index": index, "status": "preparing"}
    job.save()
    job.charge(muse.estimate_usd(r["end_sec"] - r["start_sec"] + 2 * rally_filter.PAD_SEC))
    answer, reply = score.ask_score_time(client, Path(r["clip"]))
    job.spent_usd += reply.usd
    score_sec = r["clip_start_sec"] + float(answer["score_time_sec"])
    start = max(0.0, score_sec - WINDOW_BEFORE_SEC)
    duration = score_sec + WINDOW_AFTER_SEC - start
    scene_dir = job.dir / f"scene{index:02d}"
    scene_dir.mkdir(exist_ok=True)
    win5 = _cut_blurred(Path(job.video), start, duration, scene_dir / "window5_blur.mp4", SAM_FPS)
    frame = scene_dir / "tap_frame.jpg"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(win5), "-frames:v", "1", str(frame)], check=True)
    scene = {"index": index, "status": "awaiting_tap", "score_sec": round(score_sec, 2),
             "window": [round(start, 2), round(start + duration, 2)], "score_answer": answer,
             "window5": str(win5), "tap_frame": str(frame)}
    job.scenes[str(index)] = scene
    job.save()
    return scene


def make_highlight(
    job: Job, index: int, x: float, y: float, team_label: str,
    languages: list[str], privacy: str, zoom: float, client: Any,
) -> dict[str, Any]:
    scene = job.scenes[str(index)]
    scene["status"] = "rendering"
    job.save()
    win5 = Path(scene["window5"])
    frames = render.probe(win5)[3]
    job.charge(track.estimate_usd(max(frames, 50)) + muse.estimate_usd(scene["window"][1] - scene["window"][0]))

    wire, usage, _ = track.track_players(client, win5)
    job.spent_usd += track.estimate_usd(int(usage.get("video_frames_processed") or frames))
    (win5.parent / "sam_wire.txt").write_text(wire)  # 다시 렌더할 때 SAM을 또 부르지 않는다
    result = track.parse(wire)
    me = track.find_tapped(result, x, y, frame=0)
    if me is None:
        scene["status"] = "awaiting_tap"
        job.save()
        raise TapMissed("탭한 곳에서 선수를 찾지 못했다. 선수 몸 위를 다시 탭해 주세요")

    captions, reply = score.ask_caption(client, win5, team_label, scene["score_sec"] - scene["window"][0])
    job.spent_usd += reply.usd
    start, end = scene["window"]
    win30 = _cut_blurred(Path(job.video), start, end - start, win5.parent / "window30_blur.mp4", None)
    out = render.render_video(
        win30, win5.parent / "highlight.mp4", track.boxes_by_frame(result), track.masks_by_frame(result),
        me, SAM_FPS, score.subtitle_text(captions, languages), zoom, privacy,
    )
    scene.update(status="done", me=me, tap=[x, y], team=team_label, captions=captions, window30=str(win30),
                 options={"languages": languages, "privacy": privacy, "zoom": zoom}, video=str(out))
    job.save()
    return scene


def rerender(job: Job, index: int, languages: list[str], privacy: str, zoom: float) -> dict[str, Any]:
    """자막 언어·가리기·확대만 바꿔 다시 렌더. 저장해 둔 SAM 결과와 자막을 쓰므로 API 비용이 없다."""
    scene = job.scenes[str(index)]
    scene["status"] = "rendering"
    job.save()
    win5 = Path(scene["window5"])
    result = track.parse((win5.parent / "sam_wire.txt").read_text())
    out = render.render_video(
        Path(scene["window30"]), win5.parent / "highlight.mp4", track.boxes_by_frame(result),
        track.masks_by_frame(result), scene["me"], SAM_FPS, score.subtitle_text(scene["captions"], languages),
        zoom, privacy,
    )
    scene.update(status="done", options={"languages": languages, "privacy": privacy, "zoom": zoom}, video=str(out))
    job.save()
    return scene
