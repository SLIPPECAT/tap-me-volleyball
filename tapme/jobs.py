"""하이라이트 작업의 단계와 상태. 서버(server.py)가 부른다.

작업 폴더 data/jobs/<id>/ (git 제외)에 원본·흐림본·결과와 job.json을 둔다.
단계:
  1. find_rallies    휘슬 후보 → 흐림본을 Muse로 거르기 (유료, 후보당 약 $0.006)
  2. prepare_scene   사용자가 고른 랠리의 득점 시점(Muse) → 득점 7초 전~3초 뒤 SAM용 흐림본 + 탭할 장면 2장(득점 순간·구간 시작)
  3. make_highlight  탭 → '나'만 SAM 추적(점 프롬프트) → 득점 순간과 이어서 보이는 구간만 → 자막(Muse) → 원본 fps 흐림본 렌더
외부로 나가는 영상은 흐림본뿐이다. 얼굴이 그대로인 잘린 사본은 흐림 직후 지운다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from tapme import anonymize, muse, rally, rally_filter, render, score, track

JOBS_DIR = Path(os.environ.get("TAPME_JOBS_DIR", "data/jobs"))
MAX_USD_PER_JOB = float(os.environ.get("TAPME_MAX_USD_PER_JOB", "1.0"))
WINDOW_BEFORE_SEC = 7.0  # SAM은 50프레임보다 짧아도 50으로 과금된다(비용 원장 R-10·R-17)
WINDOW_AFTER_SEC = 3.0
# 10fps: 움직이는 카메라에서 5fps는 같은 사람을 더 자주 놓쳤다(실험 H-24 B<C). 장면당 SAM 약 $0.02
SAM_FPS = 10
TAP_KINDS = ("score", "start")  # 탭할 장면: 득점 순간(필수)·구간 시작(선택)
# 하이라이트 영상은 최소 득점 4초 전~2초 뒤. '나'를 확실히 잡은 구간이 이보다 짧으면 그 밖에서는 링·확대 따라가기를 하지 않는다
CLIP_MIN_BEFORE_SEC = 4.0
CLIP_MIN_AFTER_SEC = 2.0
REEL_BEFORE_SEC = 5.0  # 모음 영상에는 장면마다 득점 5초 전~2초 뒤만. 3장면이면 약 20초
REEL_AFTER_SEC = 2.0
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
    reel: dict[str, Any] = field(default_factory=dict)
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


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}


def create_job(filename: str, data: bytes | BinaryIO) -> Job:
    """data는 바이트 또는 파일 객체. 경기 영상은 GB 단위일 수 있어 파일 객체는 조금씩 옮겨 쓴다."""
    job = Job(id=uuid.uuid4().hex[:12], video="")
    job.dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    src = job.dir / f"source{suffix if suffix in VIDEO_SUFFIXES else '.mp4'}"
    if isinstance(data, bytes):
        src.write_bytes(data)
    else:
        with src.open("wb") as f:
            shutil.copyfileobj(data, f, length=8 * 1024 * 1024)
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
    for i in range(len(job.rallies)):
        make_preview(job, i)  # 사용자가 시간만 보고는 어떤 장면인지 모른다(10/29 사용자 피드백)
    job.status = "rallies_ready"
    job.save()


PREVIEW_SHORT_SIDE = 360  # 랠리 후보 미리보기: 폰에서 고르기용 낮은 화질


def make_preview(job: Job, index: int) -> tuple[Path, Path]:
    """랠리 후보의 장면 사진(가운데)과 미리보기 영상(앞뒤 1초 포함). 로컬, API 비용 없음.

    올린 사람의 폰에만 보여 주므로 원본에서 만든다. 외부(Meta)로는 여전히 흐림본만 보낸다.
    이미 있으면 다시 만들지 않는다(예전 작업은 처음 볼 때 만들어진다).
    """
    r = job.rallies[index]
    out_dir = job.dir / "rallies"
    out_dir.mkdir(exist_ok=True)
    thumb, preview = out_dir / f"cand{index:02d}_thumb.jpg", out_dir / f"cand{index:02d}_preview.mp4"
    side = PREVIEW_SHORT_SIDE
    scale = f"scale='if(gt(iw,ih),-2,min({side},iw))':'if(gt(iw,ih),min({side},ih),-2)'"
    start = max(0.0, r["start_sec"] - rally_filter.PAD_SEC)
    if not preview.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", job.video,
                        "-t", f"{r['end_sec'] + rally_filter.PAD_SEC - start:.2f}", "-vf", scale, "-r", "30",
                        "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(preview)], check=True)
    if not thumb.exists():
        middle = (r["start_sec"] + r["end_sec"]) / 2
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{middle:.2f}", "-i", job.video,
                        "-frames:v", "1", "-vf", scale, str(thumb)], check=True)
    return thumb, preview


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
    sam_clip = _cut_blurred(Path(job.video), start, duration, scene_dir / "sam_blur.mp4", SAM_FPS)
    n = render.probe(sam_clip)[3]
    # 득점 순간 장면에서 탭하면 하이라이트의 핵심(득점 근처)은 맞는 사람을 잡는다(실험 H-25)
    frames = {"score": min(n - 1, round((score_sec - start) * SAM_FPS)), "start": 0}
    for kind, k in frames.items():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(sam_clip), "-vf", f"select=eq(n\\,{k})",
                        "-frames:v", "1", str(scene_dir / f"tap_{kind}.jpg")], check=True)
    scene = {"index": index, "status": "awaiting_tap", "score_sec": round(score_sec, 2),
             "window": [round(start, 2), round(start + duration, 2)], "score_answer": answer,
             "sam_clip": str(sam_clip), "sam_fps": SAM_FPS, "tap_frames": frames}
    job.scenes[str(index)] = scene
    job.save()
    return scene


def _track_others(job: Job, client: Any, sam_clip: Path) -> Path:
    """'몸 실루엣' 가리기에 쓸 다른 선수 위치. 이 방식을 고를 때만 SAM을 한 번 더 부른다."""
    out = sam_clip.parent / "others_wire.txt"
    if not out.exists():
        frames = render.probe(sam_clip)[3]
        job.charge(track.estimate_usd(max(frames, 50)))
        wire, usage, _ = track.track_players(client, sam_clip)
        job.spent_usd += track.estimate_usd(int(usage.get("video_frames_processed") or frames))
        out.write_text(wire)
    return out


def _render_scene(job: Job, scene: dict[str, Any], languages: list[str], privacy: str, zoom: float,
                  client: Any) -> Path:
    sam_clip = Path(scene["sam_clip"])
    me = track.parse((sam_clip.parent / "me_wire.txt").read_text())
    others = track.parse(_track_others(job, client, sam_clip).read_text()) if privacy == "silhouette" else None
    by, masks = track.merge_me_and_others(me, others)
    # '나'는 확실히 잡은 구간(span) 안에서만 쓴다. 그 밖은 화면 밖에 나갔다 들어와 다른 사람일 수 있다(실험 H-24·H-25)
    s0, s1 = scene["span_frames"]
    by = {k: [b for b in v if b.object_id != track.TAPPED_ID or s0 <= k <= s1] for k, v in by.items()}
    f0 = round((scene["clip"][0] - scene["window"][0]) * scene["sam_fps"])  # 렌더할 클립의 첫 SAM 프레임
    by = {k - f0: v for k, v in by.items() if k >= f0}
    masks = {k - f0: v for k, v in masks.items() if k >= f0}
    return render.render_video(Path(scene["clip_video"]), sam_clip.parent / "highlight.mp4", by, masks,
                               track.TAPPED_ID, scene["sam_fps"], score.subtitle_text(scene["captions"], languages),
                               zoom, privacy)


def make_highlight(
    job: Job, index: int, taps: list[tuple[str, float, float]], team_label: str,
    languages: list[str], privacy: str, zoom: float, client: Any,
) -> dict[str, Any]:
    """taps = [(장면 종류, x, y), ...]. 장면 종류는 TAP_KINDS. 득점 순간 탭은 꼭 있어야 한다."""
    scene = job.scenes[str(index)]
    if not any(kind == "score" for kind, _, _ in taps) or any(kind not in TAP_KINDS for kind, _, _ in taps):
        raise TapMissed("득점 순간 장면에서 나를 탭해 주세요")
    scene["status"] = "rendering"
    job.save()
    sam_clip = Path(scene["sam_clip"])
    w, h, _, frames = render.probe(sam_clip)
    silhouette = 2 if privacy == "silhouette" else 1
    job.charge(track.estimate_usd(max(frames, 50)) * silhouette + muse.estimate_usd(scene["window"][1] - scene["window"][0]))

    points = [(scene["tap_frames"][kind], x, y) for kind, x, y in taps]
    wire, usage, _ = track.track_players(client, sam_clip, track.tap_prompt((w, h), points))
    job.spent_usd += track.estimate_usd(int(usage.get("video_frames_processed") or frames))
    (sam_clip.parent / "me_wire.txt").write_text(wire)  # 옵션만 바꿔 다시 렌더할 때 SAM을 또 부르지 않는다
    fps = scene["sam_fps"]
    span = track.visible_span(track.boxes_by_frame(track.parse(wire)), around=scene["tap_frames"]["score"], max_gap=fps)
    if span is None:
        scene["status"] = "awaiting_tap"
        job.save()
        raise TapMissed("탭한 곳에서 선수를 찾지 못했다. 선수 몸 위를 다시 탭해 주세요")

    captions, reply = score.ask_caption(client, sam_clip, team_label, scene["score_sec"] - scene["window"][0])
    job.spent_usd += reply.usd
    win_start, win_end = scene["window"]
    clip_start = max(win_start, min(win_start + span[0] / fps, scene["score_sec"] - CLIP_MIN_BEFORE_SEC))
    clip_end = min(win_end, max(win_start + (span[1] + 1) / fps, scene["score_sec"] + CLIP_MIN_AFTER_SEC))
    clip_video = _cut_blurred(Path(job.video), clip_start, clip_end - clip_start, sam_clip.parent / "clip_blur.mp4", None)
    scene.update(taps=[list(t) for t in taps], team=team_label, captions=captions, span_frames=list(span),
                 clip=[round(clip_start, 2), round(clip_end, 2)], clip_video=str(clip_video))
    out = _render_scene(job, scene, languages, privacy, zoom, client)
    scene.update(status="done", options={"languages": languages, "privacy": privacy, "zoom": zoom}, video=str(out))
    job.save()
    return scene


def rerender(job: Job, index: int, languages: list[str], privacy: str, zoom: float, client: Any = None) -> dict[str, Any]:
    """자막 언어·가리기·확대만 바꿔 다시 렌더. 저장해 둔 SAM 결과와 자막을 써서 API 비용이 없다.
    '몸 실루엣'을 처음 고를 때만 다른 선수 추적(SAM)을 한 번 부른다."""
    scene = job.scenes[str(index)]
    scene["status"] = "rendering"
    job.save()
    out = _render_scene(job, scene, languages, privacy, zoom, client)
    scene.update(status="done", options={"languages": languages, "privacy": privacy, "zoom": zoom}, video=str(out))
    job.save()
    return scene


def _has_audio(video: Path) -> bool:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                   "-of", "csv=p=0", str(video)], text=True)
    return bool(out.strip())


def make_reel(job: Job, order: list[int]) -> dict[str, Any]:
    """완성된 장면들을 고른 순서대로, 득점 앞뒤만 잘라 한 편으로 잇는다. 이미 렌더한 영상만 써서 API 비용이 없다."""
    scenes = [job.scenes.get(str(i), {}) for i in order]
    if not scenes or any(s.get("status") != "done" for s in scenes):
        raise ValueError("완성된 장면만 모을 수 있다")
    job.reel = {"status": "rendering", "order": order}
    job.save()

    # 소리 없는 영상이 하나라도 있으면 영상만 잇는다(concat은 모든 입력에 같은 스트림이 있어야 한다)
    audio = all(_has_audio(Path(s["video"])) for s in scenes)
    inputs: list[str] = []
    parts: list[str] = []
    for k, s in enumerate(scenes):
        clip = s.get("clip", s["window"])  # 장면 영상은 '나'가 이어서 보이는 구간만 담는다
        at = s["score_sec"] - clip[0]  # 장면 영상 안에서 득점 시각
        start = max(0.0, at - REEL_BEFORE_SEC)
        end = min(clip[1] - clip[0], at + REEL_AFTER_SEC)
        inputs += ["-i", s["video"]]
        parts.append(f"[{k}:v]trim={start:.3f}:{end:.3f},setpts=PTS-STARTPTS[v{k}];")
        if audio:
            parts.append(f"[{k}:a]atrim={start:.3f}:{end:.3f},asetpts=PTS-STARTPTS[a{k}];")
    joined = "".join(f"[v{k}]" + (f"[a{k}]" if audio else "") for k in range(len(scenes)))
    graph = "".join(parts) + f"{joined}concat=n={len(scenes)}:v=1:a={int(audio)}[v]" + ("[a]" if audio else "")
    out = job.dir / "reel.mp4"
    maps = ["-map", "[v]"] + (["-map", "[a]", "-c:a", "aac"] if audio else [])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, *maps,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    job.reel = {"status": "done", "order": order, "video": str(out)}
    job.save()
    return job.reel
