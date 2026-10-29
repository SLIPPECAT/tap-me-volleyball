"""완성된 장면의 품질 확인. 로컬, API 비용 없음.

AI(Claude)가 결과를 직접 보고 확인할 수 있게 **'나'까지 모든 얼굴을 가린** 확인용 영상과 0.5초 간격 장면 묶음을 만들고,
숫자로 점검한다. 결과 영상에는 '나'의 얼굴이 그대로라, 그 영상은 AI가 보지 않는다(회원 얼굴, 10/29 원칙).

  uv run python -m tapme.qa <작업 id> [장면 번호]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tapme import anonymize, jobs, render, track

SHEET_EVERY_SEC = 0.5
SHEET_TILE = (216, 384)


def _frames(video: Path, w: int, h: int, pix: str = "bgr24") -> np.ndarray:
    raw = subprocess.check_output(["ffmpeg", "-loglevel", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", pix, "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def tracking_stats(by: dict[int, list[track.Box]], sam_fps: float, clip_sec: float) -> dict[str, Any]:
    """'나'를 잡은 비율, 가장 긴 끊김(초), SAM 프레임 사이 이동(px)."""
    total = max(1, round(clip_sec * sam_fps))
    seen = sorted(k for k, v in by.items() if 0 <= k < total and any(b.object_id == track.TAPPED_ID for b in v))
    gaps = [b - a - 1 for a, b in zip([-1, *seen], [*seen, total], strict=True)]
    centers = {k: next(b.center for b in by[k] if b.object_id == track.TAPPED_ID) for k in seen}
    jumps = [((centers[b][0] - centers[a][0]) ** 2 + (centers[b][1] - centers[a][1]) ** 2) ** 0.5
             for a, b in pairwise(seen) if b == a + 1]
    return {
        "me_seen_ratio": round(len(seen) / total, 2),
        "longest_gap_sec": round(max(gaps) / sam_fps, 2) if gaps else 0.0,
        "jump_px_median": round(statistics.median(jumps), 1) if jumps else None,
        "jump_px_max": round(max(jumps), 1) if jumps else None,
    }


def missed_face_candidates(src: Path, boxes: list[list[track.Box]], crops: list[tuple[int, int, int, int]],
                           every: int = 3) -> dict[str, int]:
    """렌더가 쓰는 얼굴 검출(기준 0.5, 2배)이 놓쳤을 수 있는 얼굴 수. 다른 설정(YuNet 0.6, 원래 크기)으로 한 번 더 찾아
    렌더 쪽 박스와 겹치지 않고, 결과 화면 안이고, '나'가 아닌 것을 센다. 얼굴을 보지 않고 숫자로만."""
    w, h, _, n = render.probe(src)
    fr = _frames(src, w, h)
    det, prof = anonymize.load_detector(), anonymize.load_profile_detector()
    alt = cv2.FaceDetectorYN.create(str(anonymize.MODEL_PATH), "", (w, h), 0.6, 0.3, 5000)
    checked = missed = 0
    for i in range(0, len(fr), every):
        k = min(i, n - 1)
        x0, y0, cw, ch = crops[k]
        mine = next((b for b in boxes[k] if b.object_id == track.TAPPED_ID), None)
        used = anonymize.detect_faces(fr[i], det, prof)
        _, found = alt.detect(fr[i])
        for v in [] if found is None else found:
            x, y, fw, fh = (float(a) for a in v[:4])
            cx, cy = x + fw / 2, y + fh / 2
            if not (x0 <= cx <= x0 + cw and y0 <= cy <= y0 + ch):
                continue  # 결과 화면 밖
            if mine and mine.left <= cx <= mine.right and mine.top <= cy <= mine.bottom:
                continue  # '나'는 일부러 안 가린다
            checked += 1
            if not any(abs(cx - (ux + uw / 2)) < max(fw, uw) * 0.6 and abs(cy - (uy + uh / 2)) < max(fh, uh) * 0.6
                       for ux, uy, uw, uh in used):
                missed += 1
    return {"faces_checked": checked, "missed_candidates": missed, "frames_checked": len(range(0, len(fr), every))}


def contact_sheet(video: Path, out: Path, score_at: float, cues: list[tuple[float, float, str]]) -> Path:
    w, h, fps, _ = render.probe(video)
    fr = _frames(video, w, h, "rgb24")
    step = max(1, round(fps * SHEET_EVERY_SEC))
    picks = list(range(0, len(fr), step))
    cols = 6
    tw, th = SHEET_TILE
    sheet = Image.new("RGB", (tw * cols, (th + 20) * ((len(picks) + cols - 1) // cols)), (20, 20, 20))
    for j, i in enumerate(picks):
        tile = Image.fromarray(fr[i]).resize((tw, th))
        t = i / fps
        label = f"{t:.1f}s" + ("  << SCORE" if abs(t - score_at) < SHEET_EVERY_SEC / 2 else "")  # 기본 글꼴에 한글 없음
        x, y = (j % cols) * tw, (j // cols) * (th + 20)
        sheet.paste(tile, (x, y + 20))
        ImageDraw.Draw(sheet).text((x + 4, y + 3), label, fill=(255, 220, 0))
    sheet.save(out, quality=85)
    return out


def check_scene(job_id: str, index: int = 0) -> dict[str, Any]:
    job = jobs.load_job(job_id)
    scene = job.scenes[str(index)]
    opts = scene.get("options", {})
    languages, privacy, zoom = opts.get("languages", []), opts.get("privacy", "face"), opts.get("zoom", 2.0)
    src, by, masks, cues = jobs.scene_render_inputs(job, scene, languages, privacy, client=None)
    out_dir = Path(scene["sam_clip"]).parent
    qa_video = render.render_video(src, out_dir / "qa.mp4", by, masks, track.TAPPED_ID, scene["sam_fps"], cues, zoom,
                                   privacy, face_mosaic=True, keep_my_face=False)
    clip_start, clip_end = scene["clip"]
    score_at = scene["score_sec"] - clip_start
    sheet = contact_sheet(qa_video, out_dir / "qa_sheet.jpg", score_at, cues)

    w, h, fps, n = render.probe(src)
    boxes = render.interpolate(by, scene["sam_fps"], fps, n)
    crops = render.zoom_crops([next((b.center for b in f if b.object_id == track.TAPPED_ID), None) for f in boxes],
                              w, h, zoom, smooth=max(1, round(fps / 2)))
    report = {
        "job": job_id, "scene": index, "options": opts,
        "clip_sec": [clip_start, clip_end], "score_sec": scene["score_sec"],
        "score_in_clip": clip_start <= scene["score_sec"] <= clip_end,
        "tracking": tracking_stats(by, scene["sam_fps"], clip_end - clip_start),
        "subtitles": [{"from": a, "to": b, "text": t} for a, b, t in cues],
        "privacy": missed_face_candidates(src, boxes, crops) if opts.get("face_mosaic", True) else "mosaic off",
        "qa_video": str(qa_video), "qa_sheet": str(sheet),
    }
    (out_dir / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="완성된 장면 품질 확인 (로컬, 무료)")
    parser.add_argument("job")
    parser.add_argument("scene", type=int, nargs="?", default=0)
    args = parser.parse_args()
    print(json.dumps(check_scene(args.job, args.scene), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
