"""영상을 60초 조각으로 훑어 랠리와 득점 순간을 모두 찾는다. 길이와 상관없이 쓰는 기본 랠리 찾기.

휘슬 규칙(rally.py)은 정모 영상에 맞춘 값이라 다른 체육관·대회 영상에서 무너졌다(7분 영상에 휘슬 85개, 실험 H-34).
조각마다 얼굴 흐림본을 Muse에 보이고 랠리 목록(서브 시각·득점 순간)을 받는다. 7분 영상에서 득점 15/15(H-34).
조각을 120초로 키우면 붙어 있는 짧은 랠리를 놓쳤고(H-35), 2fps는 판정을 바꾸지 않아 흐리기 시간을 줄이는 데 쓴다.
분석용 해상도는 줄이지 않는다(가로 영상은 360p로 줄여도 입력 토큰이 그대로였다, H-35).

사용법 (유료 API. --yes 없이 돌리면 예상 금액만 보여 준다):
  uv run python -m tapme.scan data/match.mp4 --work data/scan [--yes]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tapme import anonymize, muse

CHUNK_SEC = 60.0
STEP_SEC = 50.0  # 10초 겹침: 조각 경계에 걸친 랠리도 한 조각에는 온전히 들어간다
CLIP_FPS = 2
SAME_POINT_SEC = 2.0  # 겹친 조각에서 이 안의 득점은 같은 것
END_AFTER_POINT_SEC = 1.0  # 랠리 끝 = 득점 순간 + 이만큼
# 이보다 붙은 두 득점은 한 랠리를 둘로 짚었을 수 있다(대회 영상 A·C에서 3건, 모두 확신이 낮은 쪽이 헛것).
# 진짜로 붙은 득점도 있어(B 7.7초, 서브 범실 등) 지우지 않고 확신 낮은 쪽을 '불확실'로 표시한다(H-36)
CLOSE_POINTS_SEC = 10.0
PARALLEL = 3
SCAN_PROMPT = (
    "This {dur:.0f}-second clip is part of a volleyball match video (faces are pixelated for privacy). "
    "List every rally in it. A rally is play from the serve until the ball is dead. "
    "For each rally give times in seconds from the start of this clip: start_sec (serve contact), "
    "point_sec (the moment the ball becomes dead and the point is decided), and whether the rally was "
    "already going when the clip started (cut_start) or still going when it ended (cut_end, then point_sec is null). "
    'Answer with JSON only: {{"rallies": [{{"start_sec": 0.0, "point_sec": 0.0, "cut_start": false, '
    '"cut_end": false, "confidence": 0.0}}]}}'
)


def duration_sec(video: Path) -> float:
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]))


def chunk_starts(duration: float) -> list[float]:
    starts = [0.0]
    while starts[-1] + CHUNK_SEC < duration:
        starts.append(starts[-1] + STEP_SEC)
    return starts


def estimate_usd(duration: float) -> float:
    return sum(muse.estimate_usd(min(CHUNK_SEC, duration - s)) for s in chunk_starts(duration))


def rallies_from_answers(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """조각 답(조각 안 시각)을 원본 시각 랠리 목록으로. 같은 득점은 잘리지 않은 쪽, 확신 높은 쪽 하나만."""
    found = []
    for c in chunks:
        for x in c["answer"].get("rallies") or []:
            if x.get("point_sec") is None or x.get("cut_end"):
                continue  # 조각 끝에서 잘린 랠리는 다음 조각(겹침)에서 온전히 본다
            point = c["start"] + float(x["point_sec"])
            start = c["start"] + float(x["start_sec"]) if x.get("start_sec") is not None else point - 10.0
            found.append({"point_sec": round(point, 2), "start_sec": round(max(0.0, min(start, point - 1.0)), 2),
                          "cut_start": bool(x.get("cut_start")), "confidence": float(x.get("confidence") or 0)})
    found.sort(key=lambda r: r["point_sec"])
    out: list[dict[str, Any]] = []
    for r in found:
        if out and r["point_sec"] - out[-1]["point_sec"] <= SAME_POINT_SEC:
            if (not r["cut_start"], r["confidence"]) > (not out[-1]["cut_start"], out[-1]["confidence"]):
                out[-1] = r
        else:
            out.append(r)
    doubtful = set()
    for i in range(1, len(out)):
        if out[i]["point_sec"] - out[i - 1]["point_sec"] < CLOSE_POINTS_SEC:
            doubtful.add(i if out[i]["confidence"] <= out[i - 1]["confidence"] else i - 1)
    return [{**r, "end_sec": round(r["point_sec"] + END_AFTER_POINT_SEC, 2), "is_rally": True, "reason": None,
             "uncertain": r["confidence"] < 0.5 or i in doubtful} for i, r in enumerate(out)]


def blurred_chunk(video: Path, start: float, seconds: float, out: Path) -> Path:
    """조각을 CLIP_FPS로 잘라 얼굴을 흐린다. 얼굴이 그대로인 잘린 사본은 남기지 않는다."""
    raw = out.with_name(out.stem + "_raw.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(video), "-t", f"{seconds:.2f}",
         "-vf", f"setpts=PTS-STARTPTS,fps={CLIP_FPS}:round=up", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(raw)],
        check=True,
    )
    anonymize.anonymize_video(raw, out)
    raw.unlink()
    return out


def scan_video(
    client: Any, video: Path, duration: float, work_dir: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(랠리 목록, 조각별 답·비용). 흐리기는 CPU를 다퉈 하나씩, Muse 질문만 겹친다(H-31)."""
    starts = chunk_starts(duration)
    done = 0
    lock = threading.Lock()

    def ask(i: int, start: float, clip: Path) -> dict[str, Any]:
        nonlocal done
        seconds = min(CHUNK_SEC, duration - start)
        reply = muse.ask_about_video(client, clip, SCAN_PROMPT.format(dur=seconds))
        try:
            answer = muse.parse_json_reply(reply.text)
        except ValueError:
            answer = {"rallies": [], "unreadable": True}
        with lock:
            done += 1
            if on_progress:
                on_progress(done, len(starts))
        return {"chunk": i, "start": start, "answer": answer, "usd": reply.usd,
                "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = [
            pool.submit(ask, i, s, blurred_chunk(video, s, min(CHUNK_SEC, duration - s), work_dir / f"chunk{i:03d}_blur.mp4"))
            for i, s in enumerate(starts)
        ]
        chunks = [f.result() for f in futures]
    return rallies_from_answers(chunks), chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="영상을 60초 조각으로 훑어 랠리·득점 찾기 (유료)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--work", type=Path, required=True, help="흐림본·결과 저장 폴더 (data/ 아래)")
    parser.add_argument("--yes", action="store_true", help="예상 금액을 보고 실제로 호출")
    args = parser.parse_args()

    duration = duration_sec(args.video)
    print(f"{duration / 60:.1f}분, 조각 {len(chunk_starts(duration))}개 → 예상 ${estimate_usd(duration):.3f}")
    if not args.yes:
        print("--yes 없이 실행해 API는 부르지 않았다")
        return
    rallies, chunks = scan_video(muse.client_from_env(), args.video, duration, args.work)
    (args.work / "scan.json").write_text(json.dumps({"rallies": rallies, "chunks": chunks}, ensure_ascii=False, indent=1))
    for r in rallies:
        print(f"{int(r['point_sec'] // 60)}:{r['point_sec'] % 60:04.1f}  서브 {r['start_sec']}  확신 {r['confidence']}")
    print(f"득점 {len(rallies)}개, 합계 ${sum(c['usd'] for c in chunks):.4f}")


if __name__ == "__main__":
    main()
