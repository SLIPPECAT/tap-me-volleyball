"""휘슬 후보 구간이 진짜 랠리인지 Muse Spark로 거른다.

정모에서는 랠리가 아닐 때(자리 바꿈·인사)도 휘슬을 불어 소리만으로는 못 가른다.
후보를 앞뒤로 조금 넓혀 5fps로 자르고, 얼굴을 흐린 뒤 그 흐림본만 Muse에 보낸다.

사용법 (유료 API. --yes 없이 돌리면 예상 금액만 보여 준다):
  uv run python -m tapme.rally_filter data/match.mp4 --work data/rally_check [--yes]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tapme import anonymize, muse, rally

PAD_SEC = 1.0
# 10/28 서버 확인: 같은 흐림본(입력 토큰도 같음)에 한 번은 "랠리"(0.95), 한 번은 "영상 프레임이 없다"(0.0)고 답했다.
# 확신이 낮은 답은 한 번 더 묻고, 그래도 낮으면 '불확실'로 표시해 사용자가 판단하게 한다
MIN_CONFIDENCE = 0.5
MAX_ATTEMPTS = 2
CLIP_FPS = 5  # Muse 입력 토큰은 fps와 무관하게 길이에 비례하므로 업로드 크기만 줄인다
RALLY_CHECK_PROMPT = (
    "This clip is part of an amateur volleyball match video (faces are pixelated for privacy). "
    "Does it contain a rally, meaning play from a serve until the ball is dead? "
    "Players standing, clapping, rotating positions, or lining up at the net is not a rally. "
    "Give the reason as one short sentence in Korean, Japanese and English (shown to the user in their language). "
    'Answer with JSON only: {"is_rally": true or false, "confidence": 0.0 to 1.0, '
    '"reason": {"ko": "...", "ja": "...", "en": "..."}}'
)


def cut_clip(video: Path, start_sec: float, end_sec: float, out: Path) -> Path:
    start = max(0.0, start_sec - PAD_SEC)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(video),
         "-t", f"{end_sec + PAD_SEC - start:.2f}", "-r", str(CLIP_FPS),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        check=True,
    )
    return out


def blurred_clip(video: Path, start_sec: float, end_sec: float, out: Path) -> Path:
    """앞뒤로 넓혀 자르고 얼굴을 흐린 클립. 얼굴이 그대로인 잘린 사본은 남기지 않는다."""
    raw = cut_clip(video, start_sec, end_sec, out.with_name(out.stem + "_raw.mp4"))
    anonymize.anonymize_video(raw, out)
    raw.unlink()
    return out


JOIN_GAP_SEC = 10.0  # 이웃한 두 랠리 판정 사이(휘슬 사건 하나의 길이)가 이 안이면 잇는다
# Muse는 플레이가 조금만 보여도 랠리로 봐서, 이웃한 랠리를 모두 이으면 두 랠리가 하나가 됐다(S3, H-33).
# 이은 길이가 후보 최대 길이를 넘지 않게 한다. "점수가 났는가"를 함께 묻는 방법은 절반만 맞아 쓰지 않는다
MAX_RALLY_SEC = rally.SEGMENT_SEC[1]


def merge_rallies(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """휘슬을 사이에 두고 이어진 '랠리' 판정 구간을 하나로 합친다(랠리 중간 휘슬로 잘린 랠리를 되살림).

    합친 랠리는 'parts'에 원래 후보 번호를 담고 'clip'을 비워 둔다(호출하는 쪽이 새 흐림 클립을 만든다).
    """
    out: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        prev = out[-1] if out else None
        # 후보는 휘슬 사건 사이를 좁게 잡는다(H-31). 바로 이웃한 두 후보 사이는 휘슬 사건 하나이므로 그 사건을 건너 잇는다
        if (prev and prev["is_rally"] and r["is_rally"] and prev["parts"][-1] == i - 1
                and r["start_sec"] - prev["end_sec"] <= JOIN_GAP_SEC
                and r["end_sec"] - prev["start_sec"] <= MAX_RALLY_SEC):
            prev.update(
                end_sec=max(prev["end_sec"], r["end_sec"]), parts=[*prev["parts"], i], clip=None,
                confidence=min(float(prev.get("confidence") or 0), float(r.get("confidence") or 0)),
                uncertain=bool(prev.get("uncertain") or r.get("uncertain")),
                usd=round(prev["usd"] + r["usd"], 6),
                input_tokens=prev["input_tokens"] + r["input_tokens"],
                output_tokens=prev["output_tokens"] + r["output_tokens"],
            )
        else:
            out.append({**r, "parts": [i]})
    return out


def _ask(client: Any, c: rally.RallyCandidate, blurred: Path) -> dict[str, Any]:
    replies, answer = [], {}
    for _ in range(MAX_ATTEMPTS):
        replies.append(muse.ask_about_video(client, blurred, RALLY_CHECK_PROMPT))
        try:
            answer = muse.parse_json_reply(replies[-1].text)
        except ValueError:  # 빈 답·잘린 JSON 하나로 후보 확인 전체가 멈췄다(11/3 Muse Code 검토) → 확신 없음으로 다시 묻기
            answer = {}
        if float(answer.get("confidence") or 0) >= MIN_CONFIDENCE:
            break
    return {
        **asdict(c),
        "clip": str(blurred),
        "clip_start_sec": max(0.0, c.start_sec - PAD_SEC),  # 클립 안 시각 + 이 값 = 원본 시각
        "is_rally": bool(answer.get("is_rally")),
        "confidence": answer.get("confidence"),
        "reason": answer.get("reason"),
        "uncertain": float(answer.get("confidence") or 0) < MIN_CONFIDENCE,
        "attempts": len(replies),
        "input_tokens": sum(r.input_tokens for r in replies),
        "output_tokens": sum(r.output_tokens for r in replies),
        "usd": round(sum(r.usd for r in replies), 6),
    }


PARALLEL = 3  # 후보를 하나씩 묻느라 랠리 찾기가 오래 걸렸다(후보 3개 95초). Muse 질문은 동시에 한다


def check_candidates(
    client: Any, video: Path, candidates: list[rally.RallyCandidate], work_dir: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """후보마다 흐림 클립을 만들어 Muse에 묻는다. 결과는 후보 순서 그대로. on_progress(끝난 수, 전체).

    흐리기는 CPU를 써서 셋을 동시에 돌리면 서로 느려졌다(정모 12분, H-31). 흐리기는 하나씩, 질문만 겹친다.
    """
    done = 0
    lock = threading.Lock()

    def ask(c: rally.RallyCandidate, blurred: Path) -> dict[str, Any]:
        nonlocal done
        result = _ask(client, c, blurred)
        with lock:
            done += 1
            if on_progress:
                on_progress(done, len(candidates))
        return result

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = [
            pool.submit(ask, c, blurred_clip(video, c.start_sec, c.end_sec, work_dir / f"cand{i:02d}_blur.mp4"))
            for i, c in enumerate(candidates)
        ]
        return [f.result() for f in futures]


def main() -> None:
    parser = argparse.ArgumentParser(description="휘슬 후보 구간을 Muse로 거르기 (유료)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--work", type=Path, required=True, help="흐림본·결과 저장 폴더 (data/ 아래)")
    parser.add_argument("--yes", action="store_true", help="예상 금액을 보고 실제로 호출")
    args = parser.parse_args()

    audio = rally.load_audio(args.video)
    candidates = rally.segments(rally.find_whistles(audio), audio.size / rally.SAMPLE_RATE)
    lengths = [c.end_sec - c.start_sec + 2 * PAD_SEC for c in candidates]
    print(f"후보 {len(candidates)}개, 약 {sum(lengths):.0f}초 → 예상 ${sum(map(muse.estimate_usd, lengths)):.3f}")
    if not args.yes:
        print("--yes 없이 실행해 API는 부르지 않았다")
        return

    results = check_candidates(muse.client_from_env(), args.video, candidates, args.work)
    (args.work / "rally_check.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"합계 ${sum(r['usd'] for r in results):.4f}")


if __name__ == "__main__":
    main()
