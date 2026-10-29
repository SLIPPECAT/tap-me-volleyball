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


def merge_rallies(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """휘슬을 사이에 두고 이어진 '랠리' 판정 구간을 하나로 합친다(랠리 중간 휘슬로 잘린 랠리를 되살림).

    합친 랠리는 'parts'에 원래 후보 번호를 담고 'clip'을 비워 둔다(호출하는 쪽이 새 흐림 클립을 만든다).
    """
    out: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        prev = out[-1] if out else None
        if prev and prev["is_rally"] and r["is_rally"] and abs(prev["end_sec"] - r["start_sec"]) < 0.01:
            prev.update(
                end_sec=r["end_sec"], parts=[*prev["parts"], i], clip=None,
                confidence=min(float(prev.get("confidence") or 0), float(r.get("confidence") or 0)),
                uncertain=bool(prev.get("uncertain") or r.get("uncertain")),
                usd=round(prev["usd"] + r["usd"], 6),
                input_tokens=prev["input_tokens"] + r["input_tokens"],
                output_tokens=prev["output_tokens"] + r["output_tokens"],
            )
        else:
            out.append({**r, "parts": [i]})
    return out


def check_candidates(
    client: Any, video: Path, candidates: list[rally.RallyCandidate], work_dir: Path
) -> list[dict[str, Any]]:
    results = []
    for i, c in enumerate(candidates):
        blurred = blurred_clip(video, c.start_sec, c.end_sec, work_dir / f"cand{i:02d}_blur.mp4")
        replies, answer = [], {}
        for _ in range(MAX_ATTEMPTS):
            replies.append(muse.ask_about_video(client, blurred, RALLY_CHECK_PROMPT))
            answer = muse.parse_json_reply(replies[-1].text)
            if float(answer.get("confidence") or 0) >= MIN_CONFIDENCE:
                break
        results.append({
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
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="휘슬 후보 구간을 Muse로 거르기 (유료)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--work", type=Path, required=True, help="흐림본·결과 저장 폴더 (data/ 아래)")
    parser.add_argument("--yes", action="store_true", help="예상 금액을 보고 실제로 호출")
    args = parser.parse_args()

    candidates = rally.segments(rally.find_whistles(rally.load_audio(args.video)))
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
