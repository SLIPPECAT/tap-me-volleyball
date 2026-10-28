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
CLIP_FPS = 5  # Muse 입력 토큰은 fps와 무관하게 길이에 비례하므로 업로드 크기만 줄인다
RALLY_CHECK_PROMPT = (
    "This clip is part of an amateur volleyball match video (faces are pixelated for privacy). "
    "Does it contain a rally, meaning play from a serve until the ball is dead? "
    "Players standing, clapping, rotating positions, or lining up at the net is not a rally. "
    'Answer with JSON only: {"is_rally": true or false, "confidence": 0.0 to 1.0, "reason": "one short sentence"}'
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


def check_candidates(
    client: Any, video: Path, candidates: list[rally.RallyCandidate], work_dir: Path
) -> list[dict[str, Any]]:
    results = []
    for i, c in enumerate(candidates):
        raw = cut_clip(video, c.start_sec, c.end_sec, work_dir / f"cand{i:02d}_raw.mp4")
        blurred = work_dir / f"cand{i:02d}_blur.mp4"
        anonymize.anonymize_video(raw, blurred)
        raw.unlink()  # 얼굴이 그대로인 사본은 남기지 않는다
        reply = muse.ask_about_video(client, blurred, RALLY_CHECK_PROMPT)
        answer = muse.parse_json_reply(reply.text)
        results.append({
            **asdict(c),
            "clip": str(blurred),
            "is_rally": bool(answer.get("is_rally")),
            "confidence": answer.get("confidence"),
            "reason": answer.get("reason"),
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "usd": round(reply.usd, 6),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="휘슬 후보 구간을 Muse로 거르기 (유료)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--work", type=Path, required=True, help="흐림본·결과 저장 폴더 (data/ 아래)")
    parser.add_argument("--yes", action="store_true", help="예상 금액을 보고 실제로 호출")
    args = parser.parse_args()

    candidates = rally.pair_rallies(rally.find_whistles(rally.load_audio(args.video)))
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
