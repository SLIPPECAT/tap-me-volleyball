"""Muse Spark 호출 (Meta Model API, OpenAI SDK 호환).

영상은 얼굴 흐림본만 보낸다. Contributor 모드(입력을 학습에 씀) 모델은 거부한다.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.3"
INPUT_USD_PER_TOKEN = 1.25 / 1_000_000
OUTPUT_USD_PER_TOKEN = 4.25 / 1_000_000
# 실측(실험 레포 H-08·H-16): 입력 토큰은 fps와 무관하게 영상 길이에 비례, 초당 약 280. 출력은 답 하나에 약 2,500
INPUT_TOKENS_PER_SEC = 280
OUTPUT_TOKENS_GUESS = 2500


@dataclass(frozen=True)
class MuseReply:
    text: str
    input_tokens: int
    output_tokens: int
    usd: float
    seconds: float


def client_from_env() -> Any:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    return OpenAI(api_key=os.environ["MODEL_API_KEY"], base_url=BASE_URL)


def estimate_usd(video_sec: float) -> float:
    return video_sec * INPUT_TOKENS_PER_SEC * INPUT_USD_PER_TOKEN + OUTPUT_TOKENS_GUESS * OUTPUT_USD_PER_TOKEN


def forget_file(client: Any, file_id: str) -> None:
    """Meta에 올린 클립을 답을 받은 뒤 지운다. 블러본이라도 회원 영상은 남겨 두지 않는다(10/31 사용자 결정).
    지우기가 실패해도 받은 답은 그대로 쓴다."""
    try:
        client.files.delete(file_id)
    except Exception as e:  # noqa: BLE001 - 지우기 실패로 작업을 멈추지 않는다
        print(f"Meta 파일 지우기 실패 {file_id}: {e}")


def ask_about_video(client: Any, video: Path, prompt: str, model: str = MODEL) -> MuseReply:
    if "contributor" in model.lower():
        raise ValueError("Contributor 모드 모델은 쓰지 않는다(입력이 학습에 쓰임)")
    with video.open("rb") as f:
        file_id = client.files.create(purpose="user_data", file=f).id
    try:
        return _ask(client, file_id, prompt, model)
    finally:
        forget_file(client, file_id)


def _ask(client: Any, file_id: str, prompt: str, model: str) -> MuseReply:
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=model,
        max_output_tokens=8000,  # 3개 언어 해설처럼 긴 답은 4000에서 잘려 빈 답이 왔다(10/29)
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_file", "file_id": file_id},
            ],
        }],
    )
    in_tok = int(getattr(resp.usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(resp.usage, "output_tokens", 0) or 0)
    return MuseReply(
        text=resp.output_text or "",
        input_tokens=in_tok,
        output_tokens=out_tok,
        usd=in_tok * INPUT_USD_PER_TOKEN + out_tok * OUTPUT_USD_PER_TOKEN,
        seconds=round(time.perf_counter() - t0, 2),
    )


def parse_json_reply(text: str) -> dict[str, Any]:
    """답에서 첫 JSON 객체를 꺼낸다(코드 블록으로 감싸 와도 됨)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"JSON이 없는 답: {text[:200]}")
    return json.loads(m.group())
