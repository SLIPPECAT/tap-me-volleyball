"""Muse Spark에 랠리의 득점 시점과 하이라이트 자막(한국어·일본어·영어)을 묻는다.

실험(실험 레포 H-16): 득점 시점은 ±1초 안팎으로 맞는다. H-07: 득점 팀은 5번 중 2번만 맞아
팀은 묻지 않고 사용자가 고른 값을 자막에 넣는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tapme import muse

SCORE_TIME_PROMPT = (
    "This is one rally from an amateur volleyball match (faces are pixelated). "
    "When is the point decided, i.e. the moment the ball touches the floor, goes out, or a fault is clear? "
    'Answer with JSON only: {"score_time_sec": seconds from the start of this clip with one decimal, '
    '"confidence": 0.0 to 1.0, "reason": "one short sentence"}'
)


# 자막 언어. 한 번에 셋 다 받아 두고, 화면에는 사용자가 고른 것만 고른 순서로 넣는다
LANGUAGES = ("ko", "ja", "en")


def caption_prompt(team_label: str, score_time_sec: float) -> str:
    return (
        f"This is one rally from an amateur volleyball match. The point at {score_time_sec:.1f}s was won by "
        f"the team the user calls '{team_label}' (decided by the user, do not change it). "
        "Write a short highlight caption in Korean, Japanese and English, at most 20 characters each "
        "(English at most 30). "
        "Name the kind of play (spike, block, ace, ...) only if it is clearly visible; otherwise just say it is a point. "
        'Answer with JSON only: {"ko": "...", "ja": "...", "en": "..."}'
    )


def subtitle_text(captions: dict[str, str], languages: list[str]) -> str:
    """고른 언어의 자막을 고른 순서대로 한 줄씩. 빈 목록이면 자막 없음."""
    unknown = [lang for lang in languages if lang not in LANGUAGES]
    if unknown:
        raise ValueError(f"지원하는 자막 언어는 {LANGUAGES}: {unknown}")
    return "\n".join(captions[lang] for lang in languages if captions.get(lang))


def ask_score_time(client: Any, rally_clip: Path) -> tuple[dict[str, Any], muse.MuseReply]:
    reply = muse.ask_about_video(client, rally_clip, SCORE_TIME_PROMPT)
    return muse.parse_json_reply(reply.text), reply


def ask_caption(client: Any, clip: Path, team_label: str, score_time_sec: float) -> tuple[dict[str, Any], muse.MuseReply]:
    reply = muse.ask_about_video(client, clip, caption_prompt(team_label, score_time_sec))
    return muse.parse_json_reply(reply.text), reply
