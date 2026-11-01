"""Muse Spark에 랠리의 득점 시점과, 중계처럼 장면을 따라가는 해설 자막(한국어·일본어·영어)을 묻는다.

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
# '나'의 역할(사용자가 고름, 선택). AI가 판정하지 않는다
ROLE_TEXT = {
    "": "",
    "score": "The highlighted player (the user, ring at their feet) won the point themselves (attack, block or serve).",
    "assist": "The highlighted player (the user, ring at their feet) made the point possible with a pass or a set.",
    "defense": "The highlighted player (the user, ring at their feet) kept the rally alive with a dig or defensive save.",
}


# 득점 팀(jobs.point_winners, H-55): 전에는 늘 "우리 팀 득점"이라고 써 상대가 득점한 랠리에 틀린 해설이 붙었다(11/1)
WINNER_TEXT = {
    "mine": "The point at {t:.1f}s was won by the team the user calls '{team}' (do not change it). ",
    "opponent": ("The point at {t:.1f}s was won by the opponents of the team the user calls '{team}' (do not change it). "
                 "Call it honestly but focus on how '{team}' fought for the rally. "),
    None: ("The point is decided at {t:.1f}s; it is not known which team won it, so never say which team won. "
           "The user's team is the one they call '{team}'. "),
}


def commentary_prompt(team_label: str, score_time_sec: float, role: str, clip_sec: float, winner: str | None = "mine") -> str:
    """실제 배구 중계처럼 장면에 맞춰 나오는 해설 2~4줄. 마지막 줄은 득점 순간. winner: mine · opponent · None(모름)."""
    if role not in ROLE_TEXT:
        raise ValueError(f"역할은 {tuple(ROLE_TEXT)} 중 하나: {role}")
    # '내가 득점'은 우리 팀이 득점했을 때만 맞는 말이다
    who = (ROLE_TEXT[role] if winner == "mine" or role != "score" else "") or \
        "Do not describe the highlighted player's role; talk about the team's play."
    return (
        f"This {clip_sec:.0f}-second clip is one rally from an amateur volleyball match (faces are pixelated). "
        + WINNER_TEXT[winner].format(t=score_time_sec, team=team_label) + who + " "
        "Write 2 to 4 short lines of live TV volleyball commentary that follow the play, like a broadcaster "
        "calling the game: build up through the rally and make the last line the scoring moment, timed near "
        f"{score_time_sec:.1f}s. Only describe plays that are clearly visible; never invent a play. "
        "Each line at most 22 characters in Korean and Japanese, 40 in English. "
        "Translate the team name naturally into each language (for example a colour name '검정' becomes '黒' "
        "in Japanese and 'Black' in English). "
        'Answer with JSON only: {"lines": [{"t": seconds from the start of this clip, '
        '"ko": "...", "ja": "...", "en": "..."}]}'
    )


def subtitle_cues(
    lines: list[dict[str, Any]] | None, languages: list[str], offset: float, duration: float
) -> list[tuple[float, float, str]]:
    """해설 줄을 렌더할 클립 기준 (시작초, 끝초, 글자)로. 줄마다 다음 줄이 나올 때까지 보인다.

    offset = 해설 시각의 기준(득점 구간 시작)에서 렌더한 클립 시작까지의 초. 고른 언어가 없으면 자막 없음.
    """
    unknown = [lang for lang in languages if lang not in LANGUAGES]
    if unknown:
        raise ValueError(f"지원하는 자막 언어는 {LANGUAGES}: {unknown}")
    if not lines or not languages:
        return []
    ordered = sorted(lines, key=lambda line: float(line["t"]))
    cues = []
    for k, line in enumerate(ordered):
        start = max(0.0, float(line["t"]) - offset)
        end = duration if k + 1 == len(ordered) else min(duration, float(ordered[k + 1]["t"]) - offset)
        text = "\n".join(line[lang] for lang in languages if line.get(lang))
        if end > start and text:
            cues.append((round(start, 2), round(end, 2), text))
    return cues


# 득점 팀을 알려고 다음 랠리의 서브 쪽을 묻는다(배구는 랠리를 이긴 팀이 다음 서브). 유니폼 색은 비슷한 색 팀에서
# 틀려(H-54, 76%) 위치를 묻는다(H-55: 서브 쪽 15/17, 득점 팀 81~88%)
SERVE_SIDE_PROMPT = (
    "This short clip is the start of one rally in an amateur volleyball match, filmed from behind one end of the court "
    "(faces are pixelated). Which side serves: the team on the NEAR court (closer to the camera, players appear larger and "
    "lower in the picture) or the FAR court (beyond the net)? The server tosses and hits the ball from behind an end line. "
    'Answer with JSON only: {"server_side": "near" or "far" or "unsure", "confidence": 0.0 to 1.0, "reason": "one short sentence"}'
)


def ask_serve_side(client: Any, clip: Path) -> tuple[dict[str, Any], muse.MuseReply]:
    reply = muse.ask_about_video(client, clip, SERVE_SIDE_PROMPT)
    try:
        return muse.parse_json_reply(reply.text), reply
    except ValueError:  # 빈 답이 온 적이 있다(H-54)
        return {"server_side": "unsure", "confidence": 0.0}, reply


def ask_score_time(client: Any, rally_clip: Path) -> tuple[dict[str, Any], muse.MuseReply]:
    reply = muse.ask_about_video(client, rally_clip, SCORE_TIME_PROMPT)
    return muse.parse_json_reply(reply.text), reply


def ask_commentary(
    client: Any, clip: Path, team_label: str, score_time_sec: float, role: str, clip_sec: float, winner: str | None = "mine"
) -> tuple[list[dict[str, Any]] | None, muse.MuseReply]:
    """(해설 줄, 답). 답을 읽을 수 없으면 줄 대신 None(낸 돈은 reply로 센다)."""
    reply = muse.ask_about_video(client, clip, commentary_prompt(team_label, score_time_sec, role, clip_sec, winner))
    try:
        lines = muse.parse_json_reply(reply.text).get("lines") or []
    except ValueError:
        return None, reply
    return [line for line in lines if "t" in line], reply
