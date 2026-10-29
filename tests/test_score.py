from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tapme import score


def _client(answer: str, sent: dict) -> SimpleNamespace:
    def respond(**kw):
        sent.update(kw)
        return SimpleNamespace(output_text=answer, usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    return SimpleNamespace(files=SimpleNamespace(create=lambda purpose, file: SimpleNamespace(id="f")),
                           responses=SimpleNamespace(create=respond))


def test_commentary_prompt_fixes_team_role_and_asks_timed_lines() -> None:
    # Muse의 팀 판정은 못 믿으므로(실험 H-07) 사용자가 고른 팀·역할을 그대로 준다
    p = score.commentary_prompt("검정", 6.0, "assist", 10.0)
    assert "'검정'" in p and "6.0s" in p and "do not change it" in p
    assert score.ROLE_TEXT["assist"] in p  # 사용자가 고른 역할이 문장에 들어간다
    assert '"lines"' in p and '"t"' in p and '"ko"' in p and '"ja"' in p and '"en"' in p
    assert "Translate the team name" in p  # 10/28: "Point for 검정!"이 나왔다
    # 역할을 안 고르면 '나'의 역할을 지어내지 않는다
    assert "Do not describe the highlighted player's role" in score.commentary_prompt("검정", 6.0, "", 10.0)


def test_unknown_role_is_refused() -> None:
    with pytest.raises(ValueError):
        score.commentary_prompt("검정", 6.0, "captain", 10.0)


LINES = [
    {"t": 1.0, "ko": "서브 들어갑니다", "ja": "サーブ", "en": "Serve"},
    {"t": 4.0, "ko": "토스 올라갑니다!", "ja": "トス！", "en": "The set!"},
    {"t": 6.0, "ko": "블로킹! 득점!", "ja": "ブロック！得点！", "en": "Block! Point!"},
]


def test_cues_follow_line_times_in_the_rendered_clip() -> None:
    # 해설 시각은 득점 구간(창) 기준, 렌더한 클립은 창 2초부터 7초 길이
    cues = score.subtitle_cues(LINES, ["ko", "en"], offset=2.0, duration=7.0)
    assert cues == [(0.0, 2.0, "서브 들어갑니다\nServe"), (2.0, 4.0, "토스 올라갑니다!\nThe set!"),
                    (4.0, 7.0, "블로킹! 득점!\nBlock! Point!")]


def test_no_languages_means_no_subtitles() -> None:
    assert score.subtitle_cues(LINES, [], offset=0.0, duration=7.0) == []
    assert score.subtitle_cues(None, ["ko"], offset=0.0, duration=7.0) == []
    with pytest.raises(ValueError):
        score.subtitle_cues(LINES, ["zh"], offset=0.0, duration=7.0)


def test_ask_score_time_parses_json(tmp_path: Path) -> None:
    clip = tmp_path / "r.mp4"
    clip.write_bytes(b"x")
    sent: dict = {}
    answer, _ = score.ask_score_time(_client('{"score_time_sec": 8.4}', sent), clip)
    assert answer["score_time_sec"] == 8.4
    assert sent["input"][0]["content"][0]["text"] == score.SCORE_TIME_PROMPT
