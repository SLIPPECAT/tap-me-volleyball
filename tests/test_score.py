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


def test_caption_prompt_fixes_the_users_team_and_asks_all_languages() -> None:
    # Muse의 팀 판정은 못 믿으므로(실험 H-07) 사용자가 고른 팀을 그대로 준다
    p = score.caption_prompt("검정", 6.0)
    assert "'검정'" in p and "6.0s" in p and "do not change it" in p
    # 세 언어를 한 번에 받아 두면 사용자가 언어를 바꿔도 다시 묻지 않는다
    assert '{"ko": "...", "ja": "...", "en": "..."}' in p


def test_subtitle_uses_only_chosen_languages_in_order() -> None:
    captions = {"ko": "검정 득점!", "ja": "黒チーム得点!", "en": "Black scores!"}
    assert score.subtitle_text(captions, ["en", "ko"]) == "Black scores!\n검정 득점!"
    assert score.subtitle_text(captions, ["ja"]) == "黒チーム得点!"
    assert score.subtitle_text(captions, []) == ""  # 자막 없음


def test_subtitle_rejects_unsupported_language() -> None:
    with pytest.raises(ValueError):
        score.subtitle_text({"ko": "a"}, ["zh"])


def test_ask_score_time_parses_json(tmp_path: Path) -> None:
    clip = tmp_path / "r.mp4"
    clip.write_bytes(b"x")
    sent: dict = {}
    answer, _ = score.ask_score_time(_client('{"score_time_sec": 8.4}', sent), clip)
    assert answer["score_time_sec"] == 8.4
    assert sent["input"][0]["content"][0]["text"] == score.SCORE_TIME_PROMPT
