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


def test_commentary_follows_who_won_the_point() -> None:
    # 11/1: 전에는 늘 "우리 팀 득점"이라고 써 상대 득점 랠리에 틀린 해설이 붙었다
    assert "won by the team the user calls '검정'" in score.commentary_prompt("검정", 6.0, "", 10.0)
    opp = score.commentary_prompt("검정", 6.0, "score", 10.0, "opponent")
    assert "won by the opponents" in opp and "won the point themselves" not in opp  # '내가 득점'은 빼고
    assert "never say which team won" in score.commentary_prompt("검정", 6.0, "assist", 10.0, None)


def test_commentary_keeps_the_role_only_when_it_fits_the_winner() -> None:
    # '내가 득점'은 우리 팀이 이겼을 때만 맞는 말. 어시스트·수비는 진 랠리에서도 그대로
    assert "won the point themselves" in score.commentary_prompt("검정", 6.0, "score", 10.0, "mine")
    assert "won the point themselves" not in score.commentary_prompt("검정", 6.0, "score", 10.0, None)
    opp_assist = score.commentary_prompt("검정", 6.0, "assist", 10.0, "opponent")
    assert score.ROLE_TEXT["assist"] in opp_assist and "fought for the rally" in opp_assist
    assert score.ROLE_TEXT["defense"] in score.commentary_prompt("검정", 6.0, "defense", 10.0, "opponent")
    assert "Do not describe the highlighted player's role" in score.commentary_prompt("검정", 6.0, "", 10.0, "opponent")
    # 클립 길이와 득점 시각을 알려 장면에 맞춰 해설하게 한다
    p = score.commentary_prompt("검정", 6.0, "assist", 10.0)
    assert "10-second clip" in p and "timed near 6.0s" in p


def test_cues_sort_clamp_and_round_to_the_rendered_clip() -> None:
    # 순서를 섞어 줘도 시각 순으로, 소수는 둘째 자리로, 클립 밖에 시작하는 줄은 버린다
    lines = [
        {"t": 9.0, "ko": "뒤", "ja": "後", "en": "late"},
        {"t": 1.126, "ko": "앞", "ja": "前", "en": "early"},
    ]
    assert score.subtitle_cues(lines, ["ko"], offset=0.0, duration=7.0) == [(1.13, 7.0, "앞")]
    # 클립 시작(창 2초)보다 이른 줄은 0초부터 보인다
    line = {"t": 0.5, "ko": "시작 전", "en": "pre"}
    assert score.subtitle_cues([line], ["ko", "en"], offset=2.0, duration=7.0) == [(0.0, 7.0, "시작 전\npre")]


def test_cues_skip_lines_without_text_or_time() -> None:
    # 고른 언어의 글자가 없는 줄·다음 줄과 같은 시각의 줄은 자막에 없다
    lines = [
        {"t": 1.0, "ko": "", "en": ""},
        {"t": 2.0, "ko": "둘째", "en": ""},
        {"t": 2.0, "ko": "셋째", "en": "third"},
    ]
    assert score.subtitle_cues(lines, ["ko", "en"], offset=0.0, duration=9.0) == [(2.0, 9.0, "셋째\nthird")]
    # 없는 언어 글자는 건너뛰고 있는 것만 고른 순서로
    assert score.subtitle_cues([{"t": 1.0, "en": "only"}], ["ko", "en"], offset=0.0, duration=9.0) == [(1.0, 9.0, "only")]


def test_ask_serve_side_reports_which_side_serves(tmp_path: Path) -> None:
    clip = tmp_path / "s.mp4"
    clip.write_bytes(b"x")
    sent: dict = {}
    answer, reply = score.ask_serve_side(_client('{"server_side": "far", "confidence": 0.9}', sent), clip)
    assert (answer["server_side"], answer["confidence"]) == ("far", 0.9)
    assert sent["input"][0]["content"][0]["text"] == score.SERVE_SIDE_PROMPT
    assert reply.input_tokens == 1  # 낸 돈은 reply로 센다


def test_ask_serve_side_treats_an_empty_answer_as_unsure(tmp_path: Path) -> None:
    # 빈 답이 온 적이 있다(H-54). 서브 쪽을 모르면 득점 팀도 모른다
    clip = tmp_path / "s.mp4"
    clip.write_bytes(b"x")
    answer, _ = score.ask_serve_side(_client("", {}), clip)
    assert answer == {"server_side": "unsure", "confidence": 0.0}


def test_ask_score_time_without_json_is_an_error(tmp_path: Path) -> None:
    # 득점 시각은 다시 물으면 흔들려(H-43) 모르면 실패로. 서브 쪽·해설과 달리 대신할 값이 없다
    clip = tmp_path / "r.mp4"
    clip.write_bytes(b"x")
    with pytest.raises(ValueError):
        score.ask_score_time(_client("no json here", {}), clip)


def test_ask_commentary_returns_timed_lines(tmp_path: Path) -> None:
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    sent: dict = {}
    lines, _ = score.ask_commentary(
        _client('{"lines": [{"t": 1.0, "ko": "서브"}, {"ko": "시각 없음"}, {"t": 6.0, "ko": "득점"}]}', sent),
        clip, "검정", 6.0, "assist", 10.0)
    assert lines == [{"t": 1.0, "ko": "서브"}, {"t": 6.0, "ko": "득점"}]  # 시각 없는 줄은 버린다
    assert sent["input"][0]["content"][0]["text"] == score.commentary_prompt("검정", 6.0, "assist", 10.0)


def test_ask_commentary_distinguishes_no_lines_from_unreadable(tmp_path: Path) -> None:
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    lines, _ = score.ask_commentary(_client('{"lines": []}', {}), clip, "검정", 6.0, "", 10.0)
    assert lines == []  # 줄이 없으면 빈 목록(다시 묻지 않는다)
    lines, reply = score.ask_commentary(_client("truncated {\"lines\": [", {}), clip, "검정", 6.0, "", 10.0)
    assert lines is None and reply.input_tokens == 1  # 읽을 수 없으면 None(낸 돈은 센다)
