from __future__ import annotations

from tapme import qa, track


def _me(k: int, x: float) -> list[track.Box]:
    return [track.Box(track.TAPPED_ID, x, 100, x + 40, 200)]


def test_tracking_stats_reports_seen_ratio_gap_and_jumps() -> None:
    # 10fps 2초 = 20프레임. 0~9는 보이고 10~14 놓침, 15~19 다시 보임
    by = {k: _me(k, 100 + 10 * k) for k in [*range(10), *range(15, 20)]}
    s = qa.tracking_stats(by, sam_fps=10, clip_sec=2.0)
    assert s["me_seen_ratio"] == 0.75
    assert s["longest_gap_sec"] == 0.5
    assert s["jump_px_median"] == 10.0 and s["jump_px_max"] == 10.0  # 이어진 프레임 사이만 잰다


def test_tracking_stats_when_never_seen() -> None:
    s = qa.tracking_stats({}, sam_fps=10, clip_sec=1.0)
    assert s["me_seen_ratio"] == 0.0 and s["longest_gap_sec"] == 1.0 and s["jump_px_max"] is None
