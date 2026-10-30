from __future__ import annotations

import json
from pathlib import Path

from tapme import bench


def test_truth_merges_points_within_two_seconds_and_cuts_range(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"points": [{"t": 47.1}, {"t": 47.2}, {"t": 120.0}, {"t": 700.0}]}))
    assert bench._truth(p, 0.0, 600.0) == [47.1, 120.0]  # 2초 안 중복은 한 랠리, 600초 넘는 것은 범위 밖


def test_rally_score_uses_same_rally_rule() -> None:
    s = bench.score_rallies([10.0, 50.0, 90.0], [11.5, 49.0, 52.0, 200.0])
    assert s["matched"] == 2 and s["missed"] == [90.0] and s["extra"] == [52.0, 200.0]
