from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from tapme import rally

SR = rally.SAMPLE_RATE


def _clip(seconds: float, events: list[tuple[str, float]]) -> np.ndarray:
    """조용한 바탕에 0.3초짜리 휘슬(3.2kHz 순음) 또는 환호(넓게 퍼진 잡음)를 넣은 소리."""
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.005, int(seconds * SR)).astype(np.float32)
    t = np.arange(int(0.3 * SR)) / SR
    for kind, at in events:
        i = int(at * SR)
        burst = 0.5 * np.sin(2 * np.pi * 3200 * t) if kind == "whistle" else rng.normal(0, 0.5, t.size)
        audio[i : i + t.size] += burst.astype(np.float32)
    return audio


def test_tonality_keeps_whistles_and_drops_cheers() -> None:
    # H-17: 환호·박수도 휘슬 대역에서 크게 잡혀 랠리로 묶였다
    audio = _clip(30, [("whistle", 1.0), ("whistle", 8.0), ("cheer", 14.0), ("cheer", 22.0)])
    whistles = rally.find_whistles(audio)
    assert [round(w.time_sec) for w in whistles] == [1, 8]
    assert rally.segments(whistles) == [rally.RallyCandidate(start_sec=whistles[0].time_sec, end_sec=whistles[1].time_sec)]


def test_without_tonality_cheers_become_false_rallies() -> None:
    audio = _clip(30, [("whistle", 1.0), ("whistle", 8.0), ("cheer", 14.0), ("cheer", 22.0)])
    loud_only = rally.find_whistles(audio, tonality_min=0.0)
    assert len(rally.segments(loud_only)) == 3  # 1→8, 8→14, 14→22


def test_every_neighbouring_gap_in_range_is_a_candidate() -> None:
    w = [rally.Whistle(t, 30.0, 20.0) for t in (1.0, 2.0, 10.0, 60.0, 65.0)]
    # 1→2초는 너무 짧고 10→60초는 너무 길다. 2→10초, 60→65초가 후보
    assert rally.segments(w) == [rally.RallyCandidate(2.0, 10.0), rally.RallyCandidate(60.0, 65.0)]


def test_middle_whistle_does_not_shift_later_candidates() -> None:
    # 10/29 27초 영상: 3.0(시작) · 16.75(랠리 중간) · 26.5(끝). 두 개씩 짝지으면 26.5가 남아 랠리가 잘렸다
    w = [rally.Whistle(t, 20.0, 40.0) for t in (3.0, 16.75, 26.5)]
    assert rally.segments(w) == [rally.RallyCandidate(3.0, 16.75), rally.RallyCandidate(16.75, 26.5)]


def test_load_audio_from_video(tmp_path: Path) -> None:
    src = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=5:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=3200:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(src)],
        check=True,
    )
    audio = rally.load_audio(src)
    assert abs(audio.size / SR - 2.0) < 0.1
