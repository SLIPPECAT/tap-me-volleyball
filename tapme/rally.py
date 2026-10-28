"""휘슬 소리로 랠리 후보 구간을 찾는다. 로컬, API 호출 없음.

휘슬 대역(2.4~4.5kHz)에서 소리가 크고, 그 에너지가 한 주파수에 몰린(순음) 순간만 휘슬로 본다.
환호·박수·외침도 이 대역에서 크지만 넓게 퍼진 소리라 순음도에서 걸러진다.
이어진 두 휘슬 사이가 랠리 길이(4~40초)면 랠리 후보다. 후보가 진짜 랠리인지는 화면으로 다시 거른다.

사용법:
  uv run python -m tapme.rally data/match.mp4 [--out data/rallies.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
WIN_SEC = 0.05
HOP_SEC = 0.025
BAND_HZ = (2400.0, 4500.0)
LOUD_DB = 15.0  # 영상 전체 대역 에너지 중앙값보다 이만큼 크면 큰 소리
TONALITY_MIN = 8.0  # 대역 안 가장 센 칸 ÷ 대역 평균. 넓게 퍼진 잡음은 5 안팎
MERGE_SEC = 0.5  # 이 안에 붙은 후보는 휘슬 하나
RALLY_SEC = (4.0, 40.0)


@dataclass(frozen=True)
class Whistle:
    time_sec: float
    level_db: float  # 영상 전체 중앙값 대비
    tonality: float


@dataclass(frozen=True)
class RallyCandidate:
    start_sec: float
    end_sec: float


def load_audio(video: Path) -> np.ndarray:
    raw = subprocess.check_output(
        ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "f32le", "-"]
    )
    return np.frombuffer(raw, dtype=np.float32)


def _band_frames(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """창마다 (중심 시각, 대역 에너지 dB, 순음도)."""
    win, hop = int(WIN_SEC * SAMPLE_RATE), int(HOP_SEC * SAMPLE_RATE)
    if audio.size < win:
        empty = np.empty(0)
        return empty, empty, empty
    n = 1 + (audio.size - win) // hop
    frames = audio[np.arange(win)[None, :] + hop * np.arange(n)[:, None]] * np.hanning(win)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / SAMPLE_RATE)
    band = power[:, (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])]
    level_db = 10 * np.log10(band.sum(axis=1) + 1e-12)
    tonality = band.max(axis=1) / (band.mean(axis=1) + 1e-12)
    times = (np.arange(n) * hop + win / 2) / SAMPLE_RATE
    return times, level_db, tonality


def find_whistles(audio: np.ndarray, tonality_min: float = TONALITY_MIN) -> list[Whistle]:
    times, level_db, tonality = _band_frames(audio)
    if times.size == 0:
        return []
    base = float(np.median(level_db))
    loud = np.flatnonzero(level_db >= base + LOUD_DB)

    groups: list[list[int]] = []
    for i in loud:
        if groups and times[i] - times[groups[-1][-1]] <= MERGE_SEC:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])

    whistles = []
    for g in groups:
        # 큰 소리 한 덩어리의 순음도 중앙값으로 판정한다. 잡음도 창 하나쯤은 우연히 8을 넘는다
        tone = float(np.median(tonality[g]))
        if tone < tonality_min:
            continue
        j = max(g, key=lambda k: level_db[k])
        whistles.append(Whistle(round(float(times[j]), 2), round(float(level_db[j] - base), 1), round(tone, 1)))
    return whistles


def pair_rallies(whistles: list[Whistle]) -> list[RallyCandidate]:
    """이어진 두 휘슬 간격이 랠리 길이면 후보. 짝이 된 휘슬은 다음 짝에 다시 쓰지 않는다."""
    out: list[RallyCandidate] = []
    i = 0
    while i + 1 < len(whistles):
        a, b = whistles[i], whistles[i + 1]
        if RALLY_SEC[0] <= b.time_sec - a.time_sec <= RALLY_SEC[1]:
            out.append(RallyCandidate(a.time_sec, b.time_sec))
            i += 2
        else:
            i += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="휘슬 소리로 랠리 후보 구간 찾기 (로컬)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, help="결과 JSON 경로")
    parser.add_argument("--tonality-min", type=float, default=TONALITY_MIN, help="0이면 순음도 거르기 끔")
    args = parser.parse_args()

    whistles = find_whistles(load_audio(args.video), args.tonality_min)
    result = {
        "whistles": [asdict(w) for w in whistles],
        "rallies": [asdict(r) for r in pair_rallies(whistles)],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
