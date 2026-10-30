"""휘슬 소리로 랠리 후보 구간을 찾는다. 로컬, API 호출 없음.

휘슬 대역(2.4~4.5kHz)에서 소리가 크고, 그 에너지가 한 주파수에 몰린(순음) 순간만 휘슬로 본다.
환호·박수·외침도 이 대역에서 크지만 넓게 퍼진 소리라 순음도에서 걸러진다.
이웃한 두 휘슬 사이(2~40초)가 모두 후보다. 진짜 랠리인지는 화면으로 거르고, 이어진 랠리는 합친다.

사용법:
  uv run python -m tapme.rally data/match.mp4 [--out data/rallies.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
WIN_SEC = 0.05
HOP_SEC = 0.025
BAND_HZ = (2400.0, 4500.0)
LOUD_DB = 12.0  # 영상 전체 대역 에너지 중앙값보다 이만큼 크면 큰 소리. 15는 작은 끝 휘슬(13.6dB)을 놓쳤다(실험 H-26)
TONALITY_MIN = 8.0  # 대역 안 가장 센 칸 ÷ 대역 평균. 넓게 퍼진 잡음은 5 안팎
MERGE_SEC = 0.5  # 이 안에 붙은 후보는 휘슬 하나
SEGMENT_SEC = (2.0, 40.0)  # 이웃한 두 휘슬 사건 사이(앞 사건 끝~뒤 사건 시작)가 이 길이면 후보


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


EVENT_GAP_SEC = 2.0  # 이보다 붙은 휘슬은 한 사건(랠리 중 큰 소리가 몰려 잡히면 구간이 잘게 잘렸다, H-31)


def segments(whistles: list[Whistle], duration_sec: float | None = None) -> list[RallyCandidate]:
    """휘슬 사건 사이를 모두 후보로. 랠리인지는 화면으로 거르고(rally_filter), 이어진 랠리는 합친다.

    - 앞에서부터 두 개씩 짝지으면 중간 휘슬 하나로 뒤 짝이 다 밀렸다(H-26) → 이웃한 사건 사이를 모두 후보로
    - 2초 안에 붙은 휘슬은 한 사건으로 묶고, 후보는 앞 사건의 마지막 휘슬부터 뒤 사건의 첫 휘슬까지 좁게 잡는다.
      넉넉히 잡으면 득점 뒤 세리머니가 랠리 끝 1초와 함께 들어가 "랠리"로 판정됐다(H-31). 같은 사건을 사이에 둔
      두 "랠리"는 rally_filter.merge_rallies가 잇는다
    - duration_sec을 주면 영상 처음·끝도 사건으로 둔다(랠리 도중 시작·끝난 영상, H-31)
    """
    events: list[list[float]] = []
    for w in whistles:
        if events and w.time_sec - events[-1][1] < EVENT_GAP_SEC:
            events[-1][1] = w.time_sec
        else:
            events.append([w.time_sec, w.time_sec])
    if duration_sec is not None:
        if not events or events[0][0] >= EVENT_GAP_SEC:
            events.insert(0, [0.0, 0.0])
        else:
            events[0][0] = 0.0
        if duration_sec - events[-1][1] >= EVENT_GAP_SEC:
            events.append([duration_sec, duration_sec])
        else:
            events[-1][1] = duration_sec
    return [RallyCandidate(a[1], b[0]) for a, b in pairwise(events)
            if SEGMENT_SEC[0] <= b[0] - a[1] <= SEGMENT_SEC[1]]


def main() -> None:
    parser = argparse.ArgumentParser(description="휘슬 소리로 랠리 후보 구간 찾기 (로컬)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, help="결과 JSON 경로")
    parser.add_argument("--tonality-min", type=float, default=TONALITY_MIN, help="0이면 순음도 거르기 끔")
    args = parser.parse_args()

    audio = load_audio(args.video)
    whistles = find_whistles(audio, args.tonality_min)
    result = {
        "whistles": [asdict(w) for w in whistles],
        "rallies": [asdict(r) for r in segments(whistles, audio.size / SAMPLE_RATE)],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
