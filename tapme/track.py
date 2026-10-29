"""SAM 3.1로 선수를 추적하고, 사용자가 탭한 위치의 선수를 찾는다.

영상은 얼굴 흐림본만 보낸다. 실험(실험 레포 H-14)에서 랠리 전체보다 득점 장면 앞뒤만 추적할 때 맞았다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meta_sam_parser import (
    CompletedOutcome,
    FramePrompt,
    PromptObject,
    PromptPoint,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    VideoSegmentationResult,
    build_object_prompt,
    decode_mask_to_raster,
    video_segmentation_format,
)

MODEL = "sam-3.1"
CONCEPT = "volleyball player"  # 실험 H-02: "person"보다 선심·관중이 덜 잡힘
USD_PER_FRAME = 0.20 / 1000


@dataclass(frozen=True)
class Box:
    object_id: str
    left: float
    top: float
    right: float
    bottom: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2


def estimate_usd(frames: int) -> float:
    return frames * USD_PER_FRAME


def parse(wire_text: str) -> VideoSegmentationResult:
    parser = video_segmentation_format().create_parser()
    parser.push(wire_text)
    return parser.finish(CompletedOutcome()).result


TAPPED_ID = "0"  # 점 프롬프트로 추적한 '나'의 object_id


def tap_prompt(size: tuple[int, int], taps: list[tuple[int, float, float]]) -> str:
    """탭한 점들을 한 선수의 표시로 주는 SAM 프롬프트. taps = [(프레임, x, y), ...].

    점이 하나면 그 선수가 화면 밖에 나갔다 들어올 때 비슷한 선수(같은 색 유니폼)에게 붙었다(실험 H-24).
    여러 장면에서 탭하면 다시 들어온 뒤에도 같은 사람을 잡게 한다(H-25).
    """
    prompts = [FramePrompt(frame=f, points=[PromptPoint(x=round(x), y=round(y), label="positive")]) for f, x, y in taps]
    return build_object_prompt(size=size, objects=[PromptObject(id=TAPPED_ID, prompts=prompts)])


def track_players(client: Any, clip: Path, concept: str = CONCEPT) -> tuple[str, dict[str, Any], float]:
    """(SAM 원문, usage, 걸린 초). 원문은 파일로 남겨 다시 돈 쓰지 않고 렌더할 수 있게 한다.

    concept에는 "volleyball player" 같은 말이나 tap_prompt()가 만든 점 프롬프트를 준다.
    """
    with clip.open("rb") as f:
        file_id = client.files.create(purpose="user_data", file=f).id
    t0 = time.perf_counter()
    chunks: list[str] = []
    usage: dict[str, Any] = {}
    with client.responses.stream(
        model=MODEL,
        input=[{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": concept}, {"type": "input_video", "file_id": file_id}],
        }],
        metadata={"mask_encoding": "one_bit"},
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
            elif event.type == "response.completed" and getattr(event.response, "usage", None):
                usage = event.response.usage.model_dump()
    return "".join(chunks), usage, round(time.perf_counter() - t0, 2)


def boxes_by_frame(result: VideoSegmentationResult) -> dict[int, list[Box]]:
    out: dict[int, list[Box]] = {}
    for r in result.records:
        if isinstance(r, SegmentationBoxRecord) and r.frame is not None:
            out.setdefault(r.frame.frame_index, []).append(Box(r.object_id, r.left, r.top, r.right, r.bottom))
    return out


def masks_by_frame(result: VideoSegmentationResult) -> dict[int, dict[str, SegmentationMaskRecord]]:
    out: dict[int, dict[str, SegmentationMaskRecord]] = {}
    for r in result.records:
        if isinstance(r, SegmentationMaskRecord) and r.frame is not None:
            out.setdefault(r.frame.frame_index, {})[r.object_id] = r
    return out


def _mask_hit(rec: SegmentationMaskRecord, x: float, y: float) -> bool:
    b = rec.bounds
    if not (b.left <= x < b.right and b.top <= y < b.bottom):
        return False
    w, h = rec.mask.width, rec.mask.height
    # 마스크는 박스 크기로 오는 경우가 많다. 박스 안 상대 위치를 마스크 크기에 맞춘다
    mx = int((x - b.left) * w / max(1.0, b.right - b.left))
    my = int((y - b.top) * h / max(1.0, b.bottom - b.top))
    raster = decode_mask_to_raster(rec.mask)
    return 0 <= mx < w and 0 <= my < h and raster[my * w + mx] > 0


def find_tapped(result: VideoSegmentationResult, x: float, y: float, frame: int = 0) -> str | None:
    """탭한 점이 마스크 안에 든 선수, 없으면 점을 품은 가장 작은 박스의 선수."""
    for oid, rec in masks_by_frame(result).get(frame, {}).items():
        if _mask_hit(rec, x, y):
            return oid
    hits = [b for b in boxes_by_frame(result).get(frame, []) if b.left <= x < b.right and b.top <= y < b.bottom]
    if not hits:
        return None
    return min(hits, key=lambda b: (b.right - b.left) * (b.bottom - b.top)).object_id
