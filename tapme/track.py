"""SAM 3.1로 선수를 추적하고, 사용자가 탭한 위치의 선수를 찾는다.

영상은 얼굴 흐림본만 보낸다. 실험(실험 레포 H-14)에서 랠리 전체보다 득점 장면 앞뒤만 추적할 때 맞았다.
"""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from meta_sam_parser import (
    CompletedOutcome,
    FramePrompt,
    PromptBox,
    PromptObject,
    PromptPoint,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    VideoSegmentationResult,
    build_object_prompt,
    video_segmentation_format,
)

from tapme import muse

MODEL = "sam-3.1"
CONCEPT = "volleyball player"  # 실험 H-02: "person"보다 선심·관중이 덜 잡힘
USD_PER_FRAME = 0.20 / 1000  # 영상은 50프레임보다 짧아도 50으로 과금(공식 문서)
USD_PER_IMAGE = 2.50 / 1000


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


# 상자만 필요할 때(다른 선수 추적, '나'가 보이는 구간): 공식 해석기는 마스크를 하나하나 풀어 검사해 다른 선수 추적
# 하나에 30초 넘게 걸렸다(11/1, H-53). 마스크 자리에 1x1 빈 마스크를 넣어 해석하면 상자는 그대로 나온다.
# 마스크 글자에는 | < > 가 들어가지 않는다(meta_sam_parser의 알파벳)
_MASK_TAG = re.compile(r"<\|mask;[^|]*\|>")
_TINY_MASK = "<|mask;x=0;y=0;data=1,1,!!!!!&O.lVP|>"


def parse_boxes(wire_text: str) -> VideoSegmentationResult:
    """parse와 같은 상자(마스크는 빈 1x1). 마스크가 필요한 곳(렌더의 '나')은 parse를 쓴다."""
    return parse(_MASK_TAG.sub(_TINY_MASK, wire_text))


TAPPED_ID = "0"  # 점 프롬프트로 추적한 '나'의 object_id


def tap_prompt(size: tuple[int, int], taps: list[tuple[int, float, float] | tuple[int, Box]]) -> str:
    """사용자가 고른 선수를 표시하는 SAM 프롬프트. taps = [(프레임, x, y) 또는 (프레임, 고른 선수 박스), ...].

    점이 하나면 그 선수가 화면 밖에 나갔다 들어올 때 비슷한 선수(같은 색 유니폼)에게 붙었다(실험 H-24).
    여러 장면에서 고르면 다시 들어온 뒤에도 같은 사람을 잡게 한다(H-25). 선수 목록에서 고르면 박스를 준다.
    """
    prompts = []
    for tap in taps:
        if len(tap) == 2:
            f, b = tap
            prompts.append(FramePrompt(frame=f, box=PromptBox(left=round(b.left), top=round(b.top),
                                                              right=round(b.right), bottom=round(b.bottom))))
        else:
            f, x, y = tap
            prompts.append(FramePrompt(frame=f, points=[PromptPoint(x=round(x), y=round(y), label="positive")]))
    return build_object_prompt(size=size, objects=[PromptObject(id=TAPPED_ID, prompts=prompts)])


def _stream_text(client: Any, content: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    chunks: list[str] = []
    usage: dict[str, Any] = {}
    with client.responses.stream(
        model=MODEL, input=[{"type": "message", "role": "user", "content": content}],
        metadata={"mask_encoding": "one_bit"},
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
            elif event.type == "response.completed" and getattr(event.response, "usage", None):
                usage = event.response.usage.model_dump()
    return "".join(chunks), usage


def detect_players(client: Any, image: Path, concept: str = CONCEPT) -> list[Box]:
    """사진 한 장(얼굴 흐림본에서 뽑은 장면)의 선수를 모두 찾는다. 약 $0.0025.
    사용자가 화면을 직접 탭하는 대신 선수 목록에서 고르게 한다(10/29 사용자 제안)."""
    data = base64.b64encode(image.read_bytes()).decode()
    wire, _ = _stream_text(client, [{"type": "input_text", "text": concept},
                                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{data}"}])
    return boxes_by_frame(parse(wire)).get(0, []) if wire else []


def track_players(client: Any, clip: Path, concept: str = CONCEPT) -> tuple[str, dict[str, Any], float]:
    """(SAM 원문, usage, 걸린 초). 원문은 파일로 남겨 다시 돈 쓰지 않고 렌더할 수 있게 한다.

    concept에는 "volleyball player" 같은 말이나 tap_prompt()가 만든 점 프롬프트를 준다.
    """
    with clip.open("rb") as f:
        file_id = client.files.create(purpose="user_data", file=f).id
    t0 = time.perf_counter()
    try:
        wire, usage = _stream_text(client, [{"type": "input_text", "text": concept},
                                            {"type": "input_video", "file_id": file_id}])
    finally:
        muse.forget_file(client, file_id)
    return wire, usage, round(time.perf_counter() - t0, 2)


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


def _iou(a: Box, b: Box) -> float:
    w = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    h = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    inter = w * h
    union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
    return inter / union if union > 0 else 0.0


def merge_me_and_others(
    me: VideoSegmentationResult, others: VideoSegmentationResult | None
) -> tuple[dict[int, list[Box]], dict[int, dict[str, SegmentationMaskRecord]]]:
    """점 프롬프트로 잡은 '나'(TAPPED_ID)와, 모든 선수 추적(others, '몸 실루엣'일 때만)을 합친다.

    다른 선수 ID는 'o'를 붙여 '나'와 겹치지 않게 하고, 프레임마다 '나'와 크게 겹치는 박스(같은 사람)는 뺀다.
    빼지 않으면 실루엣 흐림이 '나'를 덮는다.
    """
    by, masks = boxes_by_frame(me), masks_by_frame(me)
    if others is None:
        return by, masks
    other_masks = masks_by_frame(others)
    for k, boxes in boxes_by_frame(others).items():
        mine = [b for b in by.get(k, []) if b.object_id == TAPPED_ID]
        for b in boxes:
            if any(_iou(b, m) > 0.5 for m in mine):
                continue
            oid = f"o{b.object_id}"
            by.setdefault(k, []).append(Box(oid, b.left, b.top, b.right, b.bottom))
            if b.object_id in other_masks.get(k, {}):
                masks.setdefault(k, {})[oid] = other_masks[k][b.object_id]
    return by, masks


def visible_span(by: dict[int, list[Box]], around: int, max_gap: int = 2) -> tuple[int, int] | None:
    """around 프레임을 품고 '나'가 이어서 보이는 구간 (첫, 끝 프레임). max_gap 프레임까지의 끊김은 잇는다.

    카메라가 움직여 '나'가 화면 밖에 나갔다 들어오면 다른 선수에게 붙을 수 있어(실험 H-24·H-25),
    하이라이트는 득점 순간과 이어진 구간만 쓴다.
    """
    seen = sorted(k for k, boxes in by.items() if any(b.object_id == TAPPED_ID for b in boxes))
    if not seen:
        return None
    runs, start = [], seen[0]
    for a, b in pairwise(seen):
        if b - a > max_gap + 1:
            runs.append((start, a))
            start = b
    runs.append((start, seen[-1]))
    return min(runs, key=lambda r: 0 if r[0] <= around <= r[1] else min(abs(around - r[0]), abs(around - r[1])))
