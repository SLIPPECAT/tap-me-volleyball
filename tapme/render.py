"""하이라이트 렌더: '나' 중심 9:16 확대 + 다른 회원 가리기 + 장면에 맞춰 바뀌는 해설 자막(선택, score.subtitle_cues).

SAM은 5fps로만 돌리고(비용), 박스를 프레임 사이에 직선으로 이어 원본 fps로 렌더한다.
실험(실험 레포 H-18): 보간 오차 중앙값 6px, 2배 확대에서 '나'가 화면 높이의 21.5%.

얼굴 모자이크(선택, 기본 켬): 입력은 원본 클립이고, 프레임마다 얼굴을 찾아 '나'의 박스 안 얼굴만 빼고 모자이크한다.
'나'를 못 잡은 프레임은 모든 얼굴을 가린다.

가리기 방식(사용자가 고른다):
  face        얼굴 모자이크만. 경기 흐름이 가장 잘 보인다 (기본)
  silhouette  SAM이 찾은 다른 선수 몸도 흐림. SAM이 놓치거나 어긋나면 빈틈이 생긴다 (H-18)
  background  '나'만 빼고 전부 흐림. 빈틈은 없지만 공·네트까지 지워진다 (H-20)
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import SegmentationMaskRecord, decode_mask_to_raster
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tapme import anonymize
from tapme.track import Box

PRIVACY_MODES = ("face", "silhouette", "background")
DEFAULT_PRIVACY = "face"
ZOOM_CHOICES = (1.0, 1.5, 2.0, 2.5)
DEFAULT_ZOOM = 1.0  # 10/30 사용자 결정: 확대하면 1080p 원본도 픽셀을 키워 흐려진다(2배 303×540, 1.5배 404×720). 결과는 포트폴리오·자랑용이라 원본 화질 그대로
SILHOUETTE_BLUR_RADIUS = 14  # 다른 선수 몸만 흐림
# '나' 빼고 전부: 14는 공·네트까지 지워져 경기가 안 보였다(실험 H-20, 10/29 사용자 테스트). 얼굴은 입력 흐림본에서
# 이미 모자이크돼 있어, 6이면 경기 흐름이 보이면서 다른 선수 등번호는 읽기 어렵다
BACKGROUND_BLUR_RADIUS = 6
# '나' 표시(10/30 사용자 C안 "좀 더 전문적인 느낌"): 발밑에 은은히 빛나는 원 + 머리 위 노란 삼각형
RING_COLOR = (0, 229, 255)
MARKER_COLOR = (255, 204, 0)

# 한글과 일본어 한자를 모두 가진 글꼴을 고른다(Apple SD Gothic Neo에는 '黒'이 없음)
FONT_CANDIDATES = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # 리눅스 서버(fonts-noto-cjk)
)
_NO_GLYPH = ""  # 사설 영역 문자. 어느 글꼴에도 없어 '없는 글자' 모양을 알려 준다


def interpolate(by_sam: dict[int, list[Box]], sam_fps: float, out_fps: float, n_out: int) -> list[list[Box]]:
    """출력 프레임마다 앞뒤 SAM 프레임 박스를 직선으로 잇는다.

    한쪽에만 있는 선수는 있는 쪽 박스를 그대로 쓴다(최대 SAM 한 프레임). 빼면 그 사이 가리기가 끊긴다.
    """
    out = []
    for j in range(n_out):
        s = j * sam_fps / out_fps
        k = int(s)
        t = s - k
        a = {b.object_id: b for b in by_sam.get(k, [])}
        b_ = {b.object_id: b for b in by_sam.get(k + 1, [])}
        frame = []
        for oid in [*a, *(o for o in b_ if o not in a)]:
            p, q = a.get(oid), b_.get(oid)
            if p and q:
                frame.append(Box(oid, p.left + (q.left - p.left) * t, p.top + (q.top - p.top) * t,
                                 p.right + (q.right - p.right) * t, p.bottom + (q.bottom - p.bottom) * t))
            else:
                frame.append(p or q)
        out.append(frame)
    return out


def zoom_crops(
    centers: list[tuple[float, float] | None], width: int, height: int, zoom: float, smooth: int
) -> list[tuple[int, int, int, int]]:
    """'나' 중심을 앞뒤 smooth 프레임 평균으로 따라가는 9:16 화면 (x0, y0, w, h).

    렌더는 영상 전체를 미리 알아 앞뒤 평균을 쓸 수 있다(뒤따라가며 늦는 문제가 없음).
    '나'가 없는 프레임은 가까운 프레임 위치, 한 번도 없으면 화면 가운데.
    """
    ch = max(2, int(height / zoom) // 2 * 2)
    cw = min(width, max(2, int(ch * 9 / 16) // 2 * 2))
    last = next((c for c in centers if c), (width / 2, height / 2))
    filled = []
    for c in centers:
        last = c or last
        filled.append(last)
    crops = []
    for i in range(len(filled)):
        # 처음·끝에서는 창을 양쪽 똑같이 줄인다. 한쪽만 줄이면 움직이는 '나'보다 앞서거나 늦었다(S1 끝, S5 처음)
        r = min(smooth, i, len(filled) - 1 - i)
        win = filled[i - r: i + r + 1]
        cx = sum(c[0] for c in win) / len(win)
        cy = sum(c[1] for c in win) / len(win)
        crops.append((max(0, min(width - cw, round(cx - cw / 2))), max(0, min(height - ch, round(cy - ch / 2))), cw, ch))
    return crops


def _covers(font: ImageFont.FreeTypeFont, text: str) -> bool:
    missing = bytes(font.getmask(_NO_GLYPH))
    return all(bytes(font.getmask(ch)) != missing for ch in set(text) if not ch.isspace())


def load_font(size: int, text: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    first = None
    for path in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        font = ImageFont.truetype(path, size=size)
        first = first or font
        if _covers(font, text):
            return font
    return first or ImageFont.load_default()


# SAM 모양(10fps)을 출력 프레임(30fps)마다 다시 풀어 렌더의 18%를 썼다(클라우드 프로파일 10/31) → 한 번 풀어 둔다.
# 렌더 하나 동안만 두고 비운다. 레코드를 함께 쥐어 같은 id가 다른 레코드에 다시 쓰이지 않게
_MASKS: dict[int, tuple[SegmentationMaskRecord, Image.Image | None]] = {}


try:  # 검사(다시 압축해 비교) 없이 푸는 내부 함수. 없으면 공식 함수(검사 포함, 두 배 느림)
    from meta_sam_parser._mask_codec import _decode_raster as _fast_decode
except ImportError:  # pragma: no cover
    _fast_decode = None


def _decode(payload: str, width: int, height: int, encoding: str, mask: object = None) -> bytes:
    if encoding == "one_bit" and _fast_decode is not None:
        return _fast_decode(payload, width, height)
    return decode_mask_to_raster(mask)  # type: ignore[arg-type]


def _decode_job(args: tuple[str, int, int]) -> bytes | None:
    """얼굴 찾기 일꾼에서 마스크 하나를 푼다(렌더 시작 전에 한꺼번에, H-67)."""
    payload, width, height = args
    try:
        return _decode(payload, width, height, "one_bit")
    except Exception:  # noqa: BLE001 - 하나가 깨져도 그 프레임만 마스크 없이
        return None


def _mask_image(raster: bytes | None, mw: int, mh: int) -> Image.Image | None:
    if raster is None or mw <= 0 or mh <= 0 or len(raster) != mw * mh:
        return None
    return Image.fromarray(np.frombuffer(bytes(raster), np.uint8).reshape(mh, mw) * np.uint8(255), "L")


def _predecode(recs: list) -> None:
    """렌더에 쓸 마스크를 미리 푼다. 일꾼이 여럿이면 나눠서(하이라이트 하나 마스크 약 280장, 한 프로세스로 약 30초)."""
    todo = [r for r in recs if r.mask.encoding == "one_bit" and id(r) not in _MASKS]
    if RENDER_WORKERS <= 1 or len(todo) < 8:
        return  # 하나씩은 _mask가 그때그때
    pool = anonymize.acquire_pool()
    try:
        rasters = list(pool.map(_decode_job, [(r.mask.payload, r.mask.width, r.mask.height) for r in todo], chunksize=4))
    finally:
        anonymize.release_pool()
    for r, raster in zip(todo, rasters, strict=True):
        _MASKS[id(r)] = (r, _mask_image(raster, r.mask.width, r.mask.height))


def _mask(rec: SegmentationMaskRecord | None, w: int, h: int) -> Image.Image | None:
    if rec is None or w <= 0 or h <= 0:
        return None
    hit = _MASKS.get(id(rec))
    if hit is None or hit[0] is not rec:
        mw, mh = rec.mask.width, rec.mask.height
        raster = _decode(rec.mask.payload, mw, mh, rec.mask.encoding, rec.mask)  # 0/1 바이트
        hit = _MASKS[id(rec)] = (rec, _mask_image(raster, mw, mh))
    return None if hit[1] is None else hit[1].resize((w, h), Image.Resampling.NEAREST)


def _paste_region(dst: Image.Image, src: Image.Image, box: Box, rec: SegmentationMaskRecord | None) -> None:
    fw, fh = dst.size
    x1, y1 = max(0, int(box.left)), max(0, int(box.top))
    x2, y2 = min(fw, int(box.right)), min(fh, int(box.bottom))
    if x2 <= x1 or y2 <= y1:
        return
    dst.paste(src.crop((x1, y1, x2, y2)), (x1, y1), _mask(rec, x2 - x1, y2 - y1))


def _draw_subtitle(img: Image.Image, text: str) -> None:
    w, h = img.size
    size = max(14, w // 22)
    font = load_font(size, text)
    lines = text.count("\n") + 1
    spacing = size // 3
    pad = max(10, w // 20)
    banner_h = size * lines + spacing * (lines - 1) + size
    y1 = h - banner_h - max(16, h // 18)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rounded_rectangle((pad, y1, w - pad, y1 + banner_h), radius=size // 2, fill=(15, 20, 30, 205))
    draw.multiline_text((pad + size // 2, y1 + size // 2), text, fill=(255, 255, 255, 255), font=font, spacing=spacing)


def _soft(img: Image.Image, box: tuple[float, float, float, float], draw: Callable[[ImageDraw.ImageDraw, float, float], None],
          blur: float) -> None:
    """box 둘레의 작은 조각에만 그리고 흐려 붙인다(프레임 전체를 흐리면 렌더가 느려진다)."""
    m = int(blur * 3) + 2
    x0, y0 = int(box[0]) - m, int(box[1]) - m
    patch = Image.new("RGBA", (int(box[2]) + m - x0, int(box[3]) + m - y0), (0, 0, 0, 0))
    draw(ImageDraw.Draw(patch), -x0, -y0)
    patch = patch.filter(ImageFilter.GaussianBlur(blur))
    img.paste(patch, (x0, y0), patch)


def _draw_ground_marker(img: Image.Image, me: Box) -> None:
    """발밑 스포트라이트(빛 + 흰색·하늘색 이중 테두리). 굵기·크기는 선수 크기에 맞춘다."""
    w, h = me.right - me.left, me.bottom - me.top
    cx = (me.left + me.right) / 2
    rx, ry = max(14.0, w * 0.75), max(6.0, h * 0.09)
    foot = (cx - rx, me.bottom - ry, cx + rx, me.bottom + ry)
    _soft(img, foot, lambda d, dx, dy: d.ellipse((foot[0] + dx, foot[1] + dy, foot[2] + dx, foot[3] + dy),
                                                 fill=RING_COLOR + (110,)), ry * 0.6)
    t = max(2, round(h / 110))
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse(foot, outline=(255, 255, 255, 230), width=t)
    d.ellipse((cx - rx * 0.62, me.bottom - ry * 0.62, cx + rx * 0.62, me.bottom + ry * 0.62), outline=RING_COLOR + (200,), width=t)


def _draw_head_marker(img: Image.Image, me: Box, head: tuple[int, int, int, int] | None = None) -> None:
    """머리 위 노란 삼각형(그림자 포함). 내 얼굴(head)을 알면 그 위에 — 앞사람과 겹쳐 내 상자가 커지면 상자 꼭대기가
    다른 사람(심판) 머리 위로 올라갔다(11/3 사용자)."""
    w = me.right - me.left
    cx = (me.left + me.right) / 2 if head is None else head[0] + head[2] / 2
    s = max(14.0, w * 0.32)
    if head is not None:  # 앞사람과 겹쳐 상자가 커지면 삼각형도 커졌다(11/3 Muse Code 검토) — 얼굴 크기로 상한
        s = max(14.0, min(s, head[2] * 1.2))
    top = me.top if head is None else head[1] - head[3] * 0.35
    tip = top - s * 0.45
    tri = [(cx - s / 2, tip - s * 0.7), (cx + s / 2, tip - s * 0.7), (cx, tip)]
    _soft(img, (cx - s / 2, tip - s * 0.7, cx + s / 2 + 2, tip + 3),
          lambda d, dx, dy: d.polygon([(x + dx + 2, y + dy + 3) for x, y in tri], fill=(0, 0, 0, 120)), 3)
    ImageDraw.Draw(img, "RGBA").polygon(tri, fill=MARKER_COLOR + (255,), outline=(20, 20, 20, 255))


def compose(
    frame: Image.Image,
    boxes: list[Box],
    masks: dict[str, SegmentationMaskRecord],
    me: str,
    crop: tuple[int, int, int, int],
    out_size: tuple[int, int],
    subtitle: str = "",
    privacy: str = DEFAULT_PRIVACY,
    head: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    if privacy not in PRIVACY_MODES:
        raise ValueError(f"가리기 방식은 {PRIVACY_MODES} 중 하나: {privacy}")
    sharp = frame.convert("RGB")
    my_box = next((b for b in boxes if b.object_id == me), None)
    if privacy == "background":
        work = sharp.filter(ImageFilter.GaussianBlur(BACKGROUND_BLUR_RADIUS))
        if my_box:
            _paste_region(work, sharp, my_box, masks.get(me))
    else:
        work = sharp.copy()
        if privacy == "silhouette":
            blurred = sharp.filter(ImageFilter.GaussianBlur(SILHOUETTE_BLUR_RADIUS))
            for b in boxes:
                if b.object_id != me:
                    _paste_region(work, blurred, b, masks.get(b.object_id))
    if my_box:
        rec = masks.get(me)
        body = work.copy() if rec is not None else None
        _draw_ground_marker(work, my_box)
        if body is not None:
            # 발밑 표시가 발을 덮지 않고 바닥에 깔린 것처럼: '나'의 몸(SAM 마스크)을 표시 위에 다시 올린다(10/30 사용자)
            _paste_region(work, body, my_box, rec)
        _draw_head_marker(work, my_box, head)
    x0, y0, cw, ch = crop
    out = work.crop((x0, y0, x0 + cw, y0 + ch)).resize(out_size, Image.Resampling.LANCZOS)
    if subtitle:
        _draw_subtitle(out, subtitle)  # 확대 뒤에 그려야 글자가 흐려지지 않는다
    return out


# 결과 영상의 얼굴 모자이크(H-40, 사용자 10/30 "살짝살짝 벗겨진다"): 프레임마다 새로 찾아 검출기가 한두 프레임 놓치면
# 그 순간만 드러났다 → 찾은 자리를 잠깐 유지하고, 박스도 넓힌다. Meta로 보내는 흐림본(anonymize)은 그대로
FACE_HOLD_SEC = 0.2
RENDER_BOX_PAD = 0.4
# 옆얼굴 Haar는 이 프레임마다 한 번(11/1 사용자 결정, H-52): 매 프레임이면 얼굴 찾기의 2/3를 쓴다. 2프레임마다면
# 장당 1,592 → 1,082ms, 옆얼굴로만 덮이던 자리의 24%가 1프레임(1/30초)씩 빠지고 다음 검출·FACE_HOLD_SEC로 다시 덮인다
PROFILE_EVERY = 2
MY_HEAD_SHARE = 0.35  # '나'의 얼굴로 보는 곳: 내 박스 위쪽 이 비율 안
# 결과 영상 모자이크의 얼굴 찾기 배율(11/2 사용자: 늘 2배였다 — 영상 크기에 따라 1·1.5배를 고르고 싶을 수 있다).
# 작을수록 빠르지만 멀리 있는 작은 얼굴을 놓친다(H-41: Meta 흐림본에서 1.5배는 얼굴 약 21% 놓침). 4K는 1배가 1080p 2배와
# 얼굴 픽셀 크기가 같다. 1배면 원래 크기 YuNet(H-29 두 번째 검출기)과 같은 일이라 한 번만 돌린다
FACE_SCALES = (1.0, 1.5, 2.0)
# 옆얼굴 Haar는 결과 화면(자른 곳) + 이만큼(프레임 높이 대비) 둘레에서만(H-58, 11/2): 화면 전체 4.9초 → 2.0초/장,
# 결과 화면 안 얼굴 덮음 99.9 → 99.8%. 결과에 안 보이는 곳의 옆얼굴은 찾을 필요가 없다. YuNet은 늘 화면 전체(H-28)
PROFILE_MARGIN = 0.15
HIGH_FPS_CAP = 60  # '원본 그대로'여도 결과 영상은 이 fps까지
DEFAULT_FACE_SCALE = 1.0  # 11/3 사용자: 가급적 1배(가장 빠름). 화면도 1배를 추천한다


def my_face_box(faces: list[tuple[int, int, int, int]], me: Box | None) -> tuple[int, int, int, int] | None:
    """얼굴 박스 중 '나'의 얼굴: 내 박스 위쪽(MY_HEAD_SHARE) 안에서 내 머리 꼭대기 가운데에 가장 가까운 것."""
    if me is None:
        return None
    head_bottom = me.top + (me.bottom - me.top) * MY_HEAD_SHARE
    top_center = ((me.left + me.right) / 2, me.top)
    heads = [f for f in faces if me.left <= f[0] + f[2] / 2 <= me.right and me.top <= f[1] + f[3] / 2 <= head_bottom]
    return min(heads, key=lambda f: abs(f[0] + f[2] / 2 - top_center[0]) + abs(f[1] - top_center[1]), default=None)


# 선수 몸 높이에 비해 이만큼보다 큰 '얼굴'은 얼굴이 아니다(11/3 사용자: 흐림이 가끔 엄청 커진다, H-73). 머리는 몸의 1/7 안팎.
# 카메라 가까운 사람은 몸도 커서 진짜 큰 얼굴은 남는다(10/31 '내 머리 3배' 기준은 그런 얼굴까지 버려 되돌렸다)
MAX_FACE_TO_BODY = 0.35


# 큰 상자라도 선수 머리 자리(몸 위쪽 이 비율)에 있으면 진짜 얼굴일 수 있어 남긴다(H-73: 크기만 보고 버리면 머리 자리의
# 진짜 얼굴까지 빠져 놓친 얼굴이 0 → 18로 늘었다). 몸통·다리에 잡힌 큰 '얼굴'만 버린다
HEAD_ZONE = 0.25


def drop_oversized(faces: list[tuple[int, int, int, int]], people: list[Box]) -> list[tuple[int, int, int, int]]:
    """얼굴 가운데가 어떤 선수 상자 안의 몸통·다리(머리 자리 아래)에 있고, 그 얼굴 높이가 (그 자리 선수 중 가장 큰) 몸 높이의
    MAX_FACE_TO_BODY를 넘으면 뺀다. 선수 상자 밖(관중 등)이나 머리 자리의 얼굴은 그대로 둔다."""
    def too_big(f: tuple[int, int, int, int]) -> bool:
        cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
        bodies = [p for p in people if p.left <= cx <= p.right and p.top <= cy <= p.bottom]
        if not bodies:
            return False
        tall = max(bodies, key=lambda p: p.bottom - p.top)
        height = tall.bottom - tall.top
        return f[3] > MAX_FACE_TO_BODY * height and cy > tall.top + HEAD_ZONE * height
    return [f for f in faces if not too_big(f)]


_PICK = object()


def faces_of_others(faces: list[tuple[int, int, int, int]], me: Box | None,
                    face: tuple[int, int, int, int] | None | object = _PICK) -> list[tuple[int, int, int, int]]:
    """얼굴 박스 (x, y, w, h) 중 '나'의 머리 쪽(내 박스 위쪽)에 있는 얼굴 하나를 뺀 나머지.
    몸 박스 전체로 빼면 내 앞뒤로 겹친 다른 사람 얼굴까지 드러났다(H-40). face를 주면 그것을 내 얼굴로 본다(이어 추적)."""
    mine = my_face_box(faces, me) if face is _PICK else face
    if mine is None:
        return list(faces)
    # 검출기 2~3개가 같은 내 얼굴을 각자 찾아 박스가 여럿이다. 하나만 빼면 나머지가 내 얼굴을 가렸다(10/30)
    mx, my = mine[0] + mine[2] / 2, mine[1] + mine[3] / 2
    return [f for f in faces
            if not (abs(f[0] + f[2] / 2 - mx) < max(mine[2], f[2]) * 0.6 and abs(f[1] + f[3] / 2 - my) < max(mine[3], f[3]) * 0.6)]


def mosaic_faces_except_me(frame: np.ndarray, faces: list[tuple[int, int, int, int]], me: Box | None) -> np.ndarray:
    return anonymize.blur_boxes(frame, faces_of_others(faces, me), pad=RENDER_BOX_PAD)


def probe(video: Path) -> tuple[int, int, float, int]:
    """(너비, 높이, fps, 프레임 수). 회전 정보가 있으면 화면에 보이는 방향 기준."""
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_frames:stream_side_data=rotation", "-of", "json", str(video)]))
    s = info["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    if any(abs(int(d.get("rotation", 0))) in (90, 270) for d in s.get("side_data_list", [])):
        w, h = h, w
    num, den = map(int, s["r_frame_rate"].split("/"))
    return w, h, num / den, int(s["nb_read_frames"])


# 얼굴 찾기를 여러 프로세스로(10/31): 클라우드 vCPU는 OpenCV 스레드를 늘려도 잘 안 빨라져(8스레드 → 16스레드가 오히려
# 느림) 프레임을 나눠 동시에 찾는다. 결과는 한 프로세스와 같다(같은 검출기, 프레임 순서대로 받음). 이 맥은 1(지금 그대로)
RENDER_WORKERS = int(os.environ.get("TAPME_RENDER_WORKERS", "1"))
WORKER_CV_THREADS = int(os.environ.get("TAPME_RENDER_WORKER_THREADS", "2"))
_W: dict[str, object] = {}


def _worker_init(threads: int, two_detectors: bool) -> None:
    cv2.setNumThreads(threads)
    _W["detector"], _W["profile"] = anonymize.load_detector(), anonymize.load_profile_detector()
    _W["plain"] = anonymize.load_plain_detector() if two_detectors else None


def _worker_faces(args: tuple) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    buf, width, height, crop, out_size, side, two_detectors, scale = args
    if "detector" not in _W or (two_detectors and _W.get("plain") is None):  # 같이 쓰는 프로세스(anonymize._worker_init)
        _worker_init(WORKER_CV_THREADS, two_detectors)
    bgr = np.ascontiguousarray(np.frombuffer(buf, np.uint8).reshape(height, width, 3)[..., ::-1])
    plain = _W["plain"] if two_detectors else None
    return _find_faces(bgr, _W["detector"], _W["profile"] if side else None, plain, crop, out_size, scale)  # type: ignore[arg-type]


def _find_faces(bgr: np.ndarray, detector: cv2.FaceDetectorYN, profile: cv2.CascadeClassifier | None,
                plain: cv2.FaceDetectorYN | None, crop: tuple[int, int, int, int],
                out_size: tuple[int, int], scale: float = DEFAULT_FACE_SCALE,
                ) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """한 프레임의 (얼굴, 옆얼굴 Haar 얼굴)(원본 좌표). YuNet은 매 프레임 화면 전체에서 찾는다. 3프레임마다(실험 H-27)나
    결과 화면 안만(H-28) 찾으면 놓치는 얼굴이 늘었다. 옆얼굴 Haar는 결과 화면 + PROFILE_MARGIN 둘레에서만(H-58),
    사람 몸 위에 잡힌 것은 렌더에서 거른다(H-48)."""
    h, w = bgr.shape[:2]
    x0, y0, cw, ch = crop
    m = round(PROFILE_MARGIN * h)
    region = (max(0, x0 - m), max(0, y0 - m), min(w, x0 + cw + m), min(h, y0 + ch + m))
    faces, side = anonymize.detect_faces_split(bgr, detector, profile, upscale=scale, profile_region=region)
    if plain is not None:  # 두 설정은 놓치는 얼굴이 달라 합친다(H-29)
        faces = faces + anonymize.detect_faces_plain(bgr, plain)
    out_w, out_h = out_size
    if cw < out_w:  # 확대하면 원본에서 작아 못 찾던 얼굴이 결과 화면에서는 커져 보인다(H-40)
        sx = out_w / cw
        part = cv2.resize(bgr[y0:y0 + ch, x0:x0 + cw], (out_w, out_h))
        faces = faces + [(int(x / sx) + x0, int(y / sx) + y0, int(w / sx), int(h / sx))
                         for x, y, w, h in anonymize.detect_faces(part, detector)]
    return faces, side


BODY_FROM = 0.35  # 사람 상자 위에서 이만큼 아래부터는 몸(어깨 아래). 얼굴은 그 위에 있다


def off_bodies(side: list[tuple[int, int, int, int]], people: list[Box]) -> list[tuple[int, int, int, int]]:
    """옆얼굴 Haar가 사람 몸 아래쪽(유니폼 등·등번호·다리)에 잡은 자리를 뺀다(H-48: 거기 생긴 모자이크가 눈에 띄었다).
    YuNet 얼굴은 빼지 않는다 — 앞사람 몸 뒤로 보이는 진짜 머리도 그 자리에 잡혀, 빼면 얼굴이 드러날 수 있다."""
    def on_body(b: tuple[int, int, int, int]) -> bool:
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        return any(p.left <= cx <= p.right and p.top + BODY_FROM * (p.bottom - p.top) < cy <= p.bottom for p in people)
    # 어느 선수에게도 붙지 않았는데 그 화면에서 가장 큰 선수 키의 MAX_FACE_TO_BODY보다 큰 옆얼굴은 선수 사이 빈자리를 잘못 잡은 것
    # (11/3 사용자 화면: 두 선수 몸을 통째로 덮는 큰 사각형, H-73). 관중석 진짜 얼굴은 작아서 남는다
    tallest = max((p.bottom - p.top for p in people), default=0)

    def stray(b: tuple[int, int, int, int]) -> bool:
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        alone = not any(p.left <= cx <= p.right and p.top <= cy <= p.bottom for p in people)
        return alone and tallest > 0 and b[3] > MAX_FACE_TO_BODY * tallest
    return [b for b in side if not on_body(b) and not stray(b)]


MY_FACE_FOLLOW_SEC = 0.5  # 내 얼굴을 잠깐 놓쳤을 때 마지막 자리를 몸 따라 옮겨 지키는 시간(H-73)


def follow_face(face: tuple[int, int, int, int], then: Box, now: Box) -> tuple[int, int, int, int]:
    """그때 내 몸(then) 안의 얼굴 자리를 지금 내 몸(now)으로 옮긴다: 발밑(상자 아래 가운데)이 움직인 만큼, 크기는 그대로.
    상자 위쪽·너비로 맞추면 앞사람과 겹쳐 상자가 위로 커질 때 얼굴 자리도 다른 사람 쪽으로 딸려 갔다(H-73). 0.5초 안이라 크기 변화는 작다."""
    dx = (now.left + now.right) / 2 - (then.left + then.right) / 2
    dy = now.bottom - then.bottom
    x, y, w, h = face
    return (int(x + dx), int(y + dy), w, h)


MY_FACE_JUMP = 1.2  # 따라오던 내 얼굴(몸 따라 옮긴 자리)에서 얼굴 크기의 이 배보다 멀면 내 얼굴이 아니다


def pick_my_face(faces: list[tuple[int, int, int, int]], me: Box | None,
                 last: tuple[tuple[int, int, int, int], Box, int] | None, i: int, follow: int
                 ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    """(이번 프레임에 찾은 내 얼굴, 지킬 자리). 최근(follow 프레임 안)에 찾은 내 얼굴이 있으면 그 자리를 몸 따라 옮긴 곳에
    가장 가까운 얼굴만 내 얼굴로 본다 — 앞사람과 겹쳐 내 상자가 커지면 상자 위쪽의 다른 사람(심판) 얼굴이 내 얼굴로 골라졌다
    (11/3 사용자, H-73). 없으면 원래 규칙(내 상자 위쪽). 못 찾았으면 옮긴 자리를 지킨다."""
    if me is None:
        return None, None
    if last is None or i - last[2] > follow:
        face = my_face_box(faces, me)
        return face, face
    pred = follow_face(last[0], last[1], me)
    px, py = pred[0] + pred[2] / 2, pred[1] + pred[3] / 2
    reach = MY_FACE_JUMP * max(pred[2], pred[3], 1)
    # 내 상자 안이어야 한다 — 없으면 옆 사람 얼굴로 옮겨 붙어 그 얼굴을 계속 안 가렸다(놓친 얼굴 8 → 155). 머리 자리로
    # 좁히지 않는다: 상자가 위로 커지면 내 진짜 얼굴이 머리 자리 아래로 내려간다
    near = [f for f in faces if abs(f[0] + f[2] / 2 - px) <= reach and abs(f[1] + f[3] / 2 - py) <= reach
            and me.left <= f[0] + f[2] / 2 <= me.right and me.top <= f[1] + f[3] / 2 <= me.bottom]
    face = min(near, key=lambda f: abs(f[0] + f[2] / 2 - px) + abs(f[1] + f[3] / 2 - py), default=None)
    return face, (face or pred)


def restore_rect(out: np.ndarray, src: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    """rect 자리를 src(흐리기 전)로 되돌린다. 화면 밖으로 걸친 부분은 잘라 낸다 — 왼쪽·위로 넘친 자리를 0에 붙이면
    엉뚱한 칸(옆 사람 흐림)을 되돌렸다(11/3 Muse Code 검토)."""
    x, y, w, h = (int(v) for v in rect)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(out.shape[1], x + w), min(out.shape[0], y + h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = src[y0:y1, x0:x1]


def off_my_body(side: list[tuple[int, int, int, int]], me: Box | None,
                front: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """탭한 '나'의 얼굴(정면 검출기로 이번 프레임에 찾은 것)보다 아래, 내 몸 상자 안에 잡힌 옆얼굴 Haar 상자를 뺀다.
    옆얼굴 검출기가 가슴팍을 얼굴로 잡아 내 몸에 흐림이 생겼다(11/3 사용자, 몸 위 33% 지점이라 BODY_FROM 0.35에 안 걸림).
    다른 선수는 팔을 들면 머리가 내려와 이 기준을 쓰지 않는다(off_bodies). 내 얼굴을 못 찾은 프레임은 그대로 둔다."""
    face = my_face_box(front, me)
    if me is None or face is None:
        return side
    below = face[1] + face[3]
    return [b for b in side if not (me.left <= b[0] + b[2] / 2 <= me.right and below < b[1] + b[3] / 2 <= me.bottom)]


def off_team(faces: list[tuple[int, int, int, int]], people: list[Box],
             clear_ids: set[str]) -> list[tuple[int, int, int, int]]:
    """모자이크를 풀어 줄 팀(clear_ids) 선수의 머리 자리(사람 상자 위 BODY_FROM)에 있는 얼굴만 뺀다(11/1 사용자: 팀별 모자이크).
    다른 사람 머리 자리와 겹치거나 추적에 없는 얼굴은 그대로 가린다 — 잘못 풀어 얼굴이 드러나는 쪽을 피한다."""
    def heads(b: tuple[int, int, int, int], ids: bool) -> bool:
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        return any((p.object_id in clear_ids) == ids and p.left <= cx <= p.right
                   and p.top <= cy <= p.top + BODY_FROM * (p.bottom - p.top) for p in people)
    return [b for b in faces if not heads(b, True) or heads(b, False)]


def render_video(
    src: Path,
    out: Path,
    by_sam: dict[int, list[Box]],
    masks_by_sam: dict[int, dict[str, SegmentationMaskRecord]],
    me: str,
    sam_fps: float,
    subtitles: list[tuple[float, float, str]] | None = None,
    zoom: float = DEFAULT_ZOOM,
    privacy: str = DEFAULT_PRIVACY,
    face_mosaic: bool = True,
    face_every: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
    keep_my_face: bool = True,
    two_detectors: bool = True,
    max_fps: int = 0,
    people_by_sam: dict[int, list[Box]] | None = None,
    clear_ids: set[str] | None = None,
    face_scale: float = DEFAULT_FACE_SCALE,
    full_frame: bool = False,
    face_cache_out: list | None = None,
    face_cache_in: list | None = None,
) -> Path:
    """subtitles = [(시작초, 끝초, 글자), ...]. 장면에 맞춰 바뀌는 해설 자막(score.subtitle_cues). 없으면 자막 없음.
    face_mosaic이면 '나'를 뺀 얼굴을 모자이크한다(src는 원본 클립). 얼굴은 face_every 프레임마다 찾고 사이는 이어 쓴다.
    clear_ids가 있으면 people_by_sam 중 그 선수들(풀어 줄 팀)의 얼굴은 가리지 않는다(off_team).
    face_scale: 얼굴 찾기 배율(FACE_SCALES).
    full_frame: 자르지 않고 원본 가로 화면 그대로(나 표시 영상, 11/2 사용자). face_cache_out이 있으면 프레임마다 찾은
    (얼굴, 옆얼굴)을 담고, face_cache_in이 있으면 얼굴을 다시 찾지 않고 그것을 쓴다(나 표시 영상에서 찾은 것을 하이라이트가 다시 씀)."""
    two_detectors = two_detectors and face_scale != 1.0  # 1배면 두 번째(원래 크기) 검출기와 같은 일
    detector = profile = plain = None
    if face_mosaic:
        detector, profile = anonymize.load_detector(), anonymize.load_profile_detector()
        plain = anonymize.load_plain_detector() if two_detectors else None
    width, height, fps, n = probe(src)
    vf: list[str] = []
    if not max_fps and fps > HIGH_FPS_CAP + 0.5:
        max_fps = HIGH_FPS_CAP  # 11/3 점검: 120·240fps 원본은 프레임마다 얼굴을 찾아 30fps의 4~8배 걸린다
    if max_fps and fps > max_fps + 0.5:
        # 결과를 max_fps로(사용자 선택). 60fps 원본은 프레임마다 얼굴을 찾아 렌더가 30fps의 두 배 걸렸다(10/30 성능 테스트)
        n, fps = int(n * max_fps / fps) + 1, float(max_fps)
        vf = ["-vf", f"fps={max_fps}"]
    out_h = height // 2 * 2
    out_w = (width // 2 * 2) if full_frame else min(width, int(out_h * 9 / 16)) // 2 * 2
    boxes = interpolate(by_sam, sam_fps, fps, n)
    if full_frame:
        crops = [(0, 0, out_w, out_h)] * n
    else:
        centers = [next((b.center for b in f if b.object_id == me), None) for f in boxes]
        crops = zoom_crops(centers, width, height, zoom, smooth=max(1, round(fps / 2)))

    out.parent.mkdir(parents=True, exist_ok=True)
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(src), *vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}",
         "-r", f"{fps}", "-i", "-", "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    assert dec.stdout and enc.stdin
    size = width * height * 3
    # 얼굴 찾기 프로세스는 흐리기와 같이 쓴다(anonymize.acquire_pool, 11/1 메모리). 일꾼 쪽 검출기는 처음 쓸 때 올린다
    pool = anonymize.acquire_pool() if face_mosaic and RENDER_WORKERS > 1 and face_cache_in is None else None
    # 앞으로 읽어 둔 프레임과 그 얼굴 찾기(여러 프로세스면 미리 맡겨 둔다). 프레임 순서대로 꺼낸다
    ahead: deque[tuple[bytes, Future | None]] = deque()
    depth, read_i = (RENDER_WORKERS * 2 if pool else 1), 0

    def fill() -> None:
        nonlocal read_i
        while len(ahead) < depth and len(buf := dec.stdout.read(size)) == size:
            job = None
            if pool is not None and read_i % face_every == 0:
                job = pool.submit(_worker_faces, (buf, width, height, crops[min(read_i, n - 1)], (out_w, out_h),
                                                  read_i % PROFILE_EVERY == 0, two_detectors, face_scale))
            ahead.append((buf, job))
            read_i += 1

    _MASKS.clear()
    want = {round(j * sam_fps / fps) for j in range(n)}  # 이 영상에 쓰일 SAM 프레임의 마스크만
    _predecode([rec for k in want for oid, rec in masks_by_sam.get(k, {}).items() if privacy == "silhouette" or oid == me])
    try:
        i = 0
        faces: list[tuple[int, int, int, int]] = []
        recent: deque[list[tuple[int, int, int, int]]] = deque(maxlen=max(1, round(FACE_HOLD_SEC * fps)))
        my_face: tuple[int, int, int, int] | None = None
        last_face: tuple[tuple[int, int, int, int], Box, int] | None = None  # 마지막으로 찾은 내 얼굴, 그때 내 몸, 프레임
        follow = max(1, round(MY_FACE_FOLLOW_SEC * fps))
        fill()
        while ahead:
            buf, job = ahead.popleft()
            fill()
            k = min(i, n - 1)
            now = i / fps
            text = next((txt for start, end, txt in subtitles or [] if start <= now < end), "")
            frame = Image.frombytes("RGB", (width, height), buf)
            if face_mosaic:
                rgb = np.asarray(frame)
                if i % face_every == 0:
                    if face_cache_in is not None:
                        faces, side = face_cache_in[i] if i < len(face_cache_in) else ([], [])
                        faces, side = [tuple(f) for f in faces], [tuple(f) for f in side]
                    elif job is not None:
                        faces, side = job.result()
                    else:
                        bgr = np.ascontiguousarray(rgb[..., ::-1])  # 검출기는 BGR
                        faces, side = _find_faces(bgr, detector, profile if i % PROFILE_EVERY == 0 else None, plain,
                                                  crops[k], (out_w, out_h), face_scale)
                    if face_cache_out is not None:
                        face_cache_out.append((faces, side))
                    if people_by_sam is not None:  # 다른 선수까지 추적했으면 몸 위에 잡힌 옆얼굴을 거른다(H-48)
                        side = off_bodies(side, people_by_sam.get(round(i * sam_fps / fps), []))
                    if keep_my_face:  # 내 얼굴보다 아래 내 몸에 잡힌 옆얼굴은 버린다(11/3 사용자: 내 가슴에 흐림, H-73)
                        side = off_my_body(side, next((b for b in boxes[k] if b.object_id == me), None), faces)
                    faces = faces + side
                    if clear_ids and people_by_sam is not None:
                        faces = off_team(faces, people_by_sam.get(round(i * sam_fps / fps), []), clear_ids)
                    if people_by_sam is not None:  # 몸에 비해 너무 큰 '얼굴'은 버린다(H-73)
                        faces = drop_oversized(faces, people_by_sam.get(round(i * sam_fps / fps), []))
                    mine = next((b for b in boxes[k] if b.object_id == me), None)
                    found, my_face = pick_my_face(faces, mine, last_face, i, follow) if keep_my_face else (None, None)
                    if found is not None and mine is not None:
                        last_face = (found, mine, i)
                    # 내 얼굴은 찾은 그 프레임에서 뺀다. 모아 둔 뒤에 빼면 0.2초 전 내 얼굴 자리가 남아 탭한 나에게
                    # 모자이크가 붙었다(10/30 사용자). keep_my_face=False는 품질 확인용(qa.py) — '나'까지 가린다
                    recent.append(faces_of_others(faces, mine, found) if keep_my_face else faces_of_others(faces, None))
                held = [f for fs in recent for f in fs]  # 검출기가 잠깐 놓쳐도 방금 찾은 다른 사람 얼굴은 가린다
                out_rgb = anonymize.blur_boxes(rgb, held, pad=RENDER_BOX_PAD)
                # 옆 사람 흐림(사방 40% 넓힘)이 내 얼굴로 넘어오지 않게 내 얼굴 자리만 원래 화면으로 되돌린다(11/3 사용자, H-73:
                # 넘어온 흐림 1,227 → 0프레임). 이번 프레임에 못 찾았으면 0.5초까지 마지막 내 얼굴을 몸 따라 옮긴 자리(걸침 221 → 0,
                # 드러날 수 있는 다른 얼굴 4,143개 중 2 → 7, 사용자 우선순위 품질 > 초상권). 옛 자리를 그대로 쓰면 18개가 드러났고,
                # 되돌린 자리를 다시 흐리면 내 얼굴에 흐림이 도로 붙었다(869프레임)
                if my_face is not None and i % face_every == 0:
                    restore_rect(out_rgb, rgb, my_face)
                frame = Image.fromarray(out_rgb)
            img = compose(frame, boxes[k],
                          masks_by_sam.get(round(i * sam_fps / fps), {}), me, crops[k], (out_w, out_h),
                          text, privacy, head=my_face if face_mosaic and keep_my_face else None)
            try:
                enc.stdin.write(img.tobytes())
            except BrokenPipeError:
                # 소리가 화면보다 조금 짧으면(-shortest) 인코더가 먼저 끝난다: 원본 경기 영상은 소리 120.0초·화면 120.07초였다
                # (11/2 나 표시 영상). 정상으로 끝났으면 남은 몇 프레임만 버리고, 아니면 아래에서 실패로 알린다
                break
            i += 1
            if on_progress and i % 30 == 0:
                on_progress(i, n)
    finally:
        _MASKS.clear()
        if pool is not None:
            for _, job in ahead:  # 같이 쓰는 프로세스라 남은 내 일만 거둔다
                if job is not None:
                    job.cancel()
            anonymize.release_pool()
        dec.stdout.close()
        dec.kill()
        dec.wait()
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg 인코딩 실패")
    return out
