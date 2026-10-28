"""하이라이트 렌더: '나' 중심 9:16 확대 + 다른 회원 가리기 + 자막(언어는 사용자가 고름, score.subtitle_text).

SAM은 5fps로만 돌리고(비용), 박스를 프레임 사이에 직선으로 이어 원본 fps로 렌더한다.
실험(실험 레포 H-18): 보간 오차 중앙값 6px, 2배 확대에서 '나'가 화면 높이의 21.5%.

가리기 방식(사용자가 고른다):
  face        입력 흐림본의 얼굴 모자이크만. 경기 흐름이 가장 잘 보인다 (기본)
  silhouette  SAM이 찾은 다른 선수 몸도 흐림. SAM이 놓치거나 어긋나면 빈틈이 생긴다 (H-18)
  background  '나'만 빼고 전부 흐림. 빈틈은 없지만 공·네트까지 지워진다 (H-20)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from meta_sam_parser import SegmentationMaskRecord, decode_mask_to_raster
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tapme.track import Box

PRIVACY_MODES = ("face", "silhouette", "background")
DEFAULT_PRIVACY = "face"
ZOOM_CHOICES = (1.0, 1.5, 2.0, 2.5)
DEFAULT_ZOOM = 2.0
SILHOUETTE_BLUR_RADIUS = 14  # 다른 선수 몸만 흐림
# '나' 빼고 전부: 14는 공·네트까지 지워져 경기가 안 보였다(실험 H-20, 10/29 사용자 테스트). 얼굴은 입력 흐림본에서
# 이미 모자이크돼 있어, 6이면 경기 흐름이 보이면서 다른 선수 등번호는 읽기 어렵다
BACKGROUND_BLUR_RADIUS = 6
RING_COLOR = (0, 230, 255, 235)

# 한글과 일본어 한자를 모두 가진 글꼴을 고른다(Apple SD Gothic Neo에는 '黒'이 없음)
FONT_CANDIDATES = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
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
        win = filled[max(0, i - smooth): i + smooth + 1]
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


def _mask(rec: SegmentationMaskRecord | None, w: int, h: int) -> Image.Image | None:
    if rec is None or w <= 0 or h <= 0:
        return None
    mw, mh = rec.mask.width, rec.mask.height
    raster = decode_mask_to_raster(rec.mask)
    if mw <= 0 or mh <= 0 or len(raster) != mw * mh:
        return None
    img = Image.frombytes("L", (mw, mh), bytes(255 if v else 0 for v in raster))
    return img.resize((w, h), Image.Resampling.NEAREST)


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


def compose(
    frame: Image.Image,
    boxes: list[Box],
    masks: dict[str, SegmentationMaskRecord],
    me: str,
    crop: tuple[int, int, int, int],
    out_size: tuple[int, int],
    subtitle: str = "",
    privacy: str = DEFAULT_PRIVACY,
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
        cx, w = (my_box.left + my_box.right) / 2, my_box.right - my_box.left
        rx, ry = max(14, w / 2 + 6), max(6, (my_box.bottom - my_box.top) / 10)
        ImageDraw.Draw(work, "RGBA").ellipse((cx - rx, my_box.bottom - ry, cx + rx, my_box.bottom + ry),
                                             outline=RING_COLOR, width=3)
    x0, y0, cw, ch = crop
    out = work.crop((x0, y0, x0 + cw, y0 + ch)).resize(out_size, Image.Resampling.LANCZOS)
    if subtitle:
        _draw_subtitle(out, subtitle)  # 확대 뒤에 그려야 글자가 흐려지지 않는다
    return out


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


def render_video(
    src: Path,
    out: Path,
    by_sam: dict[int, list[Box]],
    masks_by_sam: dict[int, dict[str, SegmentationMaskRecord]],
    me: str,
    sam_fps: float,
    subtitle: str = "",
    zoom: float = DEFAULT_ZOOM,
    privacy: str = DEFAULT_PRIVACY,
) -> Path:
    width, height, fps, n = probe(src)
    out_h = height // 2 * 2
    out_w = min(width, int(out_h * 9 / 16)) // 2 * 2
    boxes = interpolate(by_sam, sam_fps, fps, n)
    centers = [next((b.center for b in f if b.object_id == me), None) for f in boxes]
    crops = zoom_crops(centers, width, height, zoom, smooth=max(1, round(fps / 2)))

    out.parent.mkdir(parents=True, exist_ok=True)
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}",
         "-r", f"{fps}", "-i", "-", "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    assert dec.stdout and enc.stdin
    size = width * height * 3
    try:
        i = 0
        while len(buf := dec.stdout.read(size)) == size:
            k = min(i, n - 1)
            img = compose(Image.frombytes("RGB", (width, height), buf), boxes[k],
                          masks_by_sam.get(round(i * sam_fps / fps), {}), me, crops[k], (out_w, out_h),
                          subtitle, privacy)
            enc.stdin.write(img.tobytes())
            i += 1
    finally:
        dec.stdout.close()
        dec.wait()
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg 인코딩 실패")
    return out
