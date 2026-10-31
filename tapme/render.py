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
import multiprocessing
import os
import subprocess
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
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


def _mask(rec: SegmentationMaskRecord | None, w: int, h: int) -> Image.Image | None:
    if rec is None or w <= 0 or h <= 0:
        return None
    hit = _MASKS.get(id(rec))
    if hit is None or hit[0] is not rec:
        mw, mh = rec.mask.width, rec.mask.height
        raster = decode_mask_to_raster(rec.mask)  # 0/1 바이트
        img = None
        if mw > 0 and mh > 0 and len(raster) == mw * mh:
            img = Image.fromarray(np.frombuffer(bytes(raster), np.uint8).reshape(mh, mw) * np.uint8(255), "L")
        hit = _MASKS[id(rec)] = (rec, img)
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


def _draw_head_marker(img: Image.Image, me: Box) -> None:
    """머리 위 노란 삼각형(그림자 포함)."""
    w = me.right - me.left
    cx = (me.left + me.right) / 2
    s = max(14.0, w * 0.32)
    tip = me.top - s * 0.45
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
        _draw_head_marker(work, my_box)
    x0, y0, cw, ch = crop
    out = work.crop((x0, y0, x0 + cw, y0 + ch)).resize(out_size, Image.Resampling.LANCZOS)
    if subtitle:
        _draw_subtitle(out, subtitle)  # 확대 뒤에 그려야 글자가 흐려지지 않는다
    return out


# 결과 영상의 얼굴 모자이크(H-40, 사용자 10/30 "살짝살짝 벗겨진다"): 프레임마다 새로 찾아 검출기가 한두 프레임 놓치면
# 그 순간만 드러났다 → 찾은 자리를 잠깐 유지하고, 박스도 넓힌다. Meta로 보내는 흐림본(anonymize)은 그대로
FACE_HOLD_SEC = 0.2
RENDER_BOX_PAD = 0.4
MY_HEAD_SHARE = 0.35  # '나'의 얼굴로 보는 곳: 내 박스 위쪽 이 비율 안


def faces_of_others(faces: list[tuple[int, int, int, int]], me: Box | None) -> list[tuple[int, int, int, int]]:
    """얼굴 박스 (x, y, w, h) 중 '나'의 머리 쪽(내 박스 위쪽)에 있는 얼굴 하나를 뺀 나머지.
    몸 박스 전체로 빼면 내 앞뒤로 겹친 다른 사람 얼굴까지 드러났다(H-40)."""
    if me is None:
        return list(faces)
    head_bottom = me.top + (me.bottom - me.top) * MY_HEAD_SHARE
    top_center = ((me.left + me.right) / 2, me.top)
    heads = [f for f in faces if me.left <= f[0] + f[2] / 2 <= me.right and me.top <= f[1] + f[3] / 2 <= head_bottom]
    mine = min(heads, key=lambda f: abs(f[0] + f[2] / 2 - top_center[0]) + abs(f[1] - top_center[1]), default=None)
    if mine is None:
        return list(faces)
    # 검출기 2~3개가 같은 내 얼굴을 각자 찾아 박스가 여럿이다. 하나만 빼면 나머지가 내 얼굴을 가렸다(10/30)
    mx, my = mine[0] + mine[2] / 2, mine[1] + mine[3] / 2
    return [f for f in faces
            if not (abs(f[0] + f[2] / 2 - mx) < max(mine[2], f[2]) * 0.6 and abs(f[1] + f[3] / 2 - my) < max(mine[3], f[3]) * 0.6)]


def mosaic_faces_except_me(frame: np.ndarray, faces: list[tuple[int, int, int, int]], me: Box | None) -> np.ndarray:
    return anonymize.pixelate_boxes(frame, faces_of_others(faces, me), pad=RENDER_BOX_PAD)


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
    buf, width, height, crop, out_size = args
    bgr = np.ascontiguousarray(np.frombuffer(buf, np.uint8).reshape(height, width, 3)[..., ::-1])
    return _find_faces(bgr, _W["detector"], _W["profile"], _W["plain"], crop, out_size)  # type: ignore[arg-type]


def _find_faces(bgr: np.ndarray, detector: cv2.FaceDetectorYN, profile: cv2.CascadeClassifier | None,
                plain: cv2.FaceDetectorYN | None, crop: tuple[int, int, int, int],
                out_size: tuple[int, int]) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """한 프레임의 (얼굴, 옆얼굴 Haar 얼굴)(원본 좌표). 매 프레임 화면 전체에서 찾는다. 3프레임마다(실험 H-27)나
    결과 화면 안만(H-28) 찾으면 놓치는 얼굴이 늘었다. 옆얼굴은 사람 몸 위에 잡힌 것을 렌더에서 거른다(H-48)."""
    faces, side = anonymize.detect_faces_split(bgr, detector, profile)
    if plain is not None:  # 두 설정은 놓치는 얼굴이 달라 합친다(H-29)
        faces = faces + anonymize.detect_faces_plain(bgr, plain)
    x0, y0, cw, ch = crop
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
    return [b for b in side if not on_body(b)]


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
) -> Path:
    """subtitles = [(시작초, 끝초, 글자), ...]. 장면에 맞춰 바뀌는 해설 자막(score.subtitle_cues). 없으면 자막 없음.
    face_mosaic이면 '나'를 뺀 얼굴을 모자이크한다(src는 원본 클립). 얼굴은 face_every 프레임마다 찾고 사이는 이어 쓴다."""
    detector = profile = plain = None
    if face_mosaic:
        detector, profile = anonymize.load_detector(), anonymize.load_profile_detector()
        plain = anonymize.load_plain_detector() if two_detectors else None
    width, height, fps, n = probe(src)
    vf: list[str] = []
    if max_fps and fps > max_fps + 0.5:
        # 결과를 max_fps로(사용자 선택). 60fps 원본은 프레임마다 얼굴을 찾아 렌더가 30fps의 두 배 걸렸다(10/30 성능 테스트)
        n, fps = int(n * max_fps / fps) + 1, float(max_fps)
        vf = ["-vf", f"fps={max_fps}"]
    out_h = height // 2 * 2
    out_w = min(width, int(out_h * 9 / 16)) // 2 * 2
    boxes = interpolate(by_sam, sam_fps, fps, n)
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
    pool = None
    if face_mosaic and RENDER_WORKERS > 1:
        pool = ProcessPoolExecutor(RENDER_WORKERS, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_worker_init, initargs=(WORKER_CV_THREADS, two_detectors))
    # 앞으로 읽어 둔 프레임과 그 얼굴 찾기(여러 프로세스면 미리 맡겨 둔다). 프레임 순서대로 꺼낸다
    ahead: deque[tuple[bytes, Future | None]] = deque()
    depth, read_i = (RENDER_WORKERS * 2 if pool else 1), 0

    def fill() -> None:
        nonlocal read_i
        while len(ahead) < depth and len(buf := dec.stdout.read(size)) == size:
            job = None
            if pool is not None and read_i % face_every == 0:
                job = pool.submit(_worker_faces, (buf, width, height, crops[min(read_i, n - 1)], (out_w, out_h)))
            ahead.append((buf, job))
            read_i += 1

    _MASKS.clear()
    try:
        i = 0
        faces: list[tuple[int, int, int, int]] = []
        recent: deque[list[tuple[int, int, int, int]]] = deque(maxlen=max(1, round(FACE_HOLD_SEC * fps)))
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
                    if job is not None:
                        faces, side = job.result()
                    else:
                        bgr = np.ascontiguousarray(rgb[..., ::-1])  # 검출기는 BGR
                        faces, side = _find_faces(bgr, detector, profile, plain, crops[k], (out_w, out_h))
                    if people_by_sam is not None:  # 다른 선수까지 추적했으면 몸 위에 잡힌 옆얼굴을 거른다(H-48)
                        side = off_bodies(side, people_by_sam.get(round(i * sam_fps / fps), []))
                    faces = faces + side
                    mine = next((b for b in boxes[k] if b.object_id == me), None)
                    # 내 얼굴은 찾은 그 프레임에서 뺀다. 모아 둔 뒤에 빼면 0.2초 전 내 얼굴 자리가 남아 탭한 나에게
                    # 모자이크가 붙었다(10/30 사용자). keep_my_face=False는 품질 확인용(qa.py) — '나'까지 가린다
                    recent.append(faces_of_others(faces, mine if keep_my_face else None))
                held = [f for fs in recent for f in fs]  # 검출기가 잠깐 놓쳐도 방금 찾은 다른 사람 얼굴은 가린다
                frame = Image.fromarray(anonymize.pixelate_boxes(rgb, held, pad=RENDER_BOX_PAD))
            img = compose(frame, boxes[k],
                          masks_by_sam.get(round(i * sam_fps / fps), {}), me, crops[k], (out_w, out_h),
                          text, privacy)
            enc.stdin.write(img.tobytes())
            i += 1
            if on_progress and i % 30 == 0:
                on_progress(i, n)
    finally:
        _MASKS.clear()
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        dec.stdout.close()
        dec.wait()
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg 인코딩 실패")
    return out
