"""얼굴 모자이크: 외부 API로 영상을 보내기 전에 로컬에서 얼굴을 가린다.

출처: 실험 레포 baegugu-highlight `experiments/anonymize/anonymize.py` (커밋 4989e48,
2026-10-28, 대회 기간 작성)를 옮기고 경로만 바꿨다.

OpenCV YuNet 얼굴 검출기로 프레임마다 얼굴을 찾고, 박스를 조금 넓혀 모자이크한다.
API 호출은 없다. 원본 오디오는 그대로 붙인다(Muse Spark가 오디오도 본다).

모델 파일(232KB, opencv_zoo)은 git에 올리지 않는다. 처음 한 번만 받는다:
  curl -sSL -o tapme/models/face_detection_yunet_2023mar.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

사용법:
  uv run python -m tapme.anonymize data/clip.mp4 --out data/clip_blur.mp4

한계: 관중석처럼 아주 작은 얼굴은 놓칠 수 있다. 검출 누락은 보정하지 않고, 검출 통계만 보고한다.
옛 설정(기준 0.6, 원래 크기, 정면만)은 옆얼굴을 놓쳐서 지금 값(기준 0.5, 2배 검출, 옆얼굴 검출기)으로 바꿨다.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import threading
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
# OpenCV 스레드 수. 클라우드 vCPU는 하이퍼스레딩이라 vCPU 수만큼 쓰면 서로 다퉈 느려졌다
# (n2-highcpu-16에서 1분 조각 블러: 16스레드 171초, 8스레드 111초, 10/31). 배포 서버는 물리 코어 수로 정한다
if threads := int(os.environ.get("TAPME_CV_THREADS", "0")):
    cv2.setNumThreads(threads)
# 10/28 H-22 보강: 0.6·원래 크기·정면 검출만으로는 옆얼굴을 놓쳤다(H-21)
SCORE_THRESHOLD = 0.5
DETECT_UPSCALE = 2.0  # 프레임을 이만큼 키워 YuNet을 돌린다(작은 얼굴)
PROFILE_MIN_NEIGHBORS = 4
PROFILE_SCALE_FACTOR = 1.1  # 결과 영상 모자이크(render.py)는 그대로
# Meta로 보내는 흐림본: 옆얼굴 검출 간격을 넓혀 1시간 영상 랠리 찾기 78분 → 약 47분. 지금 방식이 찾는 얼굴의 99.4%를
# 그대로 가린다(H-41, 10/30 사용자 결정). 옆얼굴 검출기가 시간의 60%를 쓰고 더하는 얼굴은 0.7%였다
FAST_PROFILE_SCALE_FACTOR = 1.2
NMS_THRESHOLD = 0.3
MAX_FACES = 5000
BOX_PAD = 0.25  # 얼굴 박스를 각 변 기준 이 비율만큼 넓힌다(귀·머리카락까지 가리기 위해)
MOSAIC_DIVISOR = 12  # 모자이크 블록 크기 = 박스 폭 / 이 값. 작을수록 굵은 모자이크
MIN_MOSAIC_BLOCK = 8  # 작은 얼굴(관중석)도 알아볼 수 없도록 블록 최소 크기(px)


def load_detector() -> cv2.FaceDetectorYN:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YuNet 모델이 없습니다. docstring의 curl로 받아 두세요: {MODEL_PATH}")
    return cv2.FaceDetectorYN.create(
        str(MODEL_PATH), "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, MAX_FACES
    )


def load_profile_detector() -> cv2.CascadeClassifier:
    """OpenCV에 들어 있는 옆얼굴 검출기(Haar). 한쪽 방향만 잡으므로 뒤집은 프레임에도 돌린다."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


def detect_faces(
    frame: np.ndarray,
    detector: cv2.FaceDetectorYN,
    profile: cv2.CascadeClassifier | None = None,
    profile_scale: float = PROFILE_SCALE_FACTOR,
) -> list[tuple[int, int, int, int]]:
    """프레임에서 얼굴 박스 (x, y, w, h) 목록을 돌려준다.

    YuNet은 DETECT_UPSCALE배로 키운 프레임에서 찾고, profile이 있으면 옆얼굴도 양쪽 방향으로 찾는다.
    """
    h, w = frame.shape[:2]
    big_w, big_h = int(w * DETECT_UPSCALE), int(h * DETECT_UPSCALE)
    big = frame if (big_w, big_h) == (w, h) else cv2.resize(frame, (big_w, big_h))
    detector.setInputSize((big_w, big_h))
    _, faces = detector.detect(big)
    boxes = [] if faces is None else [
        tuple(int(v / DETECT_UPSCALE) for v in f[:4]) for f in faces
    ]
    if profile is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for flipped in (False, True):
            img = cv2.flip(gray, 1) if flipped else gray
            for x, y, bw, bh in profile.detectMultiScale(img, scaleFactor=profile_scale, minNeighbors=PROFILE_MIN_NEIGHBORS):
                x = w - int(x) - int(bw) if flipped else int(x)
                boxes.append((x, int(y), int(bw), int(bh)))
    return boxes


def load_plain_detector() -> cv2.FaceDetectorYN:
    """원래 크기로 돌리는 두 번째 YuNet. 2배로 키워 찾는 쪽과 놓치는 얼굴이 달라 결과 영상에서 함께 쓴다(H-29)."""
    return cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, MAX_FACES)


def detect_faces_plain(frame: np.ndarray, detector: cv2.FaceDetectorYN) -> list[tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    return [] if faces is None else [tuple(int(v) for v in f[:4]) for f in faces]


def pixelate_boxes(
    frame: np.ndarray, boxes: list[tuple[int, int, int, int]], pad: float = BOX_PAD
) -> np.ndarray:
    """박스 영역을 모자이크한 새 프레임을 돌려준다. 원본 프레임은 바꾸지 않는다."""
    out = frame.copy()
    h, w = frame.shape[:2]
    for x, y, bw, bh in boxes:
        x0 = max(0, x - int(bw * pad))
        y0 = max(0, y - int(bh * pad))
        x1 = min(w, x + bw + int(bw * pad))
        y1 = min(h, y + bh + int(bh * pad))
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        block = max(MIN_MOSAIC_BLOCK, (x1 - x0) // MOSAIC_DIVISOR)
        small_w = max(1, (x1 - x0) // block)
        small_h = max(1, (y1 - y0) // block)
        tiny = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out[y0:y1, x0:x1] = cv2.resize(tiny, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return out


# 얼굴 찾기를 여러 프로세스로(10/31): 클라우드 vCPU는 OpenCV 스레드를 늘려도 잘 안 빨라진다(렌더에서 8프로세스 × 1스레드가
# 한 프로세스보다 36% 빨랐다). 프레임 순서대로 받아 결과는 같다. 이 맥은 1(지금 그대로, M4는 나누면 오히려 느렸다)
BLUR_WORKERS = int(os.environ.get("TAPME_BLUR_WORKERS", "1"))
_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()
_W: dict[str, object] = {}


def _worker_init() -> None:
    cv2.setNumThreads(1)
    _W["detector"], _W["profile"] = load_detector(), load_profile_detector()


def _worker_detect(args: tuple) -> list[tuple[int, int, int, int]]:
    buf, shape = args
    frame = np.frombuffer(buf, np.uint8).reshape(shape)
    return detect_faces(frame, _W["detector"], _W["profile"], FAST_PROFILE_SCALE_FACTOR)  # type: ignore[arg-type]


def _blur_pool() -> ProcessPoolExecutor:
    """랠리 찾기는 1분 조각마다 부르므로 프로세스를 한 번 띄워 계속 쓴다."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(BLUR_WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
        return _pool


def anonymize_video(src: Path, dst: Path) -> dict[str, int]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열지 못했습니다: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = load_detector()
    profile = load_profile_detector()
    dst.parent.mkdir(parents=True, exist_ok=True)
    silent = dst.with_name(dst.stem + ".silent.mp4")

    writer: cv2.VideoWriter | None = None
    stats = {"frames": 0, "frames_with_faces": 0, "face_boxes": 0}
    pool = _blur_pool() if BLUR_WORKERS > 1 else None
    ahead: deque[tuple[np.ndarray, Future | None]] = deque()  # 미리 읽어 얼굴 찾기를 맡겨 둔 프레임(순서대로 꺼낸다)

    def fill() -> None:
        while len(ahead) < (BLUR_WORKERS * 2 if pool else 1):
            ok, f = cap.read()
            if not ok:
                return
            ahead.append((f, pool.submit(_worker_detect, (f.tobytes(), f.shape)) if pool else None))

    try:
        fill()
        while ahead:
            frame, job = ahead.popleft()
            fill()
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            boxes = job.result() if job else detect_faces(frame, detector, profile, FAST_PROFILE_SCALE_FACTOR)
            stats["frames"] += 1
            if boxes:
                stats["frames_with_faces"] += 1
                stats["face_boxes"] += len(boxes)
                frame = pixelate_boxes(frame, boxes)
            writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if writer is None:
        raise RuntimeError(f"프레임을 읽지 못했습니다: {src}")
    # 영상은 모자이크본, 오디오는 원본에서 그대로 복사한다(원본이 오디오가 없으면 영상만 남는다).
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent), "-i", str(src),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
            str(dst),
        ],
        check=True,
    )
    silent.unlink()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="얼굴 모자이크 (로컬, API 호출 없음)")
    parser.add_argument("video", type=Path, help="원본 영상 경로")
    parser.add_argument("--out", type=Path, required=True, help="모자이크본 출력 경로 (data/ 아래)")
    args = parser.parse_args()

    stats = anonymize_video(args.video, args.out)
    total = stats["frames"] or 1
    print(
        f"프레임 {stats['frames']}개 · 얼굴 검출 프레임 {stats['frames_with_faces']}개 "
        f"({stats['frames_with_faces'] / total:.0%}) · 얼굴 박스 합계 {stats['face_boxes']}개"
    )
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
