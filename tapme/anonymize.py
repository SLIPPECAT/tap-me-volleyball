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
import subprocess
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
# 10/28 H-22 보강: 0.6·원래 크기·정면 검출만으로는 옆얼굴을 놓쳤다(H-21)
SCORE_THRESHOLD = 0.5
DETECT_UPSCALE = 2.0  # 프레임을 이만큼 키워 YuNet을 돌린다(작은 얼굴)
PROFILE_MIN_NEIGHBORS = 4
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
            for x, y, bw, bh in profile.detectMultiScale(img, minNeighbors=PROFILE_MIN_NEIGHBORS):
                x = w - int(x) - int(bw) if flipped else int(x)
                boxes.append((x, int(y), int(bw), int(bh)))
    return boxes


def pixelate_boxes(
    frame: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """박스 영역을 모자이크한 새 프레임을 돌려준다. 원본 프레임은 바꾸지 않는다."""
    out = frame.copy()
    h, w = frame.shape[:2]
    for x, y, bw, bh in boxes:
        x0 = max(0, x - int(bw * BOX_PAD))
        y0 = max(0, y - int(bh * BOX_PAD))
        x1 = min(w, x + bw + int(bw * BOX_PAD))
        y1 = min(h, y + bh + int(bh * BOX_PAD))
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        block = max(MIN_MOSAIC_BLOCK, (x1 - x0) // MOSAIC_DIVISOR)
        small_w = max(1, (x1 - x0) // block)
        small_h = max(1, (y1 - y0) // block)
        tiny = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out[y0:y1, x0:x1] = cv2.resize(tiny, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return out


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
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            boxes = detect_faces(frame, detector, profile)
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
