"""성능 테스트: 정답이 있는 경기 영상을 서버에 올려 사용자처럼 처음부터 끝까지 진행하고 속도·품질·비용을 잰다.

케이스 파일(JSON)마다 영상·정답·고를 장면 수를 적는다. 실행할 때마다 data/bench/<시각>/results.json을 남겨
이전 실행과 비교한다. 서버(uvicorn)가 떠 있어야 한다. 유료 API를 쓴다(케이스당 약 $0.4~1.2).

사용법:
  uv run python -m tapme.bench <cases.json> [--only T1] [--base http://127.0.0.1:8000]
  # 배포 서버: 사용 코드를 주고, 결과 영상 검사(QA·얼굴·선명도)는 서버 파일이 없어 건너뛴다
  uv run python -m tapme.bench <cases.json> --base https://… --code XXXX-XXXX
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from tapme import anonymize, qa

SAME_RALLY_SEC = 2.0  # 정답과 이 안이면 같은 랠리(10/30 사용자 결정)


def _truth(path: Path, offset: float, until: float | None) -> list[float]:
    """정답 득점 시각(2초 안은 한 랠리로). offset을 빼 영상 안 시각으로, until 넘는 것은 뺀다."""
    pts = sorted(p["t"] - offset for p in json.loads(path.read_text())["points"])
    out: list[float] = []
    for t in pts:
        if (until is None or t <= until) and (not out or t - out[-1] > SAME_RALLY_SEC):
            out.append(t)
    return out


def score_rallies(truth: list[float], found: list[float]) -> dict[str, Any]:
    pairs = sorted((abs(t - f), i, j) for i, t in enumerate(truth) for j, f in enumerate(found) if abs(t - f) <= SAME_RALLY_SEC)
    ti, fj, errs = set(), set(), []
    for d, i, j in pairs:
        if i not in ti and j not in fj:
            ti.add(i); fj.add(j); errs.append(d)
    return {"truth": len(truth), "found_in_range": len(found), "matched": len(errs),
            "missed": [round(truth[i], 1) for i in range(len(truth)) if i not in ti],
            "extra": [round(found[j], 1) for j in range(len(found)) if j not in fj],
            "err_median_sec": round(float(np.median(errs)), 2) if errs else None}


def visible_faces(video: Path) -> dict[str, int]:
    """확인용 영상('나'까지 모두 가림)에서 다시 잡히는 얼굴 = 모자이크가 벗겨진 얼굴(H-40 측정법). 얼굴은 보지 않는다."""
    a1 = cv2.FaceDetectorYN.create(str(anonymize.MODEL_PATH), "", (320, 320), 0.6, 0.3, 5000)
    a2 = cv2.FaceDetectorYN.create(str(anonymize.MODEL_PATH), "", (320, 320), 0.5, 0.3, 5000)
    cap, frames, with_face, faces = cv2.VideoCapture(str(video)), 0, 0, 0
    while (r := cap.read())[0]:
        f = r[1]; h, w = f.shape[:2]; frames += 1
        a1.setInputSize((w, h)); _, x1 = a1.detect(f)
        a2.setInputSize((w * 2, h * 2)); _, x2 = a2.detect(cv2.resize(f, (w * 2, h * 2)))
        boxes = [] if x1 is None else [tuple(v[:4]) for v in x1]
        for v in [] if x2 is None else x2:
            b = tuple(float(c) / 2 for c in v[:4])
            if not any(abs(b[0] - o[0]) < max(b[2], o[2]) * 0.5 and abs(b[1] - o[1]) < max(b[3], o[3]) * 0.5 for o in boxes):
                boxes.append(b)
        faces += len(boxes); with_face += bool(boxes)
    return {"frames": frames, "frames_with_visible_face": with_face, "visible_faces": faces}


def sharpness(video: Path) -> dict[str, Any]:
    """결과 영상의 크기·fps와 선명도(라플라시안 분산 평균, 클수록 선명). 실행끼리 비교용."""
    cap = cv2.VideoCapture(str(video))
    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    vals, i = [], 0
    while (r := cap.read())[0]:
        if i % 10 == 0:
            vals.append(float(cv2.Laplacian(cv2.cvtColor(r[1], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
        i += 1
    return {"size": f"{w}x{h}", "fps": round(fps, 2), "sharpness": round(float(np.mean(vals)), 1) if vals else None}


class Run:
    def __init__(self, base: str, code: str = "") -> None:
        self.http = httpx.Client(base_url=base, timeout=120)
        self.code = code  # 있으면 배포 서버: 올릴 때 보내고, 로컬 파일을 보는 검사는 건너뛴다

    def status(self, job: str) -> dict[str, Any]:
        return self.http.get(f"/jobs/{job}").json()

    def wait(self, job: str, done, on_tick=None, limit: float = 4 * 3600) -> dict[str, Any]:
        t0 = time.time()
        while time.time() - t0 < limit:
            s = self.status(job)
            if on_tick:
                on_tick(s)
            if s["error"]:
                raise RuntimeError(s["error"])
            if done(s):
                return s
            time.sleep(1)
        raise TimeoutError(job)


def _upload_resumable(run: Run, video: Path, match: str) -> str:
    size = video.stat().st_size
    up = run.http.post("/uploads", json={"filename": video.name, "size": size, "match": match, "code": run.code}).raise_for_status().json()["id"]
    with video.open("rb") as f:
        while (pos := f.tell()) < size:
            run.http.put(f"/uploads/{up}", params={"offset": pos}, content=f.read(16 * 1024 * 1024), timeout=600).raise_for_status()
    return run.http.post(f"/uploads/{up}/finish").raise_for_status().json()["id"]


def run_case(run: Run, case: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    video = Path(case["video"]).expanduser()
    res: dict[str, Any] = {"case": case["id"], "video": video.name, "minutes": None}
    t0 = time.time()
    if run.code:  # 배포 서버는 이어 올리기만 받는다(10/31 보안 검토)
        job = _upload_resumable(run, video, case["id"])
    else:
        with video.open("rb") as f:
            r = run.http.post("/jobs", files={"file": (video.name, f)}, data={"match": case["id"]}, timeout=3600)
        job = r.raise_for_status().json()["id"]
    res["job"] = job
    res["upload_sec"] = round(time.time() - t0, 1)
    first: dict[str, float] = {}

    def tick(s: dict[str, Any]) -> None:
        if s["rallies"] and "first_rally" not in first:
            first["first_rally"] = time.time()

    t1 = time.time()
    s = run.wait(job, lambda s: s["status"] == "rallies_ready", tick)
    res["minutes"] = round(s["duration_sec"] / 60, 1)
    res["first_rally_sec"] = round(first.get("first_rally", time.time()) - t1, 1)
    res["find_rallies_sec"] = round(time.time() - t1, 1)
    res["rallies"] = len(s["rallies"])
    res["unread"] = s["unread"]
    res["scan_usd"] = s["spent_usd"]

    truth = _truth(Path(case["truth"]).expanduser(), case.get("truth_offset", 0.0), case.get("truth_until"))
    in_range = [r["point_sec"] for r in s["rallies"] if case.get("truth_until") is None or r["point_sec"] <= case["truth_until"] + SAME_RALLY_SEC]
    res["rally_quality"] = score_rallies(truth, in_range)

    # 정답과 맞는 득점 랠리 중 확신 높은 것부터, 서로 60초 넘게 떨어진 장면을 고른다
    good = sorted((r for r in s["rallies"] if any(abs(r["point_sec"] - t) <= SAME_RALLY_SEC for t in truth)),
                  key=lambda r: -float(r["confidence"] or 0))
    picks: list[dict[str, Any]] = []
    for r in good:
        if len(picks) < case.get("scenes", 2) and all(abs(r["point_sec"] - p["point_sec"]) > 60 for p in picks):
            picks.append(r)
    res["scenes"] = [run_scene(run, job, r["index"], out_dir, case.get("fps", 0)) for r in picks]
    res["total_usd"] = run.status(job)["spent_usd"]
    res["wall_sec"] = round(time.time() - t0, 1)
    return res


def run_scene(run: Run, job: str, index: int, out_dir: Path, fps: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {"index": index}
    t = time.time()
    run.http.post(f"/jobs/{job}/rallies/{index}/prepare")
    run.wait(job, lambda s: s["scenes"].get(str(index), {}).get("status") == "awaiting_tap")
    out["prepare_sec"] = round(time.time() - t, 1)
    t = time.time()
    players = run.http.post(f"/jobs/{job}/rallies/{index}/players?which=score").json()["players"]
    out["players_sec"] = round(time.time() - t, 1)
    n = max(range(len(players)), key=lambda k: (players[k][2] - players[k][0]) * (players[k][3] - players[k][1]))
    marks: dict[str, float] = {}

    def tick(s: dict[str, Any]) -> None:
        step = (s.get("progress") or {}).get("step")
        if step and step not in marks:
            marks[step] = time.time()

    t = time.time()
    run.http.post(f"/jobs/{job}/rallies/{index}/highlight",
                  json={"taps": [{"which": "score", "player": n}], "team": "bench", "fps": fps})
    s = run.wait(job, lambda s: s["scenes"][str(index)]["status"] in ("done", "failed"), tick)
    end = time.time()
    out["highlight_sec"] = round(end - t, 1)
    out["tracking_sec"] = round(marks.get("render", end) - marks.get("tracking", t), 1)
    out["render_sec"] = round(end - marks.get("render", end), 1)
    out["status"] = s["scenes"][str(index)]["status"]
    if run.code:
        return out
    report = qa.check_scene(job, index)
    out.update(score_in_clip=report["score_in_clip"], clip_sec=report["clip_sec"], tracking=report["tracking"])
    out["privacy"] = visible_faces(Path(report["qa_video"]))
    scene_dir = Path(report["qa_video"]).parent
    out["video"] = sharpness(scene_dir / "highlight.mp4")
    (out_dir / f"{job}_{index}_qa_sheet.jpg").write_bytes(Path(report["qa_sheet"]).read_bytes())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="성능 테스트(유료)")
    p.add_argument("cases", type=Path)
    p.add_argument("--only", nargs="*", help="이 케이스만(예: T1)")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--code", default="", help="배포 서버의 사용 코드")
    a = p.parse_args()
    cases = [c for c in json.loads(a.cases.read_text())["cases"] if not a.only or c["id"] in a.only]
    out_dir = Path("data/bench") / datetime.now().strftime("%m%d-%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    run, results = Run(a.base, a.code), []
    for c in cases:
        print("==", c["id"], flush=True)
        try:
            results.append(run_case(run, c, out_dir))
        except Exception as e:  # noqa: BLE001 - 한 케이스가 실패해도 나머지는 잰다
            results.append({"case": c["id"], "error": str(e)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("저장:", out_dir / "results.json")


if __name__ == "__main__":
    main()
