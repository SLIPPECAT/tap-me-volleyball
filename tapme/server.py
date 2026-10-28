"""하이라이트 서버 (FastAPI). 사용자 맥에서 돌린다.

  uv run uvicorn tapme.server:app --host 0.0.0.0 --port 8000

흐름: 영상 올리기 → 랠리 후보(자동) → 득점 장면 고르기 → 탭할 장면 → 나를 탭 + 옵션 → 결과 영상.
오래 걸리는 단계는 백그라운드에서 돌고, 화면은 GET /jobs/{id}로 상태를 본다.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from tapme import jobs, muse, render, score

app = FastAPI(title="tap-me-volleyball")
# 테스트에서 가짜 API 클라이언트로 바꾼다
client_factory: Callable[[], Any] = muse.client_from_env
_step_lock = threading.Lock()
STATIC_DIR = Path(__file__).parent / "static"


class RenderOptions(BaseModel):
    languages: list[str] = Field(default_factory=lambda: ["ko"])
    privacy: str = render.DEFAULT_PRIVACY
    zoom: float = render.DEFAULT_ZOOM

    @field_validator("languages")
    @classmethod
    def _languages(cls, v: list[str]) -> list[str]:
        bad = [lang for lang in v if lang not in score.LANGUAGES]
        if bad:
            raise ValueError(f"자막 언어는 {score.LANGUAGES} 중에서: {bad}")
        return v

    @field_validator("privacy")
    @classmethod
    def _privacy(cls, v: str) -> str:
        if v not in render.PRIVACY_MODES:
            raise ValueError(f"가리기 방식은 {render.PRIVACY_MODES} 중 하나")
        return v

    @field_validator("zoom")
    @classmethod
    def _zoom(cls, v: float) -> float:
        if v not in render.ZOOM_CHOICES:
            raise ValueError(f"확대 배율은 {render.ZOOM_CHOICES} 중 하나")
        return v


class TapRequest(RenderOptions):
    x: float
    y: float
    team: str = Field(min_length=1, max_length=20, description="사용자가 부르는 우리 팀 이름(예: 검정)")


def _job(job_id: str) -> jobs.Job:
    try:
        return jobs.load_job(job_id)
    except KeyError as e:
        raise HTTPException(404, "작업이 없다") from e


def _run(job_id: str, step: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """백그라운드 단계. 한 번에 하나씩 돈다(맥 한 대, 같은 job.json을 두 단계가 덮어쓰지 않게).
    실패하면 job.error에 남겨 화면이 보여 줄 수 있게 한다."""
    with _step_lock:
        job = jobs.load_job(job_id)
        try:
            step(job, *args, **kwargs)
            job.error = None
        except (jobs.BudgetExceeded, jobs.TapMissed) as e:
            job.error = str(e)
        except Exception as e:  # noqa: BLE001 - 서버가 죽지 않고 화면에 알린다
            traceback.print_exc()
            job.error = f"처리 중 오류: {e}"
        job.save()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.post("/jobs")
def upload(file: UploadFile, background: BackgroundTasks) -> dict[str, Any]:
    job = jobs.create_job(file.filename or "video.mp4", file.file)
    background.add_task(_run, job.id, jobs.find_rallies, client_factory())
    return {"id": job.id}


@app.get("/jobs/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    job = _job(job_id)
    return {"id": job.id, "status": job.status, "error": job.error, "spent_usd": round(job.spent_usd, 4),
            "rallies": [{k: r.get(k) for k in ("index", "start_sec", "end_sec", "is_rally", "confidence", "reason",
                                               "uncertain")}
                        for r in job.rallies],
            "scenes": {k: {f: v for f, v in s.items() if f not in ("window5", "tap_frame", "window30", "video")}
                       for k, s in job.scenes.items()}}


@app.post("/jobs/{job_id}/rallies/{index}/prepare")
def prepare(job_id: str, index: int, background: BackgroundTasks) -> dict[str, str]:
    job = _job(job_id)
    if not 0 <= index < len(job.rallies):
        raise HTTPException(404, "랠리가 없다")
    background.add_task(_run, job_id, jobs.prepare_scene, index, client_factory())
    return {"status": "preparing"}


@app.get("/jobs/{job_id}/rallies/{index}/tap-frame")
def tap_frame(job_id: str, index: int) -> FileResponse:
    scene = _job(job_id).scenes.get(str(index))
    if not scene or "tap_frame" not in scene:
        raise HTTPException(404, "아직 준비되지 않았다")
    return FileResponse(scene["tap_frame"], media_type="image/jpeg")


@app.post("/jobs/{job_id}/rallies/{index}/highlight")
def highlight(job_id: str, index: int, req: TapRequest, background: BackgroundTasks) -> dict[str, str]:
    scene = _job(job_id).scenes.get(str(index))
    if not scene or scene.get("status") not in ("awaiting_tap", "done"):
        raise HTTPException(409, "먼저 prepare로 탭할 장면을 만든다")
    background.add_task(_run, job_id, jobs.make_highlight, index, req.x, req.y, req.team, req.languages,
                        req.privacy, req.zoom, client_factory())
    return {"status": "rendering"}


@app.post("/jobs/{job_id}/rallies/{index}/rerender")
def rerender(job_id: str, index: int, req: RenderOptions, background: BackgroundTasks) -> dict[str, str]:
    scene = _job(job_id).scenes.get(str(index))
    if not scene or scene.get("status") != "done":
        raise HTTPException(409, "하이라이트를 먼저 만든다")
    background.add_task(_run, job_id, jobs.rerender, index, req.languages, req.privacy, req.zoom)
    return {"status": "rendering"}


@app.get("/jobs/{job_id}/rallies/{index}/video")
def video(job_id: str, index: int) -> FileResponse:
    scene = _job(job_id).scenes.get(str(index))
    if not scene or "video" not in scene:
        raise HTTPException(404, "결과 영상이 아직 없다")
    return FileResponse(scene["video"], media_type="video/mp4")
