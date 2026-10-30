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

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from tapme import jobs, muse, render, score

app = FastAPI(title="tap-me-volleyball")
# 테스트에서 가짜 API 클라이언트로 바꾼다
client_factory: Callable[[], Any] = muse.client_from_env
_step_lock = threading.Lock()
_SERVER_PATHS = ("sam_clip", "clip_video", "video")  # 서버 안 경로는 화면에 내보내지 않는다
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # 아이콘·홈 화면 설정


class RenderOptions(BaseModel):
    # 자막은 기본으로 끈다(10/29 사용자: 자막이 꼭 필요하진 않음). 언어를 고르면 해설 자막이 켜진다
    languages: list[str] = Field(default_factory=list)
    role: str = Field(default="", description="'나'의 역할(선택): '', score, assist, defense")
    # 결과 영상에서 '나'를 뺀 얼굴 모자이크(기본 켬). Meta로 보내는 클립은 이 값과 상관없이 늘 흐림본이다
    face_mosaic: bool = True
    privacy: str = render.DEFAULT_PRIVACY
    zoom: float = render.DEFAULT_ZOOM

    @field_validator("languages")
    @classmethod
    def _languages(cls, v: list[str]) -> list[str]:
        bad = [lang for lang in v if lang not in score.LANGUAGES]
        if bad:
            raise ValueError(f"자막 언어는 {score.LANGUAGES} 중에서: {bad}")
        return v

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        if v not in score.ROLE_TEXT:
            raise ValueError(f"역할은 {tuple(score.ROLE_TEXT)} 중 하나")
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


class ReelRequest(BaseModel):
    order: list[int] = Field(min_length=1, max_length=10, description="모을 장면(랠리 번호)을 보여 줄 순서대로")


class Tap(BaseModel):
    """한 장면에서 나를 고른 것: 선수 카드 번호(player) 또는 사진을 탭한 위치(x, y)."""

    which: str
    player: int | None = Field(default=None, ge=0)
    x: float | None = None
    y: float | None = None

    @model_validator(mode="after")
    def _one(self) -> Tap:
        if self.which not in jobs.TAP_KINDS:
            raise ValueError(f"장면은 {jobs.TAP_KINDS} 중 하나")
        if self.player is None and (self.x is None or self.y is None):
            raise ValueError("선수 카드를 고르거나 사진을 탭해야 한다")
        return self

    def as_tuple(self) -> tuple:
        return (self.which, self.player) if self.player is not None else (self.which, self.x, self.y)


class TapRequest(RenderOptions):
    """내가 보이는 장면(득점 순간·1.5/3/4.5초 전·구간 시작) 중 하나 이상에서 나를 고른다."""

    taps: list[Tap] = Field(min_length=1, max_length=len(jobs.TAP_KINDS))
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
def upload(file: UploadFile, background: BackgroundTasks, match: str = Form("")) -> dict[str, Any]:
    job = jobs.create_job(file.filename or "video.mp4", file.file, match)
    background.add_task(_run, job.id, jobs.find_rallies, client_factory())
    return {"id": job.id}


@app.get("/jobs/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    job = _job(job_id)
    return {"id": job.id, "status": job.status, "error": job.error, "spent_usd": round(job.spent_usd, 4),
            "progress": job.progress, "duration_sec": job.duration_sec, "estimate_usd": job.estimate_usd,
            "unread": job.unread, "match_name": job.match_name,
            "rallies": [{k: r.get(k) for k in ("index", "start_sec", "end_sec", "point_sec", "is_rally", "confidence",
                                               "reason", "uncertain")}
                        for r in job.rallies],
            "scenes": {k: {f: v for f, v in s.items() if f not in _SERVER_PATHS}
                       for k, s in job.scenes.items()},
            "reel": {k: v for k, v in job.reel.items() if k != "video"}}


def _rally_media(job_id: str, index: int) -> tuple[Path, Path]:
    job = _job(job_id)
    if not 0 <= index < len(job.rallies):
        raise HTTPException(404, "랠리가 없다")
    return jobs.make_preview(job, index)


@app.get("/jobs/{job_id}/rallies/{index}/thumb")
def rally_thumb(job_id: str, index: int) -> FileResponse:
    return FileResponse(_rally_media(job_id, index)[0], media_type="image/jpeg")


@app.get("/jobs/{job_id}/rallies/{index}/preview")
def rally_preview(job_id: str, index: int) -> FileResponse:
    return FileResponse(_rally_media(job_id, index)[1], media_type="video/mp4")


@app.post("/jobs/{job_id}/rallies/{index}/prepare")
def prepare(job_id: str, index: int, background: BackgroundTasks) -> dict[str, str]:
    job = _job(job_id)
    if not 0 <= index < len(job.rallies):
        raise HTTPException(404, "랠리가 없다")
    scene = job.scenes.get(str(index))
    failed_before = scene is not None and scene.get("status") == "preparing" and job.error
    if scene is not None and not failed_before:
        # 다시 준비하면 Muse 비용이 또 들고 만든 결과가 지워진다. 앞서 준비하다 실패했을 때만 다시 한다
        raise HTTPException(409, "이미 준비했거나 준비 중인 장면이다")
    background.add_task(_run, job_id, jobs.prepare_scene, index, client_factory())
    return {"status": "preparing"}


@app.get("/jobs/{job_id}/rallies/{index}/tap-frame")
def tap_frame(job_id: str, index: int, which: str = "score") -> FileResponse:
    scene = _job(job_id).scenes.get(str(index))
    if which not in jobs.TAP_KINDS:
        raise HTTPException(422, f"which는 {jobs.TAP_KINDS} 중 하나")
    if not scene or "tap_frames" not in scene:
        raise HTTPException(404, "아직 준비되지 않았다")
    return FileResponse(Path(scene["sam_clip"]).parent / f"tap_{which}.jpg", media_type="image/jpeg")


@app.post("/jobs/{job_id}/rallies/{index}/players")
def find_players(job_id: str, index: int, which: str = "score") -> dict[str, Any]:
    """그 장면을 처음 열 때 선수를 찾아 카드를 만든다(약 $0.0025, 한 번 만들면 다시 씀). 몇 초 걸린다."""
    if which not in jobs.TAP_KINDS:
        raise HTTPException(422, f"which는 {jobs.TAP_KINDS} 중 하나")
    with _step_lock:
        job = _job(job_id)
        scene = job.scenes.get(str(index))
        if not scene or "tap_frames" not in scene:
            raise HTTPException(404, "아직 준비되지 않았다")
        try:
            players = jobs.player_cards(job, index, which, client_factory())
        except jobs.BudgetExceeded as e:
            raise HTTPException(409, str(e)) from e
    return {"which": which, "players": players}


@app.get("/jobs/{job_id}/rallies/{index}/player")
def player_card(job_id: str, index: int, which: str = "score", n: int = 0) -> FileResponse:
    scene = _job(job_id).scenes.get(str(index))
    if which not in jobs.TAP_KINDS:
        raise HTTPException(422, f"which는 {jobs.TAP_KINDS} 중 하나")
    if not scene or not 0 <= n < len(scene.get("players", {}).get(which, [])):
        raise HTTPException(404, "그 선수 카드가 없다")
    return FileResponse(Path(scene["sam_clip"]).parent / f"player_{which}_{n:02d}.jpg", media_type="image/jpeg")


@app.post("/jobs/{job_id}/rallies/{index}/highlight")
def highlight(job_id: str, index: int, req: TapRequest, background: BackgroundTasks) -> dict[str, str]:
    scene = _job(job_id).scenes.get(str(index))
    if not scene or scene.get("status") not in ("awaiting_tap", "done"):
        raise HTTPException(409, "먼저 prepare로 탭할 장면을 만든다")
    background.add_task(_run, job_id, jobs.make_highlight, index, [t.as_tuple() for t in req.taps], req.team, req.languages, req.role,
                        req.privacy, req.zoom, client_factory(), req.face_mosaic)
    return {"status": "rendering"}


@app.post("/jobs/{job_id}/rallies/{index}/rerender")
def rerender(job_id: str, index: int, req: RenderOptions, background: BackgroundTasks) -> dict[str, str]:
    scene = _job(job_id).scenes.get(str(index))
    if not scene or scene.get("status") != "done":
        raise HTTPException(409, "하이라이트를 먼저 만든다")
    # 자막을 처음 켜거나 역할을 바꾸면 해설(Muse), '몸 실루엣'을 처음 고르면 다른 선수 추적(SAM)이 한 번 필요하다
    background.add_task(_run, job_id, jobs.rerender, index, req.languages, req.role, req.privacy, req.zoom,
                        client_factory(), req.face_mosaic)
    return {"status": "rendering"}


@app.get("/jobs/{job_id}/rallies/{index}/video")
def video(job_id: str, index: int) -> FileResponse:
    scene = _job(job_id).scenes.get(str(index))
    if not scene or "video" not in scene:
        raise HTTPException(404, "결과 영상이 아직 없다")
    return FileResponse(scene["video"], media_type="video/mp4")


@app.post("/jobs/{job_id}/reel")
def reel(job_id: str, req: ReelRequest, background: BackgroundTasks) -> dict[str, str]:
    job = _job(job_id)
    if any(job.scenes.get(str(i), {}).get("status") != "done" for i in req.order):
        raise HTTPException(409, "완성된 장면만 모을 수 있다")
    background.add_task(_run, job_id, jobs.make_reel, req.order)
    return {"status": "rendering"}


@app.get("/jobs/{job_id}/reel/video")
def reel_video(job_id: str) -> FileResponse:
    reel = _job(job_id).reel
    if reel.get("status") != "done":
        raise HTTPException(404, "모음 영상이 아직 없다")
    return FileResponse(reel["video"], media_type="video/mp4")
