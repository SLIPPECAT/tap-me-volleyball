"""하이라이트 서버 (FastAPI). 사용자 맥에서 돌린다.

  uv run uvicorn tapme.server:app --host 0.0.0.0 --port 8000

흐름: 영상 올리기 → 랠리 후보(자동) → 득점 장면 고르기 → 탭할 장면 → 나를 탭 + 옵션 → 결과 영상.
오래 걸리는 단계는 백그라운드에서 돌고, 화면은 GET /jobs/{id}로 상태를 본다.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from tapme import access, jobs, muse, render, scan, score

# 끝난 작업 영상을 이 시간 뒤에 지운다. 배포 서버에서만 켠다(이 맥의 시험 작업은 지우지 않게)
DELETE_AFTER_HOURS = float(os.environ.get("TAPME_DELETE_AFTER_HOURS", "0"))


def _sweep() -> None:
    while True:
        try:
            jobs.delete_old_jobs(DELETE_AFTER_HOURS)
        except Exception:  # noqa: BLE001 - 지우기가 실패해도 서버는 계속
            traceback.print_exc()
        time.sleep(600)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if DELETE_AFTER_HOURS > 0:
        threading.Thread(target=_sweep, daemon=True).start()
    yield


app = FastAPI(title="tap-me-volleyball", lifespan=_lifespan)
# 테스트에서 가짜 API 클라이언트로 바꾼다
client_factory: Callable[[], Any] = muse.client_from_env
_step_lock = threading.Lock()
_SERVER_PATHS = ("sam_clip", "clip_video", "video")  # 서버 안 경로는 화면에 내보내지 않는다
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # 아이콘·홈 화면 설정


@app.middleware("http")
async def _isolate_lab(request, call_next):
    """브라우저 처리 시험 페이지(/static/lab/)만 교차 출처 격리: 이래야 WASM이 여러 스레드를 쓴다."""
    response = await call_next(request)
    if request.url.path.startswith("/static/lab/"):
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


class RenderOptions(BaseModel):
    # 자막은 기본으로 끈다(10/29 사용자: 자막이 꼭 필요하진 않음). 언어를 고르면 해설 자막이 켜진다
    languages: list[str] = Field(default_factory=list)
    role: str = Field(default="", description="'나'의 역할(선택): '', score, assist, defense")
    # 결과 영상에서 '나'를 뺀 얼굴 모자이크(기본 켬). Meta로 보내는 클립은 이 값과 상관없이 늘 흐림본이다
    face_mosaic: bool = True
    privacy: str = render.DEFAULT_PRIVACY
    zoom: float = render.DEFAULT_ZOOM
    fps: int = Field(default=0, description="결과 영상 fps: 0이면 원본 그대로, 30이면 30fps까지(60fps 원본은 렌더가 약 절반)")

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

    @field_validator("fps")
    @classmethod
    def _fps(cls, v: int) -> int:
        if v not in (0, 30):
            raise ValueError("fps는 0(원본) 또는 30")
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


def _run_scan(job_id: str, client: Any) -> None:
    """랠리 찾기는 장면 단계와 따로 돈다. 1시간 영상은 오래 걸려, 찾은 랠리부터 골라 장면을 만들 수 있게(10/30).
    같은 job.json은 Job.save가 서로 바꾼 항목만 합쳐 쓴다. 여러 사용자가 올리면 온 순서대로(jobs.SCAN_TURNS).
    못 찾고 끝나면 올리기 횟수를 돌려준다."""
    with jobs.SCAN_TURNS.turn(job_id):
        job = jobs.load_job(job_id)
        try:
            jobs.find_rallies(job, client)
            job.error = None
        except (jobs.BudgetExceeded, ValueError) as e:
            job.error = str(e)
        except Exception as e:  # noqa: BLE001 - 서버가 죽지 않고 화면에 알린다
            traceback.print_exc()
            job.error = f"처리 중 오류: {e}"
        if job.error:
            access.refund(job.code, "uploads")
        job.save()


def _run_highlight(job_id: str, index: int, *args: Any) -> None:
    """하이라이트는 장면 준비 줄(_step_lock)과 따로 돈다. 여러 장면을 예약하면 추적은 겹치고 렌더만
    jobs.RENDER_LOCK으로 하나씩. 실패하면 그 장면을 'failed'로 남겨 화면이 '만드는 중'에 멈추지 않게 한다."""
    job = jobs.load_job(job_id)
    try:
        jobs.make_highlight(job, index, *args)
        job.error = None
    except (jobs.BudgetExceeded, jobs.TapMissed) as e:
        job.error = str(e)
    except Exception as e:  # noqa: BLE001 - 서버가 죽지 않고 화면에 알린다
        traceback.print_exc()
        job.error = f"처리 중 오류: {e}"
    scene = job.scenes.get(str(index), {})
    if job.error and scene.get("status") in ("queued", "rendering"):
        scene["status"] = "failed"
        access.refund(job.code, "highlights")
    job.save()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


class AccessRequest(BaseModel):
    code: str = Field(default="", max_length=20)


def _closed_reason() -> str | None:
    """새 영상을 받을 수 없는 이유. 랠리 찾기 대기가 꽉 찼거나 전체 비용이 30분 영상 하나도 못 훑을 만큼 남았을 때."""
    if jobs.SCAN_TURNS.waiting() >= jobs.MAX_WAITING_SCANS:
        return "지금 사용자가 많아요. 잠시 뒤 다시 와 주세요"
    if jobs.total_spent() + scan.estimate_usd(jobs.MAX_SCAN_SEC) > jobs.MAX_USD_TOTAL:
        return "이번 행사에 준비한 AI 비용을 다 써서 더 받을 수 없어요"
    return None


@app.post("/access")
def check_access(req: AccessRequest) -> dict[str, Any]:
    """올리기 전에 묻는다: 코드가 맞는지, 남은 횟수, 지금 받을 수 있는지(2GB를 올린 뒤 거절하지 않게)."""
    out: dict[str, Any] = {"required": access.REQUIRED, "waiting": jobs.SCAN_TURNS.waiting(), "closed": _closed_reason()}
    if access.REQUIRED and req.code:
        try:
            out["left"] = access.left(req.code)
        except access.AccessDenied as e:
            out["denied"] = str(e)
    return out


@app.get("/admin/summary")
def admin_summary(x_admin_token: str = Header("")) -> dict[str, Any]:
    """운영자용: 코드별 사용 횟수·전체 비용·대기. TAPME_ADMIN_TOKEN을 둔 서버에서만."""
    token = os.environ.get("TAPME_ADMIN_TOKEN", "")
    if not token or x_admin_token != token:
        raise HTTPException(404)
    return {"spent_usd": round(jobs.total_spent(), 3), "max_usd": jobs.MAX_USD_TOTAL, "limits": access.LIMITS,
            "waiting_scans": jobs.SCAN_TURNS.waiting(), "waiting_renders": jobs.RENDER_TURNS.waiting(),
            "codes": access._load()}


@app.post("/jobs")
def upload(file: UploadFile, background: BackgroundTasks, match: str = Form(""),
           start: float = Form(0.0), end: float = Form(0.0), code: str = Form("")) -> dict[str, Any]:
    """start·end: 훑을 구간(초). 30분(jobs.MAX_SCAN_SEC)보다 긴 영상은 화면에서 고른다. code: 사용 코드(배포 서버)."""
    if reason := _closed_reason():
        raise HTTPException(503, reason)
    try:
        access.use(code, "uploads")
    except access.AccessDenied as e:
        raise HTTPException(403, str(e)) from e
    job = jobs.create_job(file.filename or "video.mp4", file.file, match, start, end, access.normalize(code))
    background.add_task(_run_scan, job.id, client_factory())
    return {"id": job.id}


@app.get("/jobs/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    job = _job(job_id)
    return {"id": job.id, "status": job.status, "error": job.error, "spent_usd": round(job.spent_usd, 4),
            "progress": job.progress, "duration_sec": job.duration_sec, "estimate_usd": job.estimate_usd,
            "unread": job.unread, "match_name": job.match_name, "scan": job.scan, "scan_range": job.scan_range,
            "queue": jobs.SCAN_TURNS.position(job.id), "left": _left(job),
            "render_turns": {k: p for k in job.scenes if (p := jobs.RENDER_TURNS.position(f"{job.id}/{k}"))},
            "rallies": [{k: r.get(k) for k in ("index", "start_sec", "end_sec", "point_sec", "is_rally", "confidence",
                                               "reason", "uncertain")}
                        for r in job.rallies],
            "scenes": {k: {f: v for f, v in s.items() if f not in _SERVER_PATHS}
                       for k, s in job.scenes.items()},
            "reel": {k: v for k, v in job.reel.items() if k != "video"}}


def _left(job: jobs.Job) -> dict[str, int] | None:
    try:
        return access.left(job.code) if access.REQUIRED and job.code else None
    except access.AccessDenied:
        return None


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
def prepare(job_id: str, index: int, background: BackgroundTasks, ahead: bool = False) -> dict[str, str]:
    """ahead=true: 사용자가 미리보기를 열었을 때 뒤에서 미리 준비한다(고르면 탭 화면이 바로 뜨게). 비용이 없을 때
    (훑어보기가 득점 순간을 이미 찾은 랠리)와 다른 단계가 돌지 않을 때만. 아니면 아무것도 하지 않는다."""
    job = _job(job_id)
    if not 0 <= index < len(job.rallies):
        raise HTTPException(404, "랠리가 없다")
    scene = job.scenes.get(str(index))
    if ahead:
        if scene is not None or job.rallies[index].get("point_sec") is None or _step_lock.locked():
            return {"status": "skipped"}
        background.add_task(_run, job_id, jobs.prepare_scene, index, client_factory())
        return {"status": "preparing"}
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
    job = _job(job_id)
    scene = job.scenes.get(str(index))
    if not scene or scene.get("status") not in ("awaiting_tap", "done", "failed"):
        raise HTTPException(409, "먼저 prepare로 탭할 장면을 만든다(이미 만드는 중이면 기다린다)")
    active = sum(sc.get("status") in ("queued", "rendering") for sc in job.scenes.values())
    if active >= jobs.MAX_ACTIVE_SCENES:
        raise HTTPException(429, f"한 번에 {jobs.MAX_ACTIVE_SCENES}개까지 만들 수 있어요. 앞의 장면이 끝나면 다시 예약해 주세요")
    try:
        access.use(job.code, "highlights")  # 실패하면 _run_highlight가 돌려준다
    except access.AccessDenied as e:
        raise HTTPException(403, str(e)) from e
    scene["status"] = "queued"  # 예약됨. 여러 장면을 이어서 예약할 수 있다(일괄 만들기)
    job.save()
    background.add_task(_run_highlight, job_id, index, [t.as_tuple() for t in req.taps], req.team, req.languages, req.role,
                        req.privacy, req.zoom, client_factory(), req.face_mosaic, req.fps)
    return {"status": "queued"}


@app.post("/jobs/{job_id}/rallies/{index}/rerender")
def rerender(job_id: str, index: int, req: RenderOptions, background: BackgroundTasks) -> dict[str, str]:
    scene = _job(job_id).scenes.get(str(index))
    if not scene or scene.get("status") != "done":
        raise HTTPException(409, "하이라이트를 먼저 만든다")
    # 자막을 처음 켜거나 역할을 바꾸면 해설(Muse), '몸 실루엣'을 처음 고르면 다른 선수 추적(SAM)이 한 번 필요하다
    background.add_task(_run, job_id, jobs.rerender, index, req.languages, req.role, req.privacy, req.zoom,
                        client_factory(), req.face_mosaic, req.fps)
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
