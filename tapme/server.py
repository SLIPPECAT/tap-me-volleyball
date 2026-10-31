"""하이라이트 서버 (FastAPI). 사용자 맥에서 돌린다.

  uv run uvicorn tapme.server:app --host 0.0.0.0 --port 8000

흐름: 영상 올리기 → 랠리 후보(자동) → 득점 장면 고르기 → 탭할 장면 → 나를 탭 + 옵션 → 결과 영상.
오래 걸리는 단계는 백그라운드에서 돌고, 화면은 GET /jobs/{id}로 상태를 본다.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from tapme import access, jobs, muse, push, render, scan, score

# 매일 새벽 4시(jobs.DAY_START_HOUR)에 지난날 영상을 모두 지운다. 배포 서버에서만 켠다(이 맥의 시험 작업은 지우지 않게).
# 서버가 꺼져 있다가 켜지면 바로 한 번 지운다
DAILY_WIPE = os.environ.get("TAPME_DAILY_WIPE") == "1"


def _sweep() -> None:
    while True:
        try:
            jobs.delete_previous_days()
        except Exception:  # noqa: BLE001 - 지우기가 실패해도 서버는 계속
            traceback.print_exc()
        time.sleep(600)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if DAILY_WIPE:
        threading.Thread(target=_sweep, daemon=True).start()
    yield


app = FastAPI(title="tap-me-volleyball", lifespan=_lifespan)
# 테스트에서 가짜 API 클라이언트로 바꾼다
client_factory: Callable[[], Any] = muse.client_from_env
_step_lock = threading.Lock()
_SERVER_PATHS = ("sam_clip", "clip_video", "video")  # 서버 안 경로는 화면에 내보내지 않는다
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # 아이콘·홈 화면 설정


# 쓸 때만 켜는 서버(10/31): 마지막 요청 뒤 오래 조용하고 도는 작업이 없으면 VM이 스스로 꺼진다(deploy/idle-stop.sh가 /idle을 본다).
# 화면을 열어 두면 2초마다 상태를 물어 켜진 채로 남는다
_last_request = time.time()
_busy = [0]  # 백그라운드에서 도는 단계 수(랠리 찾기·장면 준비·하이라이트, 대기 중 포함)
_busy_lock = threading.Lock()


class _Working:
    def __enter__(self) -> None:
        with _busy_lock:
            _busy[0] += 1

    def __exit__(self, *exc: object) -> None:
        with _busy_lock:
            _busy[0] -= 1


# IP별 요청 속도 제한(10/31 보안 검토: 무료 요청을 1초에 수백 번 보내도 막지 못했다). 서버가 하나라 메모리로 센다.
# 배포 서버에서만 켠다(TAPME_RATE_LIMIT=1). (몇 번, 몇 초 동안). 체육관 와이파이·통신사 LTE처럼 여러 사람이 한 IP를
# 나눠 쓰므로 사람은 막지 않고 폭주(초당 10회 넘게)만 막을 만큼 넉넉히
RATE_LIMIT = os.environ.get("TAPME_RATE_LIMIT") == "1"
RATE_LIMITS = {"general": (600, 60), "light": (30, 60), "heavy": (60, 3600)}
_LIGHT_POSTS = ("/feedback", "/speedtest", "/push/test", "/access", "/access/claim", "/uploads", "/jobs/browser")
_buckets: dict[tuple[str, str], tuple[float, float]] = {}
_bucket_lock = threading.Lock()


def _rate_group(method: str, path: str, query: str) -> str:
    """light: 의견·속도 재기·알림·코드 확인처럼 가볍지만 반복될 수 있는 요청. heavy: 다시 렌더·모음·구간 올리기·장면 준비(CPU)."""
    if method != "POST":
        return "general"
    if path in _LIGHT_POSTS or path.endswith("/push"):
        return "light"
    last = path.rsplit("/", 1)[-1]
    if last in ("rerender", "reel", "segment") or (last == "prepare" and "ahead=true" not in query):
        return "heavy"
    return "general"


def _rate_wait(ip: str, group: str, now: float) -> float:
    """토큰 통: 쓸 수 있으면 0, 아니면 기다릴 초."""
    cap, per = RATE_LIMITS[group]
    with _bucket_lock:
        if len(_buckets) > 20000:  # 오래 안 쓴 IP는 잊는다
            for k in [k for k, (_, last) in _buckets.items() if now - last > 3600]:
                del _buckets[k]
        tokens, last = _buckets.get((ip, group), (float(cap), now))
        tokens = min(float(cap), tokens + (now - last) * cap / per)
        if tokens < 1:
            _buckets[(ip, group)] = (tokens, now)
            return (1 - tokens) * per / cap
        _buckets[(ip, group)] = (tokens - 1, now)
        return 0.0


@app.middleware("http")
async def _limit_rate(request, call_next):
    ip = _client_ip(request)
    if RATE_LIMIT and ip not in ("127.0.0.1", "::1"):  # 서버 안(꺼짐 확인 등)은 세지 않는다
        wait = _rate_wait(ip, _rate_group(request.method, request.url.path, request.url.query), time.monotonic())
        if wait > 0:
            return JSONResponse({"detail": "요청이 너무 많아요. 잠시 뒤 다시 해 주세요"}, status_code=429,
                                headers={"Retry-After": str(int(wait) + 1)})
    return await call_next(request)


@app.middleware("http")
async def _note_request(request, call_next):
    global _last_request
    if request.url.path != "/idle":
        _last_request = time.time()
    return await call_next(request)


@app.get("/idle")
def idle() -> dict[str, Any]:
    return {"idle_sec": round(time.time() - _last_request), "busy": _busy[0] > 0}


@app.middleware("http")
async def _isolate(request, call_next):
    """교차 출처 격리: 이래야 브라우저 블러(static/browser-blur.js)의 WASM이 여러 스레드를 쓴다. 처음엔 시험 페이지만
    켰다가 실제 화면도 기기에서 흐리게 되어(10/31) 모든 응답에 켠다. 외부 스크립트(jsdelivr)는 CORS·CORP를 보내 그대로 된다."""
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    # 기본 보안 헤더(10/31 보안 검토): 다른 사이트가 이 화면을 몰래 띄워 누르게 하지 못하게, 파일 종류를 추측하지 않게
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # 화면 파일은 쓸 때마다 바뀌었는지 묻게 한다(바뀌지 않았으면 304라 다시 받지 않음). 배포 뒤에도 예전 app.css를
    # 캐시에서 써 새 화면이 깨져 보였다(10/31)
    if request.url.path in ("/", "/sw.js") or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
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
    with _Working(), _step_lock:
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
    with _Working(), jobs.SCAN_TURNS.turn(job_id):
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
        if job.status == "rallies_ready":
            _notify_rallies(job)


def _notify_rallies(job: jobs.Job) -> None:
    """랠리를 다 찾았다고 알린다(다음 할 일: 랠리를 골라 나를 탭하기)."""
    push.notify(job.id, "rallies" if job.rallies else "no_rallies", f"/#job={job.id}", n=len(job.rallies))


def _run_highlight(job_id: str, index: int, *args: Any) -> None:
    """하이라이트는 장면 준비 줄(_step_lock)과 따로 돈다. 여러 장면을 예약하면 추적은 겹치고 렌더만
    jobs.RENDER_TURNS로 하나씩. 실패하면 그 장면을 'failed'로 남겨 화면이 '만드는 중'에 멈추지 않게 한다."""
    with _Working():
        _make_highlight(job_id, index, *args)


def _make_highlight(job_id: str, index: int, *args: Any) -> None:
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
    # 완성·실패를 알린다. 여러 장면을 예약했으면 남은 수도(다음 할 일: 열어서 저장·공유)
    left = sum(sc.get("status") in ("queued", "rendering") for k, sc in job.scenes.items() if k != str(index))
    if scene.get("status") in ("done", "failed"):
        push.notify(job_id, scene["status"], f"/#job={job_id}&rally={index}", left=left)


@app.get("/sw.js")
def service_worker() -> FileResponse:
    """완료 알림을 받는 서비스 워커. 화면 전체(/)를 맡아야 해 루트에서 준다."""
    return FileResponse(STATIC_DIR / "sw.js", media_type="text/javascript")


@app.get("/push/key")
def push_key() -> dict[str, str | None]:
    return {"key": push.public_key() or None}


class PushRequest(BaseModel):
    subscription: dict[str, Any]
    lang: str = Field(default="ko", max_length=5)


@app.post("/jobs/{job_id}/push")
def subscribe_push(job_id: str, req: PushRequest) -> dict[str, str]:
    """이 작업이 끝나면(랠리 찾기·하이라이트) 이 기기에 알린다."""
    _job(job_id)
    if not push.enabled():
        raise HTTPException(404, "알림을 쓰지 않는 서버다")
    if not push.allowed_endpoint(str(req.subscription.get("endpoint", ""))):
        raise HTTPException(422, "구독 주소가 맞지 않다")
    push.subscribe(job_id, req.subscription, req.lang)
    return {"status": "ok"}


@app.post("/push/test")
def push_test(req: PushRequest) -> dict[str, str]:
    """알림 시험: 보낸 기기에만 바로 하나(작업 없이). 그 기기 자신의 구독 주소로만 가서 다른 사람에게는 못 보낸다."""
    if not push.enabled() or not push.allowed_endpoint(str(req.subscription.get("endpoint", ""))):
        raise HTTPException(404, "알림을 쓰지 않는 서버다")
    try:
        push.send_test(req.subscription, req.lang)
    except Exception as e:  # 푸시 서비스 오류를 화면에 그대로
        raise HTTPException(502, f"알림을 보내지 못했다: {e}") from e
    return {"status": "sent"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


class AccessRequest(BaseModel):
    code: str = Field(default="", max_length=20)


def _closed_reason(code: str = "") -> str | None:
    """새 영상을 받을 수 없는 이유. 랠리 찾기 대기가 꽉 찼거나 전체 비용이 30분 영상 하나도 못 훑을 만큼 남았을 때.
    하루 올리기 한도는 심사용 코드에는 걸지 않는다(심사위원이 '오늘은 다 찼어요'를 보지 않게)."""
    if jobs.SCAN_TURNS.waiting() >= jobs.MAX_WAITING_SCANS:
        return "지금 사용자가 많아요. 잠시 뒤 다시 와 주세요"
    if jobs.MAX_UPLOADS_PER_DAY and jobs.uploads_today() >= jobs.MAX_UPLOADS_PER_DAY and not access.is_judge(code):
        return f"오늘 받을 수 있는 영상 {jobs.MAX_UPLOADS_PER_DAY}편이 다 찼어요. 내일 새벽 {jobs.DAY_START_HOUR}시 뒤에 다시 와 주세요"
    if jobs.MIN_FREE_GB and jobs.free_gb() < jobs.MIN_FREE_GB:
        return "지금 사용자가 많아요. 잠시 뒤 다시 와 주세요"
    if jobs.total_spent() + scan.estimate_usd(jobs.MAX_SCAN_SEC) > jobs.MAX_USD_TOTAL:
        return "이번 행사에 준비한 AI 비용을 다 써서 더 받을 수 없어요"
    return None


# 화면이 보여 줄 예상 시간(10/31 사용자: 대기 시간·언제 확인할 수 있는지가 관건). 클라우드 실측(BENCH.md, n2-highcpu-16·
# 스레드 8): 서버 블러 1분 조각 111초(50초 간격) → 영상 1초당 약 2.2초, 하이라이트 하나(준비+추적+렌더) 약 10분
EST_SERVER_SEC_PER_VIDEO_SEC = float(os.environ.get("TAPME_EST_SCAN_RATE", "2.2"))
EST_HIGHLIGHT_SEC = float(os.environ.get("TAPME_EST_HIGHLIGHT_SEC", "600"))


@app.post("/speedtest")
async def speedtest(request: Request) -> dict[str, int]:
    """시작 전에 업로드 속도를 잰다(화면이 2MB를 보내 걸린 시간을 잰다). 받은 것은 버린다."""
    size = 0
    async for block in request.stream():
        size += len(block)
        if size > 4 * 1024 * 1024:
            raise HTTPException(413)
    return {"bytes": size}


@app.post("/access")
def check_access(req: AccessRequest) -> dict[str, Any]:
    """올리기 전에 묻는다: 코드가 맞는지, 남은 횟수, 지금 받을 수 있는지(2GB를 올린 뒤 거절하지 않게)."""
    out: dict[str, Any] = {"required": access.REQUIRED, "waiting": jobs.SCAN_TURNS.waiting(), "closed": _closed_reason(req.code),
                           "wipe_hour": jobs.DAY_START_HOUR if DAILY_WIPE else None,
                           "est": {"scan_rate": EST_SERVER_SEC_PER_VIDEO_SEC, "highlight_sec": EST_HIGHLIGHT_SEC}}
    if access.REQUIRED and req.code:
        try:
            out["left"] = access.left(req.code)
        except access.AccessDenied as e:
            out["denied"] = str(e)
    return out


class ClaimRequest(BaseModel):
    invite: str = Field(max_length=40)


def _client_ip(request: Request) -> str:
    """배포 서버는 Caddy 뒤에 있어 Caddy가 붙인 X-Forwarded-For의 첫 주소. 다른 곳에서 온 헤더는 믿지 않는다."""
    host = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() if host == "127.0.0.1" and forwarded else host


@app.post("/access/claim")
def claim_code(req: ClaimRequest, request: Request) -> dict[str, str]:
    """초대 링크(배구구 배너·심사용)로 들어온 기기에 코드를 하나 만들어 준다."""
    try:
        code = access.claim(req.invite, _client_ip(request))
    except access.AccessDenied as e:
        raise HTTPException(403, str(e)) from e
    return {"code": f"{code[:4]}-{code[4:]}"}


class FeedbackRequest(BaseModel):
    rating: str = Field(default="", pattern="^(|good|bad)$")
    text: str = Field(default="", max_length=2000)
    contact: str = Field(default="", max_length=100)
    job: str = Field(default="", max_length=20)
    rally: int | None = None
    lang: str = Field(default="", max_length=5)


@app.post("/feedback")
def send_feedback(req: FeedbackRequest) -> dict[str, str]:
    """의견 남기기: 좋아요/아쉬워요·글·연락처(선택). 어떤 작업·장면이었는지 붙여 고치기 쉽게. 운영 현황에서 본다."""
    if not req.rating and not req.text.strip():
        raise HTTPException(422, "좋아요/아쉬워요를 고르거나 내용을 적어 주세요")
    entry = req.model_dump()
    if req.job:
        try:
            job = jobs.load_job(req.job)
            entry.update(mode=job.mode, code=job.code[:4], scene=job.scenes.get(str(req.rally), {}).get("status"))
        except KeyError:
            entry["job"] = ""
    jobs.add_feedback(entry)
    return {"status": "ok"}


@app.get("/admin/summary")
def admin_summary(x_admin_token: str = Header("")) -> dict[str, Any]:
    """운영자용: 코드별 사용 횟수·전체 비용·대기. TAPME_ADMIN_TOKEN을 둔 서버에서만."""
    token = os.environ.get("TAPME_ADMIN_TOKEN", "")
    if not token or x_admin_token != token:
        raise HTTPException(404)
    return {"spent_usd": round(jobs.total_spent(), 3), "max_usd": jobs.MAX_USD_TOTAL, "limits": access.LIMITS,
            "waiting_scans": jobs.SCAN_TURNS.waiting(), "waiting_renders": jobs.RENDER_TURNS.waiting(),
            "codes": access._load(),
            "invites": {k: {f: e.get(f) for f in ("word", "issued", "max")} for k, e in access._load_invites().items()},
            "feedback": jobs.feedback()}


class BrowserJobRequest(BaseModel):
    filename: str = Field(default="video.mp4", max_length=200)
    duration: float = Field(gt=0, le=6 * 3600)
    start: float = Field(default=0.0, ge=0)
    end: float = Field(default=0.0, ge=0)
    match: str = Field(default="", max_length=40)
    code: str = Field(default="", max_length=20)
    size: int = Field(default=0, ge=0)


MAX_CHUNK_BYTES = 80 * 1024 * 1024  # 1분 흐림본(2fps) 조각은 수십 MB 안쪽
MAX_SEGMENT_BYTES = 400 * 1024 * 1024  # 랠리 구간 원본(키프레임 간격만큼 넉넉, 4K도 들어가게)


@app.post("/jobs/browser")
def create_browser_job(req: BrowserJobRequest) -> dict[str, Any]:
    """브라우저 방식(10/31, H-45): 원본은 올리지 않고 길이·구간만 받아 작업을 만든다. 기기는 돌려준 계획대로
    1분 흐림본 조각을 /jobs/{id}/chunks/{i}로 올린다. 올리기 횟수·하루 한도·비용 상한은 원본 올리기와 같다."""
    if reason := _closed_reason(req.code):
        raise HTTPException(503, reason)
    try:
        access.use(req.code, "uploads")
    except access.AccessDenied as e:
        raise HTTPException(403, str(e)) from e
    try:
        job, plan = jobs.create_browser_job(req.filename, req.duration, req.start, req.end, req.match,
                                            access.normalize(req.code), req.size)
    except (ValueError, jobs.BudgetExceeded) as e:
        access.refund(req.code, "uploads")
        raise HTTPException(422, str(e)) from e
    jobs.count_upload()
    return {"id": job.id, "chunks": plan}


def _run_chunk(job_id: str, index: int, clip: Path, client: Any) -> None:
    with _Working():
        try:
            if jobs.add_browser_chunk(job_id, index, clip, client):  # 마지막 조각이었다
                _notify_rallies(jobs.load_job(job_id))
        except Exception as e:  # noqa: BLE001 - 서버가 죽지 않고 화면에 알린다
            traceback.print_exc()
            job = jobs.load_job(job_id)
            job.error = f"처리 중 오류: {e}"
            job.save()


@app.post("/jobs/{job_id}/chunks/{index}")
def upload_chunk(job_id: str, index: int, file: UploadFile, background: BackgroundTasks) -> dict[str, str]:
    job = _job(job_id)
    if job.mode != "browser" or job.status != "finding_rallies":
        raise HTTPException(409, "조각을 받는 작업이 아니다")
    plan_path = job.dir / "rallies" / "plan.json"
    if not 0 <= index < len(json.loads(plan_path.read_text())):
        raise HTTPException(404, "그 조각은 계획에 없다")
    clip = job.dir / "rallies" / f"chunk{index:03d}_blur.mp4"
    answered = job.dir / "rallies" / "scan.json"
    if clip.exists() or (answered.exists() and any(c["chunk"] == index for c in json.loads(answered.read_text()))):
        raise HTTPException(409, "이미 받은 조각이다")  # 같은 조각을 두 번 물어 돈을 쓰지 않게
    size = 0
    with clip.open("wb") as f:
        while block := file.file.read(1024 * 1024):
            size += len(block)
            if size > MAX_CHUNK_BYTES:
                f.close()
                clip.unlink()
                raise HTTPException(413, "조각이 너무 크다")
            f.write(block)
    background.add_task(_run_chunk, job_id, index, clip, client_factory())
    return {"status": "received"}


@app.post("/jobs")
def upload(file: UploadFile, background: BackgroundTasks, match: str = Form(""),
           start: float = Form(0.0), end: float = Form(0.0), code: str = Form("")) -> dict[str, Any]:
    """start·end: 훑을 구간(초). 30분(jobs.MAX_SCAN_SEC)보다 긴 영상은 화면에서 고른다. code: 사용 코드(배포 서버).
    한 번에 올리기는 크기 제한이 없어 디스크를 채울 수 있다(10/31 보안 검토) → 배포 서버는 이어 올리기(/uploads)만.
    이 맥에서는 성능 테스트(tapme.bench)가 쓴다."""
    if access.REQUIRED:
        raise HTTPException(410, "이어 올리기(/uploads)를 써 주세요")
    if reason := _closed_reason(code):
        raise HTTPException(503, reason)
    try:
        access.use(code, "uploads")
    except access.AccessDenied as e:
        raise HTTPException(403, str(e)) from e
    job = jobs.create_job(file.filename or "video.mp4", file.file, match, start, end, access.normalize(code))
    jobs.count_upload()
    background.add_task(_run_scan, job.id, client_factory())
    return {"id": job.id}


class UploadInit(BaseModel):
    filename: str = Field(default="video.mp4", max_length=200)
    size: int = Field(gt=0, le=20 * 1024**3)
    match: str = Field(default="", max_length=40)
    code: str = Field(default="", max_length=20)
    start: float = Field(default=0.0, ge=0)
    end: float = Field(default=0.0, ge=0)


MAX_UPLOAD_PIECE = 32 * 1024 * 1024  # 이어 올리기 한 번에 받는 크기(기기는 16MB씩)
MAX_OPEN_UPLOADS = 2  # 코드 하나가 동시에 열어 둘 수 있는 이어 올리기 자리


@app.post("/uploads")
def start_upload(req: UploadInit) -> dict[str, Any]:
    """서버에 맡기기(이어 올리기): 받을 준비만 한다. 횟수는 다 받았을 때 센다(끊겨도 횟수가 줄지 않게)."""
    if reason := _closed_reason(req.code):
        raise HTTPException(503, reason)
    if access.REQUIRED:
        try:
            if access.left(req.code)["uploads"] <= 0:
                raise HTTPException(403, "이 코드로 올릴 수 있는 영상을 다 썼어요")
        except access.AccessDenied as e:
            raise HTTPException(403, str(e)) from e
    # 코드 하나로 이어 올리기 자리를 끝없이 열어 디스크를 채우지 못하게(10/31 보안 검토)
    if access.REQUIRED and jobs.open_uploads(access.normalize(req.code)) >= MAX_OPEN_UPLOADS:
        raise HTTPException(429, "올리던 영상이 이미 있어요. 그 영상을 마저 올리거나 내일 다시 해 주세요")
    upload_id = jobs.new_upload({**req.model_dump(), "code": access.normalize(req.code)})
    return {"id": upload_id, "received": 0}


@app.get("/uploads/{upload_id}")
def upload_progress(upload_id: str) -> dict[str, int]:
    try:
        st = jobs.upload_status(upload_id)
    except KeyError as e:
        raise HTTPException(404, "이어 올릴 것이 없다(다 올렸거나 지워졌다)") from e
    return {"received": st["received"], "size": st["size"]}


@app.put("/uploads/{upload_id}")
async def upload_piece(upload_id: str, offset: int, request: Request) -> dict[str, int]:
    """원본의 offset부터 한 조각. offset이 받은 크기와 다르면 409와 받은 크기(기기가 거기서부터 다시)."""
    import io

    if jobs.MIN_FREE_GB and jobs.free_gb() < jobs.MIN_FREE_GB:  # 올리는 도중에도 디스크를 지킨다
        raise HTTPException(503, "지금 사용자가 많아요. 잠시 뒤 '이어 올리기'를 눌러 주세요")
    body = bytearray()
    async for block in request.stream():
        body += block
        if len(body) > MAX_UPLOAD_PIECE:
            raise HTTPException(413, "조각이 너무 크다")
    try:
        received = jobs.append_upload(upload_id, offset, io.BytesIO(bytes(body)), MAX_UPLOAD_PIECE)
    except KeyError as e:
        raise HTTPException(404, "이어 올릴 것이 없다") from e
    except ValueError as e:
        raise HTTPException(409, {"received": int(e.args[0])}) from e
    except OverflowError as e:
        raise HTTPException(413, "올리기로 한 크기를 넘는다") from e
    return {"received": received}


@app.post("/uploads/{upload_id}/finish")
def finish_upload(upload_id: str, background: BackgroundTasks) -> dict[str, str]:
    try:
        st = jobs.upload_status(upload_id)
    except KeyError as e:
        raise HTTPException(404, "이어 올릴 것이 없다") from e
    if reason := _closed_reason(st.get("code", "")):
        raise HTTPException(503, reason)
    try:
        access.use(st.get("code", ""), "uploads")
    except access.AccessDenied as e:
        raise HTTPException(403, str(e)) from e
    try:
        job = jobs.finish_upload(upload_id)
    except ValueError as e:
        access.refund(st.get("code", ""), "uploads")
        raise HTTPException(409, str(e)) from e
    jobs.count_upload()
    background.add_task(_run_scan, job.id, client_factory())
    return {"id": job.id}


@app.get("/jobs/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    job = _job(job_id)
    return {"id": job.id, "status": job.status, "error": job.error, "spent_usd": round(job.spent_usd, 4),
            "progress": job.progress, "duration_sec": job.duration_sec, "estimate_usd": job.estimate_usd,
            "unread": job.unread, "match_name": job.match_name, "scan": job.scan, "scan_range": job.scan_range,
            "mode": job.mode, "segments": sorted(job.segments), "source_size": job.source_size,
            "chunks_received": jobs.browser_chunks_received(job) if job.mode == "browser" else [],
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
    try:
        return jobs.make_preview(job, index)
    except FileNotFoundError as e:  # 브라우저 방식: 미리보기는 기기에서
        raise HTTPException(404, str(e)) from e


@app.get("/jobs/{job_id}/rallies/{index}/thumb")
def rally_thumb(job_id: str, index: int) -> FileResponse:
    return FileResponse(_rally_media(job_id, index)[0], media_type="image/jpeg")


@app.get("/jobs/{job_id}/rallies/{index}/preview")
def rally_preview(job_id: str, index: int) -> FileResponse:
    return FileResponse(_rally_media(job_id, index)[1], media_type="video/mp4")


@app.post("/jobs/{job_id}/rallies/{index}/segment")
def upload_segment(job_id: str, index: int, video: UploadFile, start: float = Form(...),
                   audio: UploadFile | None = None) -> dict[str, str]:
    """브라우저 방식: 고른 랠리 구간(기기가 다시 압축하지 않고 잘라 낸 영상·소리)을 받는다. 그 뒤 prepare로 장면을 준비한다."""
    job = _job(job_id)
    if job.mode != "browser" or not 0 <= index < len(job.rallies):
        raise HTTPException(404, "구간을 받을 랠리가 없다")
    tmp = job.dir / "rallies" / f"segment{index:02d}"
    files = {}
    for name, up in (("video", video), ("audio", audio)):
        if up is None:
            continue
        files[name] = tmp.with_name(f"{tmp.name}_{name}.mp4")
        size = 0
        with files[name].open("wb") as f:
            while block := up.file.read(1024 * 1024):
                size += len(block)
                if size > MAX_SEGMENT_BYTES:
                    raise HTTPException(413, "구간 파일이 너무 크다")
                f.write(block)
    try:
        jobs.save_segment(job, index, files["video"], files.get("audio"), start)
    except (ValueError, subprocess.CalledProcessError) as e:
        raise HTTPException(422, f"구간 파일을 쓸 수 없다: {e}") from e
    finally:
        for p in files.values():
            p.unlink(missing_ok=True)
    return {"status": "ok"}


@app.post("/jobs/{job_id}/rallies/{index}/prepare")
def prepare(job_id: str, index: int, background: BackgroundTasks, ahead: bool = False) -> dict[str, str]:
    """ahead=true: 사용자가 미리보기를 열었을 때 뒤에서 미리 준비한다(고르면 탭 화면이 바로 뜨게). 비용이 없을 때
    (훑어보기가 득점 순간을 이미 찾은 랠리)와 다른 단계가 돌지 않을 때만. 아니면 아무것도 하지 않는다."""
    job = _job(job_id)
    if not 0 <= index < len(job.rallies):
        raise HTTPException(404, "랠리가 없다")
    scene = job.scenes.get(str(index))
    if job.mode == "browser" and not (job.dir / f"segment{index:02d}.mp4").exists():
        # 브라우저 방식은 원본이 기기에 있어, 고른 랠리 구간을 기기가 먼저 올려야 장면을 준비할 수 있다
        if ahead:
            return {"status": "skipped"}
        raise HTTPException(409, "이 랠리 구간 영상을 기기에서 먼저 올린다")
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
    job = _job(job_id)
    scene = job.scenes.get(str(index))
    if not scene or scene.get("status") != "done":
        # 이미 다시 만드는 중이면 거절한다. 계속 눌러 렌더 줄을 막지 못하게(10/31 보안 검토)
        raise HTTPException(409, "하이라이트를 먼저 만든다(이미 만드는 중이면 기다린다)")
    scene["status"] = "rendering"
    job.save()
    # 자막을 처음 켜거나 역할을 바꾸면 해설(Muse), '몸 실루엣'을 처음 고르면 다른 선수 추적(SAM)이 한 번 필요하다
    background.add_task(_run_rerender, job_id, index, req.languages, req.role, req.privacy, req.zoom,
                        client_factory(), req.face_mosaic, req.fps)
    return {"status": "rendering"}


def _run_rerender(job_id: str, index: int, *args: Any) -> None:
    """다시 렌더가 실패해도 장면은 앞서 만든 영상으로 남긴다('만드는 중'에 멈추지 않게)."""
    _run(job_id, jobs.rerender, index, *args)
    job = jobs.load_job(job_id)
    scene = job.scenes.get(str(index), {})
    if scene.get("status") == "rendering" and scene.get("video"):
        scene["status"] = "done"
        job.save()


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
    if job.reel.get("status") == "rendering":
        raise HTTPException(409, "모음 영상을 만드는 중이다")
    job.reel = {"status": "rendering", "order": req.order}
    job.save()
    background.add_task(_run, job_id, jobs.make_reel, req.order)
    return {"status": "rendering"}


@app.get("/jobs/{job_id}/reel/video")
def reel_video(job_id: str) -> FileResponse:
    reel = _job(job_id).reel
    if reel.get("status") != "done":
        raise HTTPException(404, "모음 영상이 아직 없다")
    return FileResponse(reel["video"], media_type="video/mp4")
