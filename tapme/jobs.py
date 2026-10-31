"""하이라이트 작업의 단계와 상태. 서버(server.py)가 부른다.

작업 폴더 data/jobs/<id>/ (git 제외)에 원본·흐림본·결과와 job.json을 둔다.
단계:
  1. find_rallies    60초 조각 흐림본을 Muse로 훑어 랠리·득점 순간 찾기 (유료, 1분에 약 $0.035, scan.py)
  2. prepare_scene   사용자가 고른 랠리의 득점 7초 전~3초 뒤 SAM용 흐림본 + 탭할 장면 5장(득점 순간·그 전·구간 시작)
  3. make_highlight  탭 → '나'만 SAM 추적(점 프롬프트) → 득점 순간과 이어서 보이는 구간만 → 자막(Muse) → 원본 fps 흐림본 렌더
외부로 나가는 영상은 흐림본뿐이다. 얼굴이 그대로인 잘린 사본은 흐림 직후 지운다.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image

from tapme import anonymize, muse, rally, rally_filter, render, scan, score, track

JOBS_DIR = Path(os.environ.get("TAPME_JOBS_DIR", "data/jobs"))
MAX_USD_PER_JOB = float(os.environ.get("TAPME_MAX_USD_PER_JOB", "1.0"))
# 긴 영상도 처리하도록 상한을 길이에 비례해 올린다(10/30 사용자 결정: 1분에 $0.05, 1시간 $3). 짧은 영상은 MAX_USD_PER_JOB
MAX_USD_PER_MIN = float(os.environ.get("TAPME_MAX_USD_PER_MIN", "0.05"))
# 한 번에 훑는 길이. 30분이면 동호회 경기 한 세트가 넉넉히 들어가고 랠리 찾기 약 21분·약 $1.1(10/31 사용자 결정).
# 더 긴 영상은 화면에서 구간을 고르고, 구간 없이 오면 앞에서부터 이만큼만 훑는다
MAX_SCAN_SEC = 30 * 60
# 모든 작업을 합친 Meta API 비용 상한(10/31 사용자: 지원 크레딧 중 $100까지만). 작업이 지워져도 장부(_spend.json)는 남는다.
# 쓰기 전에 예상 금액으로 막지만 동시에 도는 작업끼리는 그 예상만큼 넘을 수 있다
MAX_USD_TOTAL = float(os.environ.get("TAPME_MAX_USD_TOTAL", "100"))
# 랠리 찾기: scan(기본, 60초 조각 훑어보기) 또는 whistle(정모 휘슬에 맞춘 예전 방식, 실험 H-17~H-33 기록·비교용)
RALLY_MODE = os.environ.get("TAPME_RALLY_MODE", "scan")
WINDOW_BEFORE_SEC = 7.0  # SAM은 50프레임보다 짧아도 50으로 과금된다(비용 원장 R-10·R-17)
# 브라우저 방식: 고른 랠리마다 기기가 원본에서 잘라 올리는 구간(장면 창보다 앞뒤 1초 넉넉히). 화면(index.html)과 같게
SEGMENT_BEFORE_SEC, SEGMENT_AFTER_SEC = 8.0, 6.0
WINDOW_AFTER_SEC = 5.0  # 훑어보기 득점 시각은 2~4초 어긋날 때가 있어(성능 테스트 4/68, H-43) 득점 뒤를 넉넉히
# 10fps: 움직이는 카메라에서 5fps는 같은 사람을 더 자주 놓쳤다(실험 H-24 B<C). 장면당 SAM 약 $0.02
SAM_FPS = 10
# 탭할 장면: 득점 순간과 그 1.5·3·4.5초 전, 구간 시작. 내가 보이는 장면을 하나 이상 골라 탭한다
# (득점 순간에 내가 화면에 없을 수 있다, 10/30 사용자 피드백). 값 = 득점 몇 초 전(None = 구간 시작)
TAP_MOMENTS: dict[str, float | None] = {"score": 0.0, "b1": 1.5, "b2": 3.0, "b3": 4.5, "start": None}
TAP_KINDS = tuple(TAP_MOMENTS)
# 하이라이트 영상은 최소 득점 4초 전~4초 뒤. 득점 시각이 ±4초 어긋나도 실제 득점이 들어가게(H-43: 다시 묻기는 답이 흔들려
# 쓰지 않음). '나'를 확실히 잡은 구간이 이보다 짧으면 그 밖에서는 링·확대 따라가기를 하지 않는다
CLIP_MIN_BEFORE_SEC = 4.0
CLIP_MIN_AFTER_SEC = 4.0
REEL_BEFORE_SEC = 5.0  # 모음 영상에는 장면마다 득점 5초 전~4초 뒤만. 3장면이면 약 27초
REEL_AFTER_SEC = 4.0
_lock = threading.Lock()


class Turns:
    """먼저 온 순서대로 slots개까지 동시에 돌린다. 기다리는 순서를 화면에 보여 준다(threading.Lock은 순서를 지키지 않는다)."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self._cv = threading.Condition()
        self._waiting: list[str] = []
        self._running = 0

    @contextmanager
    def turn(self, key: str) -> Iterator[None]:
        with self._cv:
            self._waiting.append(key)
            self._cv.wait_for(lambda: self._waiting[0] == key and self._running < self.slots)
            self._waiting.pop(0)
            self._running += 1
            self._cv.notify_all()
        try:
            yield
        finally:
            with self._cv:
                self._running -= 1
                self._cv.notify_all()

    def position(self, key: str) -> int | None:
        """기다리는 중이면 앞에서부터 몇 번째(1부터), 아니면 None."""
        with self._cv:
            return self._waiting.index(key) + 1 if key in self._waiting else None

    def waiting(self) -> int:
        with self._cv:
            return len(self._waiting)


# 렌더는 CPU를 다 써서 한 번에 하나만(동시에 돌리면 서로 느려졌다, H-42). SAM 추적·해설(네트워크)은 여러 장면이 겹쳐도 된다.
# 일괄 만들기에서 한 장면을 렌더하는 동안 다음 장면들의 추적이 끝나 있게 한다(10/30 사용자). 여러 사용자는 온 순서대로
RENDER_TURNS = Turns(1)
# 랠리 찾기도 얼굴 흐리기가 CPU를 다 써서 사용자 여럿이 한꺼번에 올리면 모두 느려진다 → 온 순서대로 이만큼씩(10/31)
SCAN_TURNS = Turns(int(os.environ.get("TAPME_SCAN_SLOTS", "1")))
# 랠리 찾기 대기가 이보다 많으면 새 영상을 받지 않는다(올리기 전에 알려 2GB를 헛되이 올리지 않게)
MAX_WAITING_SCANS = int(os.environ.get("TAPME_MAX_WAITING_SCANS", "5"))
# 서버 제한(10/30 사용자 "제한 같은 걸 둬야"): SAM 추적은 동시에 둘까지(Meta API에 한꺼번에 몰리지 않게),
# 한 작업에서 예약(대기·만드는 중)은 다섯 장면까지(한 사람이 줄을 오래 막지 않게)
TRACK_SLOTS = threading.BoundedSemaphore(2)
MAX_ACTIVE_SCENES = 5


class BudgetExceeded(RuntimeError):
    pass


class TapMissed(ValueError):
    pass


@dataclass
class Job:
    id: str
    video: str
    status: str = "uploaded"
    rallies: list[dict[str, Any]] = field(default_factory=list)
    scenes: dict[str, dict[str, Any]] = field(default_factory=dict)
    reel: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)  # 화면에 보여 줄 지금 단계: {"step", "done", "total"}
    spent_usd: float = 0.0
    error: str | None = None
    duration_sec: float = 0.0
    budget_usd: float = 0.0  # 0이면 MAX_USD_PER_JOB. 랠리 찾기 전에 영상 길이로 정한다
    estimate_usd: float = 0.0  # 랠리 찾기 예상 비용(화면에 보여 준다)
    unread: list[list[float]] = field(default_factory=list)  # AI가 답을 못 읽은 구간 [시작, 끝]. 화면에 알린다
    match_name: str = ""  # 저장 파일 이름 앞부분(경기이름_팀이름_하이라이트_01.mp4). 비우면 올린 파일 이름
    scan: dict[str, Any] = field(default_factory=dict)  # 랠리 찾기 진행 {"done", "total"}. 찾는 동안 장면 단계가 progress를 쓴다
    # "server": 원본을 올려 서버가 흐린다. "browser": 사용자 기기가 흐린 조각만 올리고 원본은 기기에 남는다(10/31, H-45)
    mode: str = "server"
    segments: dict[str, float] = field(default_factory=dict)
    source_size: int = 0  # 브라우저 방식: 기기의 원본 크기(바이트). 화면을 새로 연 뒤 같은 영상을 다시 골랐는지 확인  # 브라우저 방식: 랠리 번호 → 올린 구간 파일의 원본 시작 초
    code: str = ""  # 이 영상을 올린 사용 코드(하이라이트 횟수를 여기서 센다). 코드를 쓰지 않는 서버면 빈 값
    scan_range: list[float] = field(default_factory=list)  # 훑을 구간 [시작, 끝](원본 초). 올릴 때 받은 값을 찾기 전에 영상 길이로 맞춘다
    # 한 번에 맡기기(tapme/auto.py): 처음에 받은 나·설정과 진행 {"state", "me", "hint", "team", "options", "max", "found", ...}
    auto: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._seen: dict[str, Any] | None = None  # 마지막으로 읽거나 쓴 이 객체의 내용(save가 바뀐 항목만 고르는 기준)

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def save(self) -> None:
        """이 객체가 바꾼 항목만 디스크에 합쳐 쓴다. 랠리 찾기(긴 영상)와 장면 만들기가 동시에 돌며 같은 job.json을
        저장해도 서로의 내용을 덮어쓰지 않게. 쓴 돈은 서로 더하고, 장면은 장면별로 합친다."""
        with _lock:
            path = self.dir / "job.json"
            mine = asdict(self)
            out = mine
            if self._seen is not None and path.exists():
                disk, before = json.loads(path.read_text()), self._seen
                out = dict(disk)
                for k, v in mine.items():
                    if k == "spent_usd":
                        out[k] = round(float(disk.get(k) or 0) + v - float(before.get(k) or 0), 6)
                    elif k == "scenes":
                        out[k] = {**disk.get(k, {}), **{i: sc for i, sc in v.items() if sc != before.get(k, {}).get(i)}}
                    elif v != before.get(k):
                        out[k] = v
            # 임시 파일에 다 쓴 뒤 바꿔치기한다. 바로 쓰면 비운 순간 화면·다른 단계가 빈 파일을 읽어 오류가 났고,
            # 장면 준비가 그 오류로 멈춰 화면이 "준비 중"에 머물렀다(10/30 성능 테스트 T3)
            tmp = path.with_name(f"job.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            os.replace(tmp, path)
            _add_spend(mine["spent_usd"] - float((self._seen or {}).get("spent_usd") or 0))
            self._seen = mine

    def step(self, name: str, done: int = 0, total: int = 0) -> None:
        self.progress = {"step": name, "done": done, "total": total}
        self.save()

    def charge(self, estimate_usd: float) -> None:
        limit = self.budget_usd or MAX_USD_PER_JOB
        if self.spent_usd + estimate_usd > limit:
            raise BudgetExceeded(f"작업당 상한 ${limit:.2f}을 넘는다 (쓴 돈 ${self.spent_usd:.3f} + 예상 ${estimate_usd:.3f})")
        if total_spent() + estimate_usd > MAX_USD_TOTAL:
            raise BudgetExceeded(f"서비스 전체 비용 상한 ${MAX_USD_TOTAL:.0f}에 닿아 더 만들 수 없어요")


def _spend_path() -> Path:
    return JOBS_DIR / "_spend.json"


def uploads_today(now: float | None = None) -> int:
    path = JOBS_DIR / "_daily.json"
    d = json.loads(path.read_text()) if path.exists() else {}
    return int(d.get("uploads", 0)) if d.get("day") == today(now) else 0


def count_upload(now: float | None = None) -> None:
    with _lock:
        n = uploads_today(now) + 1
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = JOBS_DIR / f"_daily.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps({"day": today(now), "uploads": n}))
        os.replace(tmp, JOBS_DIR / "_daily.json")


def free_gb() -> float:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(JOBS_DIR).free / 1e9


def total_spent() -> float:
    """모든 작업이 쓴 Meta API 비용(지운 작업 포함)."""
    path = _spend_path()
    return float(json.loads(path.read_text())["usd"]) if path.exists() else 0.0


def _add_spend(delta: float) -> None:
    """Job.save가 _lock 안에서 부른다. 바뀐 만큼만 더해 같은 돈을 두 번 세지 않는다."""
    if not delta:
        return
    path = _spend_path()
    tmp = path.with_name(f"_spend.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"usd": round(total_spent() + delta, 6)}))
    os.replace(tmp, path)


# 하루의 시작(한국 시각). 이때 지난날 영상을 모두 지우고 하루 올리기 횟수를 새로 센다(10/31 사용자 결정).
# 자정이 아니라 새벽 4시: 정모가 밤 10~11시에 끝나 자정 무렵 올린 영상이 만드는 도중 지워지지 않게
DAY_START_HOUR = 4
KST = 9 * 3600
# 하루에 받는 영상 수(0이면 제한 없음). 디스크 100GB에 작업 하나 2~4.5GB → 20편(배포 서버에서 정한다)
MAX_UPLOADS_PER_DAY = int(os.environ.get("TAPME_MAX_UPLOADS_PER_DAY", "0"))
# 디스크 남은 공간이 이보다 적으면 새 영상을 받지 않는다(GB, 0이면 보지 않음). 이 맥은 원래 꽉 차 있어 배포 서버에서만
MIN_FREE_GB = float(os.environ.get("TAPME_MIN_FREE_GB", "0"))
IDLE_BEFORE_DELETE_SEC = 3600  # 하루가 바뀔 때 처리 중이던 작업은 한 시간 조용해진 뒤에 지운다


def day_start(now: float | None = None) -> float:
    """가장 최근 하루 시작(새벽 4시, 한국 시각)의 시각(epoch 초)."""
    now = time.time() if now is None else now
    local = now + KST - DAY_START_HOUR * 3600
    return now - local % 86400


def today(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(day_start(now) + KST))


def delete_previous_days(now: float | None = None) -> list[str]:
    """하루가 바뀌면 지난날 작업 폴더(원본·결과 영상)를 모두 지운다. 마지막 저장이 하루 시작 전이고 한 시간 넘게
    조용한 작업만 — 하루가 바뀔 때 처리 중이던 작업은 끝난 뒤 지운다. 상태로 거르지 않는다(서버가 꺼져 멈춘 작업도).
    정모 영상에는 다른 회원 얼굴이 담겨 오래 두지 않는다(10/31 사용자 결정, 배포 서버에서만 켠다)."""
    now = time.time() if now is None else now
    start, gone = day_start(now), []
    uploads = JOBS_DIR / "_uploads"
    for d in uploads.iterdir() if uploads.exists() else []:  # 다 올리지 못하고 남은 이어 올리기 조각
        data = d / "data"
        if data.exists() and data.stat().st_mtime < start and now - data.stat().st_mtime >= IDLE_BEFORE_DELETE_SEC:
            shutil.rmtree(d)
    for d in JOBS_DIR.iterdir() if JOBS_DIR.exists() else []:
        meta = d / "job.json"
        if not d.is_dir() or not meta.exists():
            continue
        mtime = meta.stat().st_mtime
        if mtime >= start or now - mtime < IDLE_BEFORE_DELETE_SEC:
            continue
        shutil.rmtree(d)
        gone.append(d.name)
    return gone


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}


def create_job(filename: str, data: bytes | BinaryIO | Path, match_name: str = "", start: float = 0.0, end: float = 0.0,
               code: str = "") -> Job:
    """data는 바이트·파일 객체·다 받은 파일 경로(이어 올리기). 경기 영상은 GB 단위일 수 있어 파일 객체는 조금씩 옮겨 쓰고,
    경로는 옮기기만 한다. start·end: 훑을 구간(초, end 0이면 끝까지)."""
    job = Job(id=uuid.uuid4().hex[:12], video="", match_name=match_name.strip()[:40] or Path(filename).stem[:40],
              scan_range=[start, end], code=code)
    job.dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    src = job.dir / f"source{suffix if suffix in VIDEO_SUFFIXES else '.mp4'}"
    if isinstance(data, bytes):
        src.write_bytes(data)
    elif isinstance(data, Path):
        shutil.move(str(data), src)
    else:
        with src.open("wb") as f:
            shutil.copyfileobj(data, f, length=8 * 1024 * 1024)
    job.video = str(src)
    job.save()
    return job


# ---- 이어 올리기(10/31 사용자: 업로드 중 화면이 꺼져도 처음부터 다시 올리지 않게) ----
# 원본을 조각으로 받아 한 파일에 이어 붙인다. 끊기면 기기가 받은 바이트 수를 묻고 그다음부터 보낸다. 다 받으면 작업이 된다
def _upload_dir(upload_id: str) -> Path:
    if not upload_id.isalnum():
        raise KeyError(upload_id)
    return JOBS_DIR / "_uploads" / upload_id


def new_upload(meta: dict[str, Any]) -> str:
    upload_id = uuid.uuid4().hex[:16]
    d = _upload_dir(upload_id)
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    (d / "data").touch()
    return upload_id


def open_uploads(code: str) -> int:
    """이 코드로 열어 둔(아직 작업이 되지 않은) 이어 올리기 자리 수."""
    d = JOBS_DIR / "_uploads"
    return sum(json.loads((p / "meta.json").read_text()).get("code") == code
               for p in (d.iterdir() if d.exists() else []) if (p / "meta.json").exists())


def upload_status(upload_id: str) -> dict[str, Any]:
    d = _upload_dir(upload_id)
    if not (d / "meta.json").exists():
        raise KeyError(upload_id)
    meta = json.loads((d / "meta.json").read_text())
    return {**meta, "received": (d / "data").stat().st_size}


def append_upload(upload_id: str, offset: int, stream: BinaryIO, limit: int) -> int:
    """offset이 지금까지 받은 크기와 같을 때만 이어 붙인다(아니면 ValueError — 기기가 받은 크기를 다시 물어 맞춘다)."""
    st = upload_status(upload_id)
    if offset != st["received"]:
        raise ValueError(st["received"])
    path = _upload_dir(upload_id) / "data"
    with path.open("ab") as f:
        n = 0
        while block := stream.read(1024 * 1024):
            n += len(block)
            if n > limit or offset + n > st["size"]:
                f.truncate(offset)  # 넘친 조각은 통째로 버린다(받은 크기를 offset으로 되돌림)
                raise OverflowError
            f.write(block)
    return path.stat().st_size


def finish_upload(upload_id: str) -> Job:
    st = upload_status(upload_id)
    if st["received"] != st["size"]:
        raise ValueError(f"아직 다 받지 않았다 ({st['received']}/{st['size']})")
    job = create_job(st["filename"], _upload_dir(upload_id) / "data", st.get("match", ""), st.get("start", 0.0),
                     st.get("end", 0.0), st.get("code", ""))
    shutil.rmtree(_upload_dir(upload_id))
    return job


# ---- 의견(10/31 사용자: 베타 의견 받기) ----
# 작업 폴더 밖(_feedback.jsonl)에 둬 새벽 4시 삭제와 상관없이 남는다. 영상은 담지 않는다
def add_feedback(entry: dict[str, Any]) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with _lock, (JOBS_DIR / "_feedback.jsonl").open("a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + KST)), **entry},
                           ensure_ascii=False) + "\n")


def feedback(limit: int = 100) -> list[dict[str, Any]]:
    path = JOBS_DIR / "_feedback.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    return [json.loads(x) for x in lines[-limit:]][::-1]  # 최근 것부터


def browser_chunks_received(job: Job) -> list[int]:
    """브라우저 방식: 서버가 받은 조각 번호(물었거나 묻는 중). 기기가 끊긴 뒤 빠진 조각부터 이어 처리한다."""
    d = job.dir / "rallies"
    asked = {c["chunk"] for c in json.loads((d / "scan.json").read_text())} if (d / "scan.json").exists() else set()
    waiting = {int(p.name[5:8]) for p in d.glob("chunk*_blur.mp4")} if d.exists() else set()
    return sorted(asked | waiting)


def load_job(job_id: str) -> Job:
    path = JOBS_DIR / job_id / "job.json"
    if not job_id.isalnum() or not path.exists():
        raise KeyError(job_id)
    text = path.read_text()
    job = Job(**json.loads(text))
    job._seen = json.loads(text)  # 따로 읽은 사본. 같은 객체면 고친 내용이 '본 내용'에도 들어가 바뀐 게 없다고 본다
    return job


def _source(job: Job, index: int, start: float) -> tuple[Path, float]:
    """장면에 쓸 원본과 그 안의 시작 초. 브라우저 방식은 기기가 올린 랠리 구간 파일(원본 시각을 빼서)."""
    if job.mode == "browser":
        return job.dir / f"segment{index:02d}.mp4", start - job.segments[str(index)]
    return Path(job.video), start


def _start_times(path: Path) -> tuple[float, float]:
    """(첫 영상 프레임 시각, 파일 시작 시각). ffmpeg -ss는 파일 시작 시각부터 센다."""
    def probe(*args: str) -> float:
        out = subprocess.check_output(["ffprobe", "-v", "error", *args, "-of", "csv=p=0", str(path)])
        return float(out.decode().strip() or 0)
    return probe("-select_streams", "v:0", "-show_entries", "stream=start_time"), probe("-show_entries", "format=start_time")


def save_segment(job: Job, index: int, video: Path, audio: Path | None, first_frame_sec: float) -> None:
    """기기가 다시 압축하지 않고 잘라 올린 랠리 구간(영상·소리 따로)을 한 파일로 합친다. first_frame_sec: 첫 영상 프레임의
    원본 시각(기기가 mp4에서 읽은 값). 조각 파일 속 시각은 만드는 도구마다 달라(원본 시각 그대로 / 0부터) 믿지 않고, 합친
    파일의 첫 영상 프레임 시각과 맞춰 기준점을 정한다. 장면 창(득점 7초 전~5초 뒤)을 덮어야 한다."""
    r = job.rallies[index]
    final, out = job.dir / f"segment{index:02d}.mp4", job.dir / f"segment{index:02d}.new.mp4"  # 맞으면 바꿔 넣는다
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video)]
    cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0"] if audio is not None else ["-map", "0:v:0"]
    subprocess.run([*cmd, "-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out)], check=True)
    video_start, file_start = _start_times(out)
    offset = first_frame_sec - video_start + file_start  # _cut(-ss)이 원본 시각 − offset으로 자르면 같은 프레임
    covers = (offset, offset + scan.duration_sec(out))
    need = (max(0.0, float(r["point_sec"]) - WINDOW_BEFORE_SEC), float(r["point_sec"]) + WINDOW_AFTER_SEC)
    if covers[0] > need[0] + 0.05 or covers[1] < need[1] - 0.05:
        out.unlink()
        raise ValueError(f"올린 구간({covers[0]:.1f}~{covers[1]:.1f}초)이 장면({need[0]:.1f}~{need[1]:.1f}초)을 덮지 않는다")
    os.replace(out, final)
    job.segments[str(index)] = round(offset, 3)
    job.save()


def _cut(video: Path, start: float, duration: float, out: Path, fps: int | None) -> Path:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(video), "-t", f"{duration:.2f}"]
    # -r는 원본보다 4프레임(0.13초) 늦은 장면을 골라, SAM 박스(프레임 k = k/fps초)가 원본 클립보다 늦게 따라갔다(S1)
    cmd += ["-vf", f"setpts=PTS-STARTPTS,fps={fps}:round=up"] if fps else []
    subprocess.run([*cmd, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], check=True)
    return out


def _cut_blurred(video: Path, start: float, duration: float, out: Path, fps: int | None) -> Path:
    """SAM·Muse로 보낼 클립. 얼굴을 흐리고, 얼굴이 그대로인 잘린 사본은 남기지 않는다."""
    raw = _cut(video, start, duration, out.with_name(out.stem + "_raw.mp4"), fps)
    anonymize.anonymize_video(raw, out)
    raw.unlink()
    return out


def scan_range(duration: float, start: float, end: float) -> list[float]:
    """훑을 구간을 영상 안, MAX_SCAN_SEC 이하로 맞춘다. 넘치면 끝을 당긴다."""
    start = min(max(0.0, start), duration)
    end = duration if end <= 0 else min(end, duration)
    if end - start < 1.0:
        raise ValueError(f"훑을 구간이 영상 밖이거나 너무 짧다 ({start:.0f}~{end:.0f}초, 영상 {duration:.0f}초)")
    return [round(start, 2), round(min(end, start + MAX_SCAN_SEC), 2)]


def find_rallies(job: Job, client: Any) -> None:
    job.status = "finding_rallies"
    job.duration_sec = round(scan.duration_sec(Path(job.video)), 2)
    job.scan_range = scan_range(job.duration_sec, *(job.scan_range or [0.0, 0.0]))
    start, length = job.scan_range[0], job.scan_range[1] - job.scan_range[0]
    job.budget_usd = round(max(MAX_USD_PER_JOB, MAX_USD_PER_MIN * length / 60), 2)
    if RALLY_MODE == "whistle":  # 남겨 둔 대안(정모 영상용). 구간 없이 영상 전체
        job.rallies = [{**r, "index": i} for i, r in enumerate(_rallies_by_whistle(job, client))]
    else:
        job.rallies = []
        job.estimate_usd = round(scan.estimate_usd(length), 3)
        total = len(scan.chunk_starts(length))
        job.scan = {"done": 0, "total": total}
        job.charge(job.estimate_usd)
        job.save()
        base = job.spent_usd

        def progress(done: int, total: int, spent: float, finished: list[dict[str, Any]]) -> None:
            # 찾은 랠리부터 목록에 붙여 1시간 영상도 바로 고를 수 있게 한다(10/30). 쓴 돈도 조각마다
            job.spent_usd = base + spent
            _add_rallies(job, scan.rallies_from_answers(finished))
            job.scan = {"done": done, "total": total}
            job.save()

        found, chunks = scan.scan_video(client, Path(job.video), length, job.dir / "rallies", progress, offset=start)
        job.spent_usd = base + sum(c["usd"] for c in chunks)
        _add_rallies(job, found)
        job.unread = scan.unread_ranges(chunks)
        (job.dir / "rallies").mkdir(exist_ok=True)
        (job.dir / "rallies" / "scan.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1))
    for i in range(len(job.rallies)):
        make_preview(job, i)  # 사용자가 시간만 보고는 어떤 장면인지 모른다(10/29 사용자 피드백). 이미 있으면 건너뜀
    job.scan = {}
    job.progress = {}
    job.status = "rallies_ready"
    job.save()


def _add_rallies(job: Job, found: list[dict[str, Any]]) -> None:
    """찾은 랠리를 목록에 붙인다. 번호(index)는 바꾸지 않는다 — 찾는 도중에 사용자가 고른 장면이 번호로 묶여 있다.
    같은 득점(2초 안)은 새 답으로 고치되, 이미 고른 장면은 그대로 둔다."""
    for r in found:
        j = next((k for k, e in enumerate(job.rallies)
                  if abs(e["point_sec"] - r["point_sec"]) <= scan.SAME_POINT_SEC), None)
        if j is None:
            job.rallies.append({**r, "index": len(job.rallies)})
            if job.mode == "server":  # 브라우저 방식은 원본이 기기에 있어 미리보기도 기기에서 만든다
                make_preview(job, len(job.rallies) - 1)
        elif str(j) not in job.scenes:
            job.rallies[j] = {**r, "index": j}


def _rallies_by_whistle(job: Job, client: Any) -> list[dict[str, Any]]:
    job.step("whistles")
    audio = rally.load_audio(Path(job.video))
    candidates = rally.segments(rally.find_whistles(audio), audio.size / rally.SAMPLE_RATE)
    job.step("rallies", 0, len(candidates))
    job.charge(sum(muse.estimate_usd(c.end_sec - c.start_sec + 2 * rally_filter.PAD_SEC) for c in candidates))
    results = rally_filter.check_candidates(client, Path(job.video), candidates, job.dir / "rallies",
                                            lambda d, n: job.step("rallies", d, n))
    job.spent_usd += sum(r["usd"] for r in results)
    # 사용자가 Muse 판정을 뒤집을 수 있게 랠리 아님으로 본 후보도 남긴다
    merged = rally_filter.merge_rallies(results)
    for m in merged:
        if m["clip"] is None:  # 합친 랠리는 득점 시점을 물을 흐림 클립을 새로 만든다
            out = job.dir / "rallies" / f"merged{m['parts'][0]:02d}_{m['parts'][-1]:02d}_blur.mp4"
            m["clip"] = str(rally_filter.blurred_clip(Path(job.video), m["start_sec"], m["end_sec"], out))
            m["clip_start_sec"] = max(0.0, m["start_sec"] - rally_filter.PAD_SEC)
    return merged


PREVIEW_SHORT_SIDE = 360  # 랠리 후보 미리보기: 폰에서 고르기용 낮은 화질


def create_browser_job(filename: str, duration: float, start: float, end: float, match_name: str = "",
                       code: str = "", size: int = 0) -> tuple[Job, list[tuple[float, float]]]:
    """브라우저 방식 작업: 원본 없이 길이·구간만 받는다. 기기가 흐린 1분 조각을 이 계획대로 올린다(조각 번호 = 순서)."""
    job = Job(id=uuid.uuid4().hex[:12], video="", mode="browser", code=code, status="finding_rallies", source_size=size,
              match_name=match_name.strip()[:40] or Path(filename).stem[:40], duration_sec=round(duration, 2))
    job.dir.mkdir(parents=True, exist_ok=True)
    job.scan_range = scan_range(job.duration_sec, start, end)
    begin, length = job.scan_range[0], job.scan_range[1] - job.scan_range[0]
    plan = scan.chunk_plan(begin, length)
    job.budget_usd = round(max(MAX_USD_PER_JOB, MAX_USD_PER_MIN * length / 60), 2)
    job.estimate_usd = round(scan.estimate_usd(length), 3)
    job.charge(job.estimate_usd)
    job.scan = {"done": 0, "total": len(plan)}
    (job.dir / "rallies").mkdir(exist_ok=True)
    (job.dir / "rallies" / "plan.json").write_text(json.dumps(plan))
    job.save()
    return job, plan


_chunk_locks: dict[str, threading.Lock] = {}


def add_browser_chunk(job_id: str, index: int, clip: Path, client: Any) -> bool:
    """기기가 올린 흐림본 조각을 Muse에 묻고(조각은 물은 뒤 지움) 찾은 랠리를 목록에 붙인다. 마지막 조각이면 끝내고 True.
    조각마다 따로 불려 겹쳐 돌 수 있어, 답을 모으는 부분만 작업별 잠금 안에서 한다."""
    job = load_job(job_id)
    start, seconds = json.loads((job.dir / "rallies" / "plan.json").read_text())[index]
    result = scan.ask_chunk(client, index, start, seconds, clip)
    with _lock:
        lock = _chunk_locks.setdefault(job_id, threading.Lock())
    with lock:
        job = load_job(job_id)
        path = job.dir / "rallies" / "scan.json"
        chunks = [c for c in (json.loads(path.read_text()) if path.exists() else []) if c["chunk"] != index] + [result]
        chunks.sort(key=lambda c: c["chunk"])
        path.write_text(json.dumps(chunks, ensure_ascii=False, indent=1))
        job.spent_usd += result["usd"]
        _add_rallies(job, scan.rallies_from_answers(chunks))
        job.scan = {"done": len(chunks), "total": job.scan.get("total") or len(chunks)}
        finished = len(chunks) >= job.scan["total"]
        if finished:
            job.unread = scan.unread_ranges(chunks)
            job.scan, job.progress, job.status = {}, {}, "rallies_ready"
        job.save()
    return finished


def make_preview(job: Job, index: int) -> tuple[Path, Path]:
    """랠리 후보의 장면 사진(가운데)과 미리보기 영상(앞뒤 1초 포함). 로컬, API 비용 없음.

    올린 사람의 폰에만 보여 주므로 원본에서 만든다. 외부(Meta)로는 여전히 흐림본만 보낸다.
    이미 있으면 다시 만들지 않는다(예전 작업은 처음 볼 때 만들어진다).
    """
    if job.mode != "server":
        raise FileNotFoundError("브라우저 방식은 원본이 기기에 있어 미리보기를 기기에서 만든다")
    r = job.rallies[index]
    out_dir = job.dir / "rallies"
    out_dir.mkdir(exist_ok=True)
    thumb, preview = out_dir / f"cand{index:02d}_thumb.jpg", out_dir / f"cand{index:02d}_preview.mp4"
    side = PREVIEW_SHORT_SIDE
    scale = f"scale='if(gt(iw,ih),-2,min({side},iw))':'if(gt(iw,ih),min({side},ih),-2)'"
    start = max(0.0, r["start_sec"] - rally_filter.PAD_SEC)
    if not preview.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", job.video,
                        "-t", f"{r['end_sec'] + rally_filter.PAD_SEC - start:.2f}", "-vf", scale, "-r", "30",
                        "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(preview)], check=True)
    if not thumb.exists():
        middle = (r["start_sec"] + r["end_sec"]) / 2
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{middle:.2f}", "-i", job.video,
                        "-frames:v", "1", "-vf", scale, str(thumb)], check=True)
    return thumb, preview


MAX_PLAYER_CARDS = 20


def player_cards(job: Job, index: int, kind: str, client: Any) -> list[list[float]]:
    """탭할 장면의 선수를 모두 찾아 카드 사진(흐림본에서 자름)을 만든다. 왼쪽에서 오른쪽 순서.
    폰에서 작은 선수를 탭하기 어려워(10/29 사용자 제안) 목록에서 고르게 한다.
    사용자가 그 장면을 처음 열 때만 만든다(장면당 약 $0.0025). 한 번 만들면 다시 쓴다."""
    scene = job.scenes[str(index)]
    cached = scene.setdefault("players", {}).get(kind)
    if cached is not None:
        return cached
    scene_dir = Path(scene["sam_clip"]).parent
    job.charge(track.USD_PER_IMAGE)
    frame = scene_dir / f"tap_{kind}.jpg"
    boxes = track.detect_players(client, frame)
    job.spent_usd += track.USD_PER_IMAGE
    boxes = sorted(boxes, key=lambda b: b.center[0])[:MAX_PLAYER_CARDS]
    img = Image.open(frame)
    for n, b in enumerate(boxes):
        pad_w, pad_h = (b.right - b.left) * 0.15, (b.bottom - b.top) * 0.08
        card = img.crop((max(0, b.left - pad_w), max(0, b.top - pad_h),
                         min(img.width, b.right + pad_w), min(img.height, b.bottom + pad_h)))
        card.thumbnail((160, 240))
        card.save(scene_dir / f"player_{kind}_{n:02d}.jpg")
    scene["players"][kind] = [[round(b.left, 1), round(b.top, 1), round(b.right, 1), round(b.bottom, 1)] for b in boxes]
    job.save()
    return scene["players"][kind]


def prepare_scene(job: Job, index: int, client: Any) -> dict[str, Any]:
    r = job.rallies[index]
    job.scenes[str(index)] = {"index": index, "status": "preparing"}
    if r.get("point_sec") is not None:  # 훑어보기가 이미 득점 순간을 찾았다(다시 묻지 않는다)
        answer = {"score_time_sec": r["point_sec"], "confidence": r.get("confidence"), "from": "scan"}
        score_sec = float(r["point_sec"])
    else:
        job.step("score_time")
        job.charge(muse.estimate_usd(r["end_sec"] - r["start_sec"] + 2 * rally_filter.PAD_SEC))
        answer, reply = score.ask_score_time(client, Path(r["clip"]))
        job.spent_usd += reply.usd
        score_sec = r["clip_start_sec"] + float(answer["score_time_sec"])
    start = max(0.0, score_sec - WINDOW_BEFORE_SEC)
    duration = score_sec + WINDOW_AFTER_SEC - start
    scene_dir = job.dir / f"scene{index:02d}"
    scene_dir.mkdir(exist_ok=True)
    job.step("blur")
    sam_clip = _cut_blurred(*_source(job, index, start), duration, scene_dir / "sam_blur.mp4", SAM_FPS)
    n = render.probe(sam_clip)[3]
    # 득점 근처 장면에서 탭하면 하이라이트의 핵심은 맞는 사람을 잡는다(실험 H-25). 장면 사진은 로컬, 무료
    frames = {kind: 0 if before is None else max(0, min(n - 1, round((score_sec - before - start) * SAM_FPS)))
              for kind, before in TAP_MOMENTS.items()}
    for kind, k in frames.items():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(sam_clip), "-vf", f"select=eq(n\\,{k})",
                        "-frames:v", "1", str(scene_dir / f"tap_{kind}.jpg")], check=True)
    scene = {"index": index, "status": "awaiting_tap", "score_sec": round(score_sec, 2),
             "window": [round(start, 2), round(start + duration, 2)], "score_answer": answer,
             "sam_clip": str(sam_clip), "sam_fps": SAM_FPS, "tap_frames": frames, "players": {}}
    job.scenes[str(index)] = scene
    job.progress = {}
    job.save()
    return scene


def _track_others(job: Job, client: Any, sam_clip: Path) -> Path:
    """'몸 실루엣' 가리기에 쓸 다른 선수 위치. 이 방식을 고를 때만 SAM을 한 번 더 부른다."""
    out = sam_clip.parent / "others_wire.txt"
    if not out.exists():
        frames = render.probe(sam_clip)[3]
        job.charge(track.estimate_usd(max(frames, 50)))
        wire, usage, _ = track.track_players(client, sam_clip)
        job.spent_usd += track.estimate_usd(int(usage.get("video_frames_processed") or frames))
        out.write_text(wire)
    return out


def scene_render_inputs(job: Job, scene: dict[str, Any], languages: list[str], privacy: str, client: Any) -> tuple[
    Path, dict[int, list[track.Box]], dict[int, dict[str, Any]], list[tuple[float, float, str]]
]:
    """렌더할 원본 클립, SAM 프레임별 박스·마스크(클립 기준), 자막. 렌더와 품질 확인(qa.py)이 같이 쓴다."""
    sam_clip = Path(scene["sam_clip"])
    if not scene.get("clip_is_original"):  # 10/29 이전 장면은 흐림본으로 잘라 둬서 '나'의 얼굴도 가려져 있다
        start, end = scene["clip"]
        scene["clip_video"] = str(_cut(*_source(job, scene["index"], start), end - start, sam_clip.parent / "clip.mp4", None))
        scene["clip_is_original"] = True
    me = track.parse((sam_clip.parent / "me_wire.txt").read_text())
    others = track.parse(_track_others(job, client, sam_clip).read_text()) if privacy == "silhouette" else None
    by, masks = track.merge_me_and_others(me, others)
    # '나'는 확실히 잡은 구간(span) 안에서만 쓴다. 그 밖은 화면 밖에 나갔다 들어와 다른 사람일 수 있다(실험 H-24·H-25)
    s0, s1 = scene["span_frames"]
    by = {k: [b for b in v if b.object_id != track.TAPPED_ID or s0 <= k <= s1] for k, v in by.items()}
    f0 = round((scene["clip"][0] - scene["window"][0]) * scene["sam_fps"])  # 렌더할 클립의 첫 SAM 프레임
    by = {k - f0: v for k, v in by.items() if k >= f0}
    masks = {k - f0: v for k, v in masks.items() if k >= f0}
    cues = score.subtitle_cues(scene.get("commentary"), languages, offset=scene["clip"][0] - scene["window"][0],
                               duration=scene["clip"][1] - scene["clip"][0])
    return Path(scene["clip_video"]), by, masks, cues


def scene_people(job: Job, scene: dict[str, Any], client: Any) -> dict[int, list[track.Box]]:
    """SAM 프레임별(렌더할 클립 기준) 모든 선수 상자. 모자이크에서 몸 위에 잡힌 옆얼굴을 거르는 데 쓴다(H-48).
    '몸 실루엣' 가리기와 같은 추적이라 장면마다 한 번(약 $0.024), 저장해 다시 쓴다."""
    sam_clip = Path(scene["sam_clip"])
    by = track.boxes_by_frame(track.parse(_track_others(job, client, sam_clip).read_text()))
    f0 = round((scene["clip"][0] - scene["window"][0]) * scene["sam_fps"])
    return {k - f0: v for k, v in by.items() if k >= f0}


# 얼굴 모자이크를 할 팀(11/1 사용자): 모두 · 상대 팀만(우리 팀은 얼굴 그대로) · 우리 팀만
MOSAIC_TEAMS = ("all", "opponent", "mine")
TEAM_QUESTION_USD = 0.03  # 선수 카드 20장 안팎 + 생각 '중간'. 넉넉히 잡고 charge, 실제 토큰으로 다시 셈
TEAM_PROMPT = """첫 번째 사진은 배구 경기 영상에서 '나'(기준 선수)를 잘라 낸 것이다. 그 뒤 사진들은 같은 장면의 다른 사람들이고,
각 사진 앞에 '후보 번호'가 있다. 얼굴은 모두 흐려져 있으니 유니폼 색·무늬·바지로 판단한다.
후보마다 '나'와 같은 팀 유니폼이면 same, 상대 팀 선수면 other, 심판·관중·코트 밖 사람이거나 잘 안 보이면 unsure.
{team}
JSON 하나로만 답한다: {{"teams": {{"0": "same" 또는 "other" 또는 "unsure", ...}}}}"""


def _card_img(path: Path) -> dict[str, str]:
    return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()}


def scene_teams(job: Job, scene: dict[str, Any], client: Any) -> dict[str, str]:
    """다른 선수 추적(scene_people와 같은 것)의 사람마다 우리 팀(mine)·상대 팀(opponent)·모름(unknown).
    사람마다 가장 크게 잡힌 순간을 얼굴 흐림본(sam_clip)에서 잘라, '나' 카드와 같은 유니폼인지 Muse Spark에 한 번 묻는다.
    장면마다 한 번(약 $0.01~0.03), 저장해 다시 쓴다. 모르는 사람은 늘 가린다(render.off_team)."""
    if scene.get("teams") is not None:
        return scene["teams"]
    sam_clip = Path(scene["sam_clip"])
    others = track.boxes_by_frame(track.parse(_track_others(job, client, sam_clip).read_text()))
    me = track.boxes_by_frame(track.parse((sam_clip.parent / "me_wire.txt").read_text()))
    s0, s1 = scene["span_frames"]

    def area(b: track.Box) -> float:
        return (b.right - b.left) * (b.bottom - b.top)
    best: dict[str, tuple[int, track.Box]] = {}  # 사람마다 가장 크게 잡힌 (프레임, 상자)
    for k, boxes in others.items():
        for b in boxes:
            if b.object_id not in best or area(b) > area(best[b.object_id][1]):
                best[b.object_id] = (k, b)
    mine = [(k, b) for k, boxes in me.items() if s0 <= k <= s1 for b in boxes if b.object_id == track.TAPPED_ID]
    if not best or not mine:
        scene["teams"] = {}
        return scene["teams"]
    people = sorted(best.items(), key=lambda kv: -area(kv[1][1]))[:MAX_PLAYER_CARDS]
    picks = [("me", max(mine, key=lambda kb: area(kb[1])))] + [(oid, kb) for oid, kb in people]
    d = sam_clip.parent / "teams"
    d.mkdir(exist_ok=True)
    content: list[dict[str, str]] = [
        {"type": "input_text", "text": TEAM_PROMPT.format(
            team=f"사용자는 우리 팀을 '{scene['team']}'(이)라고 부른다." if scene.get("team") else "")}]
    for n, (oid, (k, b)) in enumerate(picks):
        frame = d / f"frame_{k:04d}.jpg"
        if not frame.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(sam_clip), "-vf", f"select=eq(n\\,{k})",
                            "-frames:v", "1", str(frame)], check=True)
        img = Image.open(frame)
        pad_w, pad_h = (b.right - b.left) * 0.15, (b.bottom - b.top) * 0.08  # 선수 카드(player_cards)와 같게
        card = img.crop((max(0, b.left - pad_w), max(0, b.top - pad_h),
                         min(img.width, b.right + pad_w), min(img.height, b.bottom + pad_h)))
        card.thumbnail((160, 240))
        path = d / f"card_{n:02d}.jpg"
        card.save(path)
        content += [{"type": "input_text", "text": "나" if oid == "me" else f"후보 {n - 1}"}, _card_img(path)]
    job.charge(TEAM_QUESTION_USD)
    resp = client.responses.create(model=muse.MODEL, max_output_tokens=16000,
                                   input=[{"type": "message", "role": "user", "content": content}])
    job.spent_usd += (int(resp.usage.input_tokens or 0) * muse.INPUT_USD_PER_TOKEN
                      + int(resp.usage.output_tokens or 0) * muse.OUTPUT_USD_PER_TOKEN)
    try:
        answers = muse.parse_json_reply(resp.output_text or "").get("teams") or {}
    except ValueError:
        answers = {}
    names = {"same": "mine", "other": "opponent"}
    scene["teams"] = {oid: names.get(str(answers.get(str(n), "")).strip().lower(), "unknown")
                      for n, (oid, _) in enumerate(people)}
    job.save()
    return scene["teams"]


def _clear_ids(job: Job, scene: dict[str, Any], mosaic_team: str, client: Any) -> set[str] | None:
    """모자이크를 풀어 줄 사람(다른 선수 추적의 object_id). 모두 가리면 None."""
    if mosaic_team == "all":
        return None
    keep = "mine" if mosaic_team == "opponent" else "opponent"
    return {oid for oid, team in scene_teams(job, scene, client).items() if team == keep}


def _render_scene(job: Job, scene: dict[str, Any], languages: list[str], privacy: str, zoom: float,
                  face_mosaic: bool, client: Any, max_fps: int = 0, mosaic_team: str = "all") -> Path:
    src, by, masks, cues = scene_render_inputs(job, scene, languages, privacy, client)
    people = scene_people(job, scene, client) if face_mosaic else None
    clear = _clear_ids(job, scene, mosaic_team, client) if face_mosaic else None
    key = next((f"{job.id}/{k}" for k, v in job.scenes.items() if v is scene), f"{job.id}/?")  # 화면에 보일 대기 순서
    with RENDER_TURNS.turn(key):
        return _render_locked(job, scene, src, by, masks, cues, zoom, privacy, face_mosaic, max_fps, people, clear)


def _render_locked(job: Job, scene: dict[str, Any], src: Path, by: Any, masks: Any, cues: Any, zoom: float, privacy: str,
                   face_mosaic: bool, max_fps: int, people: dict[int, list[track.Box]] | None = None,
                   clear: set[str] | None = None) -> Path:
    out = render.render_video(src, Path(scene["sam_clip"]).parent / "highlight.mp4", by, masks,
                              track.TAPPED_ID, scene["sam_fps"], cues, zoom, privacy, face_mosaic,
                              on_progress=lambda d, n: job.step("render", d, n), max_fps=max_fps, people_by_sam=people,
                              clear_ids=clear)
    job.progress = {}
    return out


def _ensure_commentary(job: Job, scene: dict[str, Any], languages: list[str], role: str, client: Any) -> None:
    """자막을 켤 때만 해설을 묻는다(약 $0.01). 이미 같은 역할로 받아 뒀으면 다시 묻지 않는다."""
    if not languages or (scene.get("commentary") is not None and scene.get("role", "") == role):
        return
    window = scene["window"][1] - scene["window"][0]
    lines = None
    for _ in range(2):  # 읽을 수 없는 답이면 한 번 더 묻는다
        job.charge(muse.estimate_usd(window))
        lines, reply = score.ask_commentary(client, Path(scene["sam_clip"]), scene["team"],
                                            scene["score_sec"] - scene["window"][0], role, window)
        job.spent_usd += reply.usd  # 실패한 답도 돈은 나갔다
        if lines is not None:
            break
    # 그래도 못 받으면 자막 없이 만든다(영상 자체는 막지 않는다)
    scene.update(commentary=lines or [], role=role, commentary_failed=lines is None)


_number_lock = threading.Lock()


def assign_number(job: Job, scene: dict[str, Any]) -> int:
    """저장 파일 번호: 이 경기에서 하이라이트를 만든 순서. 다시 만들어도 그대로(10/30 사용자 요청).
    여러 장면을 동시에 만들면 각자 처음 읽은 job으로 번호를 매겨 겹쳤다(10/31 한 번에 맡기기: _02가 둘) →
    디스크의 최신 번호를 잠금 안에서 보고 바로 저장한다."""
    with _number_lock:
        if "number" not in scene:
            taken = [sc.get("number", 0) for sc in load_job(job.id).scenes.values()]
            scene["number"] = 1 + max(taken + [sc.get("number", 0) for sc in job.scenes.values()], default=0)
            job.save()
    return scene["number"]


def make_highlight(
    job: Job, index: int, taps: list[tuple[str, float, float]], team_label: str,
    languages: list[str], role: str, privacy: str, zoom: float, client: Any, face_mosaic: bool = True, max_fps: int = 0,
    mosaic_team: str = "all",
) -> dict[str, Any]:
    """taps = [(장면 종류, x, y) 또는 (장면 종류, 고른 선수 번호), ...]. 장면 종류는 TAP_KINDS, 하나 이상.
    languages가 비면 자막 없음(Muse 해설을 묻지 않는다). role은 score.ROLE_TEXT의 키(선택)."""
    scene = job.scenes[str(index)]
    if not taps or any(t[0] not in TAP_KINDS for t in taps):
        raise TapMissed("내가 보이는 장면을 하나 이상 골라 나를 탭해 주세요")
    scene["status"] = "rendering"
    job.step("tracking")
    sam_clip = Path(scene["sam_clip"])
    w, h, _, frames = render.probe(sam_clip)
    silhouette = 2 if privacy == "silhouette" else 1
    job.charge(track.estimate_usd(max(frames, 50)) * silhouette)

    points: list = []
    for tap in taps:
        frame = scene["tap_frames"][tap[0]]
        if len(tap) == 2:  # 선수 목록에서 고름 → 그 선수의 박스
            cards = scene.get("players", {}).get(tap[0], [])
            if not 0 <= tap[1] < len(cards):
                raise TapMissed("고른 선수가 목록에 없다. 다시 골라 주세요")
            points.append((frame, track.Box(track.TAPPED_ID, *cards[tap[1]])))
        else:
            points.append((frame, tap[1], tap[2]))
    with TRACK_SLOTS:
        wire, usage, _ = track.track_players(client, sam_clip, track.tap_prompt((w, h), points))
    job.spent_usd += track.estimate_usd(int(usage.get("video_frames_processed") or frames))
    (sam_clip.parent / "me_wire.txt").write_text(wire)  # 옵션만 바꿔 다시 렌더할 때 SAM을 또 부르지 않는다
    fps = scene["sam_fps"]
    # 득점 순간에 가장 가까운 탭을 기준으로 이어서 보이는 구간을 잡는다
    score_frame = scene["tap_frames"]["score"]
    anchor = min((scene["tap_frames"][t[0]] for t in taps), key=lambda k: abs(k - score_frame))
    span = track.visible_span(track.boxes_by_frame(track.parse(wire)), around=anchor, max_gap=fps)
    if span is None:
        scene["status"] = "awaiting_tap"
        job.save()
        raise TapMissed("탭한 곳에서 선수를 찾지 못했다. 선수 몸 위를 다시 탭해 주세요")

    scene["team"] = team_label
    assign_number(job, scene)
    if languages:
        job.step("commentary")
    _ensure_commentary(job, scene, languages, role, client)
    win_start, win_end = scene["window"]
    clip_start = max(win_start, min(win_start + span[0] / fps, scene["score_sec"] - CLIP_MIN_BEFORE_SEC))
    clip_end = min(win_end, max(win_start + (span[1] + 1) / fps, scene["score_sec"] + CLIP_MIN_AFTER_SEC))
    # 결과 영상은 원본에서 자르고(맥 안에서만 씀), 렌더할 때 '나'를 뺀 얼굴만 모자이크한다(선택)
    clip_video = _cut(*_source(job, index, clip_start), clip_end - clip_start, sam_clip.parent / "clip.mp4", None)
    scene.update(taps=[list(t) for t in taps], span_frames=list(span),
                 clip=[round(clip_start, 2), round(clip_end, 2)], clip_video=str(clip_video), clip_is_original=True)
    out = _render_scene(job, scene, languages, privacy, zoom, face_mosaic, client, max_fps, mosaic_team)
    scene.update(status="done", options={"languages": languages, "role": role, "privacy": privacy, "zoom": zoom,
                                         "face_mosaic": face_mosaic, "fps": max_fps, "mosaic_team": mosaic_team},
                video=str(out))
    job.save()
    return scene


def rerender(job: Job, index: int, languages: list[str], role: str, privacy: str, zoom: float,
             client: Any = None, face_mosaic: bool = True, max_fps: int = 0, mosaic_team: str = "all") -> dict[str, Any]:
    """자막·가리기·확대만 바꿔 다시 렌더. 저장해 둔 SAM 결과와 해설을 써서 API 비용이 없다.
    자막을 처음 켜거나 역할을 바꾸면 해설을, '몸 실루엣'을 처음 고르면 다른 선수 추적을 한 번 부른다."""
    scene = job.scenes[str(index)]
    scene["status"] = "rendering"
    job.save()
    _ensure_commentary(job, scene, languages, role, client)
    out = _render_scene(job, scene, languages, privacy, zoom, face_mosaic, client, max_fps, mosaic_team)
    scene.update(status="done", options={"languages": languages, "role": role, "privacy": privacy, "zoom": zoom,
                                         "face_mosaic": face_mosaic, "fps": max_fps, "mosaic_team": mosaic_team},
                video=str(out))
    job.save()
    return scene


def _has_audio(video: Path) -> bool:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                   "-of", "csv=p=0", str(video)], text=True)
    return bool(out.strip())


def make_reel(job: Job, order: list[int]) -> dict[str, Any]:
    """완성된 장면들을 고른 순서대로, 득점 앞뒤만 잘라 한 편으로 잇는다. 이미 렌더한 영상만 써서 API 비용이 없다."""
    scenes = [job.scenes.get(str(i), {}) for i in order]
    if not scenes or any(s.get("status") != "done" for s in scenes):
        raise ValueError("완성된 장면만 모을 수 있다")
    job.reel = {"status": "rendering", "order": order}
    job.save()

    # 소리 없는 영상이 하나라도 있으면 영상만 잇는다(concat은 모든 입력에 같은 스트림이 있어야 한다)
    audio = all(_has_audio(Path(s["video"])) for s in scenes)
    inputs: list[str] = []
    parts: list[str] = []
    for k, s in enumerate(scenes):
        clip = s.get("clip", s["window"])  # 장면 영상은 '나'가 이어서 보이는 구간만 담는다
        at = s["score_sec"] - clip[0]  # 장면 영상 안에서 득점 시각
        start = max(0.0, at - REEL_BEFORE_SEC)
        end = min(clip[1] - clip[0], at + REEL_AFTER_SEC)
        inputs += ["-i", s["video"]]
        parts.append(f"[{k}:v]trim={start:.3f}:{end:.3f},setpts=PTS-STARTPTS[v{k}];")
        if audio:
            parts.append(f"[{k}:a]atrim={start:.3f}:{end:.3f},asetpts=PTS-STARTPTS[a{k}];")
    joined = "".join(f"[v{k}]" + (f"[a{k}]" if audio else "") for k in range(len(scenes)))
    graph = "".join(parts) + f"{joined}concat=n={len(scenes)}:v=1:a={int(audio)}[v]" + ("[a]" if audio else "")
    out = job.dir / "reel.mp4"
    maps = ["-map", "[v]"] + (["-map", "[a]", "-c:a", "aac"] if audio else [])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, *maps,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    job.reel = {"status": "done", "order": order, "video": str(out)}
    job.save()
    return job.reel
