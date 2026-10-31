"""사용 코드: 배구구 회원에게 나눠 준 코드로만 쓰고, 코드마다 횟수를 센다(10/31 사용자 결정).

코드당 영상 올리기 2편 + 하이라이트 3개. 비용 대부분이 영상 훑기(30분 약 $1.1)라 영상 수를 따로 막는다.
TAPME_REQUIRE_CODE=1인 서버(배포)에서만 코드를 요구한다. 이 맥에서 시험할 때는 묻지 않는다.

초대 링크(10/31 사용자 결정): 배구구 배너 링크에 ?invite=<초대말>을 붙이면 들어온 기기에 코드를 하나 만들어 준다.
배너 링크는 모든 회원에게 같은 고정 주소라 퍼질 수 있어, 같은 접속 주소(IP)당 하루 3개·배너 전체 150개까지만.
심사용 초대는 해커톤 제출서에만 적고, 그 코드는 하루 올리기 한도와 상관없이 쓴다. 초대말을 새로 만들면 예전 말은 막히고
이미 받은 코드는 그대로 쓴다.

사용법 (서버에서):
  uv run python -m tapme.access new 50 --label 배구구   # 코드 50개를 만들어 보여 준다
  uv run python -m tapme.access list                     # 코드별 사용 횟수
  uv run python -m tapme.access invite banner            # 배너 초대말을 새로 만든다(예전 말은 막힘)
  uv run python -m tapme.access invite judge --max 20    # 심사용 초대말
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from tapme import jobs

REQUIRED = os.environ.get("TAPME_REQUIRE_CODE") == "1"
LIMITS = {"uploads": int(os.environ.get("TAPME_UPLOADS_PER_CODE", "2")),
          "highlights": int(os.environ.get("TAPME_HIGHLIGHTS_PER_CODE", "3"))}
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 헷갈리는 0·O·1·I는 뺐다
_lock = threading.Lock()


INVITE_MAX = {"banner": 150, "judge": 20}  # 초대말 종류별로 만들어 줄 코드 수
CLAIMS_PER_IP_PER_DAY = 3  # 배너 초대: 같은 접속 주소에서 하루에
LABELS = {"banner": "배너", "judge": "심사"}


class AccessDenied(ValueError):
    pass


def _path() -> Path:
    return jobs.JOBS_DIR / "_access.json"  # 작업 폴더를 지워도 남는다


def _load() -> dict[str, dict[str, Any]]:
    return json.loads(_path().read_text()) if _path().exists() else {}


def _save(codes: dict[str, dict[str, Any]]) -> None:
    _path().parent.mkdir(parents=True, exist_ok=True)
    tmp = _path().with_name(f"_access.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(codes, ensure_ascii=False, indent=1))
    os.replace(tmp, _path())


def normalize(code: str) -> str:
    return "".join(c for c in code.upper() if c.isalnum())


def new_codes(n: int, label: str = "") -> list[str]:
    with _lock:
        codes = _load()
        made = []
        while len(made) < n:
            c = "".join(secrets.choice(ALPHABET) for _ in range(8))
            if c not in codes:
                codes[c] = {"label": label, "uploads": 0, "highlights": 0}
                made.append(c)
        _save(codes)
    return made


def _invites_path() -> Path:
    return jobs.JOBS_DIR / "_invites.json"


def _load_invites() -> dict[str, Any]:
    return json.loads(_invites_path().read_text()) if _invites_path().exists() else {}


def _save_invites(inv: dict[str, Any]) -> None:
    tmp = _invites_path().with_name(f"_invites.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(inv, ensure_ascii=False, indent=1))
    os.replace(tmp, _invites_path())


def new_invite(kind: str, max_codes: int | None = None) -> str:
    """초대말을 새로 만든다. 같은 종류의 예전 말은 더 이상 통하지 않는다(만든 코드 수는 이어서 센다)."""
    with _lock:
        inv = _load_invites()
        entry = inv.setdefault(kind, {"issued": 0})
        entry["word"] = secrets.token_urlsafe(6)
        entry["max"] = max_codes or entry.get("max") or INVITE_MAX[kind]
        jobs.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        _save_invites(inv)
    return entry["word"]


def claim(word: str, ip: str, now: float | None = None) -> str:
    """초대말로 코드를 하나 만들어 준다. 접속 주소는 하루 횟수를 세는 데만 쓰고 해시로만 남긴다."""
    with _lock:
        inv = _load_invites()
        kind = next((k for k, e in inv.items() if word and secrets.compare_digest(e.get("word", ""), word)), None)
        if kind is None:
            raise AccessDenied("초대 링크가 맞지 않거나 바뀌었어요")
        entry = inv[kind]
        if entry["issued"] >= entry["max"]:
            raise AccessDenied("초대 링크로 받을 수 있는 코드가 모두 나갔어요")
        day = jobs.today(now)
        who = hashlib.sha256(f"{day}:{ip}".encode()).hexdigest()[:16]
        by_ip = entry.get("by_ip", {}) if entry.get("by_ip_day") == day else {}
        if kind == "banner" and by_ip.get(who, 0) >= CLAIMS_PER_IP_PER_DAY:
            raise AccessDenied("이 접속에서 오늘 받을 수 있는 코드를 다 받았어요. 내일 다시 와 주세요")
        codes = _load()
        while (code := "".join(secrets.choice(ALPHABET) for _ in range(8))) in codes:
            pass
        codes[code] = {"label": LABELS[kind], "uploads": 0, "highlights": 0}
        _save(codes)
        entry["issued"] += 1
        entry["by_ip_day"], entry["by_ip"] = day, {**by_ip, who: by_ip.get(who, 0) + 1}
        _save_invites(inv)
    return code


def is_judge(code: str) -> bool:
    """심사용 코드는 하루 올리기 한도와 상관없이 쓴다."""
    return _load().get(normalize(code), {}).get("label") == LABELS["judge"]


def left(code: str) -> dict[str, int]:
    """남은 횟수. 없는 코드면 AccessDenied."""
    entry = _load().get(normalize(code))
    if entry is None:
        raise AccessDenied("코드가 맞지 않아요")
    return {k: max(0, LIMITS[k] - entry[k]) for k in LIMITS}


def use(code: str, kind: str) -> None:
    """한 번 쓴다(올리기·하이라이트 예약 때). 남은 횟수가 없으면 AccessDenied."""
    if not REQUIRED:
        return
    with _lock:
        codes = _load()
        entry = codes.get(normalize(code))
        if entry is None:
            raise AccessDenied("코드가 맞지 않아요")
        if entry[kind] >= LIMITS[kind]:
            raise AccessDenied({"uploads": f"이 코드로 올릴 수 있는 영상 {LIMITS[kind]}편을 다 썼어요",
                                "highlights": f"이 코드로 만들 수 있는 하이라이트 {LIMITS[kind]}개를 다 썼어요"}[kind])
        entry[kind] += 1
        _save(codes)


def refund(code: str, kind: str) -> None:
    """서버 쪽 실패로 못 만들었으면 횟수를 돌려준다."""
    if not REQUIRED:
        return
    with _lock:
        codes = _load()
        entry = codes.get(normalize(code))
        if entry is not None and entry[kind] > 0:
            entry[kind] -= 1
            _save(codes)


def main() -> None:
    p = argparse.ArgumentParser(description="사용 코드 만들기·보기")
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("count", type=int)
    n.add_argument("--label", default="")
    sub.add_parser("list")
    i = sub.add_parser("invite")
    i.add_argument("kind", choices=sorted(INVITE_MAX))
    i.add_argument("--max", type=int, default=None)
    a = p.parse_args()
    if a.cmd == "new":
        print("\n".join(f"{c[:4]}-{c[4:]}" for c in new_codes(a.count, a.label)))
    elif a.cmd == "invite":
        print(new_invite(a.kind, a.max))
    else:
        for c, e in _load().items():
            print(f"{c[:4]}-{c[4:]}  영상 {e['uploads']}/{LIMITS['uploads']}  하이라이트 {e['highlights']}/{LIMITS['highlights']}  {e['label']}")
        print(f"전체 비용 ${jobs.total_spent():.2f} / ${jobs.MAX_USD_TOTAL:.0f}")
        for k, e in _load_invites().items():
            print(f"초대 {k}: {e['issued']}/{e['max']}개 발급")


if __name__ == "__main__":
    main()
