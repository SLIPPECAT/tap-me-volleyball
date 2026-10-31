"""사용 코드: 배구구 회원에게 나눠 준 코드로만 쓰고, 코드마다 횟수를 센다(10/31 사용자 결정).

코드당 영상 올리기 2편 + 하이라이트 3개. 비용 대부분이 영상 훑기(30분 약 $1.1)라 영상 수를 따로 막는다.
TAPME_REQUIRE_CODE=1인 서버(배포)에서만 코드를 요구한다. 이 맥에서 시험할 때는 묻지 않는다.

사용법 (서버에서):
  uv run python -m tapme.access new 50 --label 배구구   # 코드 50개를 만들어 보여 준다
  uv run python -m tapme.access list                     # 코드별 사용 횟수
"""

from __future__ import annotations

import argparse
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
    a = p.parse_args()
    if a.cmd == "new":
        print("\n".join(f"{c[:4]}-{c[4:]}" for c in new_codes(a.count, a.label)))
    else:
        for c, e in _load().items():
            print(f"{c[:4]}-{c[4:]}  영상 {e['uploads']}/{LIMITS['uploads']}  하이라이트 {e['highlights']}/{LIMITS['highlights']}  {e['label']}")
        print(f"전체 비용 ${jobs.total_spent():.2f} / ${jobs.MAX_USD_TOTAL:.0f}")


if __name__ == "__main__":
    main()
