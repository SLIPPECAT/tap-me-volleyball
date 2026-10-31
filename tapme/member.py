"""배구구 회원 확인(10/31 사용자: 배너는 모두에게 보이되, 쓰는 것은 배구구 회원만).

Tap Me는 tapme.highlight.baegugu.com(배구구 하위 주소)이라 배구구 로그인 쿠키(baegugu-token, 도메인 .baegugu.com)가
코드 받기 요청에 같이 온다. 그 쿠키로 배구구 /api/auth/me(읽기 전용, 로그인 안 했으면 401)를 물어 회원 번호만 본다.
(/api/auth/status는 쿠키 로그인이면 authenticated만 true고 user는 비어 와 회원 번호를 알 수 없었다, 10/31 실측)
- 쿠키는 이 확인에만 쓰고 저장·기록하지 않는다. 회원 번호도 access에서 열쇠 붙인 해시로만 남긴다
- 로그인 연장(refresh)은 하지 않는다(연장 토큰이 바뀌면 회원의 배구구 로그인이 풀릴 수 있다). 배구구 홈에서 배너를 누르는
  순간엔 보통 로그인이 살아 있다
- Caddy가 /access/claim 말고는 Cookie 헤더를 지우고 넘긴다(deploy/Caddyfile) — 앱 나머지는 배구구 쿠키를 받지 않는다
운영 배구구 API는 이 한 곳만 쓴다(CLAUDE.md 예외, 10/31 사용자 승인). 해커톤 뒤 배구구가 서명한 입장권 방식으로 바꿀 후보.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

CHECK = os.environ.get("TAPME_MEMBER_CHECK") == "1"
API = os.environ.get("TAPME_MEMBER_API", "https://api.baegugu.com/api/auth/me")
COOKIE = "baegugu-token"


class Unavailable(Exception):
    """배구구에 물어볼 수 없음(꺼짐·느림). 회원이 아니라는 뜻이 아니다."""


def member_id(cookies: dict[str, str]) -> str | None:
    """로그인한 배구구 회원이면 회원 번호, 아니면 None."""
    token = cookies.get(COOKIE)
    if not token:
        return None
    req = urllib.request.Request(API, headers={"Cookie": f"{COOKIE}={token}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return None
        raise Unavailable(f"HTTP {e.code}") from None  # 요청(쿠키)을 예외에 싣지 않는다
    except (urllib.error.URLError, OSError, ValueError):
        raise Unavailable("no answer") from None
    user = body.get("data") or {}
    return str(user["id"]) if body.get("success") and user.get("id") else None
