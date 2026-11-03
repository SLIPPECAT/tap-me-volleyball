"""배구구 회원 전용 잠금(10/31 사용자: 외부 노출 없이 무조건 배구구에서만).

TAPME_MEMBERS_ONLY=1인 서버(배포)는 입장 쿠키(tapme_session)가 없으면 모든 주소를 막고 안내 화면만 보인다.
입장 쿠키는 이 서버가 열쇠(TAPME_SESSION_SECRET, .env)로 서명해 주는 것으로, 배구구 쿠키와는 따로다. 받는 길은 둘:
1) 배구구 홈 배너(?invite=…)로 와서 회원 확인을 통과(access.claim) — 안내 화면이 코드 받기를 대신 하고 쿠키를 받는다
2) 회원 확인으로 받은 코드를 넣기(/access/enter) — 아이폰 '홈 화면에 추가' 앱은 사파리와 쿠키를 같이 쓰지 않아서
배구구 앱의 배너는 폰 기본 브라우저로 열리고 앱 로그인은 브라우저와 따로라, 처음 한 번은 브라우저에서 배구구 웹에 로그인해야 한다
(로그인 화면 returnUrl=/ → 배구구 홈 → 배너). 앱이 웹 로그인 넘기기(app-handoff)로 배너를 열면 없앨 수 있다(배구구 앱 수정, 추후)
심사용 초대는 지금 막아 둔다(TAPME_JUDGE_ACCESS=1이면 연다, GitHub 이슈로 추후 결정).

작업은 만든 사람만(10/31 사용자: 특정 사용자가 다른 사람의 영상에 접근할 수 없어야): 입장 쿠키에 그 사람 코드의 열쇠 해시(o)를
넣고, 작업·이어 올리기마다 그 작업을 만든 코드와 맞는지 본다(server._members_only). 회원은 회원당 코드가 하나라 다른 기기도 같다.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

MEMBERS_ONLY = os.environ.get("TAPME_MEMBERS_ONLY") == "1"
JUDGE_ACCESS = os.environ.get("TAPME_JUDGE_ACCESS") == "1"
SECRET = os.environ.get("TAPME_SESSION_SECRET", "")
COOKIE = "tapme_session"
MAX_AGE = 30 * 24 * 3600
# 잠가도 열어 두는 주소: 꺼짐 확인(입구가 서버 안에서 묻는다)·코드 받기·코드로 들어오기·검색 엔진 안내·운영 현황(관리 토큰)
OPEN_PATHS = ("/idle", "/access/claim", "/access/enter", "/robots.txt")
OPEN_PREFIXES = ("/admin/",)


def _sign(body: bytes) -> str:
    return base64.urlsafe_b64encode(hmac.new(SECRET.encode(), body, hashlib.sha256).digest()).decode().rstrip("=")


def owner_key(code: str) -> str:
    """코드의 열쇠 해시(쿠키에 코드를 그대로 두지 않게). 대소문자·하이픈은 무시한다."""
    norm = "".join(c for c in code.upper() if c.isalnum())
    return hmac.new(SECRET.encode(), f"owner:{norm}".encode(), hashlib.sha256).hexdigest()[:24]


def make(kind: str, now: float | None = None, owner: str = "") -> str:
    """입장 쿠키 값. kind: member(회원 확인)·code(회원 코드)·judge. owner: 그 사람의 코드(작업 주인 확인용)."""
    data = {"k": kind, "e": int((time.time() if now is None else now) + MAX_AGE), "o": owner_key(owner) if owner else ""}
    body = base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
    return f"{body}.{_sign(body.encode())}"


def session(value: str | None, now: float | None = None) -> dict | None:
    """서명이 맞고 기한이 남은 입장 쿠키의 내용. 아니면 None."""
    if not value or not SECRET or "." not in value:
        return None
    body, sig = value.rsplit(".", 1)
    if not hmac.compare_digest(sig, _sign(body.encode())):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except ValueError:
        return None
    # 주인 정보(o)가 없는 쿠키는 들어오지 않은 것으로 본다. 주인 확인을 넣기 전(10/31 21:47~22:42 배포)에 받은 30일짜리
    # 쿠키는 화면·코드 확인은 되는데 올리기만 매번 403이었다(11/3 안드로이드: '업로드 중'이 보였다가 사라짐). 없는 것으로
    # 보면 배너로 들어올 때 안내 화면이 회원 확인을 다시 해 주인 정보가 든 쿠키를 새로 받는다
    if not data.get("o"):
        return None
    return data if data.get("e", 0) > (time.time() if now is None else now) else None


def valid(value: str | None, now: float | None = None) -> bool:
    return session(value, now) is not None


def is_open(path: str) -> bool:
    return path in OPEN_PATHS or path.startswith(OPEN_PREFIXES)


# 잠긴 화면. 배너로 왔으면(?invite=…) 여기서 코드를 받고(회원 확인) 들어간다. 바깥 파일을 쓰지 않는다(잠겨 있으니)
PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex, nofollow">
<title>Tap Me Volleyball</title>
<style>
:root { color-scheme: light dark; --bg: #f4f8fd; --ink: #1f3550; --muted: #5b6f86; --line: #d6e2ef; --blue: #3d8ee8; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d1520; --ink: #e8eef6; --muted: #9fb0c4; --line: #26364a; } }
body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--ink);
       font-family: system-ui, -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }
main { max-width: 26rem; padding: 24px 16px; }
h1 { font-size: 1.3rem; margin: 0 0 12px; }
p { color: var(--muted); line-height: 1.6; margin: 0 0 12px; }
a.btn, button { display: block; width: 100%; box-sizing: border-box; padding: 12px; border-radius: 10px; border: 0;
  background: var(--blue); color: #fff; font-size: 1rem; font-weight: 600; text-align: center; text-decoration: none; cursor: pointer; }
details { margin-top: 24px; border-top: 1px solid var(--line); padding-top: 16px; }
summary { cursor: pointer; color: var(--muted); }
input { width: 100%; box-sizing: border-box; padding: 12px; margin: 12px 0; border-radius: 10px; border: 1px solid var(--line);
  font-size: 1rem; background: transparent; color: var(--ink); text-transform: uppercase; }
#s { min-height: 1.6em; color: var(--ink); }
</style></head><body><main>
<h1>배구구 회원 전용이에요</h1>
<p>Tap Me는 배구구 회원만 쓸 수 있어요. 배구구에 로그인한 뒤 홈의 Tap Me 배너를 눌러 들어와 주세요.</p>
<p>배구구 앱에서 배너를 눌렀다면, 지금 열린 이 브라우저에서 한 번만 로그인해 주세요(앱 로그인은 브라우저와 따로예요).
로그인하면 배구구 홈으로 가요. 거기서 Tap Me 배너를 다시 누르면 돼요.</p>
<p id="s" aria-live="polite"></p>
<a class="btn" href="https://baegugu.com/login?returnUrl=%2F">배구구 로그인하고 홈으로</a>
<details><summary>홈 화면에 추가한 앱에서 열었나요?</summary>
<p>배너로 들어왔을 때 받은 사용 코드를 넣으면 이 앱에서도 들어갈 수 있어요.</p>
<input id="code" maxlength="12" autocomplete="off" autocapitalize="characters" spellcheck="false" placeholder="XXXX-XXXX">
<button id="go">들어가기</button></details>
</main><script>
const s = document.getElementById("s");
const save = (code) => { try { localStorage.setItem("code", code); } catch {} };
// 들어간 뒤엔 새로 불러온다. 주소에 #job=…이 있으면 같은 주소로 옮기기는 페이지 안 이동이라 안내 화면이 그대로였다(11/1)
async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || "다시 시도해 주세요");
  return j;
}
const invite = new URLSearchParams(location.search).get("invite");
if (invite) {
  history.replaceState(null, "", location.pathname + location.hash);
  s.textContent = "배구구 회원인지 확인하는 중…";
  post("/access/claim", { invite }).then((r) => { save(r.code); location.reload(); })
    .catch((e) => { s.textContent = e.message; });
}
document.getElementById("go").onclick = () => {
  const code = document.getElementById("code").value.trim();
  if (!code) return;
  s.textContent = "확인하는 중…";
  post("/access/enter", { code }).then(() => { save(code); location.reload(); })
    .catch((e) => { s.textContent = e.message; });
};
</script></body></html>"""
