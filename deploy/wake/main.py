"""쓸 때만 켜는 서버의 입구(Cloud Run, 쓰지 않으면 0대라 거의 무료). 사용자가 오면 VM을 켜고, 준비되면 서버로 보낸다.

VM은 30분 동안 쓰지 않으면 스스로 꺼진다(deploy/idle-stop.sh). 표준 라이브러리만 쓴다.
환경 변수: PROJECT, ZONE, INSTANCE(VM 이름), TARGET(서버 https 주소)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJECT, ZONE, INSTANCE = os.environ["PROJECT"], os.environ["ZONE"], os.environ["INSTANCE"]
TARGET = os.environ["TARGET"].rstrip("/")
API = f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/instances/{INSTANCE}"
METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"


def _token() -> str:
    req = urllib.request.Request(METADATA, headers={"Metadata-Flavor": "Google"})
    return json.load(urllib.request.urlopen(req, timeout=5))["access_token"]


def _vm(method: str = "GET", action: str = "") -> dict:
    req = urllib.request.Request(API + action, method=method, data=b"" if method == "POST" else None,
                                 headers={"Authorization": f"Bearer {_token()}"})
    return json.load(urllib.request.urlopen(req, timeout=10))


def _ready() -> bool:
    """서버 앱이 응답하는지. /idle은 사용으로 치지 않아 이 확인이 VM을 켠 채로 붙잡지 않는다."""
    try:
        return urllib.request.urlopen(TARGET + "/idle", timeout=3).status == 200
    except (urllib.error.URLError, OSError):
        return False


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Tap Me Volleyball</title>
<style>
:root { color-scheme: dark; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0b0a09; color: #ece8e2;
       font-family: system-ui, -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }
main { max-width: 26rem; padding: 0 16px; text-align: center; }
h1 { font-size: 1.25rem; font-weight: 600; margin: 0 0 12px; }
p { color: #a9a39b; line-height: 1.6; margin: 0 0 8px; }
.bar { height: 3px; background: #26231f; margin: 24px 0 8px; overflow: hidden; }
.bar div { height: 100%; width: 30%; background: #f0803c; animation: go 1.4s ease-in-out infinite; }
@keyframes go { from { transform: translateX(-100%); } to { transform: translateX(340%); } }
@media (prefers-reduced-motion: reduce) { .bar div { animation: none; width: 100%; } }
</style></head><body><main>
<h1>서버를 켜는 중이에요</h1>
<p>한동안 쓰는 사람이 없으면 비용을 아끼려고 서버를 꺼 둬요. 보통 1~2분이면 켜지고, 준비되면 바로 넘어가요.</p>
<p lang="en">Starting the server (it sleeps when unused). Usually 1–2 minutes.</p>
<div class="bar" aria-hidden="true"><div></div></div>
<p id="s" aria-live="polite"></p>
</main><script>
const t0 = Date.now();
fetch("/wake", { method: "POST" }).catch(() => {});
async function tick() {
  try {
    const s = await (await fetch("/status")).json();
    if (s.ready) { location.replace(__TARGET__ + "/" + location.search + location.hash); return; }
    document.getElementById("s").textContent = Math.round((Date.now() - t0) / 1000) + "초";
  } catch {}
  setTimeout(tick, 3000);
}
tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, kind: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict) -> None:
        # 서버 화면의 서비스 워커(꺼졌을 때 보이는 깨우기 화면)가 상태를 읽을 수 있게 서버 주소에만 허용한다(10/31)
        self._send(200, json.dumps(data).encode(), "application/json", {"Access-Control-Allow-Origin": TARGET})

    def do_GET(self) -> None:
        if self.path == "/status":
            return self._json({"vm": _vm().get("status"), "ready": _ready()})
        path, _, query = self.path.partition("?")
        # 배너·심사 링크는 highlight.baegugu.com/tapme(하이라이트 기능 중 Tap Me, 10/31 사용자). / 는 지금 기능이 하나라 같게
        if path in ("/", "/tapme"):
            if _ready():  # 켜져 있으면 바로 넘긴다. ?invite=… 은 그대로, #job=… 은 브라우저가 이어 붙인다
                return self._send(302, b"", "text/plain", {"Location": TARGET + "/" + (f"?{query}" if query else "")})
            return self._send(200, PAGE.replace("__TARGET__", json.dumps(TARGET)).encode(), "text/html; charset=utf-8")
        self._send(404, b"", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/wake":
            return self._send(404, b"", "text/plain")
        status = _vm().get("status")
        if status in ("TERMINATED", "STOPPED"):
            _vm("POST", "/start")
        elif status == "SUSPENDED":
            _vm("POST", "/resume")
        self._json({"vm": status})

    def log_message(self, fmt: str, *args: object) -> None:  # 접속 기록은 남기지 않는다
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("", int(os.environ.get("PORT", "8080"))), Handler).serve_forever()
