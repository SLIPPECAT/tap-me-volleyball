// 완료 알림(웹 푸시, 10/31): 랠리를 다 찾거나 하이라이트가 끝나면 서버(tapme/push.py)가 보낸다. 누르면 그 화면을 연다
// 서버 깨우기(10/31): 서버는 30분 쉬면 꺼져, 알림·홈 화면 앱·즐겨찾기로 열면 빈 화면이었다. 화면 열기가 실패하면 서버를 깨우는
// 화면을 대신 보여 주고, 입구(highlight.baegugu.com, 늘 켜져 있음)가 서버를 켜면 다시 연다. 주소가 그대로라 홈 화면 앱
// (사파리와 쿠키가 따로)에서도 들어간 채로 이어진다
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("push", (e) => {
  const d = e.data ? e.data.json() : {};
  e.waitUntil(self.registration.showNotification(d.title || "Tap Me Volleyball", {
    body: d.body || "", tag: d.tag, renotify: true, icon: "/static/icons/icon-192.png", badge: "/static/icons/icon-192.png",
    data: { url: d.url || "/" },
  }));
});

self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  const url = new URL(e.notification.data.url, self.location.origin).href;
  e.waitUntil((async () => {
    const wins = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    for (const w of wins) {  // 열려 있는 화면이 있으면 그 화면을 해당 단계로
      if (w.url.startsWith(self.location.origin)) { await w.focus(); return w.navigate(url); }
    }
    return self.clients.openWindow(url);
  })());
});

const WAKE = "https://highlight.baegugu.com";
const WAKE_PAGE = `<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Tap Me Volleyball</title>
<style>:root { color-scheme: light dark; } body { margin: 0; min-height: 100vh; display: grid; place-items: center;
font-family: system-ui, -apple-system, "Apple SD Gothic Neo", sans-serif; } main { max-width: 26rem; padding: 0 16px; text-align: center; }
h1 { font-size: 1.2rem; } p { opacity: .75; line-height: 1.6; } .bar { height: 3px; background: #8884; overflow: hidden; margin: 20px 0 8px; }
.bar div { height: 100%; width: 30%; background: #3d8ee8; animation: go 1.4s ease-in-out infinite; }
@keyframes go { from { transform: translateX(-100%); } to { transform: translateX(340%); } }</style></head><body><main>
<h1>서버를 켜는 중이에요</h1><p>한동안 쓰는 사람이 없으면 비용을 아끼려고 서버를 꺼 둬요. 보통 1~2분이면 켜지고, 준비되면 바로 이어서 보여 드려요.</p>
<div class="bar" aria-hidden="true"><div></div></div><p id="s" aria-live="polite"></p></main><script>
const t0 = Date.now();
fetch("${WAKE}/wake", { method: "POST", mode: "no-cors" }).catch(() => {});
async function tick() {
  try { const s = await (await fetch("${WAKE}/status")).json(); if (s.ready) { location.reload(); return; } } catch {}
  document.getElementById("s").textContent = Math.round((Date.now() - t0) / 1000) + "초";
  setTimeout(tick, 3000);
}
tick();
</script></body></html>`;

self.addEventListener("fetch", (e) => {
  if (e.request.mode !== "navigate") return;  // 화면 열기만. 올리기·상태 확인 같은 요청은 그대로
  e.respondWith((async () => {
    try {
      const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 10000);
      const r = await fetch(e.request, { signal: ctl.signal });
      clearTimeout(timer);
      if ([502, 503, 504].includes(r.status)) throw new Error("down");  // Caddy는 켜졌는데 앱이 아직
      return r;
    } catch {
      return new Response(WAKE_PAGE, { headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" } });
    }
  })());
});
