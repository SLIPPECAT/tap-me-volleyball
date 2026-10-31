// 완료 알림(웹 푸시, 10/31): 랠리를 다 찾거나 하이라이트가 끝나면 서버(tapme/push.py)가 보낸다. 누르면 그 화면을 연다
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
