"""완료 알림(웹 푸시, 10/31 사용자: 끝나면 알려 주고 다음 할 일을 안내). 랠리를 다 찾았을 때, 하이라이트가 완성되거나
실패했을 때 그 작업을 구독한 기기에 알린다. 안드로이드 크롬은 브라우저에서 바로, 아이폰은 홈 화면에 추가한 앱에서만 된다.

키(VAPID)는 .env에만 둔다(TAPME_VAPID_PRIVATE·TAPME_VAPID_PUBLIC). 없으면 알림 없이 돈다.
구독(기기별 주소·키·화면 언어)은 작업 폴더에 두어 작업과 함께 지워진다(새벽 4시).
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from typing import Any

from tapme import jobs

# 푸시 서비스가 문제가 있을 때 연락할 곳. 운영자 이메일을 노출하지 않게 서비스 주소로
SUBJECT = os.environ.get("TAPME_VAPID_SUB", "https://highlight.baegugu.com")


def public_key() -> str:
    """서버 환경(배포는 systemd가 .env를 넣는다)에서 읽는다. 이 맥은 uvicorn --env-file .env로 띄울 때만."""
    return os.environ.get("TAPME_VAPID_PUBLIC", "")


def _private_key() -> str:
    return os.environ.get("TAPME_VAPID_PRIVATE", "")
_lock = threading.Lock()
# 알림 문구(기기 언어별). 다음에 할 일을 한 줄로 알려 준다
MESSAGES = {
    "rallies": {"ko": ("랠리 {n}개를 찾았어요", "랠리를 골라 나를 탭하면 하이라이트를 만들어요"),
                "ja": ("ラリーが{n}件見つかりました", "ラリーを選んで自分をタップするとハイライトを作ります"),
                "en": ("Found {n} rallies", "Pick a rally and tap yourself to make a highlight")},
    "no_rallies": {"ko": ("랠리를 찾지 못했어요", "다른 구간이나 영상으로 다시 해 보세요"),
                   "ja": ("ラリーが見つかりませんでした", "別の区間や動画で試してください"),
                   "en": ("No rallies found", "Try another part of the video or another video")},
    "done": {"ko": ("하이라이트가 완성됐어요", "열어서 저장하거나 공유해 보세요{more}"),
             "ja": ("ハイライトができました", "開いて保存・共有してください{more}"),
             "en": ("Your highlight is ready", "Open it to save or share{more}")},
    "more": {"ko": " (나머지 {left}개도 만드는 중)", "ja": "（残り{left}件も作成中）", "en": " ({left} more on the way)"},
    "auto_done": {"ko": ("하이라이트 {n}개가 완성됐어요", "열어서 한 번에 저장해 보세요{more}"),
                  "ja": ("ハイライトが{n}本できました", "開いてまとめて保存してください{more}"),
                  "en": ("{n} highlights are ready", "Open them and save them all at once{more}")},
    "auto_more": {"ko": " (나인지 확인할 장면 {review}개)", "ja": "（本人か確認するシーン{review}件）", "en": " ({review} scenes to check)"},
    "auto_none": {"ko": ("나를 찾은 장면이 없어요", "열어서 직접 고르거나 다른 장면에서 나를 다시 탭해 주세요"),
                  "ja": ("自分が映るシーンが見つかりませんでした", "開いて自分で選ぶか、別の場面で自分をタップし直してください"),
                  "en": ("Couldn't find you in any scene", "Open it to pick scenes yourself or tap yourself in another moment")},
    "failed": {"ko": ("하이라이트를 만들지 못했어요", "열어서 다시 시도해 주세요"),
               "ja": ("ハイライトを作れませんでした", "開いてもう一度お試しください"),
               "en": ("Couldn't make the highlight", "Open it and try again")},
}


# 브라우저가 주는 구독 주소는 이 푸시 서비스들뿐이다. 그 밖의 주소로는 보내지 않는다(서버가 아무 주소로나 요청을 보내는 데
# 쓰이지 않게, 10/31 보안 검토). 크롬·안드로이드(FCM), 사파리·아이폰(애플), 파이어폭스(모질라), 엣지(윈도)
PUSH_HOSTS = ("fcm.googleapis.com", "push.apple.com", "push.services.mozilla.com", "notify.windows.com")


def allowed_endpoint(url: str) -> bool:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and any(host == h or host.endswith("." + h) for h in PUSH_HOSTS)


def enabled() -> bool:
    return bool(public_key() and _private_key())


def _path(job_id: str) -> Any:
    return jobs.JOBS_DIR / job_id / "push.json"


def subscribe(job_id: str, sub: dict[str, Any], lang: str = "ko") -> None:
    """이 작업이 끝나면 알릴 기기를 더한다(같은 기기는 한 번만). lang: 알림 문구 언어(그 기기의 화면 언어)."""
    with _lock:
        path = _path(job_id)
        subs = json.loads(path.read_text()) if path.exists() else []
        endpoint = sub.get("endpoint")
        subs = [s for s in subs if s["subscription"].get("endpoint") != endpoint]
        subs.append({"subscription": sub, "lang": lang if lang in ("ko", "ja", "en") else "en"})
        path.write_text(json.dumps(subs))


def message(kind: str, lang: str, **vars: Any) -> tuple[str, str]:
    title, body = MESSAGES[kind][lang]
    if kind == "done":
        vars["more"] = MESSAGES["more"][lang].format(**vars) if vars.get("left") else ""
    if kind == "auto_done":
        vars["more"] = MESSAGES["auto_more"][lang].format(**vars) if vars.get("review") else ""
    return title.format(**vars), body.format(**vars)


def _send_all(job_id: str, kind: str, url: str, vars: dict[str, Any]) -> None:
    from pywebpush import WebPushException, webpush

    path = _path(job_id)
    with _lock:
        subs = json.loads(path.read_text()) if path.exists() else []
    gone = []
    for s in subs:
        title, body = message(kind, s["lang"], **vars)
        payload = {"title": title, "body": body, "url": url, "tag": f"job-{job_id}"}  # 같은 작업 알림은 하나만 남게
        try:
            webpush(subscription_info=s["subscription"], data=json.dumps(payload, ensure_ascii=False),
                    vapid_private_key=_private_key(), vapid_claims={"sub": SUBJECT}, ttl=6 * 3600)
        except WebPushException as e:
            if e.response is not None and e.response.status_code in (404, 410):
                gone.append(s["subscription"].get("endpoint"))  # 구독이 끝난 기기
            else:
                traceback.print_exc()
    if gone:
        with _lock:
            subs = json.loads(path.read_text()) if path.exists() else []
            path.write_text(json.dumps([s for s in subs if s["subscription"].get("endpoint") not in gone]))


def send_test(sub: dict[str, Any], lang: str) -> None:
    """알림 시험 페이지(static/lab/pushtest.html): 그 기기에만 시험 알림 하나. 비용 없음."""
    from pywebpush import webpush

    title, body = {"ko": ("알림 시험", "이렇게 끝났을 때 알려 드려요. 누르면 앱이 열려요"),
                   "ja": ("通知テスト", "終わったらこのようにお知らせします。押すとアプリが開きます"),
                   "en": ("Notification test", "This is how we'll tell you it's done. Tap to open the app")}.get(lang, ("Test", ""))
    webpush(subscription_info=sub, data=json.dumps({"title": title, "body": body, "url": "/", "tag": "test"}, ensure_ascii=False),
            vapid_private_key=_private_key(), vapid_claims={"sub": SUBJECT}, ttl=600)


def notify(job_id: str, kind: str, url: str, **vars: Any) -> None:
    """알림을 뒤에서 보낸다(느린 푸시 서비스가 처리를 막지 않게). kind는 MESSAGES의 키."""
    if not enabled() or not _path(job_id).exists():
        return
    threading.Thread(target=_send_all, args=(job_id, kind, url, vars), daemon=True).start()
