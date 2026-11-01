from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tapme import access, gate, jobs, member, render, server


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(server, "client_factory", lambda: None)

    def fake_find(job: jobs.Job, client) -> None:
        job.rallies = [{"index": 0, "start_sec": 1.0, "end_sec": 9.0, "is_rally": True, "confidence": 0.9,
                        "reason": "serve", "clip": "x", "clip_start_sec": 0.0, "usd": 0.006}]
        job.status = "rallies_ready"
        job.save()

    monkeypatch.setattr(jobs, "find_rallies", fake_find)
    return TestClient(server.app)


def _send(client: TestClient, code: str = "", data: bytes = b"v") -> Any:
    """배포 서버의 올리기(이어 올리기): 자리 열기 → 조각 → 끝내기. 처음 거절되면 그 응답, 아니면 끝내기 응답."""
    r = client.post("/uploads", json={"filename": "m.mp4", "size": len(data), "code": code})
    if r.status_code != 200:
        return r
    up = r.json()["id"]
    client.put(f"/uploads/{up}?offset=0", content=data)
    return client.post(f"/uploads/{up}/finish")


def test_upload_lists_rally_candidates(client: TestClient) -> None:
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}).json()["id"]
    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "rallies_ready"
    assert body["rallies"][0] == {"index": 0, "start_sec": 1.0, "end_sec": 9.0, "point_sec": None, "is_rally": True,
                                  "confidence": 0.9, "reason": "serve",
                                  "uncertain": None}
    assert "clip" not in body["rallies"][0]  # 서버 안 경로는 내보내지 않는다


def test_upload_keeps_the_picked_scan_range(client: TestClient) -> None:
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}, data={"start": "1500", "end": "3300"}).json()["id"]
    assert client.get(f"/jobs/{job_id}").json()["scan_range"] == [1500.0, 3300.0]


@pytest.mark.parametrize("bad", [{"privacy": "none"}, {"zoom": 3.0}, {"languages": ["zh"]}, {"role": "captain"}])
def test_invalid_options_are_rejected(client: TestClient, bad: dict) -> None:
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}).json()["id"]
    r = client.post(f"/jobs/{job_id}/rallies/0/highlight",
                    json={"taps": [{"which": "score", "x": 1, "y": 1}], "team": "검정", **bad})
    assert r.status_code == 422


def test_tap_before_prepare_is_refused(client: TestClient) -> None:
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}).json()["id"]
    r = client.post(f"/jobs/{job_id}/rallies/0/highlight", json={"taps": [{"which": "score", "x": 1, "y": 1}],
                                                                  "team": "검정"})
    assert r.status_code == 409


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/jobs/nope123").status_code == 404
    assert client.get("/jobs/..%2F..%2Fetc").status_code == 404


def test_index_page_is_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Tap Me Volleyball" in r.text


def test_pages_are_cross_origin_isolated_for_browser_blur(client: TestClient) -> None:
    for path in ("/", "/static/lab/blurtest.html", "/static/browser-blur.js"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.headers["cross-origin-opener-policy"] == "same-origin"
        assert r.headers["cross-origin-embedder-policy"] == "require-corp"
        assert r.headers["cache-control"] == "no-cache"  # 배포 뒤 예전 화면 파일을 쓰지 않게


def test_upload_streams_to_disk_with_safe_suffix(client: TestClient, tmp_path: Path) -> None:
    job_id = client.post("/jobs", files={"file": ("../../evil.sh", b"video")}).json()["id"]
    assert [p.name for p in (tmp_path / job_id).glob("source*")] == ["source.mp4"]


def test_reel_only_from_finished_scenes(client: TestClient) -> None:
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}).json()["id"]
    assert client.post(f"/jobs/{job_id}/reel", json={"order": [0]}).status_code == 409
    assert client.post(f"/jobs/{job_id}/reel", json={"order": []}).status_code == 422
    assert client.get(f"/jobs/{job_id}/reel/video").status_code == 404
    assert client.get(f"/jobs/{job_id}").json()["reel"] == {}


def test_prepare_twice_is_refused(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_prepare(job: jobs.Job, index: int, c) -> None:
        calls.append(index)
        job.scenes[str(index)] = {"index": index, "status": "awaiting_tap"}
        job.save()

    monkeypatch.setattr(jobs, "prepare_scene", fake_prepare)
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}).json()["id"]
    assert client.post(f"/jobs/{job_id}/rallies/0/prepare").status_code == 200
    assert client.post(f"/jobs/{job_id}/rallies/0/prepare").status_code == 409
    assert calls == [0]


def test_home_screen_files_are_served(client: TestClient) -> None:
    manifest = client.get("/static/manifest.json").json()
    assert manifest["display"] == "standalone"
    for icon in manifest["icons"]:
        assert client.get(icon["src"]).headers["content-type"] == "image/png"


def _awaiting_tap(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> str:
    def fake_prepare(job: jobs.Job, index: int, c) -> None:
        job.scenes[str(index)] = {"index": index, "status": "awaiting_tap", "tap_frames": {"score": 4, "start": 0}}
        job.save()

    monkeypatch.setattr(jobs, "prepare_scene", fake_prepare)
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}).json()["id"]
    client.post(f"/jobs/{job_id}/rallies/0/prepare")
    return job_id


def test_highlight_sends_taps_from_any_scenes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _awaiting_tap(client, monkeypatch)
    got: list = []
    def fake_make(job, i, taps, *a):
        got.append(taps)
        job.scenes[str(i)]["status"] = "done"

    monkeypatch.setattr(jobs, "make_highlight", fake_make)
    url = f"/jobs/{job_id}/rallies/0/highlight"
    assert client.post(url, json={"taps": [{"which": "b2", "x": 1, "y": 2}], "team": "검정"}).status_code == 200
    assert client.post(url, json={"taps": [{"which": "score", "player": 2}, {"which": "start", "x": 3, "y": 4}],
                                  "team": "검정"}).status_code == 200
    assert got == [[("b2", 1, 2)], [("score", 2), ("start", 3, 4)]]
    for bad in ([], [{"which": "../x", "x": 1, "y": 1}], [{"which": "score"}]):
        assert client.post(url, json={"taps": bad, "team": "검정"}).status_code == 422
    # 팀별 모자이크(11/1): 정해진 값만
    assert client.post(url, json={"taps": [{"which": "b2", "x": 1, "y": 2}], "team": "검정",
                                  "mosaic_team": "everyone"}).status_code == 422


def test_tap_frame_kind_is_checked(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _awaiting_tap(client, monkeypatch)
    assert client.get(f"/jobs/{job_id}/rallies/0/tap-frame?which=../../x").status_code == 422


def test_rally_thumb_and_preview_are_served(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}).json()["id"]
    thumb, preview = tmp_path / "t.jpg", tmp_path / "p.mp4"
    thumb.write_bytes(b"jpg")
    preview.write_bytes(b"mp4")
    monkeypatch.setattr(jobs, "make_preview", lambda job, i: (thumb, preview))
    assert client.get(f"/jobs/{job_id}/rallies/0/thumb").headers["content-type"] == "image/jpeg"
    assert client.get(f"/jobs/{job_id}/rallies/0/preview").headers["content-type"] == "video/mp4"
    assert client.get(f"/jobs/{job_id}/rallies/9/thumb").status_code == 404


def test_players_are_found_when_a_scene_is_opened(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _awaiting_tap(client, monkeypatch)
    monkeypatch.setattr(jobs, "player_cards", lambda job, i, which, c: [[1, 2, 3, 4]])
    assert client.post(f"/jobs/{job_id}/rallies/0/players?which=b1").json() == {"which": "b1", "players": [[1, 2, 3, 4]]}
    assert client.post(f"/jobs/{job_id}/rallies/0/players?which=zz").status_code == 422


def test_player_card_needs_a_listed_player(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _awaiting_tap(client, monkeypatch)
    assert client.get(f"/jobs/{job_id}/rallies/0/player?which=score&n=0").status_code == 404  # 목록이 비어 있음
    assert client.get(f"/jobs/{job_id}/rallies/0/player?which=../x&n=0").status_code == 422


def test_option_screen_preselects_the_server_default_zoom() -> None:
    # 기본 확대를 바꿀 때 화면과 서버 기본값이 어긋나지 않게(10/30 2배 → 1배)
    import re

    html = (Path(server.__file__).parent / "static" / "index.html").read_text()
    picked = re.search(r'<option value="([\d.]+)" selected>[\d.]+×</option>', html)
    assert picked and float(picked.group(1)) == render.DEFAULT_ZOOM == 1.0


def test_upload_keeps_the_match_name_for_saved_file_names(client: TestClient) -> None:
    # 10/30 사용자 요청: 저장 파일 이름 = 경기이름_팀이름_하이라이트_01.mp4. 비우면 올린 파일 이름
    named = client.post("/jobs", files={"file": ("IMG_0148.mov", b"v")}, data={"match": " 10/30 정모 2경기 "}).json()["id"]
    unnamed = client.post("/jobs", files={"file": ("IMG_0148.mov", b"v")}).json()["id"]
    assert client.get(f"/jobs/{named}").json()["match_name"] == "10/30 정모 2경기"
    assert client.get(f"/jobs/{unnamed}").json()["match_name"] == "IMG_0148"


def test_preparing_ahead_runs_only_when_free_and_idle(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 미리보기를 열면 뒤에서 장면을 미리 준비한다. 비용이 드는 랠리(득점 순간을 모름)나 다른 단계가 돌 때는 하지 않는다
    calls = []

    def fake_prepare(job: jobs.Job, index: int, c) -> None:
        calls.append(index)
        job.scenes[str(index)] = {"index": index, "status": "awaiting_tap"}
        job.save()

    monkeypatch.setattr(jobs, "prepare_scene", fake_prepare)
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}).json()["id"]
    assert client.post(f"/jobs/{job_id}/rallies/0/prepare?ahead=true").json()["status"] == "skipped"  # point_sec 없음
    job = jobs.load_job(job_id)
    job.rallies[0]["point_sec"] = 8.0
    job.save()
    with server._step_lock:
        assert client.post(f"/jobs/{job_id}/rallies/0/prepare?ahead=true").json()["status"] == "skipped"  # 바쁨
    assert client.post(f"/jobs/{job_id}/rallies/0/prepare?ahead=true").json()["status"] == "preparing"
    assert client.post(f"/jobs/{job_id}/rallies/0/prepare?ahead=true").json()["status"] == "skipped"  # 이미 있음
    assert calls == [0]


def test_status_shows_scan_progress_while_finding_rallies(client: TestClient) -> None:
    # 10/30: 찾는 동안 화면이 진행(30/84)을 몰라 "0/0"이 뜨고 찾은 랠리 목록도 그리지 못했다
    job_id = client.post("/jobs", files={"file": ("m.mp4", b"video")}).json()["id"]
    job = jobs.load_job(job_id)
    job.status, job.scan = "finding_rallies", {"done": 30, "total": 84}
    job.save()
    assert client.get(f"/jobs/{job_id}").json()["scan"] == {"done": 30, "total": 84}


def test_highlight_is_queued_and_a_failure_is_marked(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/30 일괄 만들기: 예약하면 바로 '대기', 만드는 중 다시 예약은 거절, 실패하면 '실패'로 남겨 화면이 멈추지 않게
    job_id = _awaiting_tap(client, monkeypatch)
    seen: list = []

    def boom(job, i, *a):
        seen.append(jobs.load_job(job.id).scenes[str(i)]["status"])
        raise RuntimeError("SAM 연결 끊김")

    monkeypatch.setattr(jobs, "make_highlight", boom)
    url = f"/jobs/{job_id}/rallies/0/highlight"
    assert client.post(url, json={"taps": [{"which": "score", "x": 1, "y": 2}], "team": "검정"}).json() == {"status": "queued"}
    assert seen == ["queued"]
    body = client.get(f"/jobs/{job_id}").json()
    assert body["scenes"]["0"]["status"] == "failed" and "SAM" in body["error"]
    # 실패한 장면은 다시 예약할 수 있다
    monkeypatch.setattr(jobs, "make_highlight", lambda job, i, *a: job.scenes[str(i)].update(status="done"))
    assert client.post(url, json={"taps": [{"which": "score", "x": 1, "y": 2}], "team": "검정"}).status_code == 200


def test_one_job_queues_at_most_five_scenes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/30 사용자 "제한 같은 걸 둬야": 한 작업에서 대기·만드는 중은 다섯 장면까지
    job_id = _awaiting_tap(client, monkeypatch)
    job = jobs.load_job(job_id)
    for k in range(1, 6):
        job.scenes[str(k)] = {"index": k, "status": "rendering" if k == 1 else "queued"}
    job.save()
    monkeypatch.setattr(jobs, "make_highlight", lambda *a: None)
    r = client.post(f"/jobs/{job_id}/rallies/0/highlight", json={"taps": [{"which": "score", "x": 1, "y": 2}], "team": "검정"})
    assert r.status_code == 429 and "5" in r.json()["detail"]


TAP = {"taps": [{"which": "score", "x": 1, "y": 2}], "team": "검정"}


def test_codes_limit_uploads_and_highlights(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자 결정: 배구구 회원에게 준 코드로만, 코드당 영상 2편 + 하이라이트 3개
    job_id = _awaiting_tap(client, monkeypatch)  # 코드를 켜기 전에 만든 장면 하나
    monkeypatch.setattr(access, "REQUIRED", True)
    code = access.new_codes(1)[0]
    assert client.post("/access", json={"code": "nope"}).json()["denied"]
    assert client.post("/access", json={"code": f"{code[:4]}-{code[4:].lower()}"}).json()["left"] == {"uploads": 2, "highlights": 3}
    assert _send(client).status_code == 403  # 코드 없이
    assert client.post("/jobs", files={"file": ("m.mp4", b"v")}, data={"code": code}).status_code == 410  # 한 번에 올리기는 닫힘
    for _ in range(2):
        assert _send(client, code).status_code == 200
    r = _send(client, code)
    assert r.status_code == 403 and "다 썼어요" in r.json()["detail"]

    job = jobs.load_job(job_id)
    job.code = code
    job.save()
    monkeypatch.setattr(jobs, "make_highlight", lambda job, i, *a: job.scenes[str(i)].update(status="done"))
    url = f"/jobs/{job_id}/rallies/0/highlight"
    assert [client.post(url, json=TAP).status_code for _ in range(4)] == [200, 200, 200, 403]
    assert client.get(f"/jobs/{job_id}").json()["left"] == {"uploads": 0, "highlights": 0}


def test_failed_work_gives_the_count_back(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _awaiting_tap(client, monkeypatch)
    monkeypatch.setattr(access, "REQUIRED", True)
    code = access.new_codes(1)[0]
    job = jobs.load_job(job_id)
    job.code = code
    job.save()

    def boom(*a):
        raise RuntimeError("SAM 연결 끊김")

    monkeypatch.setattr(jobs, "make_highlight", boom)
    client.post(f"/jobs/{job_id}/rallies/0/highlight", json=TAP)
    monkeypatch.setattr(jobs, "find_rallies", boom)
    _send(client, code)
    assert access.left(code) == {"uploads": 2, "highlights": 3}


def test_full_queue_or_spent_budget_turns_new_videos_away(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "MAX_WAITING_SCANS", 0)
    assert "많아요" in client.post("/access", json={}).json()["closed"]
    assert client.post("/jobs", files={"file": ("m.mp4", b"v")}).status_code == 503
    monkeypatch.setattr(jobs, "MAX_WAITING_SCANS", 5)
    monkeypatch.setattr(jobs, "MAX_USD_TOTAL", 0.5)  # 30분 영상 하나(약 $1.1)도 못 훑는다
    assert "비용" in client.post("/access", json={}).json()["closed"]
    monkeypatch.setattr(jobs, "MAX_USD_TOTAL", 100.0)
    monkeypatch.setattr(jobs, "MIN_FREE_GB", 1e9)  # 디스크가 꽉 찼다
    assert client.post("/access", json={}).json()["closed"]
    monkeypatch.setattr(jobs, "MIN_FREE_GB", 0.0)
    monkeypatch.setattr(jobs, "MAX_UPLOADS_PER_DAY", 2)  # 하루 두 편
    codes = [client.post("/jobs", files={"file": ("m.mp4", b"v")}).status_code for _ in range(3)]
    assert codes == [200, 200, 503] and "오늘" in client.post("/access", json={}).json()["closed"]


def test_admin_summary_needs_the_token(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.get("/admin/summary").status_code == 404  # 토큰을 두지 않은 서버
    monkeypatch.setenv("TAPME_ADMIN_TOKEN", "t0ken")
    assert client.get("/admin/summary", headers={"X-Admin-Token": "wrong"}).status_code == 404
    assert client.get("/admin/summary", headers={"X-Admin-Token": "t0ken"}).json()["max_usd"] == jobs.MAX_USD_TOTAL


def test_idle_reports_quiet_time_and_running_work(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 쓸 때만 켜는 서버: 도는 작업이 있으면 오래 조용해도 끄지 않는다
    job_id = _awaiting_tap(client, monkeypatch)
    seen: list = []
    monkeypatch.setattr(jobs, "make_highlight", lambda *a: seen.append(client.get("/idle").json()["busy"]))
    client.post(f"/jobs/{job_id}/rallies/0/highlight", json=TAP)
    assert seen == [True] and client.get("/idle").json()["busy"] is False
    monkeypatch.setattr(server, "_last_request", 0.0)
    assert client.get("/idle").json()["idle_sec"] > 3600  # /idle을 묻는 것은 요청으로 치지 않는다


def test_invite_link_gives_a_code_with_limits(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자 결정: 배구구 배너(고정 주소)로 들어오면 코드를 만들어 준다. 퍼짐은 IP당 하루 한도·총 개수로 막는다
    monkeypatch.setattr(access, "REQUIRED", True)
    monkeypatch.setattr(access, "CLAIMS_PER_IP_PER_DAY", 3)  # 실제는 10(여러 사람이 한 IP를 나눠 쓴다)
    word = access.new_invite("banner", max_codes=4)
    claim = lambda w: client.post("/access/claim", json={"invite": w})
    assert claim("wrong").status_code == 403
    got = [claim(word) for _ in range(4)]
    assert [r.status_code for r in got] == [200, 200, 200, 403]  # 같은 접속에서 하루 한도
    code = got[0].json()["code"]
    assert client.post("/access", json={"code": code}).json()["left"] == {"uploads": 2, "highlights": 3}
    monkeypatch.setattr(server, "_client_ip", lambda request: "203.0.113.9")
    assert claim(word).status_code == 200 and "모두" in claim(word).json()["detail"]  # 총 4개까지
    new_word = access.new_invite("banner")  # 초대말을 바꾸면 예전 말은 막히고 받은 코드는 그대로
    assert claim(word).status_code == 403 and new_word != word
    assert client.post("/access", json={"code": code}).json()["left"]["uploads"] == 2


def test_banner_codes_only_for_baegugu_members(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자: 배너는 모두에게 보이되 쓰는 것은 회원만. 회원당 코드 하나(다른 기기에서 다시 받으면 같은 코드), 심사용은 확인 안 함
    monkeypatch.setattr(access, "REQUIRED", True)
    monkeypatch.setattr(member, "CHECK", True)
    monkeypatch.setattr(access, "CLAIMS_PER_IP_PER_DAY", 1)  # 회원 확인 중엔 IP 한도를 보지 않는다
    who: dict[str, object] = {"id": None}

    def fake(cookies: dict[str, str]) -> str | None:
        if who["id"] == "down":
            raise member.Unavailable("no answer")
        return who["id"]  # type: ignore[return-value]

    monkeypatch.setattr(member, "member_id", fake)
    word, judge = access.new_invite("banner", max_codes=5), access.new_invite("judge")
    claim = lambda w: client.post("/access/claim", json={"invite": w})
    r = claim(word)
    assert r.status_code == 401 and "회원" in r.json()["detail"]  # 로그인 안 함
    who["id"] = "down"
    assert claim(word).status_code == 503  # 배구구가 답이 없으면 회원 아님이 아니라 잠시 뒤
    who["id"] = "u-1"
    first, again = claim(word).json()["code"], claim(word).json()["code"]
    assert first == again  # 같은 회원은 같은 코드
    who["id"] = "u-2"
    assert claim(word).json()["code"] != first  # 같은 IP여도 다른 회원은 받는다
    assert access._load_invites()["banner"]["issued"] == 2
    assert "u-1" not in access._invites_path().read_text()  # 회원 번호는 해시로만
    who["id"] = None
    assert claim(judge).status_code == 200  # 심사위원은 배구구 회원이 아니다


def test_member_check_reads_only_the_login_status(monkeypatch: pytest.MonkeyPatch) -> None:
    # 배구구 /api/auth/me에 로그인 쿠키만 실어 묻고, 회원 번호만 꺼낸다. 쿠키가 없으면 묻지도 않는다.
    # 로그인 안 했거나 만료면 401(쿠키 로그인은 익명으로 통과한 뒤 EntryPoint가 401)
    import io
    import urllib.error

    sent: list[dict[str, str]] = []

    def fake_open(req, timeout):
        sent.append(dict(req.header_items()))
        if "bad" in req.get_header("Cookie"):
            raise urllib.error.HTTPError(req.full_url, 401, "no", {}, None)  # type: ignore[arg-type]
        if "down" in req.get_header("Cookie"):
            raise urllib.error.HTTPError(req.full_url, 502, "no", {}, None)  # type: ignore[arg-type]
        body = {"success": True, "data": {"id": "abc", "email": "x@y"}} if "anon" not in req.get_header("Cookie") else {"success": False}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(member.urllib.request, "urlopen", fake_open)
    assert member.member_id({}) is None and not sent
    assert member.member_id({"baegugu-token": "tok", "other": "x"}) == "abc"
    assert sent[-1]["Cookie"] == "baegugu-token=tok"  # 다른 쿠키는 보내지 않는다
    assert member.member_id({"baegugu-token": "anon"}) is None
    assert member.member_id({"baegugu-token": "bad"}) is None
    with pytest.raises(member.Unavailable):
        member.member_id({"baegugu-token": "down"})  # 배구구 고장은 회원 아님이 아니다


def test_members_only_lock(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자: 외부 노출 없이 배구구에서만. 입장 쿠키가 없으면 화면은 안내, 요청은 401. 배너(회원 확인)나 회원 코드로만 들어온다
    monkeypatch.setattr(access, "REQUIRED", True)
    monkeypatch.setattr(member, "CHECK", True)
    monkeypatch.setattr(gate, "MEMBERS_ONLY", True)
    monkeypatch.setattr(gate, "SECRET", "test-secret")
    monkeypatch.setattr(member, "member_id", lambda cookies: "u-9")
    page = client.get("/", headers={"accept": "text/html"})
    assert page.status_code == 403 and "배구구 회원 전용" in page.text and "noindex" in page.headers["x-robots-tag"]
    assert client.get("/static/guide.html", headers={"accept": "text/html"}).status_code == 403
    assert client.post("/access", json={"code": "x"}).status_code == 401
    assert client.get("/idle").status_code == 200  # 입구가 서버 안에서 묻는다
    assert "Disallow: /" in client.get("/robots.txt").text
    client.cookies.set(gate.COOKIE, gate.make("member", now=0.0))  # 만료된 쿠키
    assert client.post("/access", json={"code": "x"}).status_code == 401
    client.cookies.set(gate.COOKIE, "forged." + "x" * 43)  # 서명이 틀린 쿠키
    assert client.post("/access", json={"code": "x"}).status_code == 401
    client.cookies.clear()
    r = client.post("/access/claim", json={"invite": access.new_invite("banner")})
    assert r.status_code == 200 and gate.COOKIE in r.cookies  # 배너 + 회원 확인 → 입장 쿠키
    client.cookies.set(gate.COOKIE, r.cookies[gate.COOKIE])  # Secure 쿠키라 시험용 http에선 저절로 안 실린다
    assert client.get("/", headers={"accept": "text/html"}).status_code == 200
    code = r.json()["code"]
    client.cookies.clear()
    assert client.post("/access/enter", json={"code": access.new_codes(1)[0]}).status_code == 403  # 회원 확인 없이 만든 코드
    entered = client.post("/access/enter", json={"code": code})
    assert entered.status_code == 200  # 홈 화면 앱: 회원 코드로 들어온다
    client.cookies.set(gate.COOKIE, entered.cookies[gate.COOKIE])
    assert client.get("/", headers={"accept": "text/html"}).status_code == 200
    assert client.post("/access/claim", json={"invite": access.new_invite("judge")}).status_code == 403  # 심사용은 추후 결정


def test_wrong_codes_lock_the_ip_for_an_hour(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자: 틀린 코드를 자꾸 넣으면 막는다. 8자리를 다 넣은 것만 센다(코드 칸은 칠 때마다 확인해 입력 중은 세지 않는다)
    monkeypatch.setattr(access, "REQUIRED", True)
    monkeypatch.setattr(server, "BAD_CODES_PER_HOUR", 3)
    monkeypatch.setattr(server, "_bad_codes", {})
    good = access.new_codes(1)[0]
    check = lambda c: client.post("/access", json={"code": c})
    for partial in ("A", "AB", "ABC", "ABCD", "ABCDE", "ABCDEF", "ABCDEFG"):
        assert check(partial).status_code == 200  # 입력 중
    assert check(good).status_code == 200
    assert [check("ZZZZ-ZZZ" + str(k)).status_code for k in range(2, 6)] == [200, 200, 200, 429]
    assert check(good).status_code == 429  # 막힌 동안은 맞는 코드도(같은 주소)
    assert client.post("/access/enter", json={"code": good}).status_code == 429
    monkeypatch.setattr(server, "_client_ip", lambda request: "203.0.113.7")
    assert check(good).status_code == 200  # 다른 주소는 그대로


def test_a_member_cannot_open_another_members_job(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자: 특정 사용자가 다른 사람의 영상에 접근할 수 없어야. 작업·이어 올리기는 만든 코드의 입장 쿠키로만, 남의 것은 404
    monkeypatch.setattr(access, "REQUIRED", True)
    monkeypatch.setattr(member, "CHECK", True)
    monkeypatch.setattr(gate, "MEMBERS_ONLY", True)
    monkeypatch.setattr(gate, "SECRET", "test-secret")
    word, who = access.new_invite("banner"), {"id": "a"}
    monkeypatch.setattr(member, "member_id", lambda cookies: who["id"])

    def enter(member_id: str) -> tuple[str, str]:
        who["id"] = member_id
        client.cookies.clear()
        r = client.post("/access/claim", json={"invite": word})
        return r.json()["code"], r.cookies[gate.COOKIE]  # Secure 쿠키라 시험용 http에선 직접 싣는다

    code_a, cookie_a = enter("a")
    code_b, cookie_b = enter("b")
    client.cookies.set(gate.COOKIE, cookie_a)
    up = client.post("/uploads", json={"filename": "m.mp4", "size": 1, "code": code_a}).json()["id"]
    client.put(f"/uploads/{up}?offset=0", content=b"v")
    job_id = client.post(f"/uploads/{up}/finish").json()["id"]
    assert client.get(f"/jobs/{job_id}").status_code == 200
    client.cookies.set(gate.COOKIE, cookie_b)  # 다른 회원
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert client.get(f"/jobs/{job_id}/rallies/0/thumb").status_code == 404
    assert client.post(f"/jobs/{job_id}/reel", json={"order": [0]}).status_code == 404
    assert client.get(f"/uploads/{up}").status_code == 404
    assert client.post("/uploads", json={"filename": "m.mp4", "size": 1, "code": code_a}).status_code == 403  # 남의 코드로
    assert client.post("/uploads", json={"filename": "m.mp4", "size": 1, "code": code_b}).status_code == 200
    client.cookies.set(gate.COOKIE, gate.make("member"))  # 주인 정보가 없는 예전 쿠키
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_judges_get_in_through_their_own_link(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자: 심사위원 접근 경로. 심사용 초대 링크는 배구구 회원 확인 없이 입장 쿠키·코드를 주고, 자기 작업만 연다
    monkeypatch.setattr(access, "REQUIRED", True)
    monkeypatch.setattr(member, "CHECK", True)
    monkeypatch.setattr(gate, "MEMBERS_ONLY", True)
    monkeypatch.setattr(gate, "JUDGE_ACCESS", True)
    monkeypatch.setattr(gate, "SECRET", "test-secret")
    monkeypatch.setattr(member, "member_id", lambda cookies: None)  # 배구구 회원이 아니다
    banner, judge = access.new_invite("banner"), access.new_invite("judge")
    assert client.post("/access/claim", json={"invite": banner}).status_code == 401  # 배너는 여전히 회원만
    r = client.post("/access/claim", json={"invite": judge})
    assert r.status_code == 200 and gate.COOKIE in r.cookies
    code = r.json()["code"]
    client.cookies.set(gate.COOKIE, r.cookies[gate.COOKIE])
    assert client.get("/", headers={"accept": "text/html"}).status_code == 200
    up = client.post("/uploads", json={"filename": "m.mp4", "size": 1, "code": code})
    assert up.status_code == 200  # 자기 코드로 올린다
    client.cookies.clear()
    entered = client.post("/access/enter", json={"code": code})  # 홈 화면 앱처럼 코드로 들어오기
    assert entered.status_code == 200


def test_one_shot_finds_me_makes_highlights_and_reel(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자: 처음에 영상·나·설정을 다 받아 맡기면 끝. 확실하면 만들고(0.7 이상), 애매하면 확인 묶음, 없으면 건너뛴다(H-46)
    import time as _time

    from tapme import auto, push

    def four_rallies(job: jobs.Job, client) -> None:
        job.rallies = [{"index": i, "start_sec": 10.0 * i, "end_sec": 10.0 * i + 8, "point_sec": 10.0 * i + 7, "is_rally": True}
                       for i in range(4)]
        job.status = "rallies_ready"
        job.save()

    monkeypatch.setattr(jobs, "find_rallies", four_rallies)
    monkeypatch.setattr(auto, "make_reference", lambda job, t, x, y, c: job.dir / "ref.jpg")
    answers = {0: {"frame": "score", "card": 2, "confidence": 0.9}, 1: {"frame": "b2", "card": 1, "confidence": 0.5},
               2: {"frame": "score", "card": -1, "confidence": 0.9}, 3: {"frame": "start", "card": 0, "confidence": 0.95}}
    monkeypatch.setattr(auto, "find_me", lambda job, i, ref, hint, c: answers[i])
    monkeypatch.setattr(jobs, "prepare_scene", lambda job, i, c: job.scenes.__setitem__(str(i), {"index": i, "status": "awaiting_tap"}))
    rendered: list[tuple[int, int, str]] = []

    def fake_render(job_id: str, i: int, frame: str, card: int, a: dict, c) -> None:
        rendered.append((i, frame, card, a["team"]))
        job = jobs.load_job(job_id)
        job.scenes[str(i)]["status"] = "done"
        job.save()

    monkeypatch.setattr(server, "_auto_render", fake_render)
    monkeypatch.setattr(jobs, "make_reel", lambda job, order: job.reel.update({"status": "done", "order": order}))
    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(push, "notify", lambda job_id, kind, url, **v: sent.append((kind, v)))

    job_id = _send(client).json()["id"]
    r = client.post(f"/jobs/{job_id}/auto", json={"t": 3.0, "x": 100, "y": 200, "hint": "초록 10번", "team": "초록"})
    assert r.status_code == 200
    for _ in range(100):
        if client.get(f"/jobs/{job_id}").json()["auto"].get("state") == "done":
            break
        _time.sleep(0.05)
    a = client.get(f"/jobs/{job_id}").json()["auto"]
    # 나는 모든 랠리에 있다(11/1 사용자): 못 찾은 랠리(-1)도 건너뛰지 않고 확인 묶음으로. 고른 장면 종류로 만든다
    assert a["state"] == "done" and a["made"] == [0, 3] and a["review"] == [1, 2] and "me" not in a
    assert rendered == [(0, "score", 2, "초록"), (3, "start", 0, "초록")]
    assert client.get(f"/jobs/{job_id}").json()["reel"]["order"] == [0, 3]  # 둘 이상이면 모음도
    # 장면마다가 아니라 끝날 때 한 번(이 시험은 찾기가 맡기기보다 먼저 끝나 랠리 알림이 앞에 하나 있다)
    assert [k for k, _ in sent if k != "rallies"] == ["auto_done"] and sent[-1] == ("auto_done", {"n": 2, "review": 2})
    assert client.post(f"/jobs/{job_id}/auto", json={"t": 1, "x": 1, "y": 1, "team": "a"}).status_code == 409  # 한 번만
    long_id = _send(client).json()["id"]
    job = jobs.load_job(long_id)
    job.scan_range = [0.0, 25 * 60.0]
    job.save()
    assert client.post(f"/jobs/{long_id}/auto", json={"t": 1, "x": 1, "y": 1, "team": "a"}).status_code == 400  # 20분까지


def test_one_shot_starts_when_settings_arrive_during_the_scan_and_resumes_after_restart(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 아이폰: 찾기는 올리자마자 시작돼 맡기기 설정(몇 초 뒤 저장)을 모른 채 끝나 멈췄다 → 끝날 때 디스크에서 다시 본다.
    # 서버가 다시 켜지면(배포) 멈춘 맡기기를 잇되, 찾은 결과·예약한 장면은 다시 하지 않는다
    import time as _time

    from tapme import auto, push

    settings = {"state": "waiting", "me": {"t": 1, "x": 1, "y": 1}, "hint": "", "team": "초록", "options": {}, "max": 8,
                "found": {}, "made": [], "review": []}

    def scan_while_settings_arrive(job: jobs.Job, client) -> None:
        other = jobs.load_job(job.id)  # 화면이 맡기기를 보낸 것(찾기 중에)
        other.auto = dict(settings)
        other.save()
        job.rallies = [{"index": i, "start_sec": 10.0 * i, "end_sec": 10.0 * i + 8, "point_sec": 10.0 * i + 7, "is_rally": True}
                       for i in range(2)]
        job.status = "rallies_ready"
        job.save()

    monkeypatch.setattr(jobs, "find_rallies", scan_while_settings_arrive)
    monkeypatch.setattr(auto, "make_reference", lambda job, t, x, y, c: job.dir / "ref.jpg")
    asked: list[int] = []

    def find_me(job, i, ref, hint, c):
        asked.append(i)
        return {"card": 0, "confidence": 0.9}

    monkeypatch.setattr(auto, "find_me", find_me)
    monkeypatch.setattr(jobs, "prepare_scene", lambda job, i, c: job.scenes.__setitem__(str(i), {"index": i, "status": "awaiting_tap"}))

    def fake_render(job_id: str, i: int, frame: str, card: int, a: dict, c) -> None:
        job = jobs.load_job(job_id)
        job.scenes[str(i)]["status"] = "done"
        job.save()

    monkeypatch.setattr(server, "_auto_render", fake_render)
    monkeypatch.setattr(jobs, "make_reel", lambda job, order: job.reel.update({"status": "done", "order": order}))
    monkeypatch.setattr(push, "notify", lambda *a, **k: None)

    def wait_done(job_id: str) -> dict:
        for _ in range(100):
            a = jobs.load_job(job_id).auto
            if a.get("state") == "done":
                return a
            _time.sleep(0.05)
        return jobs.load_job(job_id).auto

    job_id = _send(client).json()["id"]
    assert wait_done(job_id)["made"] == [0, 1] and asked == [0, 1]
    # 서버가 다시 켜진 경우: 랠리 0은 찾고 만든 뒤, 1은 찾기만 하고 멈췄다
    job = jobs.load_job(job_id)
    job.auto = {**job.auto, "state": "finding", "made": [0], "found": {"0": {"card": 0, "confidence": 0.9},
                                                                      "1": {"card": 2, "confidence": 0.8}}}
    job.scenes["1"]["status"] = "awaiting_tap"
    job.save()
    asked.clear()
    server._resume_autos()
    a = wait_done(job_id)
    assert a["made"] == [0, 1] and asked == []  # 찾은 것은 다시 묻지 않고, 만든 것은 다시 만들지 않는다


def test_judge_codes_skip_the_daily_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(access, "REQUIRED", True)
    monkeypatch.setattr(jobs, "MAX_UPLOADS_PER_DAY", 1)
    member, judge = access.new_codes(1)[0], client.post("/access/claim", json={"invite": access.new_invite("judge")}).json()["code"]
    assert _send(client, member).status_code == 200
    assert _send(client, member).status_code == 503  # 하루 1편 찼다
    assert client.post("/access", json={"code": judge}).json()["closed"] is None
    assert _send(client, judge).status_code == 200


def test_client_ip_trusts_forwarded_header_only_from_the_local_proxy() -> None:
    from types import SimpleNamespace

    proxied = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={"x-forwarded-for": "198.51.100.7, 10.0.0.1"})
    direct = SimpleNamespace(client=SimpleNamespace(host="198.51.100.8"), headers={"x-forwarded-for": "1.2.3.4"})
    assert (server._client_ip(proxied), server._client_ip(direct)) == ("198.51.100.7", "198.51.100.8")


def test_browser_job_asks_each_uploaded_chunk_and_finishes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 H-45: 원본은 기기에 두고, 기기가 흐린 1분 조각만 올린다. 서버는 조각마다 Muse에 묻고 랠리를 붙인다
    from tapme import muse, scan

    asked: list = []

    def fake_ask(c, clip, prompt):
        asked.append(clip.read_bytes())
        return muse.MuseReply(text='{"rallies": [{"start_sec": 2, "point_sec": 9, "confidence": 0.9}]}',
                              input_tokens=1, output_tokens=1, usd=0.01, seconds=0)

    monkeypatch.setattr(muse, "ask_about_video", fake_ask)
    r = client.post("/jobs/browser", json={"filename": "경기.mp4", "duration": 4000, "start": 1200, "end": 1300})
    body = r.json()
    assert body["chunks"] == [[1200.0, 60.0], [1250.0, 50.0]] and len(scan.chunk_plan(1200, 100)) == 2
    job_id = body["id"]
    assert client.get(f"/jobs/{job_id}/rallies/0/thumb").status_code == 404  # 랠리 전
    assert client.post(f"/jobs/{job_id}/chunks/0", files={"file": ("c.mp4", b"blur0")}).json() == {"status": "received"}
    assert client.post(f"/jobs/{job_id}/chunks/0", files={"file": ("c.mp4", b"blur0")}).status_code == 409  # 두 번 묻지 않는다
    s = client.get(f"/jobs/{job_id}").json()
    assert (s["status"], s["scan"], [x["point_sec"] for x in s["rallies"]]) == ("finding_rallies", {"done": 1, "total": 2}, [1209.0])
    assert client.get(f"/jobs/{job_id}/rallies/0/thumb").status_code == 404  # 미리보기는 기기에서
    client.post(f"/jobs/{job_id}/chunks/1", files={"file": ("c.mp4", b"blur1")})
    s = client.get(f"/jobs/{job_id}").json()
    assert s["status"] == "rallies_ready" and [x["point_sec"] for x in s["rallies"]] == [1209.0, 1259.0]
    assert asked == [b"blur0", b"blur1"] and s["spent_usd"] == pytest.approx(0.02) and s["mode"] == "browser"
    assert not list((jobs.JOBS_DIR / job_id / "rallies").glob("chunk*_blur.mp4"))  # 물은 조각은 지운다
    assert client.post(f"/jobs/{job_id}/chunks/5", files={"file": ("c.mp4", b"x")}).status_code == 409


def test_browser_job_needs_the_rally_segment_before_preparing(client: TestClient, tmp_path: Path) -> None:
    import subprocess

    job_id = client.post("/jobs/browser", json={"filename": "a.mp4", "duration": 30}).json()["id"]
    job = jobs.load_job(job_id)
    job.rallies = [{"index": 0, "start_sec": 12.0, "point_sec": 17.0, "end_sec": 18.0}]
    job.save()
    assert client.post(f"/jobs/{job_id}/rallies/0/prepare?ahead=true").json() == {"status": "skipped"}
    assert client.post(f"/jobs/{job_id}/rallies/0/prepare").status_code == 409  # 구간을 먼저
    seg = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=16",
                    "-c:v", "libx264", "-g", "30", "-movflags", "frag_keyframe+empty_moov", str(seg)], check=True)
    r = client.post(f"/jobs/{job_id}/rallies/0/segment", data={"start": "9"}, files={"video": ("v.mp4", seg.read_bytes())})
    assert r.json() == {"status": "ok"} and client.get(f"/jobs/{job_id}").json()["segments"] == ["0"]
    r = client.post(f"/jobs/{job_id}/rallies/0/segment", data={"start": "14"}, files={"video": ("v.mp4", seg.read_bytes())})
    assert r.status_code == 422  # 장면(득점 7초 전~5초 뒤)을 못 덮는 구간
    assert (jobs.JOBS_DIR / job_id / "segment00.mp4").exists()  # 잘못 올려도 받아 둔 구간은 그대로


def test_push_notifies_subscribers_in_their_language(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자: 끝나면 알려 주고 다음 할 일을 안내. 랠리를 다 찾았을 때·하이라이트가 끝났을 때
    import pywebpush

    from tapme import muse, push

    sent: list = []
    monkeypatch.setattr(pywebpush, "webpush", lambda subscription_info, data, **k: sent.append((subscription_info["endpoint"], json.loads(data))))
    monkeypatch.setattr(push.threading, "Thread", lambda target, args, daemon: type("T", (), {"start": lambda self: target(*args)})())
    assert client.get("/push/key").json() == {"key": None}  # 키가 없는 서버는 알림 없이
    monkeypatch.setenv("TAPME_VAPID_PUBLIC", "pub")
    monkeypatch.setenv("TAPME_VAPID_PRIVATE", "priv")
    monkeypatch.setattr(muse, "ask_about_video", lambda c, clip, prompt: muse.MuseReply(
        text='{"rallies": [{"start_sec": 2, "point_sec": 9, "confidence": 0.9}]}', input_tokens=1, output_tokens=1, usd=0.01, seconds=0))
    job_id = client.post("/jobs/browser", json={"filename": "a.mp4", "duration": 50}).json()["id"]
    assert client.post(f"/jobs/{job_id}/push", json={"subscription": {"endpoint": "http://x"}}).status_code == 422
    for ep, lang in (("https://fcm.googleapis.com/fcm/send/ko", "ko"), ("https://fcm.googleapis.com/fcm/send/ja", "ja")):
        assert client.post(f"/jobs/{job_id}/push", json={"subscription": {"endpoint": ep, "keys": {}}, "lang": lang}).json() == {"status": "ok"}
    client.post(f"/jobs/{job_id}/chunks/0", files={"file": ("c.mp4", b"blur")})  # 하나뿐인 조각 → 랠리 찾기 끝
    assert [(ep, d["title"], d["url"], d["tag"]) for ep, d in sent] == [
        ("https://fcm.googleapis.com/fcm/send/ko", "랠리 1개를 찾았어요", f"/#job={job_id}", f"job-{job_id}"),
        ("https://fcm.googleapis.com/fcm/send/ja", "ラリーが1件見つかりました", f"/#job={job_id}", f"job-{job_id}")]
    assert push.message("done", "ko", left=2)[1] == "열어서 저장하거나 공유해 보세요 (나머지 2개도 만드는 중)"


def test_service_worker_is_served_from_the_root(client: TestClient) -> None:
    r = client.get("/sw.js")
    assert r.status_code == 200 and "notificationclick" in r.text and r.headers["cache-control"] == "no-cache"


def test_push_tells_when_a_highlight_is_done_or_failed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import pywebpush

    from tapme import push

    sent: list = []
    monkeypatch.setattr(pywebpush, "webpush", lambda subscription_info, data, **k: sent.append(json.loads(data)))
    monkeypatch.setattr(push.threading, "Thread", lambda target, args, daemon: type("T", (), {"start": lambda self: target(*args)})())
    monkeypatch.setenv("TAPME_VAPID_PUBLIC", "pub")
    monkeypatch.setenv("TAPME_VAPID_PRIVATE", "priv")
    job_id = _awaiting_tap(client, monkeypatch)
    client.post(f"/jobs/{job_id}/push", json={"subscription": {"endpoint": "https://fcm.googleapis.com/fcm/send/en"}, "lang": "en"})
    monkeypatch.setattr(jobs, "make_highlight", lambda job, i, *a: job.scenes[str(i)].update(status="done"))
    client.post(f"/jobs/{job_id}/rallies/0/highlight", json=TAP)

    def boom(*a):
        raise RuntimeError("SAM 연결 끊김")

    monkeypatch.setattr(jobs, "make_highlight", boom)
    client.post(f"/jobs/{job_id}/rallies/0/highlight", json=TAP)
    assert [(d["title"], d["url"]) for d in sent] == [("Your highlight is ready", f"/#job={job_id}&rally=0"),
                                                      ("Couldn't make the highlight", f"/#job={job_id}&rally=0")]


def test_upload_resumes_from_what_the_server_already_has(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 사용자: 업로드 중 화면을 끄면 처음부터 다시 올려야 했다 → 받은 곳부터 이어 올린다
    monkeypatch.setattr(access, "REQUIRED", True)
    code = access.new_codes(1)[0]
    data = bytes(range(256)) * 400  # 102,400바이트
    up = client.post("/uploads", json={"filename": "경기.mp4", "size": len(data), "code": code}).json()["id"]
    assert client.put(f"/uploads/{up}?offset=0", content=data[:40000]).json() == {"received": 40000}
    # 화면이 꺼져 다음 조각이 끊겼다고 치고, 돌아온 기기가 엉뚱한 곳부터 보내면 받은 크기를 알려 준다
    r = client.put(f"/uploads/{up}?offset=80000", content=data[80000:])
    assert r.status_code == 409 and r.json()["detail"] == {"received": 40000}
    assert client.get(f"/uploads/{up}").json() == {"received": 40000, "size": len(data)}
    assert client.post(f"/uploads/{up}/finish").status_code == 409  # 아직 다 안 받았다
    assert client.put(f"/uploads/{up}?offset=40000", content=data[40000:] + b"x").status_code == 413  # 크기를 넘는다
    assert client.put(f"/uploads/{up}?offset=40000", content=data[40000:]).json() == {"received": len(data)}
    assert access.left(code)["uploads"] == 2  # 다 받기 전에는 횟수를 쓰지 않는다
    job_id = client.post(f"/uploads/{up}/finish").json()["id"]
    assert access.left(code)["uploads"] == 1
    job = jobs.load_job(job_id)
    assert Path(job.video).read_bytes() == data and job.match_name == "경기"
    assert client.get(f"/uploads/{up}").status_code == 404  # 작업이 되면 이어 올리기 자리는 없앤다
    assert client.get(f"/jobs/{job_id}").json()["status"] == "rallies_ready"


def test_browser_job_reports_which_chunks_arrived(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from tapme import muse

    monkeypatch.setattr(muse, "ask_about_video", lambda c, clip, prompt: muse.MuseReply(
        text='{"rallies": []}', input_tokens=1, output_tokens=1, usd=0.01, seconds=0))
    job_id = client.post("/jobs/browser", json={"filename": "a.mp4", "duration": 200}).json()["id"]
    client.post(f"/jobs/{job_id}/chunks/0", files={"file": ("c.mp4", b"x")})
    client.post(f"/jobs/{job_id}/chunks/2", files={"file": ("c.mp4", b"x")})
    assert client.get(f"/jobs/{job_id}").json()["chunks_received"] == [0, 2]  # 끊긴 기기는 1·3부터 이어 처리


def test_feedback_is_kept_with_its_scene_and_shown_to_the_operator(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _awaiting_tap(client, monkeypatch)
    assert client.post("/feedback", json={}).status_code == 422  # 빈 의견
    assert client.post("/feedback", json={"rating": "meh"}).status_code == 422
    r = client.post("/feedback", json={"rating": "bad", "text": "제가 아니라 옆 사람을 따라가요", "job": job_id, "rally": 0, "lang": "ko"})
    assert r.json() == {"status": "ok"}
    client.post("/feedback", json={"rating": "good"})
    jobs.delete_previous_days(1e12)  # 새벽 4시 삭제가 지나도 의견은 남는다
    monkeypatch.setenv("TAPME_ADMIN_TOKEN", "t")
    got = client.get("/admin/summary", headers={"X-Admin-Token": "t"}).json()["feedback"]
    assert [(f["rating"], f.get("scene")) for f in got] == [("good", None), ("bad", "awaiting_tap")]
    assert got[1]["text"] == "제가 아니라 옆 사람을 따라가요" and got[1]["job"] == job_id


def test_rate_limit_slows_down_repeated_requests(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 10/31 보안 검토: 무료 요청도 IP별로 속도를 제한한다(배포 서버에서만 켬)
    monkeypatch.setattr(server, "RATE_LIMIT", True)
    monkeypatch.setattr(server, "_buckets", {})
    monkeypatch.setattr(server, "_client_ip", lambda request: "198.51.100.1")
    codes = [client.post("/feedback", json={"rating": "good"}).status_code for _ in range(32)]
    assert codes[:30] == [200] * 30 and codes[30] == 429
    r = client.post("/feedback", json={"rating": "good"})
    assert r.json()["detail"].startswith("요청이 너무 많아요") and int(r.headers["retry-after"]) >= 1
    assert client.get("/").status_code == 200  # 다른 묶음(일반)은 따로 센다
    monkeypatch.setattr(server, "_client_ip", lambda request: "198.51.100.2")
    assert client.post("/feedback", json={"rating": "good"}).status_code == 200  # 다른 사람은 막히지 않는다
    assert server._rate_group("POST", "/jobs/x/rallies/0/prepare", "ahead=true") == "general"
    assert server._rate_group("POST", "/jobs/x/rallies/0/rerender", "") == "heavy"


def test_security_headers_and_push_hosts(client: TestClient) -> None:
    from tapme import push

    r = client.get("/")
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"
    assert push.allowed_endpoint("https://fcm.googleapis.com/fcm/send/abc")
    assert push.allowed_endpoint("https://web.push.apple.com/QGx")
    assert not push.allowed_endpoint("https://evil.example.com/fcm.googleapis.com")
    assert not push.allowed_endpoint("http://fcm.googleapis.com/x") and not push.allowed_endpoint("https://fcm.googleapis.com.evil.io/x")


def test_rerender_and_reel_cannot_be_piled_up(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 계속 눌러 렌더 줄을 막지 못하게: 만드는 중이면 거절, 실패해도 앞서 만든 영상으로 돌아간다
    job_id = _awaiting_tap(client, monkeypatch)
    job = jobs.load_job(job_id)
    job.scenes["0"].update(status="done", video="v.mp4")
    job.save()
    seen: list = []
    monkeypatch.setattr(jobs, "rerender", lambda job, i, *a: seen.append(jobs.load_job(job.id).scenes["0"]["status"]) or 1 / 0)
    assert client.post(f"/jobs/{job_id}/rallies/0/rerender", json={}).json() == {"status": "rendering"}
    assert seen == ["rendering"] and jobs.load_job(job_id).scenes["0"]["status"] == "done"  # 실패해도 그대로
    job = jobs.load_job(job_id)
    job.scenes["0"]["status"] = "rendering"
    job.reel = {"status": "rendering"}
    job.save()
    assert client.post(f"/jobs/{job_id}/rallies/0/rerender", json={}).status_code == 409
    job.scenes["0"]["status"] = "done"
    job.save()
    assert client.post(f"/jobs/{job_id}/reel", json={"order": [0]}).status_code == 409


def test_one_code_cannot_open_endless_uploads(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(access, "REQUIRED", True)
    code = access.new_codes(1)[0]
    ok = [client.post("/uploads", json={"size": 10, "code": code}).status_code for _ in range(3)]
    assert ok == [200, 200, 429]


def test_card_names_are_written_on_the_photos(tmp_path: Path) -> None:
    # H-49: 이름을 사진 앞 글로만 주면 Muse가 한 칸 밀린 이름(C1 → C2)으로 답했다. 사진 왼쪽 위에도 이름을 쓴다
    import base64
    import io

    from PIL import Image

    from tapme import auto

    card = tmp_path / "c.jpg"
    Image.new("RGB", (160, 240), (255, 255, 255)).save(card)
    url = auto._img(card, "C1")["image_url"]
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.getpixel((1, 1))[0] < 40 and img.getpixel((100, 100))[0] > 200  # 이름 칸은 검고 나머지는 그대로
    assert auto._img(card)["image_url"].endswith(base64.b64encode(card.read_bytes()).decode())


def test_candidate_photo_dims_the_neighbours_around_the_box(tmp_path: Path) -> None:
    # H-49: 카드 여유 자리에 뒤 사람이 보여 앞사람 카드를 골랐다. 상자 밖은 어둡게
    from PIL import Image

    from tapme import auto

    frame = tmp_path / "tap_score.jpg"
    Image.new("RGB", (400, 400), (200, 200, 200)).save(frame)
    auto._candidate(frame, [100, 100, 200, 300], "A0", "0")
    card = Image.open(tmp_path / "cand_A0.jpg")
    assert card.getpixel((card.width // 2, card.height // 2))[0] > 180  # 상자 안은 그대로
    assert card.getpixel((card.width - 2, card.height // 2))[0] < 110  # 여유 자리는 어둡게


def test_find_me_asks_each_moment_and_keeps_the_surest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # H-49: 세 순간 카드를 한 번에 물으면 한 칸 밀린 이름으로 답했다 → 순간마다 따로 묻고 가장 확신하는 답
    from types import SimpleNamespace

    from PIL import Image

    from tapme import auto

    (tmp_path / "scene00").mkdir()
    sizes = {"score": 2, "b2": 3, "start": 4}  # 순간마다 후보 수를 달리해 어느 질문인지 안다
    for kind in sizes:
        Image.new("RGB", (400, 400), (200, 200, 200)).save(tmp_path / "scene00" / f"tap_{kind}.jpg")
    Image.new("RGB", (160, 240)).save(tmp_path / "ref.jpg")
    monkeypatch.setattr(jobs, "player_cards", lambda job, i, kind, c: [[10, 10, 100, 200]] * sizes[kind])
    replies = {2: '{"index": 1, "confidence": 0.6}', 3: '{"index": -1, "confidence": 0.9}', 4: '{"index": 3, "confidence": 0.95}'}

    def create(**kw):
        n = sum(c["type"] == "input_image" for c in kw["input"][0]["content"]) - 1  # 기준 사진 빼고
        return SimpleNamespace(output_text=replies[n], usage=SimpleNamespace(input_tokens=1000, output_tokens=100))

    job = SimpleNamespace(dir=tmp_path, spent_usd=0.0, charge=lambda usd: None)
    found = auto.find_me(job, 0, tmp_path / "ref.jpg", "10번", SimpleNamespace(responses=SimpleNamespace(create=create)))
    assert (found["frame"], found["card"], found["confidence"]) == ("start", 3, 0.95)
    assert found["frames"] == {"score": [1, 0.6], "b2": [-1, 0.9], "start": [3, 0.95]} and job.spent_usd > 0
