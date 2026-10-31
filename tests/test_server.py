from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tapme import access, jobs, render, server


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
    assert client.post("/jobs", files={"file": ("m.mp4", b"v")}).status_code == 403  # 코드 없이
    for _ in range(2):
        assert client.post("/jobs", files={"file": ("m.mp4", b"v")}, data={"code": code}).status_code == 200
    r = client.post("/jobs", files={"file": ("m.mp4", b"v")}, data={"code": code})
    assert r.status_code == 403 and "2편" in r.json()["detail"]

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
    client.post("/jobs", files={"file": ("m.mp4", b"v")}, data={"code": code})
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
    # 10/31 사용자 결정: 배구구 배너(고정 주소)로 들어오면 코드를 만들어 준다. 퍼짐은 IP당 하루 3개·총 개수로 막는다
    monkeypatch.setattr(access, "REQUIRED", True)
    word = access.new_invite("banner", max_codes=4)
    claim = lambda w: client.post("/access/claim", json={"invite": w})
    assert claim("wrong").status_code == 403
    got = [claim(word) for _ in range(4)]
    assert [r.status_code for r in got] == [200, 200, 200, 403]  # 같은 접속에서 하루 3개
    code = got[0].json()["code"]
    assert client.post("/access", json={"code": code}).json()["left"] == {"uploads": 2, "highlights": 3}
    monkeypatch.setattr(server, "_client_ip", lambda request: "203.0.113.9")
    assert claim(word).status_code == 200 and "모두" in claim(word).json()["detail"]  # 총 4개까지
    new_word = access.new_invite("banner")  # 초대말을 바꾸면 예전 말은 막히고 받은 코드는 그대로
    assert claim(word).status_code == 403 and new_word != word
    assert client.post("/access", json={"code": code}).json()["left"]["uploads"] == 2


def test_judge_codes_skip_the_daily_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(access, "REQUIRED", True)
    monkeypatch.setattr(jobs, "MAX_UPLOADS_PER_DAY", 1)
    member, judge = access.new_codes(1)[0], client.post("/access/claim", json={"invite": access.new_invite("judge")}).json()["code"]
    assert client.post("/jobs", files={"file": ("m.mp4", b"v")}, data={"code": member}).status_code == 200
    assert client.post("/jobs", files={"file": ("m.mp4", b"v")}, data={"code": member}).status_code == 503  # 하루 1편 찼다
    assert client.post("/access", json={"code": judge}).json()["closed"] is None
    assert client.post("/jobs", files={"file": ("m.mp4", b"v")}, data={"code": judge}).status_code == 200


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
    for ep, lang in (("https://push/ko", "ko"), ("https://push/ja", "ja")):
        assert client.post(f"/jobs/{job_id}/push", json={"subscription": {"endpoint": ep, "keys": {}}, "lang": lang}).json() == {"status": "ok"}
    client.post(f"/jobs/{job_id}/chunks/0", files={"file": ("c.mp4", b"blur")})  # 하나뿐인 조각 → 랠리 찾기 끝
    assert [(ep, d["title"], d["url"], d["tag"]) for ep, d in sent] == [
        ("https://push/ko", "랠리 1개를 찾았어요", f"/#job={job_id}", f"job-{job_id}"),
        ("https://push/ja", "ラリーが1件見つかりました", f"/#job={job_id}", f"job-{job_id}")]
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
    client.post(f"/jobs/{job_id}/push", json={"subscription": {"endpoint": "https://push/en"}, "lang": "en"})
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
