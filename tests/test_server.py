from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tapme import jobs, render, server


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
    monkeypatch.setattr(jobs, "make_highlight", lambda job, i, taps, *a: got.append(taps))
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
    # 기본 확대를 바꿀 때 화면과 서버 기본값이 어긋나지 않게(10/30 2배 → 1.5배)
    import re

    html = (Path(server.__file__).parent / "static" / "index.html").read_text()
    picked = re.search(r'<option value="([\d.]+)" selected>[\d.]+×</option>', html)
    assert picked and float(picked.group(1)) == render.DEFAULT_ZOOM == 1.5
