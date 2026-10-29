from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from tapme import rally, rally_filter


def _video(path: Path, seconds: int = 6) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=64x64:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
    return path


def test_cut_clip_pads_and_downsamples(tmp_path: Path) -> None:
    src = _video(tmp_path / "m.mp4")
    out = rally_filter.cut_clip(src, 0.5, 3.0, tmp_path / "c.mp4")
    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=r_frame_rate,nb_read_frames", "-of", "csv=p=0", str(out)], text=True).strip()
    rate, frames = probe.split(",")
    assert rate == "5/1"
    # 0.5초 앞을 1초 넓히면 0초에서 잘리고, 3초 뒤로 1초 → 0~4초 = 20프레임
    assert abs(int(frames) - 20) <= 1


def test_low_confidence_answer_is_asked_again(tmp_path: Path, monkeypatch) -> None:
    # 10/28 서버 확인: 같은 클립에 "영상 프레임이 없다"(확신 0.0)고 답한 적이 있다. 한 번 더 묻는다
    src = _video(tmp_path / "m.mp4")
    monkeypatch.setattr(rally_filter.anonymize, "anonymize_video", lambda raw, dst: dst.write_bytes(raw.read_bytes()))
    answers = iter(['{"is_rally": false, "confidence": 0.0, "reason": "No video frames available."}',
                    '{"is_rally": true, "confidence": 0.95, "reason": "serve and attack"}'])

    def respond(**kw) -> SimpleNamespace:
        return SimpleNamespace(output_text=next(answers), usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    client = SimpleNamespace(files=SimpleNamespace(create=lambda purpose, file: SimpleNamespace(id="f")),
                             responses=SimpleNamespace(create=respond))
    r = rally_filter.check_candidates(client, src, [rally.RallyCandidate(1.0, 4.0)], tmp_path / "work")[0]
    assert (r["is_rally"], r["attempts"], r["uncertain"]) == (True, 2, False)
    assert r["input_tokens"] == 20  # 두 번 낸 돈을 다 센다


def test_only_blurred_clip_is_sent(tmp_path: Path, monkeypatch) -> None:
    src = _video(tmp_path / "m.mp4")
    blurred: list[Path] = []

    def fake_anonymize(raw: Path, dst: Path) -> dict:
        blurred.append(dst)
        dst.write_bytes(raw.read_bytes())
        return {}

    monkeypatch.setattr(rally_filter.anonymize, "anonymize_video", fake_anonymize)
    sent: list[str] = []

    def upload(purpose: str, file) -> SimpleNamespace:
        sent.append(file.name)
        return SimpleNamespace(id="f")

    def respond(**kw) -> SimpleNamespace:
        return SimpleNamespace(output_text='{"is_rally": false, "confidence": 0.9, "reason": "lining up"}',
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    client = SimpleNamespace(files=SimpleNamespace(create=upload), responses=SimpleNamespace(create=respond))
    results = rally_filter.check_candidates(client, src, [rally.RallyCandidate(1.0, 4.0)], tmp_path / "work")
    assert sent == [str(blurred[0])]
    assert results[0]["is_rally"] is False
    assert '"reason": {"ko": "...", "ja": "...", "en": "..."}' in rally_filter.RALLY_CHECK_PROMPT  # 화면 언어로 보여 줌
    assert results[0]["clip_start_sec"] == 0.0  # 1.0초에서 1초 앞으로 넓힘
    # 얼굴이 그대로인 자른 원본은 남기지 않는다
    assert not any("raw" in p.name for p in (tmp_path / "work").iterdir())


def test_merge_only_touching_rally_segments() -> None:
    r = lambda s, e, ok: {"start_sec": s, "end_sec": e, "is_rally": ok, "confidence": 0.9, "usd": 0.006,
                          "input_tokens": 10, "output_tokens": 5, "clip": f"{s}"}
    merged = rally_filter.merge_rallies([r(3, 16.75, True), r(16.75, 26.5, True), r(26.5, 35, False), r(35, 44, True)])
    assert [(m["start_sec"], m["end_sec"], m["parts"]) for m in merged] == [(3, 26.5, [0, 1]), (26.5, 35, [2]), (35, 44, [3])]
    assert merged[0]["clip"] is None and merged[0]["usd"] == 0.012  # 합친 랠리는 클립을 새로 만든다
    assert merged[2]["clip"] == "35"


def test_candidates_are_checked_in_parallel_but_kept_in_order(tmp_path: Path, monkeypatch) -> None:
    import threading
    import time

    running, peak = [0], [0]
    lock = threading.Lock()

    def fake_one(client, video, c, i, work_dir):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return {"start_sec": c.start_sec}

    monkeypatch.setattr(rally_filter, "_check_one", fake_one)
    progress: list = []
    cands = [rally.RallyCandidate(float(i), float(i) + 5) for i in range(6)]
    out = rally_filter.check_candidates(None, tmp_path / "v.mp4", cands, tmp_path, lambda d, n: progress.append((d, n)))
    assert [r["start_sec"] for r in out] == [float(i) for i in range(6)]
    assert peak[0] == rally_filter.PARALLEL  # 동시에 묻는다
    assert progress[-1] == (6, 6) and len(progress) == 6
