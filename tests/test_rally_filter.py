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
    assert results[0]["clip_start_sec"] == 0.0  # 1.0초에서 1초 앞으로 넓힘
    # 얼굴이 그대로인 자른 원본은 남기지 않는다
    assert not any("raw" in p.name for p in (tmp_path / "work").iterdir())
