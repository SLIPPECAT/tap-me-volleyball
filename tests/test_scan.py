from __future__ import annotations

from tapme import scan


def test_chunks_overlap_and_cover_the_whole_video() -> None:
    assert scan.chunk_starts(441.0) == [float(s) for s in range(0, 401, 50)]  # 7분 영상 B: 9조각(H-34)
    assert scan.chunk_starts(75.0) == [0.0, 50.0]
    assert scan.chunk_starts(55.0) == [0.0]
    starts = scan.chunk_starts(3600.0)
    assert starts[-1] + scan.CHUNK_SEC >= 3600.0 and len(starts) == 72


def test_same_point_seen_in_two_overlapping_chunks_is_kept_once() -> None:
    chunks = [
        {"start": 0.0, "answer": {"rallies": [
            {"start_sec": 8.5, "point_sec": 13.5, "confidence": 0.8},
            {"start_sec": 52.0, "point_sec": None, "cut_end": True, "confidence": 0.7},  # 조각 끝에서 잘림 → 다음 조각에서
        ]}},
        {"start": 50.0, "answer": {"rallies": [
            {"start_sec": 0.0, "point_sec": 8.0, "cut_start": True, "confidence": 0.9},
            {"start_sec": 2.0, "point_sec": 8.5, "cut_start": False, "confidence": 0.6},  # 같은 득점(58초), 잘리지 않은 쪽
            {"start_sec": None, "point_sec": 40.0, "confidence": 0.3},
        ]}},
    ]
    rallies = scan.rallies_from_answers(chunks)
    assert [r["point_sec"] for r in rallies] == [13.5, 58.5, 90.0]
    assert rallies[1]["start_sec"] == 52.0 and not rallies[1]["cut_start"]
    assert rallies[2]["start_sec"] == 80.0 and rallies[2]["uncertain"]  # 서브 시각이 없으면 득점 10초 전
    assert all(r["is_rally"] and r["end_sec"] == r["point_sec"] + scan.END_AFTER_POINT_SEC for r in rallies)


def test_unreadable_chunk_answer_is_skipped() -> None:
    assert scan.rallies_from_answers([{"start": 0.0, "answer": {"rallies": [], "unreadable": True}}]) == []


def test_hour_long_video_estimate_fits_the_length_based_budget() -> None:
    from tapme import jobs

    assert scan.estimate_usd(3600.0) < jobs.MAX_USD_PER_MIN * 60


def test_close_points_keep_both_but_flag_the_less_sure_one() -> None:
    # 대회 영상 C: 한 랠리를 7:30.0(0.92)·7:34.8(0.5) 둘로 짚음. B에는 7.7초 간격 진짜 득점도 있어 지우지 않는다
    def chunk(*points):
        return {"start": 0.0, "answer": {"rallies": [{"start_sec": p - 5, "point_sec": p, "confidence": c} for p, c in points]}}

    rallies = scan.rallies_from_answers([chunk((20.0, 0.92), (24.8, 0.5), (50.0, 0.8))])
    assert [(r["point_sec"], r["uncertain"]) for r in rallies] == [(20.0, False), (24.8, True), (50.0, False)]


def test_cut_off_answer_is_asked_again_then_reported(tmp_path, monkeypatch) -> None:
    # H-39: 출력 한도에서 잘린 답 하나로 조각의 득점 5개가 사라졌다. 다시 묻고, 그래도 안 되면 구간을 알린다
    from tapme import muse

    replies = iter(["{\"rallies\": [", '{"rallies": [{"start_sec": 1, "point_sec": 9, "confidence": 0.8}]}',
                    "", ""])
    monkeypatch.setattr(scan, "blurred_chunk", lambda v, s, sec, out: out)
    monkeypatch.setattr(muse, "ask_about_video", lambda c, clip, prompt: muse.MuseReply(
        text=next(replies), input_tokens=1, output_tokens=8000, usd=0.03, seconds=0))
    monkeypatch.setattr(scan, "PARALLEL", 1)  # 답 순서를 조각 순서에 맞춘다
    rallies, chunks = scan.scan_video(None, tmp_path / "v.mp4", 75.0, tmp_path)
    assert [c["attempts"] for c in chunks] == [2, 2]
    assert [r["point_sec"] for r in rallies] == [9.0]
    assert scan.unread_ranges(chunks) == [[50.0, 75.0]]
    assert sum(c["usd"] for c in chunks) == 0.12  # 다시 물은 값도 센다


def test_scanning_a_range_reports_times_in_the_whole_video(tmp_path, monkeypatch) -> None:
    from tapme import muse

    cuts: list = []
    def cut(v, s, sec, out):
        cuts.append((s, sec))
        out.write_bytes(b"blurred")
        return out

    monkeypatch.setattr(scan, "blurred_chunk", cut)
    monkeypatch.setattr(muse, "ask_about_video", lambda c, clip, prompt: muse.MuseReply(
        text='{"rallies": [{"start_sec": 1, "point_sec": 9, "confidence": 0.8}]}',
        input_tokens=1, output_tokens=1, usd=0.01, seconds=0))
    monkeypatch.setattr(scan, "PARALLEL", 1)
    rallies, chunks = scan.scan_video(None, tmp_path / "v.mp4", 75.0, tmp_path, offset=1200.0)
    assert cuts == [(1200.0, 60.0), (1250.0, 25.0)]  # 구간 끝(1275초)을 넘겨 자르지 않는다
    assert [r["point_sec"] for r in rallies] == [1209.0, 1259.0]
    assert scan.unread_ranges(chunks) == []
    assert not list(tmp_path.glob("chunk*_blur.mp4"))  # 물은 뒤 바로 지운다


def test_chunk_plan_splits_a_range_from_its_offset() -> None:
    assert scan.chunk_plan(1200.0, 100.0) == [(1200.0, 60.0), (1250.0, 50.0)]
    assert scan.chunk_plan(0.0, 30.0) == [(0.0, 30.0)]
    assert scan.chunk_plan(0.0, 60.0) == [(0.0, 60.0)]


def test_duration_sec_reads_ffprobe_output(monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"75.5\n")
    from pathlib import Path

    assert scan.duration_sec(Path("v.mp4")) == 75.5


def test_estimate_usd_adds_each_chunk() -> None:
    from tapme import muse

    assert scan.estimate_usd(30.0) == muse.estimate_usd(30.0)
    assert scan.estimate_usd(75.0) == muse.estimate_usd(60.0) + muse.estimate_usd(25.0)


def test_ask_chunk_asks_once_when_readable_and_deletes_the_clip(tmp_path, monkeypatch) -> None:
    from tapme import muse

    prompts: list[str] = []

    def fake_ask(client, clip, prompt):
        prompts.append(prompt)
        return muse.MuseReply(text='{"rallies": []}', input_tokens=5, output_tokens=5, usd=0.01, seconds=0)

    monkeypatch.setattr(muse, "ask_about_video", fake_ask)
    clip = tmp_path / "chunk000_blur.mp4"
    clip.write_bytes(b"blurred")
    r = scan.ask_chunk(None, 0, 50.0, 60.0, clip)
    assert (r["chunk"], r["start"], r["seconds"], r["attempts"], r["usd"]) == (0, 50.0, 60.0, 1, 0.01)
    assert r["answer"] == {"rallies": []} and (r["input_tokens"], r["output_tokens"]) == (5, 5)
    assert "60-second clip" in prompts[0]  # 조각 길이를 알려 조각 안 시각으로 답하게 한다
    assert not clip.exists()


def test_ask_chunk_reports_unreadable_after_two_failures(tmp_path, monkeypatch) -> None:
    import pytest

    from tapme import muse

    monkeypatch.setattr(muse, "ask_about_video", lambda c, clip, prompt: muse.MuseReply(
        text='{"rallies": [', input_tokens=1, output_tokens=8000, usd=0.03, seconds=0))
    clip = tmp_path / "chunk000_blur.mp4"
    clip.write_bytes(b"blurred")
    r = scan.ask_chunk(None, 0, 0.0, 60.0, clip)
    assert r["answer"] == {"rallies": [], "unreadable": True} and r["attempts"] == 2
    assert r["usd"] == pytest.approx(0.06) and not clip.exists()


def test_blurred_chunk_returns_the_blurred_cut_and_removes_the_raw(tmp_path, monkeypatch) -> None:
    import subprocess
    from pathlib import Path

    from tapme import anonymize

    def fake_run(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"raw")

    def fake_blur(raw, dst):
        dst.write_bytes(b"blurred")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(anonymize, "anonymize_video", fake_blur)
    out = tmp_path / "work" / "chunk000_blur.mp4"
    assert scan.blurred_chunk(tmp_path / "v.mp4", 50.0, 60.0, out) == out
    assert out.read_bytes() == b"blurred"
    assert not (tmp_path / "work" / "chunk000_blur_raw.mp4").exists()


def test_point_without_time_is_skipped_and_bad_starts_are_clamped() -> None:
    chunks = [{"start": 0.0, "answer": {"rallies": [
        {"start_sec": 5.0, "point_sec": None, "confidence": 0.9},  # 끝 표시 없이 득점 시각만 없음
        {"start_sec": 50.0, "point_sec": 20.0, "confidence": 0.8},  # 서브가 득점보다 뒤
        {"start_sec": -5.0, "point_sec": 30.0, "confidence": 0.8},  # 조각 시작보다 앞
    ]}}]
    rallies = scan.rallies_from_answers(chunks)
    assert [(r["point_sec"], r["start_sec"], r["uncertain"]) for r in rallies] == [(20.0, 19.0, False), (30.0, 0.0, False)]


def test_same_point_keeps_the_more_confident_full_view() -> None:
    def chunk(first, second):
        return {"start": 0.0, "answer": {"rallies": [
            {"start_sec": 5.0, "point_sec": 10.0, "confidence": first},
            {"start_sec": 6.0, "point_sec": 11.0, "confidence": second}]}}

    rallies = scan.rallies_from_answers([chunk(0.6, 0.9)])
    assert [(r["point_sec"], r["start_sec"]) for r in rallies] == [(11.0, 6.0)]
    rallies = scan.rallies_from_answers([chunk(0.9, 0.6)])
    assert [(r["point_sec"], r["start_sec"]) for r in rallies] == [(10.0, 5.0)]


def test_two_seconds_apart_merges_but_ten_seconds_apart_stays_clear() -> None:
    def chunk(*points):
        return {"start": 0.0, "answer": {"rallies": [{"start_sec": p - 5, "point_sec": p, "confidence": c} for p, c in points]}}

    rallies = scan.rallies_from_answers([chunk((20.0, 0.9), (22.0, 0.6))])
    assert [(r["point_sec"], r["uncertain"]) for r in rallies] == [(20.0, False)]
    rallies = scan.rallies_from_answers([chunk((20.0, 0.9), (30.0, 0.8))])
    assert [(r["point_sec"], r["uncertain"]) for r in rallies] == [(20.0, False), (30.0, False)]


def test_unread_ranges_lists_every_unread_chunk() -> None:
    chunks = [
        {"start": 0.0, "seconds": 60.0, "answer": {"rallies": []}},
        {"start": 50.0, "seconds": 60.0, "answer": {"rallies": [], "unreadable": True}},
        {"start": 100.0, "seconds": 30.5, "answer": {"rallies": [], "unreadable": True}},
    ]
    assert scan.unread_ranges(chunks) == [[50.0, 110.0], [100.0, 130.5]]


def test_scanning_reports_progress_as_chunks_finish(tmp_path, monkeypatch) -> None:
    import pytest

    from tapme import muse

    def cut(v, s, sec, out):
        out.write_bytes(b"blurred")
        return out

    monkeypatch.setattr(scan, "blurred_chunk", cut)
    monkeypatch.setattr(muse, "ask_about_video", lambda c, clip, prompt: muse.MuseReply(
        text='{"rallies": [{"start_sec": 1, "point_sec": 9, "confidence": 0.8}]}',
        input_tokens=1, output_tokens=1, usd=0.01, seconds=0))
    monkeypatch.setattr(scan, "PARALLEL", 1)
    progress: list = []
    rallies, _ = scan.scan_video(None, tmp_path / "v.mp4", 75.0, tmp_path,
                                 lambda d, n, spent, finished: progress.append((d, n, spent, [c["chunk"] for c in finished])))
    assert [r["point_sec"] for r in rallies] == [9.0, 59.0]
    assert [(d, n) for d, n, _, _ in progress] == [(1, 2), (2, 2)]
    assert [p[2] for p in progress] == pytest.approx([0.01, 0.02])
    assert progress[-1][3] == [0, 1]


def test_main_reports_the_estimate_without_calling_the_api(tmp_path, monkeypatch, capsys) -> None:
    import sys

    monkeypatch.setattr(scan, "duration_sec", lambda v: 130.0)
    monkeypatch.setattr(sys, "argv", ["scan", "match.mp4", "--work", str(tmp_path / "work")])

    def must_not_run(*a, **k):
        raise AssertionError("API를 부르면 안 된다")

    monkeypatch.setattr(scan, "scan_video", must_not_run)
    scan.main()
    out = capsys.readouterr().out
    assert "조각 3개" in out and "--yes 없이 실행해 API는 부르지 않았다" in out


def test_main_with_yes_scans_and_saves_the_results(tmp_path, monkeypatch, capsys) -> None:
    import json
    import sys

    from tapme import muse

    monkeypatch.setattr(scan, "duration_sec", lambda v: 75.0)
    monkeypatch.setattr(sys, "argv", ["scan", "match.mp4", "--work", str(tmp_path / "work"), "--yes"])
    (tmp_path / "work").mkdir()
    monkeypatch.setattr(muse, "client_from_env", lambda: object())
    rallies = [{"point_sec": 9.0, "start_sec": 1.0, "confidence": 0.8}]
    monkeypatch.setattr(scan, "scan_video", lambda *a, **k: (rallies, [{"usd": 0.01}]))
    scan.main()
    assert json.loads((tmp_path / "work" / "scan.json").read_text())["rallies"] == rallies
    out = capsys.readouterr().out
    assert "득점 1개" in out and "합계 $0.0100" in out
