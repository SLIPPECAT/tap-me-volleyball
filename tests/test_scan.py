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
