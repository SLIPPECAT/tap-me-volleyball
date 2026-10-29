from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tapme import track

# meta_sam_parser 문서 예제의 실제 마스크 페이로드(27×60, 27×48)
_MASK_A = "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
_MASK_B = "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>"
WIRE = "\n".join([
    (f"<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>{_MASK_A},"
     f"1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>{_MASK_B}"),
    f"<1f>0<|box;x1=212;y1=228;x2=271;y2=254;w=320;h=334|>{_MASK_A}",
])


def test_boxes_by_frame() -> None:
    by = track.boxes_by_frame(track.parse(WIRE))
    assert sorted(by) == [0, 1]
    assert {b.object_id for b in by[0]} == {"0", "1"}
    b = by[1][0]
    assert (b.object_id, b.left, b.top) == ("0", 212, 228)
    assert b.center == ((b.left + b.right) / 2, (b.top + b.bottom) / 2)


def test_merge_keeps_me_and_renames_others_without_my_double() -> None:
    me = track.parse("<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>" + _MASK_A)
    # 모든 선수 추적: 0번은 '나'와 같은 사람(같은 박스), 1번은 다른 선수
    merged, masks = track.merge_me_and_others(me, track.parse(WIRE))
    assert sorted(b.object_id for b in merged[0]) == ["0", "o1"]
    assert set(masks[0]) == {"0", "o1"}
    # 1프레임에는 '나'가 안 잡혀 같은 사람인지 알 수 없다 → 다른 선수로 둔다
    assert [b.object_id for b in merged[1]] == ["o0"]


def test_visible_span_takes_the_run_around_the_score() -> None:
    b = track.Box(track.TAPPED_ID, 0, 0, 1, 1)
    by = {k: [b] for k in [*range(15), *range(35, 100)]}  # 15~34 화면 밖
    assert track.visible_span(by, around=70) == (35, 99)
    assert track.visible_span(by, around=5) == (0, 14)
    assert track.visible_span(by, around=20) == (0, 14)  # 화면 밖이면 가까운 구간
    by[50] = []  # 한두 프레임 놓친 것은 잇는다
    assert track.visible_span(by, around=70) == (35, 99)
    assert track.visible_span({}, around=0) is None


def test_track_players_streams_concept_prompt(tmp_path: Path) -> None:
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    sent: dict = {}

    class Stream:
        def __enter__(self):
            return iter([
                SimpleNamespace(type="response.output_text.delta", delta=WIRE[:40]),
                SimpleNamespace(type="response.output_text.delta", delta=WIRE[40:]),
                SimpleNamespace(type="response.completed",
                                response=SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: {"frames": 2}))),
            ])

        def __exit__(self, *a):
            return False

    def stream(**kw):
        sent.update(kw)
        return Stream()

    client = SimpleNamespace(files=SimpleNamespace(create=lambda purpose, file: SimpleNamespace(id="f")),
                             responses=SimpleNamespace(stream=stream))
    wire, usage, _ = track.track_players(client, clip)
    assert wire == WIRE and usage == {"frames": 2}
    assert sent["model"] == "sam-3.1"
    assert sent["input"][0]["content"][0] == {"type": "input_text", "text": "volleyball player"}


def test_tap_prompt_marks_the_tapped_player() -> None:
    assert track.tap_prompt((1080, 1920), [(0, 958.4, 1047.6)]) == "<0f>0<|point;x=958;y=1048;w=1080;h=1920|>"


def test_tap_prompt_puts_taps_from_several_frames_on_one_player() -> None:
    p = track.tap_prompt((1080, 1920), [(0, 958, 1048), (55, 940, 1060)])
    assert "<0f>0<|point;x=958;y=1048" in p and "<55f>0<|point;x=940;y=1060" in p
