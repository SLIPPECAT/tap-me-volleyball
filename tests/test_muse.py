from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tapme import muse


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.sent: dict = {}
        self.files = SimpleNamespace(create=self._upload)
        self.responses = SimpleNamespace(create=self._respond)
        self._text = text

    def _upload(self, purpose: str, file) -> SimpleNamespace:
        self.sent["purpose"] = purpose
        return SimpleNamespace(id="file-1")

    def _respond(self, **kw) -> SimpleNamespace:
        self.sent.update(kw)
        usage = SimpleNamespace(input_tokens=2000, output_tokens=1000)
        return SimpleNamespace(output_text=self._text, usage=usage)


def test_ask_about_video_sends_prompt_and_file(tmp_path: Path) -> None:
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    client = _FakeClient('```json\n{"is_rally": true}\n```')
    reply = muse.ask_about_video(client, clip, "is it a rally?")
    content = client.sent["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "is it a rally?"}
    assert content[1] == {"type": "input_file", "file_id": "file-1"}
    assert client.sent["model"] == muse.MODEL
    assert reply.usd == pytest.approx(2000 * 1.25e-6 + 1000 * 4.25e-6)
    assert muse.parse_json_reply(reply.text) == {"is_rally": True}


def test_contributor_model_is_refused(tmp_path: Path) -> None:
    # Contributor 모드는 입력을 학습에 쓴다. 회원 영상은 보내지 않는다
    with pytest.raises(ValueError, match="Contributor"):
        muse.ask_about_video(_FakeClient("{}"), tmp_path / "c.mp4", "q", model="muse-spark-1.3-contributor")


def test_estimate_usd_scales_with_duration() -> None:
    assert muse.estimate_usd(10.0) == pytest.approx(10 * 280 * 1.25e-6 + 2500 * 4.25e-6)
