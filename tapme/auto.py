"""한 번에 맡기기(10/31 사용자: 기다렸다 누르는 단계를 없애고, 처음에 영상·나·설정을 다 받아 맡기면 끝나게).

사용자는 올리기 전에 영상에서 나를 한 번 탭하고(시각·위치) 등번호·유니폼 같은 '나에 대한 정보'를 적는다. 랠리를 다 찾으면
- 기준 사진: 탭한 순간 화면(얼굴 흐림본)에서 선수를 찾아 탭한 사람의 카드를 자른다
- 랠리마다 득점 장면을 준비하고(prepare_scene) 선수 카드(player_cards) 중 기준과 같은 사람을 Muse Spark가 고른다(실험 H-46:
  정보 있음 17/17, 확실하지 않으면 '없음'). 확신 CONFIDENT 이상이면 하이라이트를 예약, 미만이면 '나 맞나요?' 확인 묶음, 없으면 건너뜀
- 예약은 화면의 일괄 만들기와 같은 길(server._run_highlight), 다 되면 모음 영상과 알림
Meta로 가는 사진(장면·카드·기준)은 모두 얼굴 흐림본이다. 한 번에 맡기기는 서버 방식만(원본이 서버에 있어야 장면을 자른다).
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from tapme import jobs, muse, track

MAX_SCAN_SEC = 20 * 60  # 한 번에 맡기기는 20분까지(10/31 사용자). 하나씩 고르기는 jobs.MAX_SCAN_SEC(30분)
MAX_HIGHLIGHTS = 8  # 자동으로 만드는 최대 개수(개당 약 5분 반, 하나씩 만든다)
CONFIDENT = 0.7  # H-46: 맞힌 것은 0.65~0.95, 사람도 확신 못 한 장면에서 0.6. 이 아래는 사용자에게 확인받는다
USD_PER_QUESTION = 0.03  # 생각 '중간'·사진 십여 장: 실측 $0.006~0.02, 넉넉히 잡고 charge

PROMPT = """첫 번째 사진은 배구 경기 영상에서 한 선수(기준 선수)를 잘라 낸 것이다. 그 뒤의 사진들은 같은 경기의 다른 장면에서
잘라 낸 선수들이고, 각 사진 앞에 '후보 번호'가 있다. 얼굴은 모두 흐려져 있으니 얼굴은 보지 말고 유니폼 색·무늬·등번호·
바지·신발·체형·머리 모양으로 판단한다. 같은 팀 선수는 유니폼이 같으니 등번호와 체형·신발 같은 차이를 꼭 본다.
기준 선수와 같은 사람인 후보 번호를 고른다. 같은 사람이 없거나 확신할 수 없으면 -1.
{hint}
JSON 하나로만 답한다: {{"index": 후보 번호 또는 -1, "confidence": 0~1, "reason": "한 문장"}}"""


def _img(path: Path) -> dict[str, str]:
    return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()}


def make_reference(job: jobs.Job, t: float, x: float, y: float, client: Any) -> Path:
    """탭한 순간(원본 t초, 원본 화소 x·y)의 얼굴 흐림 화면에서 선수를 찾아, 탭한 곳을 가장 잘 덮는 사람의 카드를 자른다."""
    d = job.dir / "auto"
    d.mkdir(exist_ok=True)
    ref = d / "ref.jpg"
    if ref.exists():
        return ref
    clip = jobs._cut_blurred(Path(job.video), max(0.0, t), 1.0, d / "ref_blur.mp4", 10)
    frame = d / "ref_frame.jpg"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-frames:v", "1", str(frame)], check=True)
    clip.unlink()
    job.charge(track.USD_PER_IMAGE)
    boxes = track.detect_players(client, frame)
    job.spent_usd += track.USD_PER_IMAGE
    if not boxes:
        raise jobs.TapMissed("탭한 화면에서 선수를 찾지 못했어요. 내가 크게 보이는 다른 장면에서 다시 탭해 주세요")
    inside = [b for b in boxes if b.left <= x <= b.right and b.top <= y <= b.bottom]
    pick = min(inside or boxes, key=lambda b: (b.center[0] - x) ** 2 + (b.center[1] - y) ** 2)
    img = Image.open(frame)
    pad_w, pad_h = (pick.right - pick.left) * 0.15, (pick.bottom - pick.top) * 0.08  # 선수 카드(player_cards)와 같게
    card = img.crop((max(0, pick.left - pad_w), max(0, pick.top - pad_h),
                     min(img.width, pick.right + pad_w), min(img.height, pick.bottom + pad_h)))
    card.thumbnail((160, 240))
    card.save(ref)
    frame.unlink()
    return ref


def find_me(job: jobs.Job, index: int, ref: Path, hint: str, client: Any) -> dict[str, Any]:
    """그 랠리 득점 장면의 선수 카드 중 기준과 같은 사람. {"card": 번호 또는 -1, "confidence", "reason"}."""
    jobs.player_cards(job, index, "score", client)
    cards = sorted((job.dir / f"scene{index:02d}").glob("player_score_*.jpg"))
    if not cards:
        return {"card": -1, "confidence": 1.0, "reason": "선수를 찾지 못함"}
    content: list[dict[str, str]] = [
        {"type": "input_text", "text": PROMPT.format(hint=f"기준 선수 정보: {hint}" if hint else "")},
        {"type": "input_text", "text": "기준 선수"}, _img(ref)]
    for n, c in enumerate(cards):
        content += [{"type": "input_text", "text": f"후보 {n}"}, _img(c)]
    job.charge(USD_PER_QUESTION)
    # 품질 우선(10/31 사용자): 생각 '중간'(기본). 한도는 생각(1,300~4,400토큰)과 답이 다 들어가게
    resp = client.responses.create(model=muse.MODEL, max_output_tokens=16000,
                                   input=[{"type": "message", "role": "user", "content": content}])
    job.spent_usd += (int(resp.usage.input_tokens or 0) * muse.INPUT_USD_PER_TOKEN
                      + int(resp.usage.output_tokens or 0) * muse.OUTPUT_USD_PER_TOKEN)
    try:
        ans = muse.parse_json_reply(resp.output_text or "")
        card = int(ans.get("index", -1))
    except (ValueError, TypeError):
        return {"card": -1, "confidence": 0.0, "reason": "답을 읽지 못함"}
    if not -1 <= card < len(cards):
        card = -1
    return {"card": card, "confidence": float(ans.get("confidence") or 0), "reason": str(ans.get("reason") or "")[:200]}


def decide(found: dict[str, Any]) -> str:
    """make(자동으로 만들기) · review(나 맞나요?) · skip(내가 안 나옴)."""
    if found["card"] < 0:
        return "skip"
    return "make" if found["confidence"] >= CONFIDENT else "review"
