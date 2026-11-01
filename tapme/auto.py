"""한 번에 맡기기(10/31 사용자: 기다렸다 누르는 단계를 없애고, 처음에 영상·나·설정을 다 받아 맡기면 끝나게).

사용자는 올리기 전에 영상에서 나를 한 번 탭하고(시각·위치) 등번호·유니폼 같은 '나에 대한 정보'를 적는다. 랠리를 다 찾으면
- 기준 사진: 탭한 순간 화면(얼굴 흐림본)에서 선수를 찾아 탭한 사람의 카드를 자른다
- 랠리마다 장면을 준비하고(prepare_scene) 여러 순간(FRAMES)의 선수 카드 중 기준과 같은 사람을 Muse Spark가 고른다(실험 H-46:
  정보 있음 17/17). 확신 CONFIDENT 이상이면 하이라이트를 예약, 아니면 '나 맞나요?' 확인 묶음 — 건너뛰지 않는다.
  나는 모든 랠리에서 코트에 있다(11/1 사용자). 첫 실기기(H-47)에서 '없음'으로 건너뛴 3개는 그 한 장면에서 등번호·신발이
  안 보였던 것이라, 그 전제를 알려 주고 득점 순간 말고도 3초 전·구간 시작 장면도 본다(순간마다 따로 묻고 가장 확신하는 답)
- 예약은 화면의 일괄 만들기와 같은 길(server._run_highlight), 다 되면 모음 영상과 알림
Meta로 가는 사진(장면·카드·기준)은 모두 얼굴 흐림본이다. 한 번에 맡기기는 서버 방식만(원본이 서버에 있어야 장면을 자른다).
"""

from __future__ import annotations

import base64
import io
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from tapme import jobs, muse, track

# 한 번에 맡기기는 10분까지(11/1 사용자: 20분 → 10분. 입력 용량·랠리 찾기 비용이 예상 범위에 든다, 1080p 약 450MB·약 $0.37).
# 하나씩 고르기는 jobs.MAX_SCAN_SEC(30분)
MAX_SCAN_SEC = 10 * 60
# 찾은 랠리는 다 만든다(11/1 사용자: 8개 제한을 풀었다). 10분 영상이 랠리 20개 안팎이라 이 값은 안전 상한일 뿐
MAX_HIGHLIGHTS = 40
CONFIDENT = 0.7  # H-46: 맞힌 것은 0.65~0.95, 사람도 확신 못 한 장면에서 0.6. 이 아래는 사용자에게 확인받는다
USD_PER_QUESTION = 0.03  # 한 순간 질문(생각 '중간'·사진 13장 안팎): 실측 $0.006~0.02, 넉넉히 잡고 charge

FRAMES = ("score", "b2", "start")  # 득점 순간·3초 전·구간 시작(jobs.TAP_MOMENTS). 한 장면에선 등번호가 안 보이기 쉽다

PROMPT = """첫 번째 사진(REF)은 배구 경기 영상에서 한 선수(기준 선수)를 잘라 낸 것이다. 그 뒤 사진들은 같은 경기 한 랠리의
한 순간에서 잘라 낸 선수들이고, 각 사진 앞과 사진 왼쪽 위에 후보 번호가 있다(사진에 적힌 번호를 쓴다). 얼굴은 모두 흐려져
있으니 얼굴은 보지 말고 유니폼 색·무늬·등번호·바지·신발·체형·머리 모양으로 판단한다. 같은 팀 선수는 유니폼이 같으니 등번호와
체형·신발·바지 길이 같은 차이를 꼭 본다. 후보는 사진 가운데 밝은 자리의 사람이고, 테두리의 어두운 자리에 보이는 사람은 다른
후보다. 밝은 자리에 기준 선수가 다른 사람과 겹쳐 있으면 그 후보의 confidence는 0.5 이하로 준다(추적이 앞사람을 따라간다).
기준 선수는 이 랠리에도 코트에 있지만 이 순간엔 가려지거나 작아 안 보일 수 있다. 분명한 후보가 없으면 index를 -1로 한다.
{hint}
JSON 하나로만 답한다: {{"index": 후보 번호 또는 -1, "confidence": 0~1, "reason": "한 문장"}}"""


def _img(path: Path, name: str = "") -> dict[str, str]:
    """사진(얼굴 흐림본). name을 왼쪽 위에 써 넣는다: 이름을 사진 앞 글로만 주면 사진 40장 안팎에서 Muse가 한 칸 밀린 이름으로
    답했다(11/1 H-49: C1인 선수를 C2라고 답해 다른 사람을 추적)."""
    if not name:
        return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()}
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 8 + 8 * len(name), 16), fill=(0, 0, 0))
    draw.text((4, 2), name, fill=(255, 255, 0))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}


def _candidate(frame: Path, box: list[float], name: str, label: str) -> dict[str, str]:
    """후보 사진: 상자 밖은 어둡게(jobs.dimmed_card) 하고 이름을 쓴다. 여유 자리에 뒤 사람이 겹쳐 보이면 그 사람을 후보로
    착각해 골랐다(11/1 H-49: 앞 선수 카드 뒤에 보인 다른 선수 → 앞 선수를 추적)."""
    path = frame.parent / f"cand_{name}.jpg"
    jobs.dimmed_card(Image.open(frame), box).save(path)
    return _img(path, label)


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


def _ask_frame(job: jobs.Job, index: int, kind: str, ref: Path, hint: str, client: Any) -> tuple[dict[str, Any], float]:
    """한 순간의 선수 카드 중 기준과 같은 사람. ({"card", "confidence", "reason"}, 쓴 돈)."""
    boxes = jobs.player_cards(job, index, kind, client)
    if not boxes:
        return {"card": -1, "confidence": 0.0, "reason": "선수를 찾지 못함"}, 0.0
    frame = job.dir / f"scene{index:02d}" / f"tap_{kind}.jpg"
    content: list[dict[str, str]] = [
        {"type": "input_text", "text": PROMPT.format(hint=f"기준 선수 정보: {hint}" if hint else "")},
        {"type": "input_text", "text": "기준 선수 REF"}, _img(ref, "REF")]
    for n, b in enumerate(boxes):
        content += [{"type": "input_text", "text": f"후보 {n}"}, _candidate(frame, b, f"{kind[:2]}{n}", str(n))]
    # 품질 우선(10/31 사용자): 생각 '중간'(기본). 한도는 생각(1,300~4,400토큰)과 답이 다 들어가게
    resp = client.responses.create(model=muse.MODEL, max_output_tokens=16000,
                                   input=[{"type": "message", "role": "user", "content": content}])
    usd = (int(resp.usage.input_tokens or 0) * muse.INPUT_USD_PER_TOKEN
           + int(resp.usage.output_tokens or 0) * muse.OUTPUT_USD_PER_TOKEN)
    try:
        ans = muse.parse_json_reply(resp.output_text or "")
        card = int(ans.get("index", -1))
    except (ValueError, TypeError):
        return {"card": -1, "confidence": 0.0, "reason": "답을 읽지 못함"}, usd
    if not -1 <= card < len(boxes):
        card = -1
    return {"card": card, "confidence": float(ans.get("confidence") or 0), "reason": str(ans.get("reason") or "")[:200]}, usd


def find_me(job: jobs.Job, index: int, ref: Path, hint: str, client: Any) -> dict[str, Any]:
    """그 랠리 여러 순간(FRAMES)의 선수 카드 중 기준과 같은 사람. {"frame": 장면 종류, "card": 번호 또는 -1, "confidence", "reason"}.
    순간마다 따로 묻고(동시에) 가장 확신하는 답을 쓴다. 세 순간 카드 40장 안팎을 한 번에 물으면 이름을 한 칸 밀려 답해
    다른 사람을 추적했다(11/1 H-49: 6개 중 3개). 한 순간 13장 안팎은 H-46에서 17/17."""
    for kind in FRAMES:  # 선수 찾기(SAM 사진)는 job을 저장하니 먼저 차례로
        jobs.player_cards(job, index, kind, client)
    job.charge(USD_PER_QUESTION * len(FRAMES))
    with ThreadPoolExecutor(len(FRAMES)) as pool:
        answers = list(pool.map(lambda kind: _ask_frame(job, index, kind, ref, hint, client), FRAMES))
    job.spent_usd += sum(usd for _, usd in answers)
    found = [{"frame": kind, **ans} for kind, (ans, _) in zip(FRAMES, answers, strict=True)]
    best = max(found, key=lambda f: (f["card"] >= 0, f["confidence"]))  # 같으면 앞 순간(득점 순간)
    return {**best, "frames": {f["frame"]: [f["card"], f["confidence"]] for f in found}}


def decide(found: dict[str, Any]) -> str:
    """make(자동으로 만들기) · review(나 맞나요? — 사용자가 한 번 확인). 나는 모든 랠리에 있어 건너뛰지 않는다(11/1 사용자)."""
    return "make" if found["card"] >= 0 and found["confidence"] >= CONFIDENT else "review"
