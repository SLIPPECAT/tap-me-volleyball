# CLAUDE.md

Meta AI 해커톤 제출 레포. 전역 `~/.claude/CLAUDE.md` 규칙을 그대로 따르고, 아래를 더한다.

## 시작할 때

- 진행 계획·결정 기록·실험 결과는 실험 레포 `~/projects/baegugu-highlight`의 `docs/PLAN.md`, `docs/HYPOTHESES.md`에 있다. 이 레포에는 제품 코드만 둔다.

## 규칙

- **이 레포는 제출 전에 공개된다.** 한 번 커밋한 것은 공개될 것으로 본다. 영상·API 키·내부 메모는 `.gitignore`(data/, .env, internal/)에만 둔다.
- **대회 기간(10/26~11/4)에 새로 짠 코드만 둔다.** 실험 레포의 사전 작업 모듈(sam_cost·render_highlight 등)은 복사하지 않고 새로 짠다. 대회 기간에 만든 파일을 옮길 때는 파일 머리에 출처(레포·커밋)를 남긴다.
- **외부로 보내는 영상은 얼굴 흐림본만.** Meta Model API 호출 외에는 영상을 내보내지 않는다. Muse Spark Contributor 모드는 쓰지 않는다.
- **API 키는 `.env`의 `MODEL_API_KEY`에만.** 값을 출력하거나 문서에 적지 않는다.
- **유료 API는 돌리기 전에 예상 금액을 먼저 말한다.**
- 커밋은 날마다(대회 중 작성 증거). 비공개 원격이므로 **커밋하면 push까지 같이 한다**(백업, 10/28 사용자 결정). **공개 전환은 사용자가 요청할 때만**, 전환 전에 이력 전체에 키·영상이 없는지 검사한다.

## 게이트(Gate) 명령

| 범위 | 게이트 | 비고 |
|---|---|---|
| 테스트 | `uv run pytest` | `tests/` |
| 린트 | `uv run ruff check .` | |
