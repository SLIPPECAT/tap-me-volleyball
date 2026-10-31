"""브라우저용 얼굴 모델을 만든다(10/31, H-44). 서버가 쓰는 YuNet은 입력이 640×640 고정이라 onnxruntime-web에서
1080p를 넣을 수 없어, 입력·출력 크기만 가변으로 바꾼 사본을 tapme/static/models/에 둔다(가중치는 그대로).
1080p 한 장에서 OpenCV FaceDetectorYN과 14/14 같은 얼굴을 찾았다.

사용법: uv run --with onnx python -m tapme.browser_model
"""

from __future__ import annotations

from pathlib import Path

from tapme import anonymize

OUT = Path(__file__).resolve().parent / "static" / "models" / "yunet_dyn.onnx"


def main() -> None:
    import onnx  # 이 변환에만 쓴다(서버 실행에는 필요 없음)

    m = onnx.load(str(anonymize.MODEL_PATH))
    dims = m.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_param, dims[3].dim_param = "H", "W"
    for o in m.graph.output:
        o.type.tensor_type.shape.dim[1].dim_param = "N"
    del m.graph.value_info[:]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(OUT))
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
