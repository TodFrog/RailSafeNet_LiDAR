"""루트 실행 호환성을 위한 ONNX 래퍼.

실제 ONNX 추론 구현은 `src.inference.production_segformer_onnx`에 있으며,
이 파일은 과거 루트 실행 명령을 보존하기 위한 얇은 전달 계층이다.
"""

from _root_wrapper import expose_or_run

expose_or_run("src.inference.production_segformer_onnx", globals(), __name__)
