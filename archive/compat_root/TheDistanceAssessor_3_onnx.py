"""루트 실행 호환성을 위한 ONNX 추론 wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("src.inference.TheDistanceAssessor_3_onnx", globals(), __name__)

