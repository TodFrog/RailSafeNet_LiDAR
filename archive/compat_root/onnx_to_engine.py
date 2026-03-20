"""루트 실행 호환성을 위한 TensorRT 변환 wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("src.conversion.onnx_to_engine", globals(), __name__)

