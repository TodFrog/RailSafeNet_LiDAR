"""루트 실행 호환성을 위한 ONNX 변환 wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("src.conversion.original_to_onnx", globals(), __name__)

