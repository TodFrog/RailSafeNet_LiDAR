"""루트 실행 호환성을 위한 sweep wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("src.training.sweep_transfer", globals(), __name__)

