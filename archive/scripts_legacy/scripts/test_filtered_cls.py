"""구조 재편 이후 호환성을 위한 shim."""

from _root_wrapper import expose_or_run

expose_or_run("src.evaluation.test_filtered_cls", globals(), __name__)

