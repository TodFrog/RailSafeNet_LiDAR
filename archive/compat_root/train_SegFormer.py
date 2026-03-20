"""루트 실행 호환성을 위한 학습 wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("src.training.train_SegFormer", globals(), __name__)

