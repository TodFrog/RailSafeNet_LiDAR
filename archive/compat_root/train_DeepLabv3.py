"""루트 실행 호환성을 위한 DeepLabv3 학습 wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("src.training.train_DeepLabv3", globals(), __name__)

