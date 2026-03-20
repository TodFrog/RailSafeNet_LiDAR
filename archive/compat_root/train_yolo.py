"""루트 실행 호환성을 위한 YOLO 학습 wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("src.training.train_yolo", globals(), __name__)

