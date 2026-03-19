"""루트 실행 호환성을 위한 전이학습 wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("src.training.train_SegFormer_transfer_learning", globals(), __name__)

