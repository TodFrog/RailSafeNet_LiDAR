"""루트 실행 호환성을 위한 비디오 평가 wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("src.evaluation.video_frame_tester", globals(), __name__)

