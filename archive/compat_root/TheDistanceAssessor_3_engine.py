"""루트 실행 호환성을 위한 TensorRT 추론 wrapper.

통합 위험도 분석 TensorRT 경로의 실제 구현은
`src.inference.TheDistanceAssessor_3_engine`에 있다. 이 파일은 기존 실행
진입점 이름을 유지하기 위한 호환 레이어다.
"""

from _root_wrapper import expose_or_run

expose_or_run("src.inference.TheDistanceAssessor_3_engine", globals(), __name__)
