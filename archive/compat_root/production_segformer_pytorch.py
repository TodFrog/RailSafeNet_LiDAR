"""루트 실행 호환성을 위한 PyTorch 래퍼.

최종 제출 문서와 사용자 매뉴얼은 이 파일을 공식 실행 명령으로 안내한다.
실제 구현은 `src.inference.production_segformer_pytorch`에 있으므로,
이 파일은 이름 호환성과 진입점 안정성만 담당한다.
"""

from _root_wrapper import expose_or_run

expose_or_run("src.inference.production_segformer_pytorch", globals(), __name__)
