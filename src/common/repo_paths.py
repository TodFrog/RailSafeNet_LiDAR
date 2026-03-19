"""저장소 루트 기준 경로를 계산하기 위한 공용 유틸.

재구성 이후 루트 wrapper, `src/` 구현, 설정 파일, 모델 자산이 서로 다른
디렉터리에 배치되었기 때문에, 하드코딩된 상대경로를 줄이고 저장소 기준
경로 계산을 한 곳에 모으기 위해 사용한다.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """저장소 루트 기준 절대 경로를 반환한다.

    이 함수는 현재 파일 위치를 기준으로 루트 디렉터리를 계산한 뒤, 전달된
    하위 경로를 안전하게 이어 붙인다. 문서, 설정, 모델, 샘플 데이터 경로를
    여러 모듈에서 일관되게 참조하기 위한 기본 도우미다.
    """
    return REPO_ROOT.joinpath(*parts)


CONFIGS_DIR = repo_path("configs")
MODELS_DIR = repo_path("models")
DATA_SAMPLES_DIR = repo_path("data_samples")
