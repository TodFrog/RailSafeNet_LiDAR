"""루트 실행 호환성을 유지하기 위한 공통 wrapper 유틸.

저장소 재구성 이후 실제 구현은 `src/` 아래로 이동했지만, 기존 사용자나
기존 문서가 루트 스크립트 이름을 그대로 사용할 수 있도록 얇은 wrapper를
남겨 두었다. 이 유틸은 그런 루트 진입점이 다음 두 상황을 모두 지원하도록
도와준다.

1. `python some_wrapper.py`처럼 직접 실행하는 경우
2. `import some_wrapper`처럼 기존 import 경로를 유지하는 경우
"""

from importlib import import_module
from runpy import run_module


def expose_or_run(module_name: str, caller_globals: dict, caller_name: str) -> None:
    """루트 wrapper에서 `src` 모듈을 실행하거나 재노출한다.

    직접 실행 시에는 대상 모듈을 `__main__`으로 실행해 CLI 동작을 유지하고,
    import 시에는 공개 심볼을 현재 모듈 namespace로 복사해 기존 import
    경로가 즉시 깨지지 않도록 한다.
    """
    if caller_name == "__main__":
        run_module(module_name, run_name="__main__")
        return

    module = import_module(module_name)
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = [name for name in module.__dict__ if not name.startswith("_")]

    for name in public_names:
        caller_globals[name] = getattr(module, name)
