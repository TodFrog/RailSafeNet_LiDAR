"""`videoAssessor`의 PyTorch backend 사전 점검 모듈.

현재 최종 활성 런타임은 `engine` backend 기준으로 정리되어 있다. 이 모듈은
PyTorch 기반 전체 `videoAssessor` 파이프라인을 다시 구현하기보다는, 현재
저장소에 포함된 `.pth`/`.pt` 자산과 런타임 의존성이 준비되었는지를 검토하는
보수적 preflight 용도로 남긴다.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path

from src.common.repo_paths import repo_path

os.environ.setdefault("YOLO_CONFIG_DIR", str(repo_path(".ultralytics")))

SEGMENTATION_CANDIDATES = [
    repo_path("models", "converted", "SegFormer_B3_1024_finetuned.pth"),
    repo_path("models", "final", "SegFormer_B3_1024_finetuned.pth"),
]

DETECTION_CANDIDATES = [
    repo_path("models", "final", "yolov8n.pt"),
    repo_path("models", "final", "yolov8s.pt"),
]


def inspect_modules(module_names: tuple[str, ...]) -> tuple[list[tuple[str, bool, str]], bool]:
    """실제 import 기준으로 PyTorch backend 의존성을 점검한다."""
    results: list[tuple[str, bool, str]] = []
    missing = False
    for module_name in module_names:
        try:
            if importlib.util.find_spec(module_name) is None:
                raise ModuleNotFoundError(module_name)
            results.append((module_name, True, ""))
        except Exception as exc:  # pragma: no cover - 로컬 환경 차이를 그대로 보고한다.
            missing = True
            results.append((module_name, False, f"{type(exc).__name__}: {exc}"))
    return results, missing


def first_existing_path(candidates: list[Path]) -> Path | None:
    """후보 목록에서 실제 존재하는 첫 경로를 반환한다."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_check_only() -> int:
    """PyTorch backend 실행 준비 상태를 한국어로 출력한다."""
    dependency_results, has_missing = inspect_modules(("torch", "transformers", "ultralytics"))
    seg_model = first_existing_path(SEGMENTATION_CANDIDATES)
    det_model = first_existing_path(DETECTION_CANDIDATES)

    print("videoAssessor PyTorch backend 사전 점검")
    print("\n[의존성]")
    for module_name, is_available, detail in dependency_results:
        print(f"- {module_name}: {'OK' if is_available else 'MISSING'}")
        if detail:
            print(f"  상세: {detail}")

    print("\n[모델]")
    print(f"- SegFormer .pth: {seg_model if seg_model else 'MISSING'}")
    print(f"- YOLO .pt: {det_model if det_model else 'MISSING'}")

    if not has_missing and seg_model and det_model:
        print("\n결론: PyTorch backend 사전 조건이 충족되었습니다.")
        print("TODO: 전체 `videoAssessor` PyTorch 경로는 후속 통합 검토 대상입니다.")
        return 0

    print("\n결론: PyTorch backend 사전 조건이 아직 완전하지 않습니다.")
    print("TODO: 현재 저장소는 engine backend를 최종 활성 경로로 사용합니다.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    """backend별 단독 도움말을 위한 최소 parser를 만든다."""
    parser = argparse.ArgumentParser(description="videoAssessor PyTorch backend 점검 도구")
    parser.add_argument("--check-only", action="store_true", help="의존성과 모델 파일만 점검합니다.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """PyTorch backend의 공개 진입점."""
    args = build_parser().parse_args(argv)
    if args.check_only:
        return run_check_only()

    print("TODO: 전체 videoAssessor PyTorch backend 런타임은 아직 활성화하지 않았습니다.")
    print("현재 최종 사용 경로는 `videoAssessor.py --backend engine`입니다.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
