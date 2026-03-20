"""`videoAssessor`의 ONNX backend 사전 점검 모듈.

현재 저장소에는 SegFormer ONNX 산출물이 포함되어 있지만, YOLO ONNX 자산은 최종
활성 모델로 정리되어 있지 않다. 따라서 이 모듈은 ONNX runtime 전체 실행을
보장하는 대신, 실제 파일과 의존성의 준비 상태를 객관적으로 보여주는 역할만 맡는다.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from src.common.repo_paths import repo_path


SEGMENTATION_CANDIDATES = [
    repo_path("models", "converted", "segformer_b3_original_13class.onnx"),
]

DETECTION_CANDIDATES = [
    repo_path("models", "converted", "yolov8n.onnx"),
    repo_path("models", "converted", "yolov8s.onnx"),
]


def inspect_modules(module_names: tuple[str, ...]) -> tuple[list[tuple[str, bool, str]], bool]:
    """실제 import 기준으로 ONNX backend 의존성을 점검한다."""
    results: list[tuple[str, bool, str]] = []
    missing = False
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            results.append((module_name, True, ""))
        except Exception as exc:  # pragma: no cover
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
    """ONNX backend 준비 상태를 한국어로 출력한다."""
    dependency_results, has_missing = inspect_modules(("onnxruntime",))
    seg_model = first_existing_path(SEGMENTATION_CANDIDATES)
    det_model = first_existing_path(DETECTION_CANDIDATES)

    print("videoAssessor ONNX backend 사전 점검")
    print("\n[의존성]")
    for module_name, is_available, detail in dependency_results:
        print(f"- {module_name}: {'OK' if is_available else 'MISSING'}")
        if detail:
            print(f"  상세: {detail}")

    print("\n[모델]")
    print(f"- SegFormer .onnx: {seg_model if seg_model else 'MISSING'}")
    print(f"- YOLO .onnx: {det_model if det_model else 'MISSING'}")

    if not has_missing and seg_model and det_model:
        print("\n결론: ONNX backend 사전 조건이 충족되었습니다.")
        print("TODO: 전체 `videoAssessor` ONNX 경로는 후속 통합 검토 대상입니다.")
        return 0

    print("\n결론: ONNX backend 사전 조건이 아직 완전하지 않습니다.")
    if not det_model:
        print("TODO: 현재 저장소에는 최종 활성 YOLO ONNX 산출물이 없습니다.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    """backend별 단독 도움말을 위한 최소 parser를 만든다."""
    parser = argparse.ArgumentParser(description="videoAssessor ONNX backend 점검 도구")
    parser.add_argument("--check-only", action="store_true", help="의존성과 모델 파일만 점검합니다.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """ONNX backend의 공개 진입점."""
    args = build_parser().parse_args(argv)
    if args.check_only:
        return run_check_only()

    print("TODO: 전체 videoAssessor ONNX backend 런타임은 아직 활성화하지 않았습니다.")
    print("현재 최종 사용 경로는 `videoAssessor.py --backend engine`입니다.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
