#!/usr/bin/env python3
"""최종 사용자용 `videoAssessor` 단일 진입점.

이 파일은 회사 제출용으로 정리된 현재 저장소에서 사용자가 직접 실행해야 하는 유일한
루트 CLI다. 실제 통합 위험도 추론은 `engine` backend가 담당하고, `onnx`와 `pytorch`
backend는 현재 저장소에 포함된 모델/의존성 상태를 점검하는 보조 경로로 유지한다.

설계 원칙:
- 최종 사용자-facing 이름은 항상 `videoAssessor`로 통일한다.
- `TheDistanceAssessor*` 계열은 active tree에서 제거하고 archive로 보존한다.
- `engine` backend는 `001-what-why-home` branch의 최신 `videoAssessor_final.py` 흐름을
  현재 저장소 구조에 맞게 옮긴 canonical runtime이다.
- 현재 포함된 SegFormer TensorRT engine은 Linux + Titan RTX 기준 산출물이므로, 다른 GPU
  또는 다른 OS에서는 재생성 또는 재검증이 필요할 수 있다.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path

from src.common.repo_paths import repo_path


DEFAULT_TRACKER_CONFIG = repo_path("configs", "inference", "rail_tracker_config.yaml")
DEFAULT_BEV_CONFIG = repo_path("configs", "inference", "bev_config.yaml")
os.environ.setdefault("YOLO_CONFIG_DIR", str(repo_path(".ultralytics")))

ENGINE_SEG_CANDIDATES = [
    repo_path("models", "final", "segformer_b3_original_13class.engine"),
]

ENGINE_YOLO_ENGINE_CANDIDATES = [
    repo_path("models", "final", "yolov8n.engine"),
    repo_path("models", "final", "yolov8s.engine"),
]

ENGINE_YOLO_PT_CANDIDATES = [
    repo_path("models", "final", "yolov8n.pt"),
    repo_path("models", "final", "yolov8s.pt"),
]


def inspect_modules(module_names: tuple[str, ...]) -> tuple[list[tuple[str, bool, str]], bool]:
    """실제 import 기준으로 backend 실행 필수 모듈 상태를 점검한다."""
    results: list[tuple[str, bool, str]] = []
    has_missing = False
    for module_name in module_names:
        try:
            if importlib.util.find_spec(module_name) is None:
                raise ModuleNotFoundError(module_name)
            results.append((module_name, True, ""))
        except Exception as exc:  # pragma: no cover - 현재 로컬 환경 차이를 그대로 보여준다.
            has_missing = True
            results.append((module_name, False, f"{type(exc).__name__}: {exc}"))
    return results, has_missing


def first_existing_path(candidates: list[Path]) -> Path | None:
    """후보 경로 목록에서 실제로 존재하는 첫 번째 항목을 찾는다."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_engine_check_only() -> int:
    """`engine` backend의 사전 실행 가능 여부를 모듈 import 없이 점검한다."""
    dependency_results, has_missing = inspect_modules(
        ("cv2", "numpy", "yaml", "torch", "albumentations", "tensorrt", "pycuda", "ultralytics")
    )
    seg_engine = first_existing_path(ENGINE_SEG_CANDIDATES)
    yolo_engine = first_existing_path(ENGINE_YOLO_ENGINE_CANDIDATES)
    yolo_pt = first_existing_path(ENGINE_YOLO_PT_CANDIDATES)

    print("videoAssessor engine backend 사전 점검")
    print("\n[의존성]")
    for module_name, available, detail in dependency_results:
        print(f"- {module_name}: {'OK' if available else 'MISSING'}")
        if detail:
            print(f"  상세: {detail}")

    print("\n[모델]")
    print(f"- SegFormer engine: {seg_engine if seg_engine else 'MISSING'}")
    print(f"- YOLO engine: {yolo_engine if yolo_engine else 'MISSING'}")
    print(f"- YOLO pt fallback: {yolo_pt if yolo_pt else 'MISSING'}")

    print("\n[환경 제약]")
    print("- 포함된 SegFormer TensorRT engine은 Linux + Titan RTX 기준 산출물이다.")
    print("- 현재 Windows workspace에서는 구조 점검과 경로 검증을 우선 수행한다.")
    print("- 최종 배포 장비에서는 TensorRT/CUDA 조합에 맞는 재검증이 필요할 수 있다.")

    ready = seg_engine is not None and (yolo_engine is not None or yolo_pt is not None)
    if ready and not has_missing:
        print("\n결론: engine backend 기본 실행 조건이 충족되었다.")
        return 0

    if ready:
        print("\n결론: 모델 경로는 준비되어 있으나 현재 환경 의존성이 부족하다.")
        return 1

    print("\n결론: engine backend 실행 준비가 아직 완전하지 않다.")
    return 1


def load_backend(module_name: str):
    """선택한 backend 모듈을 실제 실행 시점에만 import한다."""
    return importlib.import_module(module_name)


def build_parser() -> argparse.ArgumentParser:
    """최종 사용자용 공통 CLI parser를 구성한다."""
    parser = argparse.ArgumentParser(
        description="RailSafeNet videoAssessor 최종 실행 진입점",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", choices=["engine", "onnx", "pytorch"], default="engine", help="실행 backend")
    parser.add_argument("--mode", choices=["video", "camera"], default="video", help="입력 모드")
    parser.add_argument("--video", type=str, default=None, help="직접 지정할 비디오 경로")
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 번호")
    parser.add_argument("--config", type=str, default=str(DEFAULT_TRACKER_CONFIG), help="선로 추적 설정 파일")
    parser.add_argument("--bev-config", type=str, default=str(DEFAULT_BEV_CONFIG), help="BEV 설정 파일")
    parser.add_argument("--fullscreen", action="store_true", help="전체 화면 표시")
    parser.add_argument("--output", type=str, default=None, help="화면 표시 없이 결과 비디오 저장 경로")
    parser.add_argument("--start-time", type=float, default=0.0, help="비디오 시작 시각(초)")
    parser.add_argument("--calibrate", action="store_true", help="BEV calibration 실행")
    parser.add_argument("--calibrate-vp", action="store_true", help="소실점 calibration 실행")
    parser.add_argument("--check-only", action="store_true", help="실행 준비 상태만 점검")
    parser.add_argument("--seg-model", type=str, default=None, help="SegFormer 모델 경로 override")
    parser.add_argument("--det-model", type=str, default=None, help="YOLO 모델 경로 override")
    return parser


def namespace_to_engine_argv(args: argparse.Namespace) -> list[str]:
    """engine backend로 넘길 CLI 인자 목록을 재구성한다."""
    argv: list[str] = [
        "--mode", args.mode,
        "--camera", str(args.camera),
        "--config", args.config,
        "--bev-config", args.bev_config,
        "--start-time", str(args.start_time),
    ]
    if args.video:
        argv.extend(["--video", args.video])
    if args.fullscreen:
        argv.append("--fullscreen")
    if args.output:
        argv.extend(["--output", args.output])
    if args.calibrate:
        argv.append("--calibrate")
    if args.calibrate_vp:
        argv.append("--calibrate-vp")
    if args.check_only:
        argv.append("--check-only")
    if args.seg_model:
        argv.extend(["--seg-model", args.seg_model])
    if args.det_model:
        argv.extend(["--det-model", args.det_model])
    return argv


def main(argv: list[str] | None = None) -> int:
    """선택한 backend에 따라 최종 실행 또는 사전 점검을 수행한다."""
    args = build_parser().parse_args(argv)

    if args.backend == "engine":
        if args.check_only:
            return run_engine_check_only()
        engine_backend = load_backend("src.inference.video_assessor")
        return engine_backend.main(namespace_to_engine_argv(args))

    if args.backend == "onnx":
        onnx_backend = load_backend("src.inference.video_assessor_onnx")
        if args.check_only:
            return onnx_backend.run_check_only()
        print("TODO: ONNX backend의 전체 videoAssessor 런타임은 아직 활성 경로로 통합되지 않았다.")
        print("현재 최종 사용 경로는 `videoAssessor.py --backend engine`이다.")
        return 1

    pytorch_backend = load_backend("src.inference.video_assessor_pytorch")
    if args.check_only:
        return pytorch_backend.run_check_only()

    print("TODO: PyTorch backend의 전체 videoAssessor 런타임은 아직 활성 경로로 통합되지 않았다.")
    print("현재 최종 사용 경로는 `videoAssessor.py --backend engine`이다.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
