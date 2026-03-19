#!/usr/bin/env python3
"""
PyTorch 기반 공식 재현 추론 진입점.

이 모듈은 최종 납품 검토에서 가장 먼저 확인할 수 있는 공식 실행 경로다.
기본 동작은 SegFormer `.pth` 모델을 로드해 더미 입력 1회 추론을 수행하는
smoke test이며, `--check-only`를 사용하면 실제 모델 로드 없이도 다음을 점검한다.

1. `torch`, `transformers` 런타임 의존성 존재 여부
2. `--model-path` 또는 `models/final/` 기준 모델 파일 존재 여부
3. 과거 운영 환경용 외부 절대경로 fallback 의존 여부

주의:
- 외부 `/home/mmc-server4/...` 경로는 과거 운영 환경 호환용 fallback이다.
- 최종 납품 기준에서는 저장소 내부 `models/final/` 경로 사용을 우선한다.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from src.common.repo_paths import repo_path


torch = None
SegformerConfig = None
SegformerForSemanticSegmentation = None


DEFAULT_MODEL_CANDIDATES = [
    repo_path("models", "final", "segformer_b3_production_optimized_rail_0.7500.pth"),
    repo_path("models", "final", "SegFormer_B3_1024_finetuned.pth"),
    repo_path("models", "converted", "SegFormer_B3_1024_finetuned.pth"),
    Path("/home/mmc-server4/RailSafeNet/models/segformer_b3_production_optimized_rail_0.7500.pth"),
    Path("/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth"),
]


def build_model_candidates(model_path: str | Path | None = None) -> list[Path]:
    """사용자가 지정한 경로와 기본 후보 경로를 우선순위대로 정리한다."""
    candidates: list[Path] = []
    if model_path is not None:
        candidates.append(Path(model_path).expanduser())
    candidates.extend(DEFAULT_MODEL_CANDIDATES)

    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """명시 경로, 저장소 내부 경로, 기존 절대경로 순으로 모델을 찾는다.

    최종 납품 기준으로는 저장소 내부 `models/final` 경로가 우선이며,
    외부 절대경로는 과거 운영 환경과의 호환을 위한 fallback이다.
    """
    unique_candidates = build_model_candidates(model_path=model_path)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    searched_paths = "\n".join(f"  - {candidate}" for candidate in unique_candidates)
    raise FileNotFoundError(
        "사용 가능한 SegFormer .pth 모델을 찾지 못했습니다.\n"
        "다음 경로를 확인했습니다.\n"
        f"{searched_paths}\n"
        "원하는 모델이 있다면 --model-path로 직접 지정하세요."
    )


def inspect_runtime_dependencies() -> tuple[list[tuple[str, bool, str]], bool]:
    """공식 실행 경로의 핵심 런타임 의존성을 실제 import 기준으로 점검한다.

    단순 문자열 비교가 아니라 실제 import를 시도해, 설치는 되어 있어도
    런타임 오류가 나는 경우까지 사전 점검 결과에 반영한다.
    """
    dependency_results: list[tuple[str, bool, str]] = []
    has_missing_dependency = False

    for module_name in ("torch", "transformers"):
        try:
            importlib.import_module(module_name)
            dependency_results.append((module_name, True, ""))
        except Exception as exc:  # pragma: no cover - 환경 의존 오류를 그대로 보고한다.
            has_missing_dependency = True
            detail = f"{type(exc).__name__}: {exc}"
            dependency_results.append((module_name, False, detail))

    return dependency_results, has_missing_dependency


def run_preflight_check(model_path: str | Path | None = None) -> int:
    """실제 모델 로드 없이 실행 준비 상태를 점검한다.

    반환 코드는 납품 검토 자동화에서 재사용하기 쉽도록 단순화한다.

    - `0`: 의존성과 모델 경로가 모두 준비됨
    - `1`: 의존성 누락 또는 모델 부재
    """
    print("RailSafeNet 실행 준비 사전 점검")

    dependency_results, has_missing_dependency = inspect_runtime_dependencies()
    print("\n[의존성 점검]")
    for module_name, is_available, detail in dependency_results:
        status = "OK" if is_available else "MISSING"
        print(f"- {module_name}: {status}")
        if detail:
            print(f"  상세: {detail}")

    print("\n[모델 경로 점검]")
    candidates = build_model_candidates(model_path=model_path)
    existing_candidates = [candidate for candidate in candidates if candidate.exists()]

    if existing_candidates:
        print(f"- 사용 가능한 모델 발견: {existing_candidates[0]}")
    else:
        print("- 사용 가능한 SegFormer .pth 모델을 찾지 못했습니다.")

    print("- 확인한 후보 경로:")
    for candidate in candidates:
        status = "FOUND" if candidate.exists() else "MISSING"
        print(f"  - [{status}] {candidate}")

    if not has_missing_dependency and existing_candidates:
        print("\n결론: 실행 준비 가능")
        return 0

    unresolved_items: list[str] = []
    if has_missing_dependency:
        unresolved_items.append("런타임 의존성 설치")
    if not existing_candidates:
        unresolved_items.append("SegFormer .pth 모델 배치 또는 --model-path 지정")

    print("\n결론: 실행 준비 미완료")
    print("필요 조치:")
    for item in unresolved_items:
        print(f"- {item}")
    return 1


def ensure_runtime_dependencies():
    """실행 시점에만 무거운 의존성을 로드한다.

    `--help` 같은 가벼운 명령에서 `transformers` 미설치로 즉시 실패하지 않도록
    lazy import 구조를 유지한다.
    """
    global torch, SegformerConfig, SegformerForSemanticSegmentation
    if torch is not None:
        return

    import torch as torch_module
    from transformers import SegformerConfig as segformer_config
    from transformers import SegformerForSemanticSegmentation as segformer_model

    torch = torch_module
    SegformerConfig = segformer_config
    SegformerForSemanticSegmentation = segformer_model


def resolve_device(device_name: str | None):
    """요청된 장치를 검증하고 torch.device로 변환한다."""
    ensure_runtime_dependencies()

    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 현재 환경에서 CUDA를 사용할 수 없습니다.")

    return torch.device(device_name)


class ProductionSegFormerPyTorch:
    """
    원본 .pth 가중치를 직접 사용하는 PyTorch 추론 래퍼.
    """

    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        self.model_path = resolve_model_path(model_path)
        self.device = resolve_device(device)
        self.num_labels = 13
        self._load_pytorch_model()

    def _load_pytorch_model(self) -> None:
        """모델 파일 형식에 따라 전체 모델 또는 state dict를 로드한다.

        저장 포맷이 일관되지 않으므로 파일명과 checkpoint key를 함께 보고 분기한다.
        """
        ensure_runtime_dependencies()
        print(f"모델 로드 경로: {self.model_path}")

        if "SegFormer_B3_1024_finetuned" in self.model_path.name:
            self.model = torch.load(self.model_path, map_location="cpu")
            print("원본 전체 SegFormer 모델을 로드했습니다.")
        else:
            config = SegformerConfig.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
            config.num_labels = self.num_labels
            self.model = SegformerForSemanticSegmentation(config)

            checkpoint = torch.load(self.model_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            print("학습 체크포인트 state dict를 로드했습니다.")

        self.model.to(self.device)
        self.model.eval()

        num_labels = getattr(getattr(self.model, "config", None), "num_labels", "Unknown")
        print(f"추론 장치: {self.device}")
        print(f"num_labels: {num_labels}")

    def __call__(self, pixel_values):
        """기존 호출 방식과 호환되는 추론 인터페이스."""
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(self.device)
        else:
            pixel_values = torch.from_numpy(pixel_values).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        return outputs

    def eval(self):
        """기존 코드 호환용 메서드."""
        self.model.eval()
        return self

    @property
    def config(self):
        """기존 코드 호환용 config 노출."""
        return self.model.config


def load_model(model_path: str | Path | None = None, device: str | None = None):
    """PyTorch 공식 재현 모델을 로드한다."""
    return ProductionSegFormerPyTorch(model_path=model_path, device=device)


def load_pytorch_model(model_path: str | Path | None = None, device: str | None = None):
    """기존 이름을 유지하는 호환 함수."""
    return load_model(model_path=model_path, device=device)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다.

    이 인터페이스는 정식 서비스 API가 아니라 재현성 확인용 smoke test 진입점이다.
    """
    parser = argparse.ArgumentParser(description="RailSafeNet PyTorch 재현 추론 smoke test")
    parser.add_argument("--model-path", type=str, default=None, help="명시적으로 사용할 .pth 모델 경로")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="추론 장치. 생략하면 CUDA 가능 여부를 자동 판별한다.",
    )
    parser.add_argument("--input-height", type=int, default=1024, help="더미 입력 높이")
    parser.add_argument("--input-width", type=int, default=1024, help="더미 입력 너비")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="실제 모델 로드 없이 의존성과 모델 경로만 사전 점검한다.",
    )
    return parser.parse_args()


def main() -> int:
    """모델 로드와 더미 입력 1회 추론을 수행하거나 사전 점검만 수행한다."""
    args = parse_args()

    if args.check_only:
        return run_preflight_check(model_path=args.model_path)

    ensure_runtime_dependencies()

    print("RailSafeNet PyTorch 재현 추론 smoke test")
    model = load_model(model_path=args.model_path, device=args.device)

    dummy_input = torch.randn(1, 3, args.input_height, args.input_width)
    output = model(dummy_input)

    print(f"출력 shape: {output.logits.shape}")
    print("PyTorch 모델 smoke test를 마쳤습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
