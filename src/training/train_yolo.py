#!/usr/bin/env python3
"""최종 유지용 YOLO 학습 진입 스크립트.

이 스크립트는 현재 저장소에서 active 상태로 유지하는 두 개의 학습 엔트리 중 YOLO 담당
파일이다. 복잡한 학습 파이프라인을 다시 구현하지 않고, 최종 제출 시점에 어떤 데이터 설정과
기본 모델을 사용해 학습을 시작하는지 명확하게 남기는 역할에 집중한다.

주의:
- 실제 학습은 `ultralytics` CLI에 위임한다.
- `comet_ml` 초기화 여부와 학습 epoch 등은 현재 코드의 기존 가정을 그대로 유지한다.
- 데이터 설정 파일은 `configs/training/pilsen.yaml`을 canonical 경로로 사용한다.
"""

from __future__ import annotations

import os

from src.common.repo_paths import repo_path

os.environ.setdefault("YOLO_CONFIG_DIR", str(repo_path(".ultralytics")))

import ultralytics
import comet_ml


DATA_CONFIG = repo_path("configs", "training", "pilsen.yaml")
DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMAGE_SIZE = 640


def main() -> int:
    """Ultralytics YOLO 학습 CLI를 호출한다."""
    ultralytics.checks()
    comet_ml.init()

    command = (
        f'yolo train model={DEFAULT_MODEL} '
        f'data="{DATA_CONFIG}" '
        f"epochs={DEFAULT_EPOCHS} "
        f"imgsz={DEFAULT_IMAGE_SIZE}"
    )
    return os.system(command)


if __name__ == "__main__":
    raise SystemExit(main())
