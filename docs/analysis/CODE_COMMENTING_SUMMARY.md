# 코드 주석 보강 요약

## 문서화한 파일 목록
- `production_segformer_pytorch.py`
- `production_segformer_onnx.py`
- `TheDistanceAssessor_3_engine.py`
- `_root_wrapper.py`
- `src/inference/TheDistanceAssessor_3_engine.py`
- `src/inference/TheDistanceAssessor_3.py`
- `src/inference/TheDistanceAssessor_3_onnx.py`
- `src/inference/production_segformer_pytorch.py`
- `src/inference/production_segformer_onnx.py`
- `src/common/repo_paths.py`
- `src/common/dataloader_RailSem19.py`
- `src/common/dataloader_SegFormer.py`
- `src/common/metrics_filtered_cls.py`
- `src/conversion/onnx_to_engine.py`
- `src/conversion/yolo_onnx_to_engine.py`
- `src/conversion/original_to_onnx.py`
- `src/conversion/yolo_original_to_onnx.py`
- `src/evaluation/SegFormer_test.py`
- `src/evaluation/video_frame_tester.py`
- `src/evaluation/test_filtered_cls.py`
- `src/training/train_SegFormer.py`
- `src/training/train_DeepLabv3.py`
- `src/training/train_yolo.py`
- `src/training/sweep_transfer.py`
- `src/training/train_SegFormer_transfer_learning.py`

## 각 파일에서 보강한 내용 요약
- `production_segformer_pytorch.py`, `production_segformer_onnx.py`, `TheDistanceAssessor_3_engine.py`
  - 루트 wrapper가 왜 남아 있는지와 실제 구현 위치가 `src/` 아래라는 점을 한국어로 명시했다.
- `_root_wrapper.py`
  - 직접 실행과 import 호환을 동시에 유지하는 이유를 설명했다.
- `src/inference/TheDistanceAssessor_3_engine.py`
  - 전체 추론 파이프라인 개요를 모듈 상단에 추가했다.
  - TensorRT 엔진 래퍼, 전처리/후처리, 선로 경계 추정, 위험 구역 생성, 객체 위험도 분류, 시각화 흐름에 한국어 docstring을 추가했다.
  - `4`, `9`, `min_width=19`, `threshold=7`, 거리 구역 `[100, 400, 1000]`, 이동/비이동 클래스 집합 같은 휴리스틱의 의미를 주석으로 설명했다.
- `src/inference/TheDistanceAssessor_3.py`
  - 레거시 TensorRT 구현이라는 위치를 명시했다.
  - 엔진/YOLO 래퍼와 핵심 위험도 분류 함수에 유지보수 관점 설명을 추가했다.
- `src/inference/TheDistanceAssessor_3_onnx.py`
  - ONNX Runtime provider 선택과 출력 해석 방식 차이를 설명했다.
  - TensorRT 버전과 공통인 레일 추정/위험도 분류 로직의 역할을 한국어 docstring으로 정리했다.
- `src/inference/production_segformer_pytorch.py`
  - 모델 탐색 순서, 외부 절대경로 fallback의 의미, lazy import 의도를 명시했다.
  - CLI가 재현성 smoke test용이라는 점을 설명했다.
  - `--check-only` 사전 점검 경로를 추가하고, 의존성/모델 경로 점검 목적을 한국어로 문서화했다.
- `src/inference/production_segformer_onnx.py`
  - ONNX 보조 추론 모듈이라는 위치를 명시했다.
  - 전처리, 배치 예측, benchmark 함수의 목적과 한계를 한국어로 정리했다.
- `src/common/repo_paths.py`
  - 저장소 재구성 이후 공용 경로 유틸이 필요한 이유와 사용 범위를 보강했다.
- `src/common/dataloader_RailSem19.py`
  - DeepLabv3 학습용 데이터셋이라는 역할과 클래스 무시/재번호 부여 규칙을 설명했다.
- `src/common/dataloader_SegFormer.py`
  - `SegformerImageProcessor`와 결합되는 데이터셋이라는 점과 `255 -> 21` 라벨 처리 이유를 설명했다.
- `src/common/metrics_filtered_cls.py`
  - morphology, AP/mAP, IoU 계산 함수의 입력 형식과 배경 클래스 제외 규칙을 설명했다.
  - `remap_mask`가 현재 미사용 보존 함수라는 점을 명시했다.
- `src/conversion/onnx_to_engine.py`
  - 대상 GPU에서 직접 엔진을 빌드해야 하는 이유와 optimization profile 가정을 보강했다.
- `src/conversion/yolo_onnx_to_engine.py`
  - YOLO ONNX가 batch=1 고정 export라는 가정을 설명했다.
- `src/conversion/original_to_onnx.py`
  - 512x896 더미 입력이 후속 TensorRT 변환 및 런타임 규격과 연결된다는 점을 설명했다.
- `src/conversion/yolo_original_to_onnx.py`
  - 고정 해상도 export 의도와 후속 엔진 변환 연계를 설명했다.
- `src/evaluation/SegFormer_test.py`
  - 평가용 모델 로드, 클래스 매핑, 추론 후처리의 목적을 보강했다.
- `src/evaluation/video_frame_tester.py`
  - 프레임 기반 수동 검토용 스크립트라는 점과 단순 위험 구역 근사의 한계를 명시했다.
- `src/evaluation/test_filtered_cls.py`
  - 평가/시각화 도우미 함수의 역할과 수동 검토 성격을 설명했다.
- `src/training/train_SegFormer.py`
  - 기본 SegFormer 학습 스크립트의 위치, 하드코딩된 경로/하이퍼파라미터 성격, 핵심 함수 역할을 정리했다.
- `src/training/train_DeepLabv3.py`
  - DeepLabv3 학습 스크립트의 실험용 성격과 주요 helper 함수 역할을 보강했다.
- `src/training/train_yolo.py`
  - Ultralytics CLI를 호출하는 얇은 엔트리포인트라는 점과 `configs/training/pilsen.yaml` 사용 의도를 설명했다.
- `src/training/sweep_transfer.py`
  - WandB sweep 스크립트의 목적, GPU 고정 가정, subprocess 기반 실행 이유를 한국어로 정리했다.
- `src/training/train_SegFormer_transfer_learning.py`
  - 학습/평가/추론이 공유하는 19->13 클래스 매핑 가정을 강조했다.
  - freeze/unfreeze 전략의 의도를 한국어로 정리했다.

## 공통 휴리스틱/가정 정리
- 선로 후보 클래스는 주로 `4`, `9`를 사용하며 일부 평가/비디오 스크립트는 `1`도 함께 참고한다.
- 선로 폭 필터링의 `min_width=19`와 rail strip 탐색의 `min_width=5`는 노이즈 제거용 경험적 임계값이다.
- `threshold=7`은 rail side x 이동량의 급격한 점프를 이상치로 볼지 결정하는 계수다.
- 선로 crossing 병합에서 쓰는 간격 `50`과 보조 판정 `30`은 분기/가드레일에 의한 과분할을 줄이기 위한 경험 규칙이다.
- 거리 구역 기본값 `[100, 400, 1000]`은 위험 구역 계층을 만들기 위한 하드코딩 값이다.
- 객체 위험도 분류는 COCO 클래스 전체가 아니라 `accepted_moving`, `accepted_stationary` 집합만 사용한다.
- 일부 실행 경로는 저장소 내부 모델보다 외부 절대경로를 fallback으로 유지한다. 이는 과거 운영 환경 호환을 위한 보수적 선택이다.

## 여전히 불명확한 항목
- `TODO`: 클래스 ID `4`, `9`, `1`이 실제 데이터셋 라벨에서 어떤 의미로 최종 확정되었는지 근거 문서가 부족하다.
- `TODO`: 거리 구역 `100/400/1000`이 어떤 운영 시나리오나 제동 거리 기준에서 왔는지 코드 외 설명이 없다.
- `TODO`: `min_width=19`, `threshold=7`, crossing 간격 `50` 같은 하드코딩 임계값의 실험 로그 또는 튜닝 기록이 저장소에 없다.
- `ASSUMPTION`: 외부 절대경로 fallback은 과거 서버 운영 환경 호환을 위해 남아 있는 것으로 보이나, 최종 납품 기준에서 반드시 필요한지는 추가 확인이 필요하다.
- `ASSUMPTION`: 이동체/비이동체 클래스 집합은 시각화 경보 정책을 위한 선택으로 보이나, 안전성 기준 문서와 연결된 근거는 확인되지 않았다.

## 범위 메모
- 이번 작업은 `archive/`를 제외한 활성 코드만 대상으로 했다.
- 루트 wrapper는 동작 보존을 해치지 않는 범위에서 최소 설명을 보강했다.
- `__init__.py`와 `archive/` 내부 파일은 문서화 범위에서 제외했다.
