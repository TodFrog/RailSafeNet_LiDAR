# 소프트웨어 요구사항 명세서

- 문서명: `REQUIREMENTS_SPEC.md`
- 작성 기준일: `2026-03-19`
- 대상 독자: 회사 제출 검토자, 개발 인수자
- 기준 저장소 범위: `RailSafeNet_LiDAR`
- 문서 목적: 현재 저장소에 실제 존재하는 구현과 아티팩트를 기준으로 소프트웨어 요구사항을 명세합니다.

## 1. 문서 목적 및 범위

본 문서는 철도 환경 영상 기반 세그멘테이션 및 위험도 분석 저장소 `RailSafeNet_LiDAR`의 최종 제출용 소프트웨어 요구사항 명세서입니다. 요구사항은 현재 저장소에 실제로 존재하는 코드, 모델 파일, 문서, 환경 파일을 기준으로 작성하며, 저장소 근거가 없는 기능은 구현 완료로 서술하지 않습니다.

범위는 다음과 같습니다.

- 주 범위: 루트 wrapper `production_segformer_pytorch.py`를 통한 공식 PyTorch smoke path
- 보조 범위: ONNX/TensorRT 추론, 학습, 평가, 변환 기능
- 제외 범위: `archive/` 하위 보관 자산, 저장소에 없는 외부 서비스나 미제공 산출물

## 2. 시스템 개요

`RailSafeNet_LiDAR`는 철도 환경 영상을 입력으로 받아 선로 관련 영역을 분할하고, 필요 시 객체 검출 결과와 결합해 위험 구역을 추정하는 저장소입니다. 현재 저장소의 기능군은 다음과 같습니다.

- 추론
  - 공식 경로: `production_segformer_pytorch.py` 기반 PyTorch smoke path
  - 보조 경로: `src/inference/`의 ONNX/TensorRT 통합 추론
- 위험도 분석
  - 세그멘테이션 마스크 기반 선로 후보 및 ego track 추정
  - danger zone 생성
  - YOLO 기반 객체 위험도 분류
- 학습
  - SegFormer, DeepLabv3, YOLO 학습/전이학습 스크립트 제공
- 변환
  - PyTorch/YOLO -> ONNX
  - ONNX -> TensorRT
- 평가
  - IoU/mAP 계열 지표 계산
  - 샘플 시각화 및 프레임 단위 검토

## 3. 용어 정의

| 용어 | 정의 |
|---|---|
| SegFormer | 본 저장소에서 세그멘테이션 모델로 사용하는 Transformer 계열 구조 |
| YOLO | 객체 검출에 사용하는 모델 계열 |
| ONNX Runtime | ONNX 모델 추론을 위한 실행 환경 |
| TensorRT | NVIDIA GPU 대상 최적화 추론 엔진 및 변환 도구 |
| Ego Track | 여러 선로 후보 중 현재 주행 선로로 간주하는 중심 선로 |
| Danger Zone | 선로 폭과 거리 규칙을 바탕으로 생성한 위험 구역 |
| Smoke Test | 전체 성능 검증이 아니라 최소 실행 가능 여부를 확인하는 시험 |
| Root Wrapper | 루트 실행 명령을 유지하기 위해 `src/` 내부 구현을 노출하는 얇은 스크립트 |

## 4. 기능 요구사항

| ID | 요구사항 | 구현 근거 | 검증 방법 | 상태/비고 |
|---|---|---|---|---|
| FR-01 | 시스템은 루트 wrapper를 통해 공식 PyTorch 추론 진입점을 제공해야 한다. | `production_segformer_pytorch.py`, `_root_wrapper.py`, `src/inference/production_segformer_pytorch.py` | 코드 경로 확인, `python production_segformer_pytorch.py --help` | 구현됨 |
| FR-02 | 시스템은 SegFormer 기반 세그멘테이션 전처리 및 추론을 수행해야 한다. | `src/inference/production_segformer_pytorch.py`, `src/inference/production_segformer_onnx.py`, `src/inference/TheDistanceAssessor_3_engine.py` | 코드 검토, 모델 준비 후 수동 실행 | 구현됨. ONNX/TensorRT는 보조 경로 |
| FR-03 | 시스템은 세그멘테이션 마스크에서 선로 후보와 ego track을 추정해야 한다. | `src/inference/TheDistanceAssessor_3.py`, `src/inference/TheDistanceAssessor_3_onnx.py`, `src/inference/TheDistanceAssessor_3_engine.py` | 코드 검토, 샘플 입력 기반 수동 검토 | 구현됨(보조) |
| FR-04 | 시스템은 추정된 선로 폭을 바탕으로 거리 기반 위험 구역 경계를 생성해야 한다. | `border_handler`, `find_zone_border`, `create_danger_zone_mask` 관련 구현 | 코드 검토, 시각화 결과 확인 | 구현됨(제약 있음). 거리값 근거는 `TODO` |
| FR-05 | 시스템은 YOLO 기반 객체 검출과 위험 구역 비교를 통해 위험도를 분류해야 한다. | `manage_detections`, `classify_detections` 관련 구현 | 코드 검토, 통합 경로 수동 실행 | 구현됨(보조) |
| FR-06 | 시스템은 위험 구역과 검출 결과를 시각화하거나 결과 화면으로 표시해야 한다. | `show_result`, `video_frame_tester.py`, 평가 스크립트 | 코드 검토, 수동 실행 | 구현됨(보조) |
| FR-07 | 시스템은 세그멘테이션 성능 평가용 정량 지표 및 시각화 스크립트를 제공해야 한다. | `src/evaluation/SegFormer_test.py`, `src/evaluation/test_filtered_cls.py`, `src/common/metrics_filtered_cls.py` | 코드 경로 확인, 평가 스크립트 실행 | 구현됨(보조) |
| FR-08 | 시스템은 SegFormer/DeepLabv3/YOLO 학습 또는 전이학습 스크립트를 제공해야 한다. | `src/training/train_SegFormer.py`, `src/training/train_SegFormer_transfer_learning.py`, `src/training/train_DeepLabv3.py`, `src/training/train_yolo.py`, `src/training/sweep_transfer.py` | 코드 경로 확인 | 구현됨(보조) |
| FR-09 | 시스템은 모델을 ONNX 및 TensorRT 형식으로 변환하는 스크립트를 제공해야 한다. | `src/conversion/original_to_onnx.py`, `src/conversion/yolo_original_to_onnx.py`, `src/conversion/onnx_to_engine.py`, `src/conversion/yolo_onnx_to_engine.py` | 코드 경로 확인, 대상 환경 수동 실행 | 구현됨(보조) |

## 5. 비기능 요구사항

| ID | 요구사항 | 근거 | 검증 방법 | 비고 |
|---|---|---|---|---|
| NFR-01 | 우선 지원 환경은 `Linux + Nvidia Orin NX`여야 한다. | `README.md`, `docs/user_manual/SETUP_AND_RUN.md` | 문서 검토 | 구현 문서 존재. `ASSUMPTION`: 현재 포함 `.engine`는 Titan RTX 기준 산출물 |
| NFR-02 | 재현 설치를 위해 `requirements.txt`와 `environment.yml`이 제공되어야 한다. | 루트 `requirements.txt`, `environment.yml`, `requirements/` | 파일 존재 확인, 설치 절차 문서 검토 | 구현됨 |
| NFR-03 | 저장소는 `src/`, `configs/`, `models/`, `data_samples/`, `docs/` 중심 구조를 유지해야 한다. | 현재 디렉터리 구조, `README.md` | 디렉터리 구조 확인 | 구현됨 |
| NFR-04 | 모델/의존성 누락 시 시스템은 명시적 오류 또는 실패 메시지를 제공해야 한다. | `production_segformer_pytorch.py`의 `--check-only`, 예외 처리 | `--check-only`, 직접 실행 시험 | 구현됨(부분 검증) |
| NFR-05 | TensorRT 엔진은 대상 GPU 환경에서 생성/사용된다는 제약을 명시해야 한다. | `docs/user_manual/SETUP_AND_RUN.md`, `docs/design/SW_ARCHITECTURE.md`, `models/MODEL_MANIFEST.md` | 문서 검토 | 구현됨 |
| NFR-06 | 공식 문서와 사용자 안내는 한국어 기준으로 유지되어야 한다. | `README.md`, `docs/user_manual/`, `docs/requirements/`, `docs/design/`, `docs/test_report/`, `docs/final_report/`, `docs/delivery/` | 문서 검토 | 구현됨 |
| NFR-07 | 현재 재현 한계와 환경 차이를 명시해야 한다. | `README.md`, `SETUP_AND_RUN.md`, `SW_TEST_REPORT.md`, `FINAL_REPORT.md` | 문서 검토 | 구현됨 |

## 6. 입력/출력 정의

| 기능군 | 입력 | 출력 | 비고 |
|---|---|---|---|
| 공식 PyTorch 추론 | `.pth` 모델 경로, `--device`, `--input-height`, `--input-width` | 콘솔 로그, `logits.shape` | 공식 smoke path |
| 통합 위험도 추론 | 이미지 파일, 세그멘테이션 모델, 검출 모델, 거리 규칙 | 위험 구역 경계, 검출 분류 결과, 시각화 화면 | 보조 경로 |
| 학습 | 이미지/마스크 디렉터리, 하이퍼파라미터, sweep 설정 | 체크포인트, 로그 | 실제 데이터셋 필요 |
| 변환 | `.pth`, `.pt`, `.onnx` 입력 파일 | `.onnx`, `.engine` 파일 | 대상 GPU 제약 존재 |
| 평가 | 이미지, GT 마스크, 모델 경로 | IoU/mAP 계열 지표, 그래프, 샘플 이미지 | 수동 검토 스크립트 포함 |

## 7. 제약사항 및 가정

- `ASSUMPTION`: 공식 요구사항 명세 범위는 “공식 재현 경로 + 저장소 내 보조 구현”입니다.
- 현재 저장소에는 실제 SegFormer `.pth`, `.onnx`, `.engine` 파일이 포함되어 있습니다.
- `ASSUMPTION`: 현재 Windows workspace는 문서/검토용 환경이며, 실제 학습/최적화는 별도 Linux 환경에서 수행되었습니다.
- `ASSUMPTION`: `models/final/segformer_b3_original_13class.engine`는 Titan RTX 기준 산출물입니다.
- `TODO`: 실제 배포 장비가 `Orin NX`인 경우 현재 `.engine`를 그대로 사용할 수 있는지 검증이 필요합니다.
- `TODO`: 클래스 ID `4`, `9`, `1`의 최종 데이터셋 의미 근거는 별도 확인이 필요합니다.
- `TODO`: 거리 구역 `100/400/1000`의 도메인 기준은 저장소에 명시되어 있지 않습니다.
- `ASSUMPTION`: 외부 `/home/mmc-server4/...` 절대경로는 과거 운영 환경 호환용 fallback입니다.
- Docker 기반 재현 환경은 현재 제공되지 않습니다.
- 통합 YOLO 경로의 대표 모델 기준은 `yolov8n`과 `yolov8s` 참조가 혼재해 있어 추가 정리가 필요합니다.

## 8. 요구사항 검증 방법 요약

본 저장소의 요구사항 검증은 다음 방법의 조합으로 수행합니다.

- 문서 검토
  - README, 사용자 매뉴얼, 요구사항 명세서, 아키텍처 문서, 시험 보고서의 일관성 확인
- 코드 경로 확인
  - 각 요구사항이 실제 활성 코드에 연결되는지 확인
- Smoke Test
  - `--help`, `--check-only`, import, `compileall` 중심의 최소 실행 검증
- 수동 실행 검토
  - ONNX/TensorRT, 평가, 변환 스크립트는 대상 환경에서 수동 실행으로 검증
- 산출물 존재 확인
  - `requirements.txt`, `environment.yml`, `models/`, `docs/`, `configs/` 존재 여부 확인

주요 요구사항별 검증 방식은 4장과 5장의 표에 직접 연결되어 있으며, 공식 기본 경로와 보조 경로를 혼동하지 않도록 문서와 시험 보고서를 함께 참조합니다.
