# 소프트웨어 요구사항 명세서

- 문서명: `REQUIREMENTS_SPEC.md`
- 작성 기준일: `2026-03-19`
- 대상 독자: 회사 제출 검토자, 개발 인수자
- 기준 저장소: `RailSafeNet_LiDAR`
- 문서 목적: 현재 `main`에 남겨진 active deliverable을 기준으로 소프트웨어 요구사항을 정리한다.

## 1. 문서 목적 및 범위

본 문서는 현재 저장소에 실제로 남아 있는 active 구성만을 대상으로 한다. 최종 사용자 진입점은
루트 `videoAssessor.py`이며, 최종 runtime은 `engine` backend를 기준으로 정의한다.

범위 구분:

- 주 범위: `videoAssessor.py --backend engine`
- 보조 범위: `videoAssessor.py --backend onnx`, `videoAssessor.py --backend pytorch`
- 부가 범위: `src/training/train_segformer.py`, `src/training/train_yolo.py`
- 범위 제외: `archive/` 하위의 과거 runtime, 평가, 변환, delivery 과정 문서

## 2. 시스템 개요

`RailSafeNet_LiDAR`는 전방 철도 영상을 입력으로 받아 다음 순서의 처리를 수행하는 시스템이다.

1. SegFormer 기반 선로/배경 세그멘테이션 수행
2. 레일 후보 검출 및 ego track 추정
3. BEV 및 위험 구역 계산
4. YOLO 기반 객체 검출과 위험 구역 비교
5. 시각화 화면 출력 또는 결과 비디오 저장

현재 저장소는 위 파이프라인을 `videoAssessor.py` 단일 진입점 아래에서 backend 선택 방식으로
정리해 둔 상태다.

## 3. 용어 정의

| 용어 | 정의 |
|---|---|
| `videoAssessor.py` | 최종 사용자용 단일 CLI 진입점 |
| engine backend | TensorRT 기반 canonical runtime |
| onnx backend | ONNX 아티팩트와 dependency 상태를 점검하는 보조 backend |
| pytorch backend | PyTorch `.pth` 아티팩트와 dependency 상태를 점검하는 보조 backend |
| Ego Track | 현재 주행 중인 레일로 판단된 중심 선로 |
| BEV | Bird's Eye View. 상단 투영 기반 시각화/경로 분석 표현 |
| Danger Zone | 선로와 거리 규칙을 기반으로 계산되는 위험 구역 |
| Preflight / Check-only | 실제 runtime 대신 dependency와 모델 상태만 점검하는 사전 확인 절차 |
| Archive | active 사용 경로에서 제외됐지만 참고용으로 보존한 과거 구현 |

## 4. 기능 요구사항

| ID | 요구사항 | 구현 근거 | 검증 방법 | 상태/비고 |
|---|---|---|---|---|
| FR-01 | 시스템은 `videoAssessor.py` 단일 진입점을 제공해야 한다. | `videoAssessor.py` | `python videoAssessor.py --help` | 구현됨 |
| FR-02 | 시스템은 `engine`, `onnx`, `pytorch` backend 선택 옵션을 제공해야 한다. | `videoAssessor.py --backend` | 도움말 확인, 각 backend preflight 실행 | 구현됨 |
| FR-03 | 시스템은 `engine` backend에서 비디오/카메라 모드와 calibration 모드를 지원해야 한다. | `src/inference/video_assessor.py` | CLI 옵션 검토, Linux runtime에서 수동 실행 | 구현됨(환경 의존) |
| FR-04 | 시스템은 SegFormer 모델과 YOLO 모델을 조합해 통합 위험도 분석을 수행해야 한다. | `src/inference/video_assessor.py`, `src/rail_detection/` | engine runtime 수동 실행 | 구현됨(환경 의존) |
| FR-05 | 시스템은 backend별 preflight를 통해 dependency 누락과 모델 경로 상태를 명확히 보고해야 한다. | `videoAssessor.py`, `src/inference/video_assessor_onnx.py`, `src/inference/video_assessor_pytorch.py` | `--check-only` 실행 | 구현됨 |
| FR-06 | 시스템은 SegFormer 학습 엔트리와 YOLO 학습 엔트리를 active 상태로 유지해야 한다. | `src/training/train_segformer.py`, `src/training/train_yolo.py` | 파일 존재 확인, 문법 검사 | 구현됨 |
| FR-07 | 시스템은 active runtime과 archive 자산을 분리해 사용자 혼동을 줄여야 한다. | 루트 구조, `archive/` | 구조 검토 | 구현됨 |
| FR-08 | 시스템은 모델, 실행 방법, 구조, 시험 결과를 한국어 문서로 제공해야 한다. | `README.md`, `docs/*`, `models/MODEL_MANIFEST.md` | 문서 검토 | 구현됨 |

## 5. 비기능 요구사항

| ID | 요구사항 | 근거 | 검증 방법 | 비고 |
|---|---|---|---|---|
| NFR-01 | 최종 배포 목표 환경은 `Linux + Nvidia Orin NX`를 우선으로 한다. | `README.md`, `SETUP_AND_RUN.md` | 문서 검토 | 구현됨 |
| NFR-02 | 현재 포함된 TensorRT engine의 원래 산출 환경을 명시해야 한다. | `MODEL_MANIFEST.md`, `SETUP_AND_RUN.md` | 문서 검토 | 구현됨 |
| NFR-03 | 문서와 사용자 안내는 한국어 기준으로 유지해야 한다. | active docs 전체 | 문서 검토 | 구현됨 |
| NFR-04 | 모델/의존성 누락 시 과장 없이 명시적 상태를 출력해야 한다. | `--check-only` 결과 | preflight 실행 | 구현됨 |
| NFR-05 | active runtime과 archive reference를 구조적으로 분리해야 한다. | 루트 구조 | 디렉터리 검토 | 구현됨 |
| NFR-06 | provenance를 명시해 upstream 출처를 추적 가능하게 해야 한다. | `README.md`, `FINAL_REPORT.md` | 문서 검토 | 구현됨 |
| NFR-07 | 실제로 검증되지 않은 성능이나 완성도를 완료 상태로 기술하지 않아야 한다. | `README.md`, `SW_TEST_REPORT.md`, `FINAL_REPORT.md` | 문서 검토 | 구현됨 |

## 6. 입력/출력 정의

| 기능 | 입력 | 출력 | 비고 |
|---|---|---|---|
| `videoAssessor.py --backend engine --check-only` | dependency 상태, model path 후보 | 콘솔 상태 보고 | 실제 runtime 실행 없음 |
| `videoAssessor.py --backend engine --mode video` | 비디오 파일, 설정 파일, 모델 경로 | 시각화 창 또는 저장 비디오 | GPU/TensorRT 환경 필요 |
| `videoAssessor.py --backend engine --mode camera` | 카메라 장치 번호, 설정 파일, 모델 경로 | 실시간 시각화 창 | GPU/TensorRT 환경 필요 |
| `videoAssessor.py --backend onnx --check-only` | ONNX dependency 및 아티팩트 상태 | 콘솔 상태 보고 | 현재 보조 경로 |
| `videoAssessor.py --backend pytorch --check-only` | PyTorch dependency 및 아티팩트 상태 | 콘솔 상태 보고 | 현재 보조 경로 |
| `src/training/train_segformer.py` | 데이터셋 경로, epoch, learning rate 등 | 체크포인트, 학습 로그 | Linux 학습 환경 가정 잔존 |
| `src/training/train_yolo.py` | `configs/training/pilsen.yaml`, 기본 YOLO 모델 | 학습 로그, 학습 결과물 | Ultralytics CLI 위임 |

## 7. 제약사항 및 가정

- `ASSUMPTION`: 현재 canonical runtime 기준은 `engine` backend다.
- `ASSUMPTION`: `001-what-why-home` branch는 최신 runtime 설계 참고용이지만 현재 active main과 동일한 구조는 아니다.
- `ASSUMPTION`: 현재 포함된 `segformer_b3_original_13class.engine`는 Titan RTX/Linux 기준 산출물이다.
- `TODO`: 최종 배포 장비에서 현재 engine을 그대로 사용할지, 대상 장비에서 재생성할지 결정이 필요하다.
- `TODO`: active YOLO `.engine` 또는 `.onnx` 아티팩트는 현재 repo에 없다.
- `TODO`: `pytorch`와 `onnx` backend의 전체 runtime 통합은 현재 완료되지 않았고 preflight 수준만 active 상태다.
- `TODO`: training 스크립트의 기본 절대경로는 과거 Linux 학습 환경 흔적이므로 최종 재학습 시 재정리가 필요하다.

## 8. 요구사항 검증 방법 요약

주요 요구사항의 검증은 아래 방식으로 수행한다.

- CLI 검증: `python videoAssessor.py --help`
- backend preflight 검증:
  - `python videoAssessor.py --backend engine --check-only`
  - `python videoAssessor.py --backend onnx --check-only`
  - `python videoAssessor.py --backend pytorch --check-only`
- 정적 검증:
  - `python -m compileall videoAssessor.py src/inference src/common src/training src/rail_detection src/utils`
- 문서 검증:
  - README, 사용자 매뉴얼, 구조 문서, 시험 보고서, 최종 보고서, 모델 매니페스트 상호 정합성 확인
- 수동 검증:
  - Linux GPU runtime에서 `engine` backend 실제 비디오/카메라 실행
