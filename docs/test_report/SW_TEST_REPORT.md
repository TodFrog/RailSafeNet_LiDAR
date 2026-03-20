# 소프트웨어 시험 항목 및 결과 보고서

- 문서명: `SW_TEST_REPORT.md`
- 작성 기준일: `2026-03-19`
- 대상 독자: 회사 제출 검토자, 개발 인수자
- 기준 저장소: `RailSafeNet_LiDAR`
- 보고서 성격: 현재 active main 기준의 증거 기반 시험 결과 보고

## 1. 시험 목적

본 보고서는 최종 사용자 진입점이 `videoAssessor.py` 하나로 통일된 이후, active runtime과
학습 엔트리가 현재 저장소 상태에서 어느 수준까지 검증되었는지 객관적으로 기록하기 위한 문서다.

핵심 목표:

- 루트 CLI가 정상적으로 노출되는지 확인
- backend별 preflight가 실제 모델/의존성 상태를 정확히 보고하는지 확인
- active Python 소스가 문법적으로 유효한지 확인
- 아직 미실행인 항목을 명확히 분리

## 2. 시험 환경

### 2.1 실제 검증 환경

- 운영체제: `Windows`
- Python: 현재 workspace의 `python`
- 저장소 상태: `videoAssessor` 중심 최소화 구조

### 2.2 목표 배포 환경

- 운영체제: `Linux`
- GPU: `NVIDIA`
- 우선 대상: `Linux + Nvidia Orin NX`

`ASSUMPTION`: 현재 세션은 목표 GPU 배포 환경이 아니므로 TensorRT 실동작과 카메라/비디오 전체
runtime 검증은 수행하지 않았다.

## 3. 시험 항목 목록

| TC ID | 시험 항목 | 대상 기능 | 근거 | 상태 |
|---|---|---|---|---|
| TC-01 | CLI 도움말 출력 확인 | root entry | `videoAssessor.py --help` | 실행됨 |
| TC-02 | engine backend preflight | canonical runtime 준비 상태 | `videoAssessor.py --backend engine --check-only` | 부분 검증 |
| TC-03 | onnx backend preflight | ONNX 참조 경로 준비 상태 | `videoAssessor.py --backend onnx --check-only` | 부분 검증 |
| TC-04 | pytorch backend preflight | PyTorch 참조 경로 준비 상태 | `videoAssessor.py --backend pytorch --check-only` | 부분 검증 |
| TC-05 | active Python 소스 문법 검증 | active source tree | `python -m compileall ...` | 실행됨 |
| TC-06 | engine backend 실제 비디오 실행 | canonical runtime 전체 동작 | `videoAssessor.py --backend engine --mode video` | 계획됨(미실행) |
| TC-07 | engine backend 카메라 실행 | camera mode | `videoAssessor.py --backend engine --mode camera` | 계획됨(미실행) |
| TC-08 | calibration 실행 | BEV/소실점 calibration | `--calibrate`, `--calibrate-vp` | 계획됨(미실행) |
| TC-09 | SegFormer training entry smoke test | `train_segformer.py` | training entry 존재/구문 | 계획됨(미실행) |
| TC-10 | YOLO training entry smoke test | `train_yolo.py` | training entry 존재/구문 | 계획됨(미실행) |

## 4. 시험 절차

| TC ID | 시험 절차 |
|---|---|
| TC-01 | `python videoAssessor.py --help` |
| TC-02 | `python videoAssessor.py --backend engine --check-only` |
| TC-03 | `python videoAssessor.py --backend onnx --check-only` |
| TC-04 | `python videoAssessor.py --backend pytorch --check-only` |
| TC-05 | `python -m compileall videoAssessor.py src/inference src/common src/training src/rail_detection src/utils` |
| TC-06 | Linux GPU runtime에서 `python videoAssessor.py --backend engine --mode video --video <path>` 실행 |
| TC-07 | Linux GPU runtime에서 `python videoAssessor.py --backend engine --mode camera --camera 0` 실행 |
| TC-08 | Linux GPU runtime에서 calibration 명령 실행 |
| TC-09 | Linux 학습 환경에서 `python src/training/train_segformer.py ...` 실행 |
| TC-10 | Linux 학습 환경에서 `python src/training/train_yolo.py` 실행 |

## 5. 기대 결과

| TC ID | 기대 결과 |
|---|---|
| TC-01 | root CLI help가 오류 없이 출력된다. |
| TC-02 | engine backend가 dependency와 모델 상태를 명확히 보고한다. |
| TC-03 | onnx backend가 ONNX dependency와 모델 상태를 명확히 보고한다. |
| TC-04 | pytorch backend가 `.pth`, `.pt`, `transformers` 상태를 명확히 보고한다. |
| TC-05 | active Python 소스가 문법 오류 없이 compile된다. |
| TC-06 | 비디오 입력 기반 engine runtime이 정상 동작한다. |
| TC-07 | 카메라 입력 기반 engine runtime이 정상 동작한다. |
| TC-08 | calibration 화면이 실행되고 설정 파일 저장이 가능하다. |
| TC-09 | SegFormer training entry가 지정한 경로로 학습을 시작한다. |
| TC-10 | YOLO training entry가 Ultralytics CLI를 호출해 학습을 시작한다. |

## 6. 실제 결과

| TC ID | 실제 결과 | 판정 | 증거 |
|---|---|---|---|
| TC-01 | `videoAssessor.py --help`가 정상 출력되었고 `backend`, `mode`, `check-only` 등 핵심 옵션이 노출되었다. | PASS | 이번 세션 명령 출력 |
| TC-02 | engine preflight가 SegFormer engine과 YOLO `.pt` fallback을 발견했고, `albumentations`, `tensorrt`, `pycuda` 누락을 명시적으로 보고했다. | CONDITIONAL | 이번 세션 명령 출력 |
| TC-03 | ONNX preflight가 SegFormer `.onnx`는 발견했으나 active YOLO `.onnx` 부재를 명시적으로 보고했다. | CONDITIONAL | 이번 세션 명령 출력 |
| TC-04 | PyTorch preflight가 SegFormer `.pth`와 YOLO `.pt`는 발견했으나 `transformers` 누락을 명시적으로 보고했다. | CONDITIONAL | 이번 세션 명령 출력 |
| TC-05 | `compileall`이 `videoAssessor.py`, `src/inference`, `src/common`, `src/training`, `src/rail_detection`, `src/utils`에 대해 문법 오류 없이 완료되었다. | PASS | 이번 세션 명령 출력 |
| TC-06 | 미실행. 현재 Windows workspace에는 TensorRT runtime dependency가 없고, 포함된 engine도 Titan RTX/Linux 기준 산출물이다. | CONDITIONAL | 미실행 사유 기록 |
| TC-07 | 미실행. 현재 검증 환경은 카메라 및 GPU runtime 기준이 아니다. | CONDITIONAL | 미실행 사유 기록 |
| TC-08 | 미실행. calibration은 interactive GUI와 실제 입력 영상 검증이 필요하다. | CONDITIONAL | 미실행 사유 기록 |
| TC-09 | 미실행. training runtime은 별도 Linux 학습 환경과 dataset 경로 정리가 필요하다. | CONDITIONAL | 미실행 사유 기록 |
| TC-10 | 미실행. training runtime은 별도 Linux 학습 환경과 Ultralytics/Comet 설정이 필요하다. | CONDITIONAL | 미실행 사유 기록 |

## 7. 판정 요약

### 7.1 실행됨

- `TC-01`
- `TC-05`

### 7.2 부분 검증

- `TC-02`
- `TC-03`
- `TC-04`

### 7.3 계획됨(미실행)

- `TC-06`
- `TC-07`
- `TC-08`
- `TC-09`
- `TC-10`

## 8. 리스크, 한계, 후속 조치

### 리스크 및 한계

- 현재 canonical runtime은 `engine` backend지만, Windows workspace에는 `albumentations`, `tensorrt`, `pycuda`가 없다.
- 포함된 `segformer_b3_original_13class.engine`는 Titan RTX/Linux 기준 산출물이다.
- ONNX backend는 SegFormer `.onnx`는 있으나 active YOLO `.onnx`가 없다.
- PyTorch backend는 `.pth`와 `.pt`는 있으나 `transformers`가 없다.
- training entry는 active 상태로 남아 있으나 실제 학습 runtime 검증은 수행하지 않았다.

### 후속 조치

1. Linux GPU runtime에서 `requirements.txt`와 TensorRT/CUDA 조합을 다시 설치한다.
2. 대상 배포 장비에서 `videoAssessor.py --backend engine --check-only`를 재실행한다.
3. 실제 비디오 입력으로 `TC-06`, `TC-07`, `TC-08`을 수행한다.
4. 필요 시 대상 장비 기준으로 SegFormer `.engine`를 재생성한다.
5. ONNX 경로를 활성화하려면 YOLO `.onnx` 산출물 정리가 필요하다.
6. 학습 환경에서 `train_segformer.py`, `train_yolo.py`를 별도 smoke test 한다.
