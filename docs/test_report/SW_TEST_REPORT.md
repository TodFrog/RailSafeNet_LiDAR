# 소프트웨어 시험 항목 및 결과 보고서

- 문서명: `SW_TEST_REPORT.md`
- 작성 기준일: `2026-03-19`
- 대상 독자: 회사 제출 검토자, 개발 인수자
- 기준 저장소: `RailSafeNet_LiDAR`
- 문서 성격: 현재 시점의 실제 실행 증거와 미실행 항목을 함께 기록하는 시험 보고서

## 1. 시험 목적

본 보고서는 현재 저장소 기준으로 실제 실행 가능한 시험 항목과 아직 미실행인 시험 항목을 구분해 기록하기 위한 문서입니다. 특히 다음을 확인하는 데 목적이 있습니다.

- 공식 PyTorch smoke path의 진입 가능 여부
- `--check-only` 기반 사전 점검 기능의 유효성
- 활성 Python 코드의 문법 상태
- 현재 workspace와 실제 학습/최적화 환경의 차이로 인한 제약
- ONNX/TensorRT/평가/변환 경로의 남은 검증 범위

## 2. 시험 환경

### 2.1 실제 검증 환경

- 운영체제: `Windows`
- Python: `Python 3.13`
- 저장소 경로: `C:\Users\user\.vscode\python\RailSafeNet_LiDAR`
- 실제 실행한 명령:
  - `python production_segformer_pytorch.py --help`
  - `python production_segformer_pytorch.py --check-only`
  - `python -c "from src.inference.production_segformer_pytorch import load_model, resolve_model_path; print(load_model.__name__); print(resolve_model_path.__name__)"`
  - `python -m compileall src/inference src/common src/evaluation src/conversion src/training`
  - `python production_segformer_pytorch.py --device cpu`

### 2.2 모델/아티팩트 상태

- 실제 포함 파일:
  - `models/converted/SegFormer_B3_1024_finetuned.pth`
  - `models/converted/segformer_b3_original_13class.onnx`
  - `models/final/segformer_b3_original_13class.engine`
  - `models/final/yolov8n.pt`

### 2.3 목표 배포 환경

- 문서상 우선 지원 환경: `Linux + Nvidia Orin NX`
- 사용자 제공 정보 기준:
  - 현재 Windows 환경은 문서/정리용 workspace
  - 학습 및 변환 환경은 별도 Linux 환경
  - 포함된 `segformer_b3_original_13class.engine`는 `Titan RTX` 기준 산출물

`ASSUMPTION`: 현재 세션은 목표 GPU 배포 환경이 아니므로 GPU/TensorRT 실동작 검증은 수행되지 않았습니다.

## 3. 시험 항목 목록

| TC ID | 시험 항목 | 대상 기능 | 근거 | 상태 |
|---|---|---|---|---|
| TC-01 | 공식 CLI 도움말 출력 확인 | 공식 PyTorch 추론 진입점 | `production_segformer_pytorch.py`, `_root_wrapper.py` | 실행됨 |
| TC-02 | 공식 사전 점검 경로 확인 | 의존성/모델 경로 preflight | `src/inference/production_segformer_pytorch.py` | 실행됨 |
| TC-03 | 공식 PyTorch 모듈 import smoke test | 공식 PyTorch 추론 모듈 import 가능성 | `src/inference/production_segformer_pytorch.py` | 실행됨 |
| TC-04 | 활성 Python 소스 문법 검증 | `src/` 하위 활성 코드 문법 상태 | `src/inference`, `src/common`, `src/evaluation`, `src/conversion`, `src/training` | 실행됨 |
| TC-05 | 공식 smoke path 직접 실행 | 공식 PyTorch 경로의 실제 런타임 진행 상태 | `production_segformer_pytorch.py` | 부분 검증 |
| TC-06 | 실제 SegFormer `.pth` 모델 로드 및 더미 추론 | 공식 PyTorch smoke path의 모델 기반 실행 | `src/inference/production_segformer_pytorch.py` | 계획됨(미실행) |
| TC-07 | 통합 TensorRT 위험도 분석 경로 실행 | 세그멘테이션, 선로 추정, YOLO 분류, 시각화 | `src/inference/TheDistanceAssessor_3_engine.py` | 계획됨(미실행) |
| TC-08 | ONNX 추론 또는 benchmark 경로 실행 | ONNX Runtime 기반 추론/benchmark | `src/inference/production_segformer_onnx.py`, `src/inference/TheDistanceAssessor_3_onnx.py` | 계획됨(미실행) |
| TC-09 | 평가 스크립트 기반 정량 지표 산출 | mAP/IoU 계열 평가 | `src/evaluation/SegFormer_test.py`, `src/evaluation/test_filtered_cls.py` | 계획됨(미실행) |
| TC-10 | 모델 변환 스크립트 실행 | PyTorch/YOLO -> ONNX, ONNX -> TensorRT | `src/conversion/*.py` | 계획됨(미실행) |

## 4. 시험 절차

| TC ID | 시험 절차 |
|---|---|
| TC-01 | `python production_segformer_pytorch.py --help` 실행 |
| TC-02 | `python production_segformer_pytorch.py --check-only` 실행 |
| TC-03 | `python -c "from src.inference.production_segformer_pytorch import load_model, resolve_model_path; print(load_model.__name__); print(resolve_model_path.__name__)"` 실행 |
| TC-04 | `python -m compileall src/inference src/common src/evaluation src/conversion src/training` 실행 |
| TC-05 | `python production_segformer_pytorch.py --device cpu` 실행 |
| TC-06 | Linux runtime에서 `models/converted/SegFormer_B3_1024_finetuned.pth` 또는 `--model-path`를 사용해 공식 PyTorch smoke path 실행 |
| TC-07 | TensorRT 엔진과 입력 자산을 준비한 뒤 `TheDistanceAssessor_3_engine.py` 경로 실행 |
| TC-08 | ONNX 모델 자산을 준비한 뒤 `production_segformer_onnx.py` 또는 `TheDistanceAssessor_3_onnx.py` 경로 실행 |
| TC-09 | 평가용 모델과 GT 자산을 준비한 뒤 `SegFormer_test.py` 또는 `test_filtered_cls.py` 실행 |
| TC-10 | 대상 GPU 환경에서 `original_to_onnx.py`, `onnx_to_engine.py`, `yolo_original_to_onnx.py`, `yolo_onnx_to_engine.py` 실행 |

## 5. 기대 결과

| TC ID | 기대 결과 |
|---|---|
| TC-01 | CLI 사용법과 옵션 목록이 출력되어야 한다. |
| TC-02 | 의존성 상태와 모델 경로 상태가 명시적으로 출력되어야 한다. |
| TC-03 | 모듈 import가 성공하고 `load_model`, `resolve_model_path`가 출력되어야 한다. |
| TC-04 | 활성 Python 소스에 문법 오류가 없어야 한다. |
| TC-05 | 런타임 실패 시에도 중단 지점과 실패 원인이 명확해야 한다. |
| TC-06 | 실제 모델 로드 후 `logits shape`가 출력되어야 한다. |
| TC-07 | 세그멘테이션, 선로 분석, danger zone 계산, YOLO 분류, 시각화가 수행되어야 한다. |
| TC-08 | ONNX 세션 초기화 후 추론 또는 benchmark 출력이 표시되어야 한다. |
| TC-09 | 정량 지표와 시각화 결과가 산출되어야 한다. |
| TC-10 | `.onnx` 또는 `.engine` 산출물이 생성되어야 한다. |

## 6. 실제 결과

### 6.1 실행됨

| TC ID | 실제 결과 | 판정 | 증거 |
|---|---|---|---|
| TC-01 | `production_segformer_pytorch.py --help` 실행 시 usage, 옵션 목록, `--check-only` 설명이 정상 출력되었다. | PASS | 이번 세션 명령 출력 |
| TC-02 | `--check-only` 실행 시 `torch: OK`, `transformers: MISSING`가 출력되었고, SegFormer 모델 후보 중 `models/converted/SegFormer_B3_1024_finetuned.pth`가 `FOUND`로 표시되었다. 종료 코드는 `1`이었다. 즉, 현재 workspace 기준으로 모델은 존재하지만 런타임 의존성은 미설치 상태임이 확인되었다. | PASS | 이번 세션 명령 출력 |
| TC-03 | import smoke test 결과 `load_model`, `resolve_model_path` 문자열이 출력되었다. | PASS | 이번 세션 명령 출력 |
| TC-04 | `compileall` 실행 시 `src/inference`, `src/common`, `src/evaluation`, `src/conversion`, `src/training`에 대해 문법 오류가 보고되지 않았다. | PASS | 이번 세션 명령 출력 |

### 6.2 부분 검증

| TC ID | 실제 결과 | 판정 | 증거 |
|---|---|---|---|
| TC-05 | `production_segformer_pytorch.py --device cpu` 실행 시 `ModuleNotFoundError: No module named 'transformers'`가 발생했다. 현재 Windows workspace는 문서/정리용 환경이며 원래 학습/실행 환경과 다르므로, 이 결과는 “현재 workspace에서 런타임 의존성이 준비되지 않음”을 보여준다. 모델 부재 때문은 아니며, 실제 로컬 `.pth` 존재는 TC-02에서 확인되었다. | CONDITIONAL | 이번 세션 Traceback 출력 |

### 6.3 계획됨(미실행)

| TC ID | 실제 결과 | 상태 | 증거 |
|---|---|---|---|
| TC-06 | 미실행. 로컬 `SegFormer_B3_1024_finetuned.pth`는 존재하지만, 현재 Windows workspace에서는 `transformers` 런타임이 준비되지 않아 실제 모델 로드까지 진행하지 않았다. | 미실행 | 저장소 모델 파일 상태, TC-02 결과 |
| TC-07 | 미실행. `segformer_b3_original_13class.engine`는 실제 파일이 존재하지만 Linux + Titan RTX 기준 엔진이므로 현재 Windows workspace에서 실행 증거를 만들지 않았다. | 미실행 | 사용자 제공 정보, 저장소 모델 파일 상태 |
| TC-08 | 미실행. `segformer_b3_original_13class.onnx`는 실제 파일이 존재하지만, ONNX Runtime 기반 실행/benchmark는 이번 세션에서 수행하지 않았다. | 미실행 | 저장소 모델 파일 상태 |
| TC-09 | 미실행. 평가 스크립트는 존재하지만 GT와 런타임 환경을 준비해 정량 지표를 다시 산출하지 않았다. | 미실행 | `src/evaluation/` 스크립트 존재 |
| TC-10 | 미실행. 변환 스크립트는 존재하지만 이번 세션에서는 재변환을 수행하지 않았다. 현재 `.onnx`와 `.engine`은 “실행 결과물”이 아니라 “가져온 산출물”로만 확인했다. | 미실행 | 저장소 모델 파일 상태, `src/conversion/` 스크립트 존재 |

## 7. 판정 요약

### 7.1 상태별 요약

| 구분 | TC ID |
|---|---|
| 실행됨 | `TC-01`, `TC-02`, `TC-03`, `TC-04` |
| 부분 검증 | `TC-05` |
| 계획됨(미실행) | `TC-06`, `TC-07`, `TC-08`, `TC-09`, `TC-10` |

### 7.2 종합 판단

- 공식 CLI, 사전 점검, import, 문법 검증은 현재 workspace 기준으로 확인되었다.
- `--check-only`는 “모델 존재”와 “의존성 부족”을 분리해 보여주므로 납품 검토용 preflight로 유효하다.
- 현재 Windows workspace에서는 `transformers`가 없으므로 공식 PyTorch smoke path의 end-to-end 실행은 완료되지 않았다.
- ONNX/TensorRT/평가/변환 경로는 코드와 일부 아티팩트는 존재하지만, 이번 세션 실행 결과는 없다.

따라서 현재 시험 상태는 다음과 같이 판단한다.

- 공식 preflight 경로: `적합`
- 공식 모델 기반 smoke path: `부분 적합`
- ONNX/TensorRT/평가/변환 경로: `증거 존재, 실행 검증 미완료`

## 8. 리스크, 한계, 후속 조치

### 8.1 리스크 및 한계

- 현재 Windows workspace는 원래 학습/최적화가 수행된 Linux 환경과 다르다.
- `transformers` 미설치 상태에서는 공식 PyTorch smoke path를 실제 모델 로드까지 진행할 수 없다.
- 포함된 TensorRT 엔진은 Titan RTX 기준 산출물이므로 다른 GPU에서의 동작 여부를 현재 보고서로 보장할 수 없다.
- ONNX/TensorRT benchmark, FPS, latency, accuracy 수치는 이번 보고서에 포함하지 않았다.
- 평가 스크립트의 최신 정량 결과 로그와 별도 시각화 산출물은 이번 세션에 생성하지 않았다.

### 8.2 후속 조치

1. Linux 런타임 환경에 `requirements.txt` 또는 `environment.yml` 기준 의존성을 설치한다.
2. `python production_segformer_pytorch.py --check-only`를 Linux 런타임에서 다시 수행한다.
3. `models/converted/SegFormer_B3_1024_finetuned.pth` 기준으로 공식 PyTorch smoke path를 end-to-end 재실행한다.
4. 대상 GPU에서 `segformer_b3_original_13class.engine`를 재검증하거나 재생성한다.
5. ONNX/TensorRT 경로와 평가 스크립트에 대한 실행 로그, 스크린샷, 산출물을 별도 보관한다.

## 증거 출처

- 이번 세션의 실제 명령 출력
  - `production_segformer_pytorch.py --help`
  - `production_segformer_pytorch.py --check-only`
  - import smoke test
  - `compileall`
  - `ModuleNotFoundError: No module named 'transformers'`
- 저장소 문서
  - `README.md`
  - `docs/user_manual/SETUP_AND_RUN.md`
  - `docs/requirements/REQUIREMENTS_SPEC.md`
- 저장소 코드 및 모델 파일
  - `src/inference/`
  - `src/evaluation/`
  - `src/conversion/`
  - `models/converted/`
  - `models/final/`
