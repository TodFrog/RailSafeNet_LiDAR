# RailSafeNet_LiDAR

## Project Overview

`RailSafeNet_LiDAR`는 철도 환경 영상 기반 세그멘테이션, 선로 추정, 위험 구역 계산, 객체 검출을 결합한 안전 보조 연구/개발 저장소입니다. 현재 저장소는 회사 제출용 최종 패키지 구조를 기준으로 `src/`, `configs/`, `models/`, `data_samples/`, `docs/`, `requirements/` 중심으로 정리되어 있습니다.

공식 재현 경로는 루트 wrapper `production_segformer_pytorch.py`에서 시작하는 PyTorch smoke path입니다. ONNX/TensorRT 통합 추론, 학습, 평가, 변환 스크립트는 저장소에 실제로 포함되어 있으나, 문서 기준으로는 보조 경로로 구분합니다.

## Key Features

### 구현됨

- `src/inference/` 아래에 `PyTorch`, `ONNX`, `TensorRT` 기반 추론 스크립트가 분리되어 있습니다.
- `src/training/` 아래에 `SegFormer`, `DeepLabv3`, `YOLO` 학습 및 전이학습 스크립트가 포함되어 있습니다.
- `src/conversion/` 아래에 PyTorch/YOLO -> ONNX, ONNX -> TensorRT 변환 스크립트가 포함되어 있습니다.
- `src/evaluation/` 아래에 정량 지표 계산과 수동 검토용 평가 스크립트가 포함되어 있습니다.
- 루트 wrapper를 유지해 기존 실행 방식과 재구성된 `src/` 구현을 함께 지원합니다.
- 요구사항 명세서, 아키텍처 문서, 시험 보고서, 최종 보고서, 납품 체크리스트가 한국어 Markdown으로 정리되어 있습니다.

### 구현 상태에 대한 주의

- 공식 기본 경로는 전체 위험도 분석 파이프라인이 아니라 `production_segformer_pytorch.py` 기반 smoke path입니다.
- `archive/`는 보관 영역이며 기본 실행 경로가 아닙니다.
- 저장소에는 실제 SegFormer `.pth`, `.onnx`, `.engine` 산출물이 포함되어 있지만, 생성 환경이 현재 Windows workspace와 다릅니다.
- 포함된 TensorRT 엔진 `models/final/segformer_b3_original_13class.engine`는 Linux + Titan RTX 환경에서 가져온 산출물입니다. 다른 GPU에서 그대로 동작한다고 단정할 수 없으며 대상 장비에서 재생성이 필요할 수 있습니다.

## Repository Structure

```text
.
├─ README.md
├─ src/
│  ├─ inference/
│  ├─ training/
│  ├─ conversion/
│  ├─ evaluation/
│  └─ common/
├─ configs/
│  ├─ training/
│  └─ sweeps/
├─ models/
│  ├─ final/
│  ├─ converted/
│  ├─ references/
│  └─ MODEL_MANIFEST.md
├─ data_samples/
├─ docs/
│  ├─ user_manual/
│  ├─ requirements/
│  ├─ design/
│  ├─ test_report/
│  ├─ final_report/
│  ├─ delivery/
│  └─ analysis/
├─ requirements/
├─ requirements.txt
├─ environment.yml
└─ archive/
```

- `src/`: 실제 구현 코드
- `configs/`: 학습 및 sweep 설정
- `models/`: 실제 포함 모델, 변환 산출물, 포인터 파일, 모델 문서
- `data_samples/`: 제출용 샘플 데이터
- `docs/`: 사용자 안내, 요구사항, 설계, 시험, 최종 보고, 납품 검토 문서
- `archive/`: 보관용 레거시 자산

## Setup and Installation

자세한 절차는 `docs/user_manual/SETUP_AND_RUN.md`를 기준으로 합니다.

### `pip`

```bash
python -m pip install -r requirements.txt
```

### `conda`

```bash
conda env create -f environment.yml
conda activate railsafenet-delivery
```

- 문서상 우선 지원 환경: `Linux + Nvidia Orin NX`
- 현재 정리/검토 workspace: Windows
- `ASSUMPTION`: `environment.yml`의 `python=3.13`은 저장소 기준 환경 값이며, GPU 런타임 완전 호환 버전으로 확정된 값은 아닙니다.

## Model Preparation

현재 저장소에서 실제 확인되는 주요 모델/아티팩트는 다음과 같습니다.

- `models/converted/SegFormer_B3_1024_finetuned.pth`
- `models/converted/segformer_b3_original_13class.onnx`
- `models/final/segformer_b3_original_13class.engine`
- `models/final/yolov8n.pt`
- `models/references/segformer/SegFormer_B3_1024_finetuned.pth.txt`
- `models/references/yolo/yolov8s.pt.txt`

공식 PyTorch smoke path는 아래 순서로 SegFormer `.pth`를 찾습니다.

1. `--model-path`로 지정한 경로
2. `models/final/segformer_b3_production_optimized_rail_0.7500.pth`
3. `models/final/SegFormer_B3_1024_finetuned.pth`
4. `models/converted/SegFormer_B3_1024_finetuned.pth`
5. 기존 Linux 운영 환경 fallback 경로

`models/final/segformer_b3_original_13class.engine`는 실제 포함 파일이지만, 현재 저장소의 공식 기본 경로는 아닙니다. 이 엔진은 Linux + Titan RTX 기준 산출물이므로 현재 Windows workspace 또는 다른 GPU에서 그대로 사용하는 것을 전제로 하지 않습니다.

모델 상세 목록은 `models/MODEL_MANIFEST.md`를 참조하십시오.

## How to Run

공식 시작점은 루트 wrapper `production_segformer_pytorch.py`입니다.

### 1. 도움말 확인

```bash
python production_segformer_pytorch.py --help
```

### 2. 사전 점검

```bash
python production_segformer_pytorch.py --check-only
```

이 명령은 실제 모델 로드 없이 다음을 점검합니다.

- `torch`, `transformers` 의존성 존재 여부
- SegFormer `.pth` 후보 경로 존재 여부
- 현재 workspace 기준 실행 준비 상태

### 3. 공식 smoke path 예시

Linux 예시:

```bash
python production_segformer_pytorch.py --model-path models/converted/SegFormer_B3_1024_finetuned.pth --device cuda
```

Windows PowerShell 예시:

```powershell
python production_segformer_pytorch.py --model-path "models\converted\SegFormer_B3_1024_finetuned.pth" --device cpu
```

### 4. 보조 경로

- `production_segformer_onnx.py`
- `TheDistanceAssessor_3_onnx.py`
- `TheDistanceAssessor_3_engine.py`

위 경로는 저장소에 포함되어 있으나, 기본 사용자 안내는 공식 PyTorch smoke path를 중심으로 제공합니다.

## Test and Validation Overview

현재 저장소에는 `pytest`나 `unittest` 기반의 통합 자동화 체계가 확인되지 않았습니다. 대신 다음과 같은 실행/점검 수단이 정리되어 있습니다.

- `python production_segformer_pytorch.py --help`
- `python production_segformer_pytorch.py --check-only`
- `python -c "from src.inference.production_segformer_pytorch import load_model, resolve_model_path"`
- `python -m compileall src/inference src/common src/evaluation src/conversion src/training`
- `src/evaluation/SegFormer_test.py`
- `src/evaluation/test_filtered_cls.py`
- `src/evaluation/video_frame_tester.py`

현재 확인된 시험 결과와 미실행 항목은 `docs/test_report/SW_TEST_REPORT.md`를 참조하십시오.

## Final Deliverables Included in This Repository

현재 저장소 기준 주요 납품 구성은 다음과 같습니다.

- 소스 코드와 루트 wrapper
- 설정 파일: `configs/`, `requirements.txt`, `environment.yml`, `requirements/`
- 모델 및 아티팩트 문서: `models/MODEL_MANIFEST.md`
- 실제 포함 모델/아티팩트: `models/converted/SegFormer_B3_1024_finetuned.pth`, `models/converted/segformer_b3_original_13class.onnx`, `models/final/segformer_b3_original_13class.engine`, `models/final/yolov8n.pt`
- 사용자 실행 문서: `docs/user_manual/SETUP_AND_RUN.md`
- 요구사항 명세서: `docs/requirements/REQUIREMENTS_SPEC.md`
- 아키텍처 문서: `docs/design/SW_ARCHITECTURE.md`
- 시험 항목 및 결과 보고서: `docs/test_report/SW_TEST_REPORT.md`
- 최종 프로젝트 보고서: `docs/final_report/FINAL_REPORT.md`
- 납품 체크리스트 및 보완 로그: `docs/delivery/`
- 샘플 데이터: `data_samples/`

다만 `model files`와 `software test items and results report`는 현재 저장소 기준으로 여전히 `PARTIAL` 상태이며, 세부 사유는 `docs/delivery/DELIVERY_CHECKLIST.md`를 참조하십시오.

## Known Limitations

- 현재 Windows workspace는 문서 정리/리뷰 중심 환경이며, 원래 학습 및 변환이 수행된 Linux 환경과 동일하지 않습니다.
- 현재 Windows workspace에 `transformers`가 없는 것은 자연스러운 상태일 수 있으며, 이 경우 `--help`와 `--check-only` 중심 점검만 가능합니다.
- `models/final/segformer_b3_original_13class.engine`는 Linux + Titan RTX 기준 TensorRT 엔진이므로 다른 GPU, 다른 TensorRT 버전, 다른 운영체제에서 그대로 사용할 수 있다고 단정할 수 없습니다.
- 통합 위험도 분석 경로의 YOLO 자산은 여전히 `yolov8s` 외부 참조와 `yolov8n.pt` 실파일이 혼재합니다.
- 일부 코드에는 기존 `/home/mmc-server4/...` fallback 경로가 남아 있습니다.
- ONNX/TensorRT 경로는 코드와 파일은 존재하지만, 현재 Windows workspace 기준 재검증 결과는 없습니다.
- 클래스 ID 매핑과 danger zone 거리값(`100/400/1000`)의 최종 도메인 근거는 별도 검증이 필요합니다.

## Future Work

- 대상 Linux 런타임에서 `requirements.txt` 또는 `environment.yml` 기준 설치를 재검증합니다.
- `models/converted/SegFormer_B3_1024_finetuned.pth` 기준 공식 PyTorch smoke path를 end-to-end로 다시 실행합니다.
- 실제 배포 대상 GPU에 맞춰 TensorRT 엔진을 재생성하거나 재검증합니다.
- 통합 위험도 분석 경로에 필요한 YOLO 아티팩트 전략을 `yolov8n` 또는 `yolov8s` 기준으로 정리합니다.
- 평가 스크립트 기반 정량 결과와 스냅샷을 최종 제출용 부속 자료로 축적합니다.
