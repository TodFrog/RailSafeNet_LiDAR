# RailSafeNet_LiDAR 최종 납품용 저장소 재구성 계획

작성일: 2026-03-18  
기준: 회사 제출용 최종 프로젝트 납품 패키지 설계  
원칙: 이번 단계는 구조 설계 문서화만 수행하며, 실제 파일 이동/삭제는 수행하지 않음

## 1. 계획 목적과 설계 원칙

이 문서는 현재 저장소를 기준으로 회사 제출용 최종 패키지 구조를 설계하기 위한 계획서다. 목표는 연구/실험 흔적이 섞여 있는 현재 구조를 납품 중심 구조로 재정렬할 수 있도록 기준을 만드는 것이다.

핵심 원칙은 다음과 같다.

- 최종 제출본은 실행 코드, 모델, 요구사항 문서, 아키텍처 문서, 시험 항목 및 결과 보고서, 최종 프로젝트 보고서를 중심으로 단순하게 구성한다.
- 실험, 레거시, 참조가 끊긴 스크립트, 내부 분석 문서는 `archive/` 또는 내부 보관 영역으로 격리한다.
- 1차 재구성은 파일명 변경 없이 폴더 재배치 중심으로 설계한다.
- 불확실한 항목은 `ASSUMPTION` 또는 `TODO`로 표시하고, 실제 이동 전 별도 검증을 요구한다.
- 이번 단계에서는 파일을 이동하지 않는다.

## 2. 목표 최상위 디렉터리 구조

최종 납품용 목표 구조는 아래와 같이 제안한다.

```text
/
├─ README.md
├─ LICENSE.txt
├─ .gitignore
├─ src/
│  ├─ inference/
│  ├─ training/
│  ├─ conversion/
│  ├─ evaluation/
│  └─ common/
├─ models/
│  ├─ final/
│  ├─ converted/
│  └─ references/
├─ configs/
│  ├─ inference/
│  ├─ training/
│  └─ sweeps/
├─ data_samples/
│  ├─ rs19/
│  ├─ pilsen/
│  └─ outputs/
├─ docs/
│  ├─ requirements/
│  ├─ architecture/
│  ├─ tests/
│  ├─ final_report/
│  ├─ analysis/
│  └─ assets/
├─ requirements/
│  ├─ base.txt
│  ├─ inference.txt
│  ├─ training.txt
│  └─ conversion.txt
└─ archive/
   ├─ legacy/
   ├─ experiments/
   ├─ old_docs/
   └─ broken_or_unverified/
```

### 2.1 최상위 구조의 역할

- `src/`
  - 실행 가능한 Python 소스코드의 공식 위치
  - 추론, 학습, 변환, 평가, 공용 유틸을 하위 폴더로 분리
- `models/`
  - 최종 납품 모델, 변환 산출물, 참조용 포인터/출처 메모를 분리 저장
- `configs/`
  - 학습, 추론, sweep 설정 파일을 코드와 분리
- `data_samples/`
  - 회사 제출용 샘플 데이터와 예시 결과만 포함
- `docs/`
  - 제출 필수 문서를 목적별로 구분
- `requirements/`
  - 실행 목적별 의존성 파일을 분리
- `archive/`
  - 내부 보관용 영역
  - 최종 제출 패키지에는 포함하지 않음

### 2.2 필수 납품 산출물과 목표 위치

| 납품 항목 | 목표 위치 | 상태 |
|---|---|---|
| 소스코드(상세 주석 포함) | `src/` | 기존 코드 재배치 대상 |
| 모델 파일 | `models/final/`, `models/converted/` | 일부 실제 파일 부재 |
| 요구사항 명세서 | `docs/requirements/` | `TODO` 신규 문서 필요 |
| 소프트웨어 아키텍처 설계서 | `docs/architecture/` | `TODO` 신규 문서 필요 |
| 시험 항목 및 결과 보고서 | `docs/tests/` | `TODO` 신규 문서 필요 |
| 최종 프로젝트 보고서 | `docs/final_report/` | `TODO` 신규 문서 필요 |

## 3. 파일 이동 계획

### 3.1 안전한 1차 이동 전략

이번 재구성의 1차 목표는 “파일명을 유지한 채 카테고리별 디렉터리로 재배치”하는 것이다.

- 1단계: `src/`, `models/`, `configs/`, `data_samples/`, `docs/`, `requirements/`, `archive/` 구조를 먼저 만든다.
- 2단계: 루트 파일을 목적별 폴더로 이동하되, 파일명은 유지한다.
- 3단계: import 경로와 실행 경로를 검증한다.
- 4단계: 검증 후에만 선택적으로 이름 정리 또는 공통화 리팩터링을 검토한다.

이 방식은 현재 코드베이스가 중복 파일과 절대경로 의존을 많이 갖고 있기 때문에, 바로 파일명까지 바꾸는 것보다 롤백이 쉽고 안전하다.

### 3.2 루트 파일 매핑 계획

| 현재 경로 | 목표 경로 | 분류 | 최종 제출 포함 여부 | 근거 | 확신도/메모 |
|---|---|---|---|---|---|
| `LICENSE.txt` | `LICENSE.txt` | 루트 유지 | 포함 | 표준 라이선스 파일 | 높음 |
| `.gitignore` | `.gitignore` | 루트 유지 | 포함 | 납품 저장소 관리용 | 높음 |
| `TheDistanceAssessor_3.py` | `src/inference/TheDistanceAssessor_3.py` | 추론 | 포함 | 현재 최신 후보군 | 높음 |
| `TheDistanceAssessor_3_onnx.py` | `src/inference/TheDistanceAssessor_3_onnx.py` | 추론 | 포함 | ONNX 대체 추론 경로 | 높음 |
| `TheDistanceAssessor_3_engine.py` | `src/inference/TheDistanceAssessor_3_engine.py` | 추론 | 포함 후보 | TensorRT 후보 경로 | `ASSUMPTION`, 미추적 파일 |
| `production_segformer_onnx.py` | `src/inference/production_segformer_onnx.py` | 추론 지원 | 포함 | ONNX 추론 래퍼 | 높음 |
| `production_segformer_pytorch.py` | `src/inference/production_segformer_pytorch.py` | 추론 지원 | 포함 후보 | PyTorch fallback/디버그 경로 | 중간 |
| `train_SegFormer.py` | `src/training/train_SegFormer.py` | 학습 | 포함 | 기본 SegFormer 학습 | 높음 |
| `train_SegFormer_transfer_learning.py` | `src/training/train_SegFormer_transfer_learning.py` | 학습 | 포함 | 전이학습 핵심 스크립트 | 높음 |
| `train_DeepLabv3.py` | `src/training/train_DeepLabv3.py` | 학습 | 포함 | 대체 모델 학습 경로 | 높음 |
| `train_yolo.py` | `src/training/train_yolo.py` | 학습 | 포함 후보 | 매우 단순한 래퍼 | 중간 |
| `original_to_onnx.py` | `src/conversion/original_to_onnx.py` | 변환 | 포함 | SegFormer ONNX 변환 | 높음 |
| `onnx_to_engine.py` | `src/conversion/onnx_to_engine.py` | 변환 | 포함 | SegFormer TensorRT 변환 | 높음 |
| `yolo_original_to_onnx.py` | `src/conversion/yolo_original_to_onnx.py` | 변환 | 포함 | YOLO ONNX 변환 | 높음 |
| `yolo_onnx_to_engine.py` | `src/conversion/yolo_onnx_to_engine.py` | 변환 | 포함 | YOLO TensorRT 변환 | 높음 |
| `SegFormer_test.py` | `src/evaluation/SegFormer_test.py` | 평가 | 포함 | 평가 및 결과 생성 | 높음 |
| `video_frame_tester.py` | `src/evaluation/video_frame_tester.py` | 평가 | 포함 | 영상 프레임 기반 평가 | 높음 |
| `sweep_transfer.py` | `src/training/sweep_transfer.py` | 학습 지원 | 포함 후보 | 현재 전이학습 sweep 스크립트 | 높음 |
| `sweep_transfer.yaml` | `configs/sweeps/sweep_transfer.yaml` | sweep 설정 | 포함 후보 | 현재 유효 후보 설정 | 높음 |
| `sweep_transfer_aggressive.yaml` | `configs/sweeps/sweep_transfer_aggressive.yaml` | sweep 설정 | 포함 후보 | 실험 설정 | 중간 |
| `sweep_transfer_balanced.yaml` | `configs/sweeps/sweep_transfer_balanced.yaml` | sweep 설정 | 포함 후보 | 실험 설정 | 중간 |
| `yolov8n.pt` | `models/final/yolov8n.pt` | 모델 | 포함 후보 | 저장소 내 유일한 실제 모델 바이너리 | `TODO`, 실제 최종 사용 모델 확인 필요 |
| `ESSENTIAL_FILES_ROADMAP.md` | `archive/old_docs/ESSENTIAL_FILES_ROADMAP.md` | 구 문서 | 제외 | 현재 저장소와 참조 불일치 | 높음 |
| `result.txt` | `archive/experiments/result.txt` | 실험 로그 | 제외 | 과거 실험 결과 요약 | 높음 |
| `sweep.yaml` | `archive/broken_or_unverified/sweep.yaml` | 구 sweep | 제외 | 현재 없는 스크립트 참조 | 높음 |
| `sweep_add_agent.py` | `archive/broken_or_unverified/sweep_add_agent.py` | 구 실험 스크립트 | 제외 | 현재 없는 모듈 참조 | 높음 |
| `TheDistanceAssessor.py` | `archive/legacy/TheDistanceAssessor.py` | 레거시 | 제외 | PyTorch 기반 초기 통합 버전 | 높음 |
| `TheDistanceAssessor_2.py` | `archive/legacy/TheDistanceAssessor_2.py` | 레거시 | 제외 | 중간 세대 통합 버전 | 높음 |

### 3.3 `scripts/` 디렉터리 매핑 계획

| 현재 경로 | 목표 경로 | 분류 | 최종 제출 포함 여부 | 근거 | 확신도/메모 |
|---|---|---|---|---|---|
| `scripts/dataloader_RailSem19.py` | `src/common/dataloader_RailSem19.py` | 공용 유틸 | 포함 | 학습/평가 공용 데이터로더 | 높음 |
| `scripts/dataloader_SegFormer.py` | `src/common/dataloader_SegFormer.py` | 공용 유틸 | 포함 | SegFormer 전용 데이터로더 | 높음 |
| `scripts/metrics_all_cls.py` | `src/common/metrics_all_cls.py` | 공용 유틸 | 포함 후보 | 평가 함수군 | 중간 |
| `scripts/metrics_filtered_cls.py` | `src/common/metrics_filtered_cls.py` | 공용 유틸 | 포함 | 현재 참조 빈도 높음 | 높음 |
| `scripts/test_all_cls.py` | `src/evaluation/test_all_cls.py` | 평가 | 포함 후보 | 구 평가 스크립트 | 중간 |
| `scripts/test_filtered_cls.py` | `src/evaluation/test_filtered_cls.py` | 평가 | 포함 | 현재 여러 파일에서 참조 | 높음 |
| `scripts/test_pilsen.py` | `src/evaluation/test_pilsen.py` | 평가 | 포함 후보 | Pilsen 평가 스크립트 | 중간 |
| `scripts/pilsen.yaml` | `configs/training/pilsen.yaml` | 학습 설정 | 포함 후보 | YOLO/Pilsen 데이터 설정 | 중간 |

### 3.4 `assets/` 디렉터리 매핑 계획

| 현재 경로 | 목표 경로 | 분류 | 최종 제출 포함 여부 | 근거 | 확신도/메모 |
|---|---|---|---|---|---|
| `assets/README/*` | `docs/assets/` | 문서 자산 | 포함 | 문서 삽화/GIF/구조도 | 높음 |
| `assets/pilsen_railway_dataset/` | `data_samples/pilsen/` | 샘플 데이터 | 포함 후보 | 샘플 데이터/메타데이터 | 중간 |
| `assets/rs19val/` | `data_samples/rs19/` | 샘플 데이터 | 포함 후보 | 샘플 이미지/마스크/예시 스크립트 | 중간 |
| `assets/models_pretrained/segformer/*.txt` | `models/references/segformer/` | 모델 참조 | 선택 포함 | 실제 모델이 아닌 포인터 | 중간 |
| `assets/models_pretrained/ultralyticsplus/*.txt` | `models/references/yolo/` | 모델 참조 | 선택 포함 | 실제 모델이 아닌 포인터 | 중간 |
| `assets/requirements.txt` | `requirements/base.txt` | 의존성 초안 | 포함 | 현재 유일한 의존성 목록 | 높음 |

### 3.5 `docs/` 및 신규 문서 구조 계획

| 현재 경로 | 목표 경로 | 분류 | 최종 제출 포함 여부 | 근거 | 확신도/메모 |
|---|---|---|---|---|---|
| `docs/analysis/REPO_AUDIT.md` | `docs/analysis/REPO_AUDIT.md` | 내부 분석 | 제외 | 내부 감사 문서 | 높음 |
| `docs/analysis/RESTRUCTURING_PLAN.md` | `docs/analysis/RESTRUCTURING_PLAN.md` | 내부 분석 | 제외 | 내부 재구성 계획 | 높음 |
| `README.md` | `README.md` | 제출 문서 | `TODO` 생성 | 최종 제출본 개요 문서 필요 | 높음 |
| `docs/requirements/requirements_specification.md` | 동일 | 제출 문서 | `TODO` 생성 | 요구사항 명세서 필요 | 높음 |
| `docs/architecture/software_architecture_design.md` | 동일 | 제출 문서 | `TODO` 생성 | 아키텍처 설계서 필요 | 높음 |
| `docs/tests/test_items_and_results_report.md` | 동일 | 제출 문서 | `TODO` 생성 | 시험 항목 및 결과 보고서 필요 | 높음 |
| `docs/final_report/final_project_report.md` | 동일 | 제출 문서 | `TODO` 생성 | 최종 프로젝트 보고서 필요 | 높음 |

## 4. 아카이브 또는 격리 대상

### 4.1 `archive/`로 격리할 대상

아래 항목은 최종 제출본 중심 구조에 직접 두지 않는 것이 적절하다.

- 레거시 실행 경로
  - `TheDistanceAssessor.py`
  - `TheDistanceAssessor_2.py`
- 구 문서/불일치 문서
  - `ESSENTIAL_FILES_ROADMAP.md`
- 실험성 결과 로그
  - `result.txt`
- 참조가 끊긴 sweep/실험 스크립트
  - `sweep.yaml`
  - `sweep_add_agent.py`

### 4.2 격리 기준

- 현재 저장소에 없는 스크립트나 모듈을 참조하면 `archive/broken_or_unverified/`
- 과거 세대 실행 파이프라인이면 `archive/legacy/`
- 실험 결과 요약, 로그, 메모면 `archive/experiments/`
- 최신 구조와 맞지 않는 문서는 `archive/old_docs/`

## 5. 최종 제출에서 제외할 항목

최종 회사 제출 패키지에서는 아래 항목을 제외하는 것을 기본값으로 제안한다.

- `archive/` 전체
- `docs/analysis/` 전체
- `.git/`
- 가상환경
  - `.venv/`, `venv/`, `env/`, `ENV/`, `env.bak/`, `venv.bak/`
- 캐시/빌드 부산물
  - `__pycache__/`, `*.pyc`
- 로컬 실험 결과 디렉터리
  - `evaluation_results/`, `comparison_results/`, `video_results/`, `optimized_results/`, `fixed_results/`, `simple_results/`, `robust_results/`, `enhanced_results/`, `class_analysis/`
- 로컬 로그/추적 폴더
  - `wandb/`
  - `geckodriver.log`
- 현재 참조가 끊긴 구형 스크립트와 설정
  - `sweep.yaml`
  - `sweep_add_agent.py`
  - `ESSENTIAL_FILES_ROADMAP.md`

`ASSUMPTION`: `models/references/`는 최종 제출본에 꼭 필요하지 않을 수 있다. 실제 모델 바이너리를 제출할 수 있다면 참조 포인터는 부록 또는 내부 보관으로 돌리는 편이 더 깔끔하다.

## 6. 리스크와 롤백 고려사항

### 6.1 주요 리스크

1. 메인 추론 엔트리포인트 미확정
   - `TheDistanceAssessor_3.py`, `TheDistanceAssessor_3_onnx.py`, `TheDistanceAssessor_3_engine.py` 중 최종 대표 경로가 아직 확정되지 않았다.
2. 실제 모델 파일 부족
   - 현재 저장소에 실제 모델 바이너리는 `yolov8n.pt`만 확인된다.
   - 다수 스크립트는 `/home/mmc-server4/...` 외부 경로 모델을 전제한다.
3. import 경로 파손 가능성
   - `scripts.*` 기준 import가 많아 단순 폴더 이동만으로 실행이 깨질 수 있다.
4. 문서와 코드의 불일치
   - 기존 로드맵 문서와 일부 sweep 설정은 저장소 현재 상태와 맞지 않는다.
5. 샘플 데이터 범위 불명확
   - `assets/pilsen_railway_dataset/`, `assets/rs19val/`를 최종 제출에 어느 수준으로 포함할지 결정이 필요하다.

### 6.2 롤백 전략

실제 재구성 단계에서는 아래 롤백 절차를 기본값으로 둔다.

1. 파일 이동 시작 전 Git 커밋으로 기준 스냅샷을 고정한다.
2. 카테고리 단위로 이동한다.
   - 예: `src/inference/` 먼저
   - 다음 `src/common/`
   - 그 다음 `configs/`, `models/`
3. 각 이동 단위마다 import와 실행 검증을 수행한다.
4. 문제가 생기면 해당 디렉터리 단위 커밋만 되돌린다.
5. 파일명 변경은 최종 검증 후 별도 커밋으로 분리한다.

### 6.3 보수적 실행 순서 제안

실제 재구성이 승인될 경우, 이동 순서는 아래가 가장 안전하다.

1. `docs/`, `requirements/`, `configs/` 신설
2. `src/common/`로 유틸 이동
3. `src/evaluation/`로 평가 스크립트 이동
4. `src/conversion/`로 변환 스크립트 이동
5. `src/inference/`로 추론 스크립트 이동
6. 마지막에 `archive/`로 레거시/구형 자산 격리

## 7. 실행 전 확인해야 할 TODO

- `TODO`: 최종 대표 추론 엔트리포인트 1개 확정
- `TODO`: 실제 제출할 SegFormer/YOLO 모델 파일 확보 여부 확인
- `TODO`: `assets/requirements.txt`를 기준으로 `requirements/base.txt`, `inference.txt`, `training.txt`, `conversion.txt` 분리
- `TODO`: 회사 제출용 `README.md`와 4종 필수 문서 작성
- `TODO`: 샘플 데이터의 포함 범위와 외부 배포 가능 여부 확인
- `TODO`: 미추적 파일 `TheDistanceAssessor_3_engine.py`의 채택 여부 확정

## 8. 결론

현재 저장소는 연구/실험/운영 코드가 혼재한 상태이며, 회사 제출용 최종 패키지로는 구조가 평면적이고 역할 구분이 약하다. 따라서 최종 제출본은 `src`, `models`, `configs`, `data_samples`, `docs`, `requirements` 중심으로 재구성하고, 레거시 및 실험 자산은 `archive`로 분리하는 것이 가장 보수적이고 안전한 방향이다.

이번 문서는 그 재구성을 위한 기준선이며, 실제 이동은 엔트리포인트 확정과 모델 확보 이후 단계적으로 수행하는 것이 적절하다.
