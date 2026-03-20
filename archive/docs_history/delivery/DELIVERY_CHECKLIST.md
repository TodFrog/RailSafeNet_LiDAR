# 최종 납품 검토 체크리스트

- 문서명: `DELIVERY_CHECKLIST.md`
- 작성 기준일: `2026-03-19`
- 대상 독자: 회사 제출 검토자, 내부 납품 점검자
- 기준 저장소: `RailSafeNet_LiDAR`
- 문서 성격: 현재 저장소 상태 기준의 엄격한 납품 점검 문서

## 1. 문서 목적 및 판정 기준

본 문서는 “문서가 존재하는가”만이 아니라, 실제 납품 적합성까지 함께 판단하기 위한 체크리스트입니다.

- `COMPLETE`
  - 저장소 안에 요구 산출물이 존재하고, 납품 관점의 기본 요건을 충족함
- `PARTIAL`
  - 산출물은 존재하지만 핵심 자산, 실행 검증, 환경 적합성, 범위 정리가 일부 부족함
- `MISSING`
  - 저장소 안에서 요구 산출물을 확인할 수 없음

## 2. 현재 저장소 검토 기준 상태

`git status --short --branch` 기준 현재 상태는 다음과 같습니다.

- 브랜치 상태: `main...origin/main [behind 1]`
- 작업 트리 상태: clean 상태가 아닌 dirty worktree
- 기존 루트 파일 수정, 신규 `src/`, `docs/`, `models/`, `configs/`, `data_samples/`, `archive/` 구조가 함께 존재

납품 관점 주의사항:

- 현재 브랜치는 `origin/main`보다 1커밋 뒤처져 있습니다.
- 작업 트리가 정리되지 않았으므로 제출 직전에는 정리와 재확인이 필요합니다.
- 문서와 구조는 정리되었지만, 제출 직전 저장소 상태 관리까지 완료된 것은 아닙니다.

## 3. 필수 납품 산출물 점검표

| 산출물 | 상태 | 근거 | 판정 사유 | 관련 경로 |
|---|---|---|---|---|
| source code with detailed comments | `COMPLETE` | `docs/analysis/CODE_COMMENTING_SUMMARY.md` | 최종 제출 범위로 보는 활성 entrypoint, 공용 유틸, 주요 학습 스크립트에 한국어 주석과 docstring이 보강되어 있음 | `src/`, 루트 wrapper, `docs/analysis/CODE_COMMENTING_SUMMARY.md` |
| model files | `PARTIAL` | `models/MODEL_MANIFEST.md`, `models/converted/SegFormer_B3_1024_finetuned.pth`, `models/converted/segformer_b3_original_13class.onnx`, `models/final/segformer_b3_original_13class.engine`, `models/final/yolov8n.pt` | 핵심 SegFormer `.pth/.onnx/.engine`와 YOLO `.pt`는 존재한다. 다만 포함된 TensorRT 엔진은 Titan RTX/Linux 기준 산출물이며, 통합 YOLO 경로의 대표 아티팩트 기준이 아직 정리되지 않았다. | `models/final/`, `models/converted/`, `models/references/`, `models/MODEL_MANIFEST.md` |
| requirements specification | `COMPLETE` | `docs/requirements/REQUIREMENTS_SPEC.md` | 요구사항 문서가 존재하고 범위, 기능 요구사항, 비기능 요구사항, 검증 방법이 정리되어 있음 | `docs/requirements/REQUIREMENTS_SPEC.md` |
| software architecture design | `COMPLETE` | `docs/design/SW_ARCHITECTURE.md` | 실제 저장소 구조와 실행 흐름에 맞춘 아키텍처 문서가 존재함 | `docs/design/SW_ARCHITECTURE.md` |
| software test items and results report | `PARTIAL` | `docs/test_report/SW_TEST_REPORT.md` | 문서와 실제 실행 증거는 존재한다. 다만 공식 PyTorch 경로의 end-to-end 실행, ONNX/TensorRT, 평가, 변환 경로는 아직 미실행이다. | `docs/test_report/SW_TEST_REPORT.md` |
| final project report | `COMPLETE` | `docs/final_report/FINAL_REPORT.md` | 최종 보고서가 존재하고 구현 결과, 관찰된 실행 수준, 한계, 향후 과제가 분리되어 있음 | `docs/final_report/FINAL_REPORT.md` |

### 현재 기준 총평

- 현재 납품 준비 상태 총평: `PARTIAL`

총평이 `PARTIAL`인 이유는 다음과 같습니다.

- 필수 문서 4종과 사용자 문서, 모델 문서는 존재합니다.
- 핵심 모델 파일도 실제로 포함되어 있습니다.
- 그러나 현재 저장소는 dirty worktree 상태이고, 실제 런타임 검증은 아직 부분 수준입니다.
- 포함된 TensorRT 엔진의 장비 적합성과 통합 YOLO 아티팩트 전략이 아직 확정되지 않았습니다.

## 4. 누락 또는 미완료 항목 상세

### 4.1 model files

- `models/converted/SegFormer_B3_1024_finetuned.pth` 존재
- `models/converted/segformer_b3_original_13class.onnx` 존재
- `models/final/segformer_b3_original_13class.engine` 존재
- `models/final/yolov8n.pt` 존재

하지만 다음 이유로 `PARTIAL` 유지가 적절합니다.

- 포함된 `.engine`는 Titan RTX/Linux 기준 산출물이라 다른 GPU에서의 즉시 사용성을 보장할 수 없습니다.
- 통합 추론 코드는 여전히 `yolov8s` 외부 자산 참조를 포함합니다.
- 최종 제출에서 대표 YOLO 모델 전략이 정리되지 않았습니다.

### 4.2 software test items and results report

- `TC-01`~`TC-04`는 실제 실행됨
- `TC-05`는 부분 검증
- `TC-06`~`TC-10`은 미실행

현재 `PARTIAL`인 이유:

- `--check-only` 기준 모델 존재는 확인했으나, 현재 Windows workspace에서는 `transformers`가 없어 실제 PyTorch 모델 로드까지 진행하지 않았습니다.
- ONNX/TensorRT/평가/변환 경로는 파일과 스크립트는 존재하지만 재실행 증거가 없습니다.

### 4.3 제출 직전 저장소 상태 관리

- 현재 브랜치는 `origin/main`보다 1커밋 뒤처져 있습니다.
- 작업 트리가 dirty 상태입니다.
- 따라서 문서와 구조는 갖추었더라도 “제출 직전 상태 관리 완료”로 보기는 어렵습니다.

### 4.4 `001-what-why-home` branch 비교 결과

- `origin/001-what-why-home`는 Jetson/Orin 배포를 전제로 재설계된 별도 branch입니다.
- 현재 납품 정리본은 `src/inference`, `src/common`, `src/training`, `src/evaluation`, `src/conversion`, `docs/`, `models/` 중심 구조입니다.
- 반면 `001-what-why-home`는 `src/rail_detection`, `docker/`, `config/`, `videoAssessor_final.py` 중심 구조이며, 실행 경로와 배포 전제가 다릅니다.
- 따라서 이번 `main` 승격에서는 `001-what-why-home`를 병합 기준으로 사용하지 않고 `reference-only`로 유지하는 것이 안전합니다.
- 후속 검토 후보:
  - Docker 기반 배포 문서
  - Jetson용 TensorRT 재생성 절차
  - `config/rail_tracker_config.yaml` 파라미터 비교

## 5. 권장 최종 제출 패키지 구조

다음은 회사 제출용 패키지와 내부 보관 자료를 분리하는 권장 구조입니다. 이번 단계에서는 실제 이동/삭제를 수행하지 않습니다.

### 5.1 제출 패키지 포함 권장

```text
RailSafeNet_LiDAR/
├─ README.md
├─ src/
├─ configs/
├─ requirements.txt
├─ environment.yml
├─ requirements/
├─ models/
│  ├─ final/
│  ├─ converted/
│  ├─ references/         # 필요 시만 포함
│  └─ MODEL_MANIFEST.md
├─ data_samples/
└─ docs/
   ├─ user_manual/SETUP_AND_RUN.md
   ├─ requirements/REQUIREMENTS_SPEC.md
   ├─ design/SW_ARCHITECTURE.md
   ├─ test_report/SW_TEST_REPORT.md
   └─ final_report/FINAL_REPORT.md
```

### 5.2 내부 보관 또는 제출 제외 권장

- `archive/`
- `docs/analysis/`
- `docs/delivery/`
- 비어 있는 `docs/architecture/`
- 비어 있는 `docs/tests/`

## 6. 최종 제출 전 체크리스트

- [ ] 작업 트리를 clean 상태로 정리했는가
- [ ] `origin/main`과의 차이를 확인했는가
- [ ] `models/converted/SegFormer_B3_1024_finetuned.pth` 기준 공식 smoke path를 다시 검증했는가
- [ ] 현재 포함된 TensorRT 엔진을 그대로 제출할지, 대상 GPU에서 재생성할지 결정했는가
- [ ] `requirements.txt` 또는 `environment.yml` 기준 설치를 목표 런타임에서 재검증했는가
- [ ] `python production_segformer_pytorch.py --check-only`가 목표 런타임에서 기대한 결과를 반환하는가
- [ ] 공식 PyTorch smoke path를 실제 모델로 재실행했는가
- [ ] `TC-06`~`TC-10`을 그대로 미실행으로 둘지, 추가 실행할지 결정했는가
- [ ] 통합 YOLO 경로의 대표 모델 기준을 `yolov8n` 또는 `yolov8s`로 정리했는가
- [ ] `001-what-why-home`를 reference-only로 유지할지, 후속 Jetson 통합 branch로 다시 검토할지 결정했는가
- [ ] 최종 제출 패키지에서 `archive/`, `docs/analysis/`, `docs/delivery/`를 제외할지 확정했는가
- [ ] 요구사항 명세서가 최신 상태인가
- [ ] 아키텍처 문서가 최신 상태인가
- [ ] 시험 보고서가 최신 상태인가
- [ ] 최종 보고서가 최신 상태인가
- [ ] 모델 매니페스트와 실제 포함 파일이 일치하는가
