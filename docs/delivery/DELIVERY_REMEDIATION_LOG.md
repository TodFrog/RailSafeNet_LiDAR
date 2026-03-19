# 납품 보완 작업 로그

- 문서명: `DELIVERY_REMEDIATION_LOG.md`
- 작성 기준일: `2026-03-19`
- 기준 문서: `docs/delivery/DELIVERY_CHECKLIST.md`
- 문서 목적: 체크리스트 기준 보완 작업, 검증 상태, 남은 이슈를 추적합니다.

## 1. 보완 이슈 요약

| 체크리스트 이슈 | 우선순위 | 영향 파일 | 조치 내용 | 검증 상태 | 남은 이슈 | 다음 권장 조치 |
|---|---|---|---|---|---|---|
| 공식 실행 경로와 실제 모델 파일 상태 불일치 | P0 | `src/inference/production_segformer_pytorch.py`, `README.md`, `docs/user_manual/SETUP_AND_RUN.md`, `models/MODEL_MANIFEST.md`, `docs/design/SW_ARCHITECTURE.md` | 로컬 `models/converted/SegFormer_B3_1024_finetuned.pth`를 공식 PyTorch 후보 경로에 반영하고 관련 문서를 모두 갱신했다. | 검증 완료 | 현재 Windows workspace에는 `transformers`가 없음 | Linux runtime에서 실제 smoke path 재실행 |
| 모델 아티팩트 설명 부정확 | P0 | `README.md`, `docs/user_manual/SETUP_AND_RUN.md`, `models/MODEL_MANIFEST.md`, `docs/final_report/FINAL_REPORT.md`, `docs/delivery/DELIVERY_CHECKLIST.md` | 실제 존재하는 `.pth`, `.onnx`, `.engine`, `.pt` 파일을 기준으로 설명을 다시 작성했다. Titan RTX 기준 엔진이라는 제약도 명시했다. | 검증 완료 | 대표 YOLO 계열과 대상 GPU 적합성은 아직 미확정 | 대상 GPU별 엔진 전략 확정 |
| 시험/최종 보고서의 모델 상태 및 환경 해석 보완 필요 | P1 | `docs/test_report/SW_TEST_REPORT.md`, `docs/final_report/FINAL_REPORT.md`, `docs/delivery/DELIVERY_CHECKLIST.md` | `--check-only` 결과를 “모델 존재 + 의존성 부족”으로 분리해 반영하고, 현재 Windows workspace가 원래 런타임이 아님을 명시했다. | 검증 완료 | ONNX/TensorRT/평가/변환 경로는 미실행 | 대상 환경에서 `TC-06`~`TC-10` 추가 수행 |
| 납품 판정 근거 보강 필요 | P1 | `docs/delivery/DELIVERY_CHECKLIST.md` | `model files`와 `software test items and results report`의 `PARTIAL` 판정 사유를 새 근거에 맞춰 수정했다. | 검증 완료 | dirty worktree, `behind 1` 상태 지속 | 제출 직전 작업 트리 정리 |
| `001-what-why-home` branch와의 역할 혼선 | P1 | `docs/delivery/DELIVERY_CHECKLIST.md`, `docs/delivery/DELIVERY_REMEDIATION_LOG.md` | `001-what-why-home`를 Jetson 배포용 별도 재설계 branch로 정의하고, 이번 `main` 승격에서는 `reference-only`로 유지한다는 결론을 기록했다. | 검증 완료 | Jetson 자산을 후속에 얼마나 통합할지 미정 | 필요 시 `codex/jetson-sync-*` branch에서 별도 비교 |

## 2. 상세 로그

### 2.1 공식 실행 경로와 실제 모델 파일 상태 불일치

- 우선순위: `P0`
- 영향 파일:
  - `src/inference/production_segformer_pytorch.py`
  - `README.md`
  - `docs/user_manual/SETUP_AND_RUN.md`
  - `models/MODEL_MANIFEST.md`
  - `docs/design/SW_ARCHITECTURE.md`
- 수행 내용:
  - 공식 PyTorch 후보 경로에 `models/converted/SegFormer_B3_1024_finetuned.pth`를 추가했다.
  - 문서에서 더 이상 “SegFormer `.pth`가 저장소에 없음”이라고 서술하지 않도록 수정했다.
  - ONNX와 TensorRT 산출물이 실제로 존재함을 반영했다.
- 검증 상태: `검증 완료`
- 남은 이슈:
  - 현재 Windows workspace에는 `transformers`가 없다.
  - 현재 `.engine`는 Titan RTX 기준 산출물이다.
- 다음 권장 조치:
  - Linux runtime에서 `python production_segformer_pytorch.py --check-only`
  - Linux runtime에서 실제 `.pth` 기반 smoke path 재실행

### 2.2 모델 아티팩트 설명 부정확

- 우선순위: `P0`
- 영향 파일:
  - `README.md`
  - `docs/user_manual/SETUP_AND_RUN.md`
  - `models/MODEL_MANIFEST.md`
  - `docs/final_report/FINAL_REPORT.md`
  - `docs/delivery/DELIVERY_CHECKLIST.md`
- 수행 내용:
  - 실제 포함 파일 목록을 다음 기준으로 재정리했다.
    - `models/converted/SegFormer_B3_1024_finetuned.pth`
    - `models/converted/segformer_b3_original_13class.onnx`
    - `models/final/segformer_b3_original_13class.engine`
    - `models/final/yolov8n.pt`
  - TensorRT 엔진이 Linux + Titan RTX 기준 산출물이라는 제약을 명시했다.
- 검증 상태: `검증 완료`
- 남은 이슈:
  - 통합 YOLO 경로의 대표 모델 기준 미확정
  - 대상 배포 GPU에서 엔진 재생성 필요 여부 미확정
- 다음 권장 조치:
  - `yolov8n` / `yolov8s` 기준 정리
  - 대상 GPU에서 TensorRT 재검증

### 2.3 시험/최종 보고서의 모델 상태 및 환경 해석 보완 필요

- 우선순위: `P1`
- 영향 파일:
  - `docs/test_report/SW_TEST_REPORT.md`
  - `docs/final_report/FINAL_REPORT.md`
  - `docs/delivery/DELIVERY_CHECKLIST.md`
- 수행 내용:
  - `TC-02` 설명을 “모델 부재”가 아니라 “모델 발견 + `transformers` 미설치”로 수정했다.
  - `TC-05`의 부분 검증 사유를 현재 Windows workspace 특성에 맞춰 수정했다.
  - 최종 보고서의 구현 상세, 배포 현황, 한계 항목을 최신 모델 상태와 환경 차이에 맞춰 수정했다.
- 검증 상태: `검증 완료`
- 남은 이슈:
  - 실제 Linux runtime에서의 end-to-end 검증 없음
  - ONNX/TensorRT/평가/변환 경로 미실행
- 다음 권장 조치:
  - Linux runtime에서 `TC-06` 재실행
  - 필요 시 `TC-07`~`TC-10` 증거 수집

### 2.4 납품 판정 근거 보강 필요

- 우선순위: `P1`
- 영향 파일:
  - `docs/delivery/DELIVERY_CHECKLIST.md`
  - `docs/delivery/DELIVERY_REMEDIATION_LOG.md`
- 수행 내용:
  - `source code with detailed comments`는 `COMPLETE` 유지
  - `model files`는 실제 파일 존재를 반영하되, 엔진 이식성과 YOLO 기준 미정 때문에 `PARTIAL` 유지
  - `software test items and results report`는 실행 증거 존재를 반영하되 미실행 시험이 많아 `PARTIAL` 유지
- 검증 상태: `검증 완료`
- 남은 이슈:
  - 작업 트리 clean 상태 아님
  - `origin/main` 대비 `behind 1`
- 다음 권장 조치:
  - 제출 직전 브랜치 정리
  - 최종 패키지 범위 확정

### 2.5 `001-what-why-home` branch와의 역할 혼선

- 우선순위: `P1`
- 영향 파일:
  - `docs/delivery/DELIVERY_CHECKLIST.md`
  - `docs/delivery/DELIVERY_REMEDIATION_LOG.md`
- 수행 내용:
  - `origin/001-what-why-home`를 Jetson/Orin 배포용 별도 재설계 branch로 정의했다.
  - 현재 회사 제출용 정리본과 `001-what-why-home`의 구조 차이(`src/inference` vs `src/rail_detection`, `configs/` vs `config/`, 공식 실행 경로 차이)를 명시했다.
  - 이번 `main` 승격에서는 `001-what-why-home`를 병합 기준으로 사용하지 않고 `reference-only`로 유지한다는 결론을 기록했다.
- 검증 상태: `검증 완료`
- 남은 이슈:
  - Docker/Jetson 자산을 후속에 얼마나 통합할지 결정되지 않음
  - `config/rail_tracker_config.yaml` 파라미터 비교는 아직 상세 검토하지 않음
- 다음 권장 조치:
  - 필요 시 `codex/jetson-sync-*` branch를 별도로 생성
  - Docker 배포 문서와 Jetson용 TensorRT 재생성 절차를 후속 통합 후보로 검토
