# RailSafeNet_LiDAR 저장소 감사 보고서

작성일: 2026-03-18  
기준: 재현 포함형 산업 납품 패키지 사전 감사  
원칙: 이번 단계는 분석 전용이며, 코드 이동/삭제/대규모 리팩터링은 수행하지 않음

## 1. 현재 저장소 상태 요약

- Git 상태
  - 현재 브랜치 상태: `main...origin/main [behind 1]`
  - 추적 파일 삭제 1건: `train_SegFormer_transfer_production.py`
  - 미추적 파일 1건: `TheDistanceAssessor_3_engine.py`
- 저장소는 크게 다음 3개 영역으로 구성되어 있다.
  - 루트: `TheDistanceAssessor*` 계열 실행 스크립트, 학습 스크립트, 모델 변환 스크립트, sweep 설정/실행 스크립트, 실험 결과 텍스트
  - `scripts/`: dataloader, metrics, 테스트/평가 스크립트
  - `assets/`: README 이미지, 샘플 데이터셋, `models_pretrained` 포인터 파일, `rs19val` 샘플 자산
- 루트에 `docs/` 디렉터리가 없었으며, 본 감사 문서 추가를 위해 `docs/analysis/`를 생성했다.
- `.gitignore`에 `*.md` 전역 무시 규칙이 있어 문서 자산이 기본적으로 추적되지 않는 상태였고, 본 문서 추적을 위해 `docs/analysis/*.md` 예외 규칙을 추가했다.

### 1.1 현재 디렉터리 구조 요약

```text
.
├─ TheDistanceAssessor*.py
├─ train_*.py
├─ original_to_onnx.py / onnx_to_engine.py
├─ yolo_original_to_onnx.py / yolo_onnx_to_engine.py
├─ production_segformer_*.py
├─ sweep*.py / sweep*.yaml
├─ result.txt / ESSENTIAL_FILES_ROADMAP.md / LICENSE.txt
├─ scripts/
│  ├─ dataloader_RailSem19.py
│  ├─ dataloader_SegFormer.py
│  ├─ metrics_all_cls.py
│  ├─ metrics_filtered_cls.py
│  ├─ test_all_cls.py
│  ├─ test_filtered_cls.py
│  ├─ test_pilsen.py
│  └─ pilsen.yaml
└─ assets/
   ├─ README/
   ├─ pilsen_railway_dataset/
   ├─ rs19val/
   ├─ models_pretrained/
   └─ requirements.txt
```

### 1.2 핵심 모듈 식별

- 메인 파이프라인 계열
  - `TheDistanceAssessor.py`: PyTorch SegFormer + YOLO(`ultralyticsplus`) 기반 통합 파이프라인
  - `TheDistanceAssessor_2.py`: `scripts.test_filtered_cls`를 재사용하는 중간 세대 통합 파이프라인
  - `TheDistanceAssessor_3.py`: TensorRT 엔진 기반 통합 파이프라인
  - `TheDistanceAssessor_3_onnx.py`: ONNX Runtime 기반 통합 파이프라인
  - `TheDistanceAssessor_3_engine.py`: 미추적 TensorRT 변형 파이프라인
- 모델 래퍼/변환 계열
  - `production_segformer_pytorch.py`
  - `production_segformer_onnx.py`
  - `original_to_onnx.py`, `onnx_to_engine.py`
  - `yolo_original_to_onnx.py`, `yolo_onnx_to_engine.py`
- 학습/평가 계열
  - `train_SegFormer.py`
  - `train_SegFormer_transfer_learning.py`
  - `train_DeepLabv3.py`
  - `SegFormer_test.py`
  - `video_frame_tester.py`
- 공용 유틸 계열
  - `scripts/dataloader_RailSem19.py`
  - `scripts/dataloader_SegFormer.py`
  - `scripts/metrics_all_cls.py`
  - `scripts/metrics_filtered_cls.py`

## 2. 추정 메인 실행 흐름

### 2.1 메인 엔트리포인트 후보

저장소 내 `if __name__ == "__main__"`가 확인된 주요 실행 스크립트는 다음과 같다.

- 운영/추론
  - `TheDistanceAssessor.py`
  - `TheDistanceAssessor_2.py`
  - `TheDistanceAssessor_3.py`
  - `TheDistanceAssessor_3_onnx.py`
  - `TheDistanceAssessor_3_engine.py`
  - `production_segformer_onnx.py`
  - `production_segformer_pytorch.py`
  - `video_frame_tester.py`
- 학습/튜닝
  - `train_SegFormer.py`
  - `train_SegFormer_transfer_learning.py`
  - `train_DeepLabv3.py`
  - `train_yolo.py`
  - `sweep_transfer.py`
  - `sweep_add_agent.py`
- 변환
  - `original_to_onnx.py`
  - `onnx_to_engine.py`
  - `yolo_original_to_onnx.py`
  - `yolo_onnx_to_engine.py`
- 평가/테스트
  - `SegFormer_test.py`
  - `scripts/test_all_cls.py`
  - `scripts/test_filtered_cls.py`
  - `scripts/test_pilsen.py`
  - `assets/rs19val/example_vis.py`

### 2.2 현재 최종 진입점에 대한 판단

- `ESSENTIAL_FILES_ROADMAP.md`는 `TheDistanceAssessor_3.py`를 최신 최적화 버전으로 설명한다.
- 그러나 현재 작업 트리에는 미추적 파일 `TheDistanceAssessor_3_engine.py`가 존재하며, 내용상 `TheDistanceAssessor_3.py`의 TensorRT 변형/정리 버전으로 보인다.
- `TheDistanceAssessor_3.py`, `TheDistanceAssessor_3_onnx.py`, `TheDistanceAssessor_3_engine.py`는 동일한 문제 영역을 공유하고, 백엔드만 TensorRT/ONNX/세부 구현으로 갈리는 구조다.

`ASSUMPTION`: 현재 산업 납품 기준의 1순위 메인 실행 경로는 `_3` 계열(TensorRT/ONNX)일 가능성이 높다. 다만 `TheDistanceAssessor_3_engine.py`가 미추적 상태이므로, 현 시점에서 “최종 단일 메인 엔트리포인트”를 확정할 수는 없다.

### 2.3 세대별 실행 흐름 추정

1. `TheDistanceAssessor.py`
   - `production_segformer_pytorch.py`에서 PyTorch 모델 로드
   - YOLO는 `ultralyticsplus` 기반
   - 절대경로 결과 저장 로직과 디버그 출력이 포함됨
   - 레거시/디버그 경로로 보는 것이 타당함
2. `TheDistanceAssessor_2.py`
   - `scripts.test_filtered_cls`의 `load`, `load_model`, `process`를 재사용
   - `_1` 대비 중간 단계 전환 버전으로 보임
3. `TheDistanceAssessor_3.py`
   - TensorRT(`tensorrt`, `pycuda`) 엔진 사용
   - `PATH_model_seg`, `PATH_model_det` 모두 `.engine` 기준
   - 현재 로드맵 문서상 최신 버전으로 간주됨
4. `TheDistanceAssessor_3_onnx.py`
   - ONNX Runtime 기반 백엔드
   - TensorRT 미사용 환경에서의 대체 경로로 보임
5. `TheDistanceAssessor_3_engine.py`
   - TensorRT 백엔드를 별도 클래스로 재구성한 변형
   - 일부 파라미터/입력 크기/예외 처리 방식이 `_3.py`와 다름
   - 미추적 상태이므로 납품 산출물 포함 여부를 우선 결정해야 함

## 3. 모델 및 아티팩트 인벤토리

### 3.1 저장소 내부에 실제로 존재하는 모델 파일

| 항목 | 위치 | 상태 | 비고 |
|---|---|---|---|
| YOLO 가중치 | `yolov8n.pt` | 저장소 포함 | 루트에 실제 바이너리 존재 |

### 3.2 저장소 내부 포인터 파일

| 항목 | 위치 | 상태 | 비고 |
|---|---|---|---|
| SegFormer 포인터 | `assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth.txt` | 저장소 포함 | 실제 모델 아님 |
| YOLO 포인터 | `assets/models_pretrained/ultralyticsplus/yolov8s.pt.txt` | 저장소 포함 | 실제 모델 아님 |

### 3.3 스크립트가 기대하지만 저장소에 없는 모델/산출물

- SegFormer `.pth`
  - 다수 스크립트가 `/home/mmc-server4/.../*.pth` 절대경로를 직접 참조
- ONNX 산출물
  - `segformer_b3_transfer_best_0.7961_896x512.onnx`
  - `segformer_b3_original_13class_896x512.onnx`
  - `yolov8s_896x512.onnx`
- TensorRT 엔진
  - `segformer_b3_transfer_best_0.7961.engine`
  - `segformer_b3_original_13class.engine`
  - `yolov8s.engine`, `yolov8s_896x512.engine`

위 산출물은 스크립트상 기대 경로만 존재하며, 현재 저장소에는 실제 파일이 포함되어 있지 않다.

### 3.4 데이터/샘플 자산

- `assets/pilsen_railway_dataset/`
  - `train.json`, `test.json`, `eda_table.table.json` 포함
  - 대용량 JSON 메타데이터가 저장소에 포함됨
- `assets/rs19val/`
  - 샘플 이미지, 마스크, JSON, 시각화 스크립트 포함
  - 전체 데이터셋이 아니라 최소 샘플/구조 예시로 보임
- `assets/README/`
  - README용 시각화 이미지/GIF 포함

`TODO`: 산업 납품 시 실제로 포함해야 하는 모델 바이너리, 데이터셋 샘플 범위, 외부 배포 금지 자산 범위를 별도 확정해야 한다.

## 4. 실험 / 임시 / 중복 / 데드코드 의심 파일

### 4.1 중복 파이프라인 후보

- `TheDistanceAssessor_3.py`
- `TheDistanceAssessor_3_onnx.py`
- `TheDistanceAssessor_3_engine.py`

관찰 근거:

- 세 파일 모두 동일한 거리 평가/위험 구역/검출-분류 흐름을 공유한다.
- 차이는 주로 모델 로딩 백엔드(TensorRT/ONNX)와 일부 입출력 처리 방식이다.
- `git diff --no-index --stat` 기준으로 `_3.py` 대비 `_3_engine.py`, `_3_onnx.py`는 대규모 공통 코드를 유지한 변형 파일로 확인됐다.

정리 판단:

- 즉시 삭제 대상이 아니라 역할 명세가 필요한 중복군이다.
- 납품 전 “대표 실행 경로 1개 + 대체 백엔드 1개 + 레거시 보관” 구조로 격리 검토가 필요하다.

### 4.2 거의 동일한 유틸 파일 쌍

- `scripts/metrics_all_cls.py` / `scripts/metrics_filtered_cls.py`
  - diff 기준 소규모 차이만 존재
  - `remap_mask`는 두 파일 모두 `# not used` 주석 존재
- `scripts/dataloader_RailSem19.py` / `scripts/dataloader_SegFormer.py`
  - 공통 변환/분할 로직이 크고, 차이는 이미지 프로세서/클래스 무시 정책 쪽에 집중됨

정리 판단:

- 즉시 통합보다는 “공통부/차이점 문서화”가 우선이다.
- 실제 사용 경로가 정해진 뒤 공통 유틸로 분리 가능성이 높다.

### 4.3 실험 잔재 또는 임시 파일 의심

- `result.txt`
  - 과거 실험 요약 로그 성격
  - 현재 저장소에 없는 스크립트 다수를 언급함
- `sweep.yaml`
  - 현재 저장소에 존재하지 않는 `train_SegFormer_B3_improved.py`를 가리킴
- `sweep_add_agent.py`
  - `sweep_SegFormer` 모듈을 import하지만 현재 저장소에 없음
- `train_yolo.py`
  - 매우 짧은 래퍼 스크립트이며, `os.system('yolo train ...')` 호출만 수행

정리 판단:

- 삭제가 아니라 “실행 검증 전 격리 후보”로 분류하는 것이 안전하다.
- 납품 패키지에는 “지원 대상 스크립트”와 “보관용 실험 스크립트”를 구분해야 한다.

### 4.4 문서/참조 불일치

- `ESSENTIAL_FILES_ROADMAP.md`는 다음 파일을 언급하지만 현재 저장소에 존재하지 않는다.
  - `evaluate_segformer_rail_performance.py`
  - `create_production_model.py`
  - `test_model_comparison.py`
  - `train_SegFormer_transfer_production.py`

정리 판단:

- 현재 문서는 저장소 기준 최신 사실과 어긋난다.
- 납품 전 반드시 갱신하거나 별도 보관 문서로 이동해야 한다.

### 4.5 데드코드 가능성

- `scripts/metrics_all_cls.py`, `scripts/metrics_filtered_cls.py`의 `remap_mask`
  - 코드 주석상 `not used`
- `TheDistanceAssessor_3.py`, `TheDistanceAssessor_3_onnx.py`의 `model_type`
  - 주석상 호환성 유지용이며 실제 TensorRT/ONNX 경로에서는 사용되지 않음

`ASSUMPTION`: 위 항목은 제거 가능성이 높지만, 실제 외부 사용자 스크립트 참조 여부를 확인하기 전까지는 보류가 안전하다.

## 5. 의존성 및 환경 리스크

### 5.1 의존성 파일 리스크

- 루트 레벨 표준 의존성 파일(`requirements.txt`, `pyproject.toml`, `environment.yml`)이 없다.
- 현재 존재하는 의존성 목록은 `assets/requirements.txt`뿐이며, 내용도 최소 목록 수준이다.
- `assets/requirements.txt`에는 다음과 같은 한계가 있다.
  - 버전 고정 없음
  - 선택 의존성 구분 없음
  - TensorRT/PyCUDA/ONNX Runtime GPU 같은 환경 특화 패키지 미기재
  - `pathlib`처럼 표준 라이브러리성 항목이 포함됨

### 5.2 환경 종속성 리스크

- TensorRT 및 CUDA 계열
  - `tensorrt`
  - `pycuda`
- ONNX 계열
  - `onnx`
  - `onnxruntime`
- 학습/추론 계열
  - `torch`, `torchvision`
  - `transformers`
  - `albumentations`
  - `opencv-python`
- 실험 추적/YOLO 계열
  - `wandb`
  - `ultralytics`
  - `ultralyticsplus`
  - `comet_ml`

산업 납품 관점에서는 위 의존성을 최소 3개 프로필로 분리하는 것이 바람직하다.

- 추론 전용
- 학습/재현 전용
- 변환/TensorRT 빌드 전용

### 5.3 경로 및 실행 환경 리스크

- 다수 스크립트가 `/home/mmc-server4/...` 절대경로를 직접 사용한다.
- 일부 스크립트는 `Grafika/Video_export/...` 같은 로컬 상대경로를 사용한다.
- Linux 서버 전제를 강하게 가진 경로와 Windows 개발 환경이 혼재한다.

정리 판단:

- 현재 상태로는 “다른 시스템에 바로 복제 가능한 납품 패키지”라고 보기 어렵다.
- 환경 변수, 설정 파일, CLI 인자 기반 경로 분리가 필요하다.

### 5.4 import 구조 리스크

- `scripts.test_all_cls.py`는 `from metrics_all_cls import ...`, `from rs19_val.example_vis import ...`처럼 루트 기준이 아닌 과거 상대 구조를 사용한다.
- 반면 다른 스크립트는 `from scripts.metrics_filtered_cls import ...`처럼 현재 구조를 직접 참조한다.
- 모듈 import 스타일이 일관되지 않아 실행 위치에 따라 import 실패 가능성이 있다.

## 6. 제안 정리 우선순위

### P0. 납품 차단 리스크 정리

- 외부 절대경로를 설정값 또는 CLI 인자로 치환할 기준 수립
- 실제 모델/엔진/ONNX 파일 포함 범위 결정
- 저장소에 없는 스크립트를 참조하는 문서와 스윕 설정 정리
- `.gitignore`의 Markdown 전역 무시 정책을 문서 자산 기준으로 보정

### P1. 실행 경로 명세 정리

- `TheDistanceAssessor` 계열 중 대표 실행 경로 1개를 명시
- `_3.py`, `_3_onnx.py`, `_3_engine.py`의 역할을 문서로 구분
- 재현용 최소 실행 순서 문서화
  - 학습
  - 평가
  - ONNX 변환
  - TensorRT 변환
  - 최종 추론

### P2. 실험 자산 격리

- `result.txt`, `sweep.yaml`, `sweep_add_agent.py`, `train_yolo.py` 등 실험성 파일을 별도 실험 구역으로 분리 검토
- `assets/README/` 자산과 샘플 데이터셋 자산의 납품 포함 목적을 문서화
- `ESSENTIAL_FILES_ROADMAP.md`를 최신 저장소 기준으로 갱신하거나 아카이브 처리 검토

### P3. 후속 리팩터링 후보 식별

- 중복 파이프라인 공통화
- dataloader/metrics 공용화
- import 경로 정규화
- 루트 스크립트와 라이브러리 코드 분리

이번 감사 단계에서는 위 항목을 “후속 검토 과제”로만 남기고, 실제 구조 변경은 수행하지 않는다.

## 7. 최종 납품 전 리스크

- 메인 실행 경로가 문서와 작업 트리 사이에서 일치하지 않는다.
- 실제 필수 모델 바이너리가 저장소에 포함되어 있지 않다.
- 절대경로 의존 때문에 제3자 환경에서 즉시 실행이 어렵다.
- 일부 sweep/문서가 이미 저장소 현황과 불일치한다.
- 의존성 정의가 산업 납품 기준으로 부족하다.
- Git 작업 트리에 삭제 1건과 미추적 1건이 남아 있어 기준 스냅샷이 고정되지 않았다.

## 8. 권장 후속 액션

1. 납품 기준 메인 엔트리포인트를 1개 확정한다.
2. 실제 포함할 모델/엔진/샘플 데이터 범위를 결정한다.
3. 외부 절대경로를 설정 파일 또는 CLI 인자로 분리한다.
4. 실험 스크립트와 납품 지원 스크립트를 분리 표기한다.
5. 의존성 정의를 루트 기준으로 재작성한다.
6. 누락 참조 문서와 구식 sweep 설정을 정리한다.

## 9. 메모

- 본 문서는 현재 저장소 관찰 결과만 반영했다.
- 확실하지 않은 항목은 `ASSUMPTION` 또는 `TODO`로 남겼다.
- 안전성 검토 전에는 파일 삭제 대신 `보류`, `격리`, `후속 검토` 원칙을 유지하는 것이 바람직하다.
