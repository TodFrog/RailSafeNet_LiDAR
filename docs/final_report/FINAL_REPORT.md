# 최종 프로젝트 보고서

- 문서명: `FINAL_REPORT.md`
- 작성 기준일: `2026-03-19`
- 대상 독자: 회사 제출 검토자, 개발 인수자
- 기준 저장소: `RailSafeNet_LiDAR`

## 1. 프로젝트 배경

철도/트램 전방 영상 기반 위험도 분석은 선로 인식과 객체 검출을 동시에 다뤄야 하며, 실제 운행
환경에 적용하려면 runtime 경로가 명확하고 유지보수 가능한 구조가 필요하다. 본 저장소는
[oValach/RailSafeNet](https://github.com/oValach/RailSafeNet) 계열 코드에서 출발했고, 이후 여러
실험 branch를 거치며 분산된 runtime과 문서가 혼재해 있었다.

이번 정리의 핵심 목표는 최신 `videoAssessor` 계열 코드를 중심으로 최종 사용자 관점의 구조를
단순화하고, 회사 제출용 문서와 모델 자산 상태를 일치시키는 것이었다.

## 2. 목표

- 최종 사용자 진입점을 `videoAssessor.py` 하나로 통일
- 최신 `videoAssessor` runtime을 기준으로 active runtime 재정렬
- active training 엔트리를 SegFormer 1개, YOLO 1개로 축소
- 과거 runtime, 변환, 평가, 과정 문서를 archive로 분리
- 모델 상태, 환경 제약, 문서를 실제 저장소 상태와 일치시키기

## 3. 기술적 접근 방법

이번 정리는 두 가지 축으로 진행했다.

### 3.1 runtime 축

- `001-what-why-home` branch의 `videoAssessor_final.py`와 helper 코드를 기준으로 현재 active runtime을 재구성했다.
- 최종 사용자 entry는 루트 `videoAssessor.py` 하나로 고정했다.
- `engine` backend를 canonical runtime으로 두고, `onnx`와 `pytorch`는 preflight 중심 보조 backend로 유지했다.

### 3.2 구조 축

- `TheDistanceAssessor*`, `production_segformer_*`, 변환/평가 wrapper, 과정 문서를 active tree에서 제거했다.
- 이들은 삭제하지 않고 `archive/`로 이동해 참고 자산으로 보존했다.
- training은 `train_segformer.py`, `train_yolo.py` 두 개만 active 상태로 유지했다.

## 4. 모델 및 알고리즘 개요

### 4.1 SegFormer 기반 선로 세그멘테이션

- TensorRT runtime용 engine: `models/final/segformer_b3_original_13class.engine`
- ONNX 아티팩트: `models/converted/segformer_b3_original_13class.onnx`
- PyTorch 체크포인트: `models/converted/SegFormer_B3_1024_finetuned.pth`

### 4.2 YOLO 기반 객체 검출

- 현재 repo에 실제 포함된 바이너리: `models/final/yolov8n.pt`
- engine backend는 YOLO `.engine`가 있을 경우 우선 사용하고, 없으면 `.pt` fallback을 사용하도록 정리했다.

### 4.3 위험도 분석 흐름

- 세그멘테이션 결과 후처리
- rail edge 탐색 및 ego track 추정
- BEV 변환 및 path analyzer 적용
- danger zone 생성
- 객체 검출 결과와 위험 구역 비교
- alert panel / mini BEV / overlay 렌더링

## 5. 구현 상세

현재 active 저장소 구조는 아래와 같이 단순화됐다.

- 사용자 진입점: `videoAssessor.py`
- runtime 코드: `src/inference`, `src/rail_detection`, `src/utils`, `src/common`
- training 코드: `src/training/train_segformer.py`, `src/training/train_yolo.py`
- 설정: `configs/inference`
- 모델: `models/final`, `models/converted`, `models/references`
- 문서: `docs/user_manual`, `docs/requirements`, `docs/design`, `docs/test_report`, `docs/final_report`
- legacy/reference: `archive/`

특히 기존 root surface에 흩어져 있던 `TheDistanceAssessor*`, 평가/변환 wrapper를 active tree에서 제거해
최종 사용자가 무엇을 실행해야 하는지 명확히 했다.

## 6. 최적화 및 배포 현황

- canonical runtime은 TensorRT 기반 `engine` backend다.
- 하지만 현재 포함된 SegFormer engine은 Linux + Titan RTX 기준 산출물이다.
- `onnx`와 `pytorch` backend는 현재 full runtime이 아니라 모델/의존성 상태 점검용으로 유지한다.
- Docker/Jetson 배포 자산은 이번 active main에 직접 포함하지 않았고, reference branch 성격으로만 취급했다.

`ASSUMPTION`: 최종 배포 장비가 Jetson 또는 다른 Linux GPU일 경우, 현재 engine을 그대로 쓸 수 있을지
재검증이 필요하다.

## 7. 시험 및 검증 요약

현재 active main 기준 실제 확인된 항목은 다음과 같다.

### 실행됨

- `python videoAssessor.py --help`
- `python -m compileall videoAssessor.py src/inference src/common src/training src/rail_detection src/utils`

### 부분 검증

- `python videoAssessor.py --backend engine --check-only`
  - SegFormer engine 존재 확인
  - YOLO `.pt` fallback 존재 확인
  - `albumentations`, `tensorrt`, `pycuda` 누락 확인
- `python videoAssessor.py --backend onnx --check-only`
  - SegFormer `.onnx` 존재 확인
  - active YOLO `.onnx` 부재 확인
- `python videoAssessor.py --backend pytorch --check-only`
  - SegFormer `.pth` 존재 확인
  - `transformers` 누락 확인

### 계획됨(미실행)

- engine backend 실제 비디오/카메라 실행
- calibration 실행
- training entry runtime smoke test

정량 성능, FPS, latency는 현재 active main 기준으로 새로 측정하지 않았으므로 보고하지 않는다.

## 8. 성과

- `videoAssessor.py` 단일 사용자 entrypoint를 확립했다.
- 최신 `videoAssessor` 계열 runtime을 기준으로 active runtime을 정리했다.
- `train_segformer.py`, `train_yolo.py` 두 개만 active training 엔트리로 유지했다.
- 과거 runtime, 평가, 변환, delivery 과정 문서를 `archive/`로 이동해 혼선을 줄였다.
- 모델 상태, 환경 제약, README, 사용자 매뉴얼, 요구사항/아키텍처/시험/최종 보고서의 정합성을 다시 맞췄다.

## 9. 한계

- canonical runtime은 `engine` backend지만 현재 Windows workspace에는 TensorRT 관련 dependency가 없다.
- 포함된 SegFormer engine은 Titan RTX/Linux 기준 산출물이라 대상 장비 호환성이 미확정이다.
- `onnx` backend에는 active YOLO `.onnx`가 없다.
- `pytorch` backend에는 `transformers`가 없어 full runtime 검증이 중단된다.
- training 스크립트는 active 상태로 정리했지만 실제 학습 runtime은 별도 검증이 필요하다.

## 10. 향후 과제

1. 목표 배포 장비에서 `engine` backend 실제 비디오/카메라 runtime을 검증한다.
2. 필요 시 대상 장비 기준으로 SegFormer TensorRT engine을 재생성한다.
3. ONNX 경로를 활성화하려면 YOLO `.onnx` 산출물을 정리한다.
4. PyTorch full runtime이 필요하면 `transformers`를 포함한 runtime 환경을 별도로 구성한다.
5. training 엔트리에 대해 Linux 학습 환경 기준 smoke test를 수행한다.
