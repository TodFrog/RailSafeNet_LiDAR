# 최종 프로젝트 보고서

- 문서명: `FINAL_REPORT.md`
- 작성 기준일: `2026-03-19`
- 대상 독자: 회사 제출 검토자, 개발 인수자
- 기준 저장소: `RailSafeNet_LiDAR`
- 문서 성격: 회사 제출용 최종 프로젝트 보고서

## 작성 기준

본 보고서는 현재 저장소에 실제로 포함된 코드, 모델 파일, 문서, 그리고 이번 세션에서 다시 확인한 실행 결과만을 기준으로 작성합니다. 구현 결과, 관찰된 실행 수준, 한계, 향후 과제를 분리해 기술하며, 저장소 근거가 없는 성능 수치나 완료 주장은 포함하지 않습니다.

## 1. 프로젝트 배경

`RailSafeNet_LiDAR`는 철도 환경 영상에서 선로 영역을 분할하고, 필요 시 객체 검출 결과와 결합해 위험 구역을 추정하기 위한 프로젝트 저장소입니다. 저장소에는 SegFormer 기반 세그멘테이션, YOLO 기반 객체 검출, ONNX/TensorRT 변환, 평가 스크립트, 학습 스크립트가 함께 존재합니다.

최종 제출 준비 과정에서 저장소는 납품 지향 구조로 재정리되었으며, 현재는 공식 재현 경로와 보조 구현 경로가 공존하는 형태입니다.

- 공식 재현 경로: `production_segformer_pytorch.py` 기반 PyTorch smoke path
- 보조 구현 경로: `TheDistanceAssessor_3_*`, `src/training/`, `src/evaluation/`, `src/conversion/`

## 2. 목표

현재 저장소 기준 프로젝트 목표는 다음과 같습니다.

- 공식 PyTorch smoke path 제공
- SegFormer 기반 선로 분할 및 위험도 분석 파이프라인 보존
- 학습/평가/변환 스크립트 포함형 납품 구조 제공
- 한국어 중심 문서 세트와 제출 검토 문서 정리
- 실제 포함 모델/아티팩트의 상태를 명확히 문서화

## 3. 기술적 접근 방법

프로젝트는 공식 경로와 보조 경로를 분리해 정리했습니다.

### 3.1 공식 경로

- 루트 wrapper `production_segformer_pytorch.py`
- 실제 구현 `src/inference/production_segformer_pytorch.py`

이 경로는 전체 서비스 파이프라인이 아니라, 모델 로드와 최소 추론 가능 여부를 확인하는 smoke path입니다. 이번 보완 과정에서 `--check-only` preflight를 추가해 모델 경로와 런타임 의존성 상태를 실제 로드 없이 점검할 수 있게 했습니다.

### 3.2 보조 경로

- `TheDistanceAssessor_3_engine.py`
- `TheDistanceAssessor_3.py`
- `TheDistanceAssessor_3_onnx.py`
- `src/training/`, `src/evaluation/`, `src/conversion/`

이 경로들은 세그멘테이션, 선로 추정, danger zone 계산, 객체 검출, 시각화, 학습, 평가, 변환을 포함하는 실험/통합 구현입니다.

## 4. 모델 및 알고리즘 개요

### 4.1 SegFormer 기반 세그멘테이션

공식 PyTorch 경로와 통합 ONNX/TensorRT 경로 모두 SegFormer 계열 세그멘테이션을 핵심 전제로 둡니다. 현재 저장소에는 실제 SegFormer 관련 아티팩트가 포함되어 있습니다.

- `models/converted/SegFormer_B3_1024_finetuned.pth`
- `models/converted/segformer_b3_original_13class.onnx`
- `models/final/segformer_b3_original_13class.engine`

### 4.2 YOLO 기반 객체 검출

통합 위험도 분석 경로는 YOLO 기반 객체 검출을 사용합니다. 저장소에는 `models/final/yolov8n.pt`가 실제 바이너리로 포함되어 있습니다. 다만 통합 파이프라인 일부는 여전히 `yolov8s` 외부 자산을 참조합니다.

`TODO`: 최종 납품 기준 대표 검출 모델을 `yolov8n` 또는 `yolov8s` 중 무엇으로 볼지 별도 확정이 필요합니다.

### 4.3 위험도 분석 로직

통합 경로의 핵심 단계는 다음과 같습니다.

- 세그멘테이션 마스크 생성
- morphology 기반 후처리
- rail candidate 탐색
- ego track 식별
- danger zone 생성
- YOLO 검출 결과와 danger zone 비교
- 위험도 분류 및 시각화

`TODO`: 클래스 ID `4`, `9`, `1`과 거리 구역 `100/400/1000`의 도메인 의미는 저장소만으로 최종 확정할 수 없습니다.

### 4.4 최적화 경로

저장소에는 ONNX Runtime 및 TensorRT 기반 경로가 구현되어 있으며, 실제 `.onnx`와 `.engine` 파일도 포함되어 있습니다. 다만 이 파일들은 현재 Windows workspace에서 생성된 것이 아니라 별도 Linux 환경에서 가져온 산출물입니다.

## 5. 구현 상세

현재 저장소 구조는 다음과 같이 정리되어 있습니다.

- `src/`: 추론, 학습, 평가, 변환, 공용 유틸
- `configs/`: 학습 및 sweep 설정
- `models/`: 실제 포함 모델, 변환 산출물, 포인터 파일, 모델 문서
- `data_samples/`: 샘플 데이터
- `docs/`: 사용자 안내, 요구사항, 설계, 시험, 최종 보고, 납품 검토 문서
- `requirements/`, `requirements.txt`, `environment.yml`: 재현 환경 정의
- `archive/`: 보관용 자산

기능군 기준 구현 상태는 다음과 같습니다.

### 5.1 추론

- 공식 경로: `production_segformer_pytorch.py`, `src/inference/production_segformer_pytorch.py`
- 보조 경로: `TheDistanceAssessor_3_engine.py`, `TheDistanceAssessor_3.py`, `TheDistanceAssessor_3_onnx.py`, `production_segformer_onnx.py`

### 5.2 학습

- `train_SegFormer.py`
- `train_SegFormer_transfer_learning.py`
- `train_DeepLabv3.py`
- `train_yolo.py`
- `sweep_transfer.py`

### 5.3 평가

- `SegFormer_test.py`
- `test_filtered_cls.py`
- `video_frame_tester.py`

### 5.4 변환

- `original_to_onnx.py`
- `yolo_original_to_onnx.py`
- `onnx_to_engine.py`
- `yolo_onnx_to_engine.py`

### 5.5 문서 및 납품 정리

현재 저장소에는 README, 사용자 매뉴얼, 요구사항 명세서, 아키텍처 문서, 시험 보고서, 최종 보고서, 모델 매니페스트, 납품 체크리스트, 보완 로그가 포함되어 있습니다.

## 6. 최적화 및 배포 현황

### 6.1 최적화 구현 현황

- 공식 PyTorch 경로는 smoke test 기준으로 정리되어 있습니다.
- `--check-only` preflight를 통해 모델 후보와 의존성 상태를 점검할 수 있습니다.
- ONNX Runtime 및 TensorRT 추론/변환 스크립트가 실제로 존재합니다.
- SegFormer 관련 `.pth`, `.onnx`, `.engine` 파일이 저장소에 포함되어 있습니다.

### 6.2 배포 관점 상태

- 문서상 우선 지원 환경은 `Linux + Nvidia Orin NX`로 정리되어 있습니다.
- 현재 포함된 TensorRT 엔진은 사용자 제공 정보 기준으로 `Titan RTX`용 Linux 산출물입니다.
- 따라서 현재 엔진은 “존재”와 “대상 장비에서 즉시 사용 가능”을 분리해 해석해야 합니다.
- Docker는 현재 제공되지 않습니다.

### 6.3 배포 준비 상태에 대한 판단

저장소 구조, 문서, 모델 목록, 사전 점검 경로는 납품 검토에 충분한 수준까지 정리되었습니다. 다만 실제 배포 대상 장비에서의 런타임 검증과 엔진 재생성 여부 판단은 아직 남아 있습니다.

## 7. 시험 및 검증 요약

시험 상태는 `docs/test_report/SW_TEST_REPORT.md`를 기준으로 요약합니다.

### 7.1 실행됨

- `TC-01`: `production_segformer_pytorch.py --help`
- `TC-02`: `production_segformer_pytorch.py --check-only`
- `TC-03`: import smoke test
- `TC-04`: `compileall`

위 항목은 모두 `PASS`입니다.

### 7.2 부분 검증

- `TC-05`: `python production_segformer_pytorch.py --device cpu`

실행 결과는 `ModuleNotFoundError: No module named 'transformers'`였습니다. 현재 Windows workspace는 문서/정리용 환경이므로, 이는 “현재 workspace에서 런타임 의존성이 준비되지 않음”을 보여주는 결과로 해석했습니다. 모델 부재는 아니며, 로컬 `.pth` 존재는 `--check-only`에서 확인되었습니다.

### 7.3 계획됨(미실행)

- `TC-06`: 로컬 `.pth` 기반 공식 PyTorch smoke path end-to-end 실행
- `TC-07`: TensorRT 통합 위험도 분석 경로 실행
- `TC-08`: ONNX 추론/benchmark 경로 실행
- `TC-09`: 평가 스크립트 기반 정량 지표 산출
- `TC-10`: 변환 스크립트 재실행

### 7.4 관찰된 실행 수준

이번 세션에서 실제 확인된 수준은 다음과 같습니다.

- 공식 CLI 동작 확인
- 공식 preflight 동작 확인
- 로컬 SegFormer `.pth` 후보 발견 확인
- 활성 Python 코드 문법 상태 확인
- 현재 Windows workspace에서 `transformers` 부재 시 실패 지점 확인

FPS, latency, accuracy, benchmark 수치는 저장소 근거가 없으므로 본 보고서에 포함하지 않습니다.

## 8. 성과

현재 저장소 기준으로 확인 가능한 성과는 다음과 같습니다.

- 납품 지향 디렉터리 구조로 저장소를 재정리했습니다.
- 공식 실행 경로와 보조 경로를 분리해 문서화했습니다.
- `requirements.txt`, `environment.yml`, 사용자 매뉴얼을 통해 재현 환경 안내를 정리했습니다.
- 실제 SegFormer `.pth`, `.onnx`, `.engine` 파일과 YOLO `.pt` 파일을 저장소 내 `models/` 구조에 포함해 상태를 명확히 했습니다.
- 공식 PyTorch 경로에 `--check-only` preflight를 추가해 납품 검토자가 준비 상태를 즉시 확인할 수 있게 했습니다.
- 요구사항 명세서, 아키텍처 문서, 시험 보고서, 최종 보고서, 납품 체크리스트, 보완 로그를 한국어 Markdown으로 정리했습니다.
- 활성 delivery-scope 코드에 한국어 주석과 docstring을 보강했습니다.

## 9. 한계

- 현재 Windows workspace는 원래 학습/최적화가 수행된 Linux 환경과 다릅니다.
- `transformers` 미설치 상태에서는 공식 PyTorch 경로의 end-to-end 실행이 불가능합니다.
- 포함된 TensorRT 엔진은 Titan RTX 기준 산출물이므로 대상 GPU에서 그대로 사용할 수 있다고 보장할 수 없습니다.
- 통합 YOLO 경로는 여전히 `yolov8s` 외부 자산 참조와 `yolov8n.pt` 실파일이 혼재합니다.
- ONNX/TensorRT 경로의 실제 benchmark와 장비별 실측 결과는 이번 보고서에 포함하지 않았습니다.
- 평가 스크립트의 최신 정량 결과 리포트는 별도로 축적되지 않았습니다.

## 10. 향후 과제

1. Linux 런타임 환경에서 `requirements.txt` 또는 `environment.yml` 기준 설치를 재검증합니다.
2. `models/converted/SegFormer_B3_1024_finetuned.pth` 기준 공식 PyTorch smoke path를 end-to-end로 다시 실행합니다.
3. 실제 배포 장비에 맞춰 TensorRT 엔진을 재생성하거나 재검증합니다.
4. 통합 위험도 경로에 필요한 YOLO 아티팩트 전략을 정리합니다.
5. 평가 스크립트 기반 정량 결과와 스냅샷을 제출용 부속 자료로 보강합니다.
