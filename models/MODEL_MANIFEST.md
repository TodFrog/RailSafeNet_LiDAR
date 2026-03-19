# 모델 아티팩트 목록

- 문서명: `MODEL_MANIFEST.md`
- 작성 기준일: `2026-03-19`
- 기준 저장소: `RailSafeNet_LiDAR`
- 문서 목적: 저장소에 실제 존재하거나, 코드와 문서에서 명시적으로 참조되는 모델/아티팩트를 정리합니다.

## 1. 작성 기준

이 문서는 아래 조건을 만족하는 항목만 기록합니다.

- 저장소 안에 실제 파일이 존재하는 모델/아티팩트
- 저장소 코드 또는 공식 문서에서 파일명/경로가 명시적으로 참조되는 모델/아티팩트

이 문서는 존재하지 않는 체크포인트를 새로 가정하지 않습니다. 불확실한 항목은 `TODO` 또는 `ASSUMPTION`으로만 표시합니다.

## 2. 저장소에 실제 포함된 모델/아티팩트

| 모델명 | 시스템 역할 | 관련 경로 | 입력 / 출력 형식 | 학습 또는 변환 상태 | 런타임 사용 | 누락 또는 불확실성 |
|---|---|---|---|---|---|---|
| `SegFormer_B3_1024_finetuned.pth` | 공식 PyTorch smoke path에서 사용할 수 있는 SegFormer 체크포인트 | `models/converted/SegFormer_B3_1024_finetuned.pth` | 입력은 `1x3x1024x1024` 계열 이미지 텐서로 가정, 출력은 세그멘테이션 `logits` 텐서 | 사용자 제공 정보 기준으로 Linux 환경에서 가져온 학습 결과 파일 | `production_segformer_pytorch.py`의 로컬 후보 경로로 사용 가능 | `TODO`: 최종 클래스 매핑 근거와 저장 형식(전체 모델/순수 state dict)의 생성 이력은 별도 확인 필요 |
| `segformer_b3_original_13class.onnx` | ONNX 추론 또는 TensorRT 엔진 생성의 입력 아티팩트 | `models/converted/segformer_b3_original_13class.onnx` | 입력은 세그멘테이션용 이미지 텐서, 출력은 세그멘테이션 결과 텐서로 추정 | 사용자 제공 정보 기준으로 Linux 환경에서 가져온 ONNX 산출물 | ONNX Runtime 경로 또는 TensorRT 재생성 입력으로 사용 가능 | `ASSUMPTION`: `13class`는 13개 클래스 구성을 의미하는 것으로 보이나 최종 라벨 매핑 문서는 저장소에 없음 |
| `segformer_b3_original_13class.engine` | TensorRT 기반 SegFormer 추론 엔진 | `models/final/segformer_b3_original_13class.engine` | 입력/출력은 ONNX 원본과 동일 목적의 TensorRT 바인딩 텐서 | 사용자 제공 정보 기준으로 Linux + Titan RTX 환경에서 최적화된 산출물 | TensorRT 경로에서 사용할 수 있는 실파일이지만, 환경 호환성 검증이 필요 | `ASSUMPTION`: Titan RTX 기준 엔진이므로 다른 GPU, 다른 TensorRT/CUDA 조합에서는 재생성 필요 가능성이 큼 |
| `yolov8n.pt` | 저장소에 실제 포함된 YOLO 바이너리 | `models/final/yolov8n.pt` | 입력은 이미지, 출력은 검출 박스/점수/클래스 정보 계열로 가정 | 실제 바이너리 파일 존재 | 문서상 “실제 포함 모델”로 확인 가능 | `TODO`: 통합 파이프라인의 대표 YOLO 모델을 `yolov8n`으로 확정할지, `yolov8s` 계열 외부 자산을 기준으로 볼지 결정 필요 |

## 3. 저장소에 실제 포함된 포인터 파일

아래 파일은 모델 바이너리가 아니라 참조 위치를 설명하는 포인터 파일입니다.

| 모델명 | 시스템 역할 | 관련 경로 | 입력 / 출력 형식 | 학습 또는 변환 상태 | 런타임 사용 | 누락 또는 불확실성 |
|---|---|---|---|---|---|---|
| `SegFormer_B3_1024_finetuned.pth.txt` | SegFormer `.pth`의 참조 위치 설명 | `models/references/segformer/SegFormer_B3_1024_finetuned.pth.txt` | 포인터 파일 자체에는 적용되지 않음 | 실제 `.pth`에 대한 안내 파일 | 문서적 참조 용도 | 현재는 실제 `.pth` 실파일도 저장소 안에 존재하므로 포인터와 실파일이 함께 존재함 |
| `yolov8s.pt.txt` | `yolov8s.pt` 외부 자산 참조 설명 | `models/references/yolo/yolov8s.pt.txt` | 포인터 파일 자체에는 적용되지 않음 | 실제 `yolov8s.pt`는 저장소에 없음 | 통합 YOLO 경로의 외부 참조 설명 | `TODO`: 최종 제출에 `yolov8s` 계열 실파일을 포함할지 여부 결정 필요 |

## 4. 코드가 명시적으로 참조하지만 저장소 외부에 있는 모델/아티팩트

| 모델명 | 시스템 역할 | 관련 경로 | 입력 / 출력 형식 | 학습 또는 변환 상태 | 런타임 사용 | 누락 또는 불확실성 |
|---|---|---|---|---|---|---|
| 외부 SegFormer `.pth` 후보 | 과거 Linux 운영 환경 fallback | `/home/mmc-server4/RailSafeNet/models/segformer_b3_production_optimized_rail_0.7500.pth`, `/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth` | PyTorch SegFormer 입력/로그잇 출력 구조를 가정 | 과거 운영 환경 참조 | 공식 PyTorch smoke path의 fallback 후보 | `ASSUMPTION`: 현재는 로컬 `models/converted/SegFormer_B3_1024_finetuned.pth`가 우선 사용될 수 있음 |
| 외부 `yolov8s` 계열 | 통합 추론 및 변환 경로의 YOLO 자산 | `/home/mmc-server4/.../yolov8s.pt`, `/home/mmc-server4/.../yolov8s_896x512.onnx`, `/home/mmc-server4/.../yolov8s_896x512.engine` | 입력은 이미지, 출력은 검출 결과 | 외부 학습/변환 산출물로 추정 | `TheDistanceAssessor_3_*`, `yolo_original_to_onnx.py`, `yolo_onnx_to_engine.py`에서 참조 | 저장소 실파일은 현재 `yolov8n.pt`만 존재하므로 대표 YOLO 기준이 혼재함 |

## 5. 현재 해석 요약

- 저장소에는 실제 SegFormer `.pth`, `.onnx`, `.engine` 파일이 포함되어 있습니다.
- 저장소에는 실제 YOLO `.pt` 파일(`yolov8n.pt`)도 포함되어 있습니다.
- 공식 PyTorch smoke path는 이제 저장소 내부 `models/converted/SegFormer_B3_1024_finetuned.pth`까지 후보로 탐색할 수 있습니다.
- 포함된 TensorRT 엔진은 “존재”와 “현재 workspace에서 즉시 사용 가능”을 동일하게 볼 수 없습니다.
- 통합 YOLO 경로는 여전히 `yolov8s` 외부 참조와 `yolov8n.pt` 실파일이 혼재합니다.

## 6. 납품 관점 판정

- 실제 포함된 핵심 SegFormer 아티팩트: 존재
- 실제 포함된 YOLO 바이너리: 존재
- 공식 기본 경로용 PyTorch 체크포인트: 존재
- ONNX 아티팩트: 존재
- TensorRT 엔진: 존재
- 환경 호환성 검증: 미완료
- 대표 YOLO 모델 기준: 미확정

따라서 “모델 파일 존재 여부”는 상당 부분 보완되었지만, “모든 런타임 경로에 대해 즉시 재현 가능한 상태”라고 단정하기는 어렵습니다.

## 7. 다음 조치

1. `models/converted/SegFormer_B3_1024_finetuned.pth` 기준 공식 PyTorch smoke path를 Linux runtime에서 재검증합니다.
2. `models/final/segformer_b3_original_13class.engine`를 실제 배포 대상 GPU에서 재사용할지, 재생성할지 결정합니다.
3. 통합 추론 경로의 YOLO 기준을 `yolov8n` 또는 `yolov8s` 중 하나로 정리합니다.
4. `TODO`: 클래스 ID와 `13class` 구성의 최종 근거 문서를 별도로 보완합니다.
