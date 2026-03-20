# 모델 아티팩트 목록

- 문서명: `MODEL_MANIFEST.md`
- 작성 기준일: `2026-03-19`
- 기준 저장소: `RailSafeNet_LiDAR`
- 문서 목적: 현재 저장소에 실제 존재하거나 active 코드에서 명시적으로 참조하는 모델/아티팩트를 정리한다.

## 1. 작성 원칙

이 문서는 아래 조건을 만족하는 항목만 기록한다.

- 저장소 안에 실제 파일이 존재한다.
- 또는 active 코드가 명시적으로 해당 아티팩트를 참조한다.

존재하지 않는 체크포인트를 새로 가정하지 않으며, 불확실한 항목은 `TODO` 또는 `ASSUMPTION`으로 표시한다.

## 2. active runtime 기준 모델 아티팩트

| 모델명 | 시스템 역할 | 관련 경로 | 입력 / 출력 형식 | 학습/변환 상태 | runtime 사용 | 누락/불확실성 |
|---|---|---|---|---|---|---|
| `segformer_b3_original_13class.engine` | canonical `engine` backend의 SegFormer 추론 엔진 | `models/final/segformer_b3_original_13class.engine` | 입력: 영상 프레임 기반 segmentation tensor / 출력: segmentation 결과 | 이미 변환 완료된 TensorRT engine | `videoAssessor.py --backend engine`에서 기본 SegFormer 모델로 사용 | `ASSUMPTION`: Linux + Titan RTX 기준 산출물이며 대상 배포 장비에서 재검증 필요 |
| `SegFormer_B3_1024_finetuned.pth` | PyTorch backend용 SegFormer 체크포인트 | `models/converted/SegFormer_B3_1024_finetuned.pth` | 입력: PyTorch image tensor / 출력: segmentation logits | 학습 결과 체크포인트로 간주 | `videoAssessor.py --backend pytorch --check-only`에서 경로 점검 | `TODO`: full PyTorch runtime 검증에는 `transformers` 설치 필요 |
| `segformer_b3_original_13class.onnx` | ONNX backend용 SegFormer 아티팩트 | `models/converted/segformer_b3_original_13class.onnx` | 입력: ONNX image tensor / 출력: segmentation 결과 | ONNX 변환 완료 | `videoAssessor.py --backend onnx --check-only`에서 경로 점검 | `TODO`: active YOLO `.onnx`가 없어 전체 ONNX runtime은 미완성 |
| `yolov8n.pt` | 객체 검출용 YOLO PyTorch 모델 | `models/final/yolov8n.pt` | 입력: 이미지 / 출력: detection boxes, class, score | 실제 바이너리 포함 | engine backend에서 YOLO `.engine`가 없을 때 fallback, pytorch backend preflight | `TODO`: 최종 active YOLO 기준을 `yolov8n`으로 확정할지 별도 판단 필요 |

## 3. reference 파일

| 파일명 | 역할 | 관련 경로 | 비고 |
|---|---|---|---|
| `SegFormer_B3_1024_finetuned.pth.txt` | SegFormer 체크포인트 reference 메모 | `models/references/segformer/SegFormer_B3_1024_finetuned.pth.txt` | 실제 `.pth`는 현재 repo에 이미 존재 |
| `yolov8s.pt.txt` | YOLO `yolov8s.pt` reference 메모 | `models/references/yolo/yolov8s.pt.txt` | active repo에는 실제 `yolov8s.pt`가 없음 |

## 4. backend별 사용 관계

### 4.1 `engine` backend

- SegFormer: `models/final/segformer_b3_original_13class.engine`
- YOLO:
  - 우선: `models/final/yolov8n.engine`, `models/final/yolov8s.engine`
  - fallback: `models/final/yolov8n.pt`, `models/final/yolov8s.pt`

현재 실제 상태:

- SegFormer engine: 존재
- YOLO engine: 없음
- YOLO pt fallback: `yolov8n.pt` 존재

### 4.2 `onnx` backend

- SegFormer ONNX: `models/converted/segformer_b3_original_13class.onnx`
- YOLO ONNX: active repo 기준 없음

### 4.3 `pytorch` backend

- SegFormer `.pth`: `models/converted/SegFormer_B3_1024_finetuned.pth`
- YOLO `.pt`: `models/final/yolov8n.pt`

## 5. 현재 상태 요약

- SegFormer는 `engine`, `onnx`, `pytorch` 세 형식이 모두 repo에 존재한다.
- YOLO는 active repo 기준 `.pt`만 존재한다.
- 따라서 canonical runtime은 `SegFormer engine + YOLO .pt fallback` 조합을 포함한 형태로 이해하는 것이 안전하다.

## 6. 제약과 TODO

- `TODO`: 최종 배포 장비에서 현재 SegFormer engine을 재사용할지, 새로 빌드할지 결정이 필요하다.
- `TODO`: ONNX full runtime을 활성화하려면 YOLO `.onnx` 아티팩트가 필요하다.
- `TODO`: engine backend에서 사용할 최종 YOLO 모델 계열(`yolov8n` vs `yolov8s`)을 명확히 확정할 필요가 있다.
- `ASSUMPTION`: 본 저장소의 runtime/모델 계보는 [oValach/RailSafeNet](https://github.com/oValach/RailSafeNet)와 그 후속 branch 정리본을 기반으로 한다.
