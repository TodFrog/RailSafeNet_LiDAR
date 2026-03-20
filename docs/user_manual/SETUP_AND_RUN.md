# videoAssessor 설정 및 실행 안내

## 1. 지원 환경

### 1.1 목표 실행 환경

- 운영체제: `Linux`
- GPU: `NVIDIA` 계열 GPU
- 우선 대상: `Linux + Nvidia Orin NX`

### 1.2 현재 문서 검증 workspace

- 운영체제: `Windows`
- 목적: 저장소 정리, 모델 경로 점검, CLI 및 preflight 검증

`ASSUMPTION`: 현재 Windows workspace는 원래 학습/최적화 환경과 다르다. 따라서 TensorRT와 CUDA,
학습용 dependency, 실제 카메라/비디오 입출력은 Linux runtime에서 다시 검증해야 한다.

## 2. Python 버전

- 기준 파일: `environment.yml`
- 현재 기준값: `python=3.13`

`ASSUMPTION`: 이 값은 현재 저장소 기준 환경 설정값이다. TensorRT/CUDA와의 실제 호환 버전은
최종 배포 장비 기준으로 다시 확인해야 한다.

## 3. 의존성 설치

### 3.1 `pip`

Linux 예시:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 예시:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3.2 `conda`

```bash
conda env create -f environment.yml
conda activate railsafenet-delivery
```

### 3.3 training 전용 확장

```bash
python -m pip install -r requirements/training.txt
```

## 4. 모델 준비

### 4.1 현재 저장소에 포함된 모델

- `models/final/segformer_b3_original_13class.engine`
- `models/converted/segformer_b3_original_13class.onnx`
- `models/converted/SegFormer_B3_1024_finetuned.pth`
- `models/final/yolov8n.pt`

### 4.2 backend별 기본 사용 모델

| backend | 기본 모델 | 비고 |
|---|---|---|
| `engine` | `models/final/segformer_b3_original_13class.engine` | SegFormer canonical runtime |
| `engine` | `models/final/yolov8n.engine` 또는 `models/final/yolov8s.engine` 우선, 없으면 `models/final/yolov8n.pt` fallback | 현재 repo에는 `.pt` fallback만 존재 |
| `onnx` | `models/converted/segformer_b3_original_13class.onnx` | SegFormer ONNX 존재 |
| `pytorch` | `models/converted/SegFormer_B3_1024_finetuned.pth` | PyTorch SegFormer 체크포인트 존재 |

### 4.3 중요한 제약

- `models/final/segformer_b3_original_13class.engine`는 Linux + Titan RTX 기준 산출물이다.
- Jetson 또는 다른 GPU에서 바로 사용 가능하다고 보장하지 않는다.
- `onnx` backend는 active YOLO `.onnx`가 없어 현재 preflight 수준으로만 유지한다.

## 5. 설정 파일

기본 설정 경로:

- 선로 추적 설정: `configs/inference/rail_tracker_config.yaml`
- BEV 설정: `configs/inference/bev_config.yaml`

필요 시 CLI에서 override할 수 있다.

## 6. 실행 준비 점검

### 6.1 도움말 확인

```bash
python videoAssessor.py --help
```

### 6.2 engine backend preflight

```bash
python videoAssessor.py --backend engine --check-only
```

이 명령은 아래를 점검한다.

- `cv2`, `numpy`, `yaml`, `torch`, `albumentations`
- `tensorrt`, `pycuda`
- `ultralytics`
- SegFormer engine 경로
- YOLO engine 또는 `.pt` fallback 경로

### 6.3 onnx backend preflight

```bash
python videoAssessor.py --backend onnx --check-only
```

점검 대상:

- `onnxruntime`
- SegFormer `.onnx`
- YOLO `.onnx` 존재 여부

### 6.4 pytorch backend preflight

```bash
python videoAssessor.py --backend pytorch --check-only
```

점검 대상:

- `torch`
- `transformers`
- `ultralytics`
- SegFormer `.pth`
- YOLO `.pt`

## 7. 실제 실행 예시

### 7.1 비디오 파일 실행

```bash
python videoAssessor.py --backend engine --mode video --video <video_path>
```

### 7.2 카메라 실행

```bash
python videoAssessor.py --backend engine --mode camera --camera 0
```

### 7.3 전체 화면

```bash
python videoAssessor.py --backend engine --mode video --video <video_path> --fullscreen
```

### 7.4 결과 비디오 저장

```bash
python videoAssessor.py --backend engine --mode video --video <video_path> --output output.mp4
```

### 7.5 calibration

```bash
python videoAssessor.py --backend engine --mode video --video <video_path> --calibrate
python videoAssessor.py --backend engine --mode video --video <video_path> --calibrate-vp
```

## 8. 학습 엔트리

현재 active training 엔트리는 아래 두 개만 유지한다.

- `src/training/train_segformer.py`
- `src/training/train_yolo.py`

과거 학습 실험, DeepLabv3, sweep 스크립트는 `archive/`로 이동했다.

## 9. 자주 발생하는 오류와 대응 방법

### 9.1 `albumentations`, `tensorrt`, `pycuda` 누락

증상:

- `python videoAssessor.py --backend engine --check-only` 결과에서 `MISSING`
- 실제 `engine` runtime 실행 시 import 오류 또는 초기화 실패

대응:

- Linux GPU runtime에서 `requirements.txt` 및 TensorRT/CUDA 조합을 다시 설치
- 현재 Windows workspace에서는 preflight 결과만 확인하는 것이 정상일 수 있음

### 9.2 `transformers` 누락

증상:

- `python videoAssessor.py --backend pytorch --check-only` 결과에서 `MISSING`

대응:

- PyTorch backend 재현이 필요하면 `transformers` 설치
- 현재 제출 구조에서는 `pytorch` backend를 보조 점검 경로로 간주

### 9.3 YOLO `.onnx` 부재

증상:

- `python videoAssessor.py --backend onnx --check-only` 결과에서 YOLO `.onnx`가 `MISSING`

대응:

- 현재 저장소 기준 `onnx` backend는 SegFormer `.onnx`만 확보된 상태다.
- 최종 ONNX runtime을 활성화하려면 YOLO ONNX 산출물을 별도로 정리해야 한다.

### 9.4 TensorRT engine 호환성 문제

증상:

- engine 파일이 있어도 대상 장비에서 로드 실패 가능

원인:

- 현재 포함된 engine은 Titan RTX/Linux 기준 산출물

대응:

- 최종 배포 장비에서 `.onnx` 기준으로 engine 재생성 또는 재검증 수행

## 10. 알려진 제한사항

- 현재 canonical runtime은 `engine` backend 하나다.
- `onnx`와 `pytorch`는 현재 전체 video pipeline이 아니라 preflight 중심 보조 경로다.
- Windows workspace는 검토/정리 환경이며, 실제 배포 환경과 동일하지 않다.
- `archive/`에는 평가/변환/과거 runtime과 과정 문서가 남아 있으나 active 사용 경로가 아니다.
