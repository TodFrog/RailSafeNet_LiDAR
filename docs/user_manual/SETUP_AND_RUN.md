# RailSafeNet 설정 및 실행 안내

## 1. 지원 환경

### 1.1 문서상 우선 대상 환경

- 운영체제: `Linux`
- GPU: `NVIDIA` 계열 GPU
- 문서상 우선 제출 환경: `Linux + Nvidia Orin NX`

### 1.2 현재 저장소 정리/검토 workspace

- 운영체제: `Windows`
- 용도: 문서 보완, 모델 파일 정리, 기본 smoke path 점검

`ASSUMPTION`: 현재 Windows workspace는 원래 학습/변환이 수행된 런타임 환경과 다르며, 포함된 ONNX/TensorRT 산출물도 별도 Linux 환경에서 가져온 파일입니다.

### 1.3 TensorRT 엔진 관련 주의

- `models/final/segformer_b3_original_13class.engine`는 Linux 환경에서 가져온 TensorRT 엔진입니다.
- 사용자 제공 정보 기준으로 이 엔진은 `Titan RTX` 기준으로 최적화되었습니다.
- 따라서 현재 Windows workspace나 다른 GPU에서 그대로 재사용하는 것을 기본 전제로 두면 안 됩니다.
- `TODO`: 실제 최종 배포 장비가 `Orin NX`인지, 다른 Linux GPU 서버인지 확인한 뒤 대상 장비에서 엔진 재생성 또는 재검증이 필요합니다.

## 2. Python 버전

- 저장소 기준 환경 파일: `environment.yml`
- 명시 값: `python=3.13`

`ASSUMPTION`: `3.13`은 현재 저장소 기준 환경 값입니다. TensorRT/CUDA 조합이 필요한 실제 배포 환경에서는 별도 호환성 검토가 필요합니다.

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

### 3.3 추가 프로파일

- 학습 확장 패키지: `requirements/training.txt`
- 변환/TensorRT 관련 패키지: `requirements/conversion.txt`

예시:

```bash
python -m pip install -r requirements/training.txt
python -m pip install -r requirements/conversion.txt
```

## 4. 모델 준비

### 4.1 현재 저장소에 실제 포함된 주요 파일

- `models/converted/SegFormer_B3_1024_finetuned.pth`
- `models/converted/segformer_b3_original_13class.onnx`
- `models/final/segformer_b3_original_13class.engine`
- `models/final/yolov8n.pt`

### 4.2 포인터 파일

- `models/references/segformer/SegFormer_B3_1024_finetuned.pth.txt`
- `models/references/yolo/yolov8s.pt.txt`

### 4.3 공식 PyTorch 경로의 모델 탐색 순서

공식 스크립트 `production_segformer_pytorch.py`는 아래 순서로 SegFormer `.pth`를 찾습니다.

1. `--model-path`로 지정한 경로
2. `models/final/segformer_b3_production_optimized_rail_0.7500.pth`
3. `models/final/SegFormer_B3_1024_finetuned.pth`
4. `models/converted/SegFormer_B3_1024_finetuned.pth`
5. 기존 Linux 운영 환경 fallback 경로

즉, 현재 저장소만 기준으로도 `models/converted/SegFormer_B3_1024_finetuned.pth`를 공식 smoke path 후보로 사용할 수 있습니다.

### 4.4 포함된 아티팩트의 해석

- `SegFormer_B3_1024_finetuned.pth`
  - PyTorch 기반 SegFormer 체크포인트
  - 공식 PyTorch smoke path에서 직접 사용할 수 있는 후보
- `segformer_b3_original_13class.onnx`
  - ONNX 기반 추론 또는 TensorRT 변환의 입력 아티팩트
- `segformer_b3_original_13class.engine`
  - TensorRT 엔진
  - Linux + Titan RTX 환경에서 가져온 파일
  - 현재 Windows workspace나 다른 GPU에서 그대로 동작한다고 보장할 수 없음
- `yolov8n.pt`
  - 저장소에 실제 포함된 YOLO 바이너리
  - 다만 통합 파이프라인 코드는 여전히 `yolov8s` 외부 자산을 참조하는 부분이 있음

## 5. 메인 실행 방법

### 5.1 도움말 확인

```bash
python production_segformer_pytorch.py --help
```

### 5.2 사전 점검

설치 직후에는 먼저 아래 명령으로 준비 상태를 확인합니다.

```bash
python production_segformer_pytorch.py --check-only
```

사전 점검 항목:

- `torch`, `transformers` 의존성 존재 여부
- SegFormer `.pth` 후보 경로 존재 여부
- 현재 workspace 기준 실행 준비 상태

종료 코드:

- `0`: 의존성과 모델 후보가 모두 충족됨
- `1`: 의존성 누락 또는 모델 부재

### 5.3 공식 PyTorch smoke path 실행

Linux 예시:

```bash
python production_segformer_pytorch.py --model-path models/converted/SegFormer_B3_1024_finetuned.pth --device cuda
```

Windows PowerShell 예시:

```powershell
python production_segformer_pytorch.py --model-path "models\converted\SegFormer_B3_1024_finetuned.pth" --device cpu
```

`--model-path`를 생략하면 저장소 내부 기본 후보와 외부 fallback을 순서대로 탐색합니다.

## 6. 자주 발생하는 오류와 대응 방법

### 6.1 `transformers` 또는 `torch`가 없다고 표시되는 경우

원인:

- 현재 workspace에 런타임 의존성이 설치되지 않음
- 문서 검토용 Windows 환경이라 의도적으로 설치하지 않았을 가능성

확인:

```bash
python production_segformer_pytorch.py --check-only
```

대응:

```bash
python -m pip install -r requirements.txt
```

주의:

- 현재 Windows workspace에서 `transformers: MISSING`이 나오는 것은 “이 workspace가 원래 학습/실행 환경이 아님”을 반영할 수 있습니다.
- 실제 runtime 검증은 Linux CUDA 환경에서 다시 수행하는 것이 안전합니다.

### 6.2 `사용 가능한 SegFormer .pth 모델을 찾지 못했습니다.`가 표시되는 경우

원인:

- `.pth` 파일이 후보 경로에 없음
- `--model-path`가 잘못 지정됨

대응:

- `models/converted/SegFormer_B3_1024_finetuned.pth` 존재 여부 확인
- 필요 시 `--model-path`로 명시적 지정

### 6.3 TensorRT 엔진이 동작하지 않는 경우

원인:

- 현재 포함된 `.engine` 파일이 Linux + Titan RTX 환경 기준 산출물임
- GPU, TensorRT, CUDA, 드라이버 조합이 다를 수 있음

대응:

- 현재 `.engine`를 범용 파일로 가정하지 않음
- 대상 배포 GPU에서 `.onnx` 기준으로 다시 `.engine`을 생성
- `onnx_to_engine.py` 실행 전 대상 장비의 TensorRT/CUDA 조합 확인

### 6.4 `torch.load` 또는 체크포인트 로드 오류

원인:

- 모델 생성 환경과 현재 PyTorch 환경 차이
- 전체 모델 객체 저장 방식과 state dict 저장 방식 차이

대응:

- 모델 생성 환경과 유사한 PyTorch 환경 사용
- `--check-only`로 먼저 경로 존재만 확인한 뒤 실제 로드는 Linux runtime에서 수행

## 7. 알려진 제한사항

- 공식 기본 경로는 전체 위험도 분석 파이프라인이 아니라 PyTorch smoke path입니다.
- 현재 Windows workspace는 문서/검토 환경으로 사용 중이며, 원래 학습 및 최적화가 수행된 Linux 환경과 다릅니다.
- 포함된 TensorRT 엔진은 Titan RTX 기준 산출물이므로 대상 장비가 다르면 재생성이 필요할 수 있습니다.
- `models/converted/segformer_b3_original_13class.onnx`와 `models/final/segformer_b3_original_13class.engine`는 실제 파일이 존재하지만, 현재 Windows workspace에서 재검증된 결과는 아닙니다.
- 통합 추론 코드에는 여전히 `yolov8s` 외부 참조가 남아 있습니다.
- `TODO`: 최종 배포 대상 장비 기준 TensorRT 엔진 재생성 여부와 대표 YOLO 모델 계열을 확정해야 합니다.
