# RailSafeNet_LiDAR

## 프로젝트 개요

`RailSafeNet_LiDAR`는 철도/트램 전방 영상을 기반으로 선로를 추정하고, 위험 구역을 생성하며,
객체 검출 결과를 결합해 운행 위험 상황을 시각화하는 저장소다. 현재 `main`은 회사 제출용으로
정리된 최소 구조를 기준으로 하며, 최종 사용자 진입점은 루트의 `videoAssessor.py` 하나로 통일돼 있다.

본 저장소는 [oValach/RailSafeNet](https://github.com/oValach/RailSafeNet) 계열 코드와 그 후속 정리
branch를 바탕으로 재구성되었다. 특히 `001-what-why-home` branch의 `videoAssessor` 계열 코드를
최신 runtime 기준으로 참고했지만, 현재 `main`은 Jetson/Docker 중심 branch를 그대로 복제한 것이
아니라 회사 제출용 최소 구조로 다시 정리한 버전이다.

## 현재 활성 구성

- 최종 사용자 진입점: `videoAssessor.py`
- 기본 실행 backend: `engine`
- 보조 backend: `onnx`, `pytorch`
- 활성 학습 엔트리: `src/training/train_segformer.py`, `src/training/train_yolo.py`
- archive 참고 자산: 과거 `TheDistanceAssessor*`, 변환 스크립트, 평가 스크립트, 과정 문서

## 저장소 구조

```text
.
├─ videoAssessor.py
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ src/
│  ├─ common/
│  ├─ inference/
│  ├─ rail_detection/
│  ├─ training/
│  └─ utils/
├─ configs/
│  ├─ inference/
│  ├─ training/
│  └─ sweeps/
├─ models/
│  ├─ converted/
│  ├─ final/
│  ├─ references/
│  └─ MODEL_MANIFEST.md
├─ data_samples/
├─ docs/
│  ├─ user_manual/
│  ├─ requirements/
│  ├─ design/
│  ├─ test_report/
│  └─ final_report/
├─ requirements/
└─ archive/
```

### active / archive 구분

- `src/`, `configs/`, `models/`, `docs/`는 현재 제출용 active tree다.
- `archive/`는 과거 runtime, 평가, 변환, delivery 과정 문서, root wrapper를 보존하는 참고 영역이다.
- `TheDistanceAssessor*` 계열 파일은 active tree에서 제거됐고 archive로 이동했다.

## 포함 모델과 현재 상태

현재 저장소에 실제로 포함된 주요 모델/아티팩트는 다음과 같다.

- `models/final/segformer_b3_original_13class.engine`
- `models/converted/segformer_b3_original_13class.onnx`
- `models/converted/SegFormer_B3_1024_finetuned.pth`
- `models/final/yolov8n.pt`

모델 상세는 [MODEL_MANIFEST.md](C:/Users/user/.vscode/python/RailSafeNet_LiDAR/models/MODEL_MANIFEST.md)를
참조한다.

### 중요한 제약

- 포함된 `segformer_b3_original_13class.engine`는 Linux + Titan RTX 기준 산출물이다.
- 현재 Windows workspace는 문서 정리 및 사전 점검용이며, 원래 학습/최적화 환경과 동일하지 않다.
- 따라서 TensorRT engine은 최종 배포 장비에서 재검증 또는 재생성이 필요할 수 있다.

## 설치

상세 절차는 [SETUP_AND_RUN.md](C:/Users/user/.vscode/python/RailSafeNet_LiDAR/docs/user_manual/SETUP_AND_RUN.md)를
참조한다. 기본 설치 명령은 아래와 같다.

### `pip`

```bash
python -m pip install -r requirements.txt
```

### `conda`

```bash
conda env create -f environment.yml
conda activate railsafenet-delivery
```

## 실행 방법

### 1. 도움말

```bash
python videoAssessor.py --help
```

### 2. backend별 사전 점검

```bash
python videoAssessor.py --backend engine --check-only
python videoAssessor.py --backend onnx --check-only
python videoAssessor.py --backend pytorch --check-only
```

### 3. 기본 engine backend 실행 예시

```bash
python videoAssessor.py --backend engine --mode video --video <video_path>
```

### 4. 카메라 모드 예시

```bash
python videoAssessor.py --backend engine --mode camera --camera 0
```

### 5. calibration 예시

```bash
python videoAssessor.py --backend engine --mode video --video <video_path> --calibrate
python videoAssessor.py --backend engine --mode video --video <video_path> --calibrate-vp
```

## 학습 엔트리

현재 active training 엔트리는 아래 두 개만 유지한다.

- `src/training/train_segformer.py`
- `src/training/train_yolo.py`

과거 `DeepLabv3`, 추가 SegFormer 실험 스크립트, sweep 스크립트는 모두 `archive/`로 이동했다.

## 문서 구성

최종 제출용 핵심 문서는 아래 다섯 개를 기준으로 유지한다.

- [SETUP_AND_RUN.md](C:/Users/user/.vscode/python/RailSafeNet_LiDAR/docs/user_manual/SETUP_AND_RUN.md)
- [REQUIREMENTS_SPEC.md](C:/Users/user/.vscode/python/RailSafeNet_LiDAR/docs/requirements/REQUIREMENTS_SPEC.md)
- [SW_ARCHITECTURE.md](C:/Users/user/.vscode/python/RailSafeNet_LiDAR/docs/design/SW_ARCHITECTURE.md)
- [SW_TEST_REPORT.md](C:/Users/user/.vscode/python/RailSafeNet_LiDAR/docs/test_report/SW_TEST_REPORT.md)
- [FINAL_REPORT.md](C:/Users/user/.vscode/python/RailSafeNet_LiDAR/docs/final_report/FINAL_REPORT.md)

## 알려진 한계

- `engine` backend는 현재 active canonical runtime이지만, 현재 Windows workspace에는 `albumentations`,
  `tensorrt`, `pycuda`가 설치되어 있지 않아 실제 실행까지는 바로 이어지지 않는다.
- `pytorch` backend는 `.pth`와 `yolov8n.pt`가 준비돼 있지만 `transformers`가 없으면 runtime 검증이 중단된다.
- `onnx` backend는 SegFormer `.onnx`는 있으나, active YOLO `.onnx` 아티팩트는 없다.
- TensorRT engine은 Titan RTX/Linux 기준 산출물이므로 Jetson 또는 다른 GPU에서 그대로 동작한다고 단정하지 않는다.
- `archive/`에는 참고 가치가 있는 과거 구현이 남아 있으나, 현재 active 사용 경로로 간주하지 않는다.

## provenance

본 저장소의 코드 계보는 [oValach/RailSafeNet](https://github.com/oValach/RailSafeNet)에서 출발했다.
이후 프로젝트 진행 과정에서 여러 branch와 실험 버전이 생성되었고, 현재 `main`은 그 중
`001-what-why-home`의 최신 `videoAssessor` 계열 코드를 적극 참고해 회사 제출용으로 최소화한 정리본이다.
