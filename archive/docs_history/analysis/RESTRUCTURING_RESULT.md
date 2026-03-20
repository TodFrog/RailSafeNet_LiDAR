# 저장소 재구성 결과 요약

작성일: 2026-03-18

## 1. 이동한 항목

- 루트 실행 스크립트를 `src/` 하위 목적별 디렉터리로 이동했다.
  - 추론: `src/inference/`
  - 학습: `src/training/`
  - 변환: `src/conversion/`
  - 평가: `src/evaluation/`
  - 공용 유틸: `src/common/`
- 설정 파일을 `configs/`로 이동했다.
  - `configs/training/pilsen.yaml`
  - `configs/sweeps/sweep_transfer*.yaml`
- 모델과 참조 파일을 `models/`로 이동했다.
  - 실제 모델: `models/final/yolov8n.pt`
  - 포인터 파일: `models/references/`
- 샘플 데이터와 문서 자산을 목적별 위치로 이동했다.
  - `data_samples/rs19/`
  - `data_samples/pilsen/`
  - `docs/assets/`
- 의존성 파일을 `requirements/base.txt`로 이동했고, 목적별 requirements 초안 파일을 추가했다.

## 2. archive로 보관한 항목

- 레거시 실행 스크립트
  - `archive/legacy/TheDistanceAssessor.py`
  - `archive/legacy/TheDistanceAssessor_2.py`
- 구형 또는 불일치 문서
  - `archive/old_docs/ESSENTIAL_FILES_ROADMAP.md`
- 실험 로그
  - `archive/experiments/result.txt`
- 현재 참조가 끊겼거나 검증되지 않은 스크립트/설정
  - `archive/broken_or_unverified/sweep.yaml`
  - `archive/broken_or_unverified/sweep_add_agent.py`
  - `archive/broken_or_unverified/test_all_cls.py`
  - `archive/broken_or_unverified/test_pilsen.py`
  - `archive/broken_or_unverified/metrics_all_cls.py`

## 3. 그대로 유지한 항목

- 루트 실행 파일명은 유지했다.
  - 다만 실제 구현은 `src/`로 이동했고, 루트에는 호환성 wrapper를 남겼다.
- `scripts/` 디렉터리는 완전히 제거하지 않았다.
  - 현재 실제로 참조되는 모듈에 한해 shim 파일을 남겨 import 호환성을 유지했다.
- 외부 `/home/mmc-server4/...` 절대경로 fallback은 즉시 제거하지 않았다.
  - 현재 저장소에 실제 모델 파일이 부족하므로 보수적으로 유지했다.
- `docs/analysis/` 내부 분석 문서는 그대로 유지했다.

## 4. 알려진 breakage 리스크

- 현재 실행 환경에는 Python 런타임이 없어 실제 실행 검증은 수행하지 못했다.
- 루트 wrapper 방식은 유지했지만, 일부 스크립트는 여전히 외부 모델 경로와 외부 데이터 경로에 의존한다.
- `TheDistanceAssessor_3_engine.py`는 미추적 상태에서 이동되었으므로 최종 대표 엔트리포인트로 확정된 것은 아니다.
- `scripts/test_all_cls.py`, `scripts/test_pilsen.py`, `metrics_all_cls.py`는 archive로 이동했기 때문에 해당 구형 직접 실행 경로는 기본 구조에서 제외됐다.
- 빈 디렉터리 일부는 작업 트리에는 존재하지만, Git 추적 기준에서는 별도 파일이 없으면 유지되지 않을 수 있다.

## 5. 다음 검증 단계

1. Python이 설치된 환경에서 루트 wrapper import가 정상 동작하는지 확인한다.
2. 대표 추론 경로 1개를 실제 모델 파일과 함께 실행한다.
3. 대표 학습 경로 1개와 변환 경로 1개를 각각 smoke test 한다.
4. 외부 절대경로 대신 `models/`, `data_samples/`, `configs/` 기반 실행이 가능한지 점검한다.
5. 최종 제출용 문서 4종을 `docs/requirements/`, `docs/architecture/`, `docs/tests/`, `docs/final_report/`에 작성한다.
