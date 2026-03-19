"""Ultralytics YOLO CLI를 호출하는 얇은 학습 진입 스크립트.

이 스크립트는 복잡한 파이프라인을 직접 구현하지 않고, 최종 제출 시 어떤 데이터
설정 파일과 기본 모델을 사용했는지 명확히 남기기 위한 실행 기록용 엔트리포인트다.
실제 학습 파라미터는 현재 코드에 하드코딩되어 있으므로, 재사용 전 검토가 필요하다.
"""

import os
import ultralytics
from src.common.repo_paths import repo_path

ultralytics.checks()
import comet_ml; comet_ml.init()

logger = 'Comet' #@param ['Comet', 'TensorBoard']

# 재구성 이후 데이터 설정은 저장소 내부 configs 경로를 우선 사용한다.
data_config = repo_path("configs", "training", "pilsen.yaml")
os.system(f'yolo train model=yolov8s.pt data=\"{data_config}\" epochs=50 imgsz=640')
