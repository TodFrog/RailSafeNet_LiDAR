#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""최종 활성 `videoAssessor` engine backend 구현.

이 모듈은 `001-what-why-home` branch의 최신 `videoAssessor_final.py`를 기준으로
현재 저장소 구조와 실제 모델 자산 경로에 맞게 조정한 통합 런타임이다.

핵심 원칙:
- 최종 사용자 진입점은 루트 `videoAssessor.py` 하나로 통일한다.
- 실제 통합 위험도 추론은 이 모듈이 담당한다.
- SegFormer는 저장소에 포함된 TensorRT `.engine`를 우선 사용한다.
- 객체 검출은 YOLO TensorRT 엔진이 없으면 `yolov8n.pt` 기반 Ultralytics 경로로
  fallback 한다.
- 현재 포함된 SegFormer TensorRT 엔진은 Linux + Titan RTX 환경에서 가져온
  산출물이므로, 다른 GPU/OS에서는 재생성 또는 재검증이 필요할 수 있다.
"""

import cv2
import os
import sys
import time
import argparse
import importlib
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import deque

# Fix DISPLAY environment variable
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':1'

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TENSORRT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - platform/runtime에 따라 달라진다.
    trt = None
    cuda = None
    TENSORRT_IMPORT_ERROR = exc

# Project imports
from src.common.metrics_filtered_cls import image_morpho
from src.common.repo_paths import repo_path
from src.inference.video_assessor_helpers import (
    get_clues,
    find_edges,
    select_video_file,
    identify_ego_track,
    find_rail_sides,
)

# Phase 4 - Simple Rail Tracker (Direct Midpoint Calculation)
from src.rail_detection.simple_rail_tracker import SimpleRailTracker

# Phase 6 - BEV
from src.rail_detection.bev_transform import (
    BEVTransformer, BEVConfig, BEVCalibrator, create_default_bev_config
)
from src.rail_detection.bev_path_analyzer import (
    BEVPathAnalyzer, PathDirection, JunctionType
)

# Final components
from src.rail_detection.danger_zone_detector import (
    DangerZoneDetector, DangerLevel, convert_yolo_results_to_detections,
    ALLOWED_HAZARD_CLASSES, VanishingPointCalibrator
)
from src.rail_detection.alert_panel import AlertPanel
from src.rail_detection.mini_bev_renderer import MiniBEVRenderer


SEG_ENGINE_CANDIDATES = [
    repo_path("models", "final", "segformer_b3_original_13class.engine"),
]

YOLO_ENGINE_CANDIDATES = [
    repo_path("models", "final", "yolov8n.engine"),
    repo_path("models", "final", "yolov8s.engine"),
]

YOLO_PT_CANDIDATES = [
    repo_path("models", "final", "yolov8n.pt"),
    repo_path("models", "final", "yolov8s.pt"),
]

DEFAULT_VIDEO_DIR = repo_path("data_samples")
DEFAULT_TRACKER_CONFIG = repo_path("configs", "inference", "rail_tracker_config.yaml")
DEFAULT_BEV_CONFIG = repo_path("configs", "inference", "bev_config.yaml")


def inspect_runtime_dependencies() -> tuple[list[tuple[str, bool, str]], bool]:
    """engine backend가 요구하는 Python 의존성을 실제 import 기준으로 점검한다."""
    results: list[tuple[str, bool, str]] = []
    has_missing_dependency = False
    for module_name in ("tensorrt", "pycuda", "ultralytics"):
        try:
            importlib.import_module(module_name)
            results.append((module_name, True, ""))
        except Exception as exc:  # pragma: no cover - 현재 환경 차이를 그대로 기록한다.
            has_missing_dependency = True
            results.append((module_name, False, f"{type(exc).__name__}: {exc}"))
    return results, has_missing_dependency


def first_existing_path(candidates: list[Path]) -> Path | None:
    """후보 경로 목록에서 실제 존재하는 첫 항목을 반환한다."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_runtime_paths() -> dict[str, Path | None]:
    """현재 저장소 기준의 모델 경로 후보를 정리한다."""
    return {
        "seg_engine": first_existing_path(SEG_ENGINE_CANDIDATES),
        "yolo_engine": first_existing_path(YOLO_ENGINE_CANDIDATES),
        "yolo_pt": first_existing_path(YOLO_PT_CANDIDATES),
    }


def run_check_only() -> int:
    """실제 모델 로드 없이 engine backend 실행 준비 상태를 출력한다."""
    dependency_results, has_missing_dependency = inspect_runtime_dependencies()
    runtime_paths = build_runtime_paths()

    print("videoAssessor engine backend 사전 점검")
    print("\n[의존성]")
    for module_name, is_available, detail in dependency_results:
        print(f"- {module_name}: {'OK' if is_available else 'MISSING'}")
        if detail:
            print(f"  상세: {detail}")

    print("\n[모델]")
    for label, path in runtime_paths.items():
        print(f"- {label}: {path if path else 'MISSING'}")

    print("\n[환경 제약]")
    print("- 포함된 SegFormer TensorRT 엔진은 Linux + Titan RTX 기준 산출물입니다.")
    print("- 현재 Windows workspace에서는 실제 TensorRT 실행보다 구조 검증과 사전 점검이 우선입니다.")

    if runtime_paths["seg_engine"] and (runtime_paths["yolo_engine"] or runtime_paths["yolo_pt"]):
        if runtime_paths["yolo_engine"]:
            print("\n결론: TensorRT 기반 engine backend 준비 조건이 기본적으로 충족되었습니다.")
            return 0 if not has_missing_dependency else 1
        print("\n결론: SegFormer engine은 준비되었고 YOLO는 `.pt` fallback 경로를 사용할 수 있습니다.")
        print("TODO: 최종 배포 장비에서는 YOLO TensorRT 엔진 재생성 여부를 다시 판단해야 합니다.")
        return 0 if not has_missing_dependency else 1

    print("\n결론: engine backend 실행 준비가 아직 완전하지 않습니다.")
    return 1


# =============================================================================
# TensorRT Engine Classes
# =============================================================================
class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


def allocate_buffers(engine, stream=None):
    inputs, outputs, bindings = [], [], []
    if stream is None:
        stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(tensor_name)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


class TRTEngine:
    def __init__(self, engine_path, cuda_stream=None):
        if TENSORRT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "TensorRT 또는 PyCUDA를 import하지 못했습니다. "
                f"상세: {type(TENSORRT_IMPORT_ERROR).__name__}: {TENSORRT_IMPORT_ERROR}"
            )
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)

        if -1 in self.input_shape:
            self.context.set_input_shape(self.input_name, self.input_shape)

        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.engine, cuda_stream
        )


class TRTSegmentationEngine(TRTEngine):
    def __init__(self, engine_path, cuda_stream=None):
        super().__init__(engine_path, cuda_stream)
        self.expected_height = self.input_shape[2]
        self.expected_width = self.input_shape[3]
        print(f"  SegFormer INT8 [{self.expected_height}x{self.expected_width}]")

    def infer(self, input_data):
        np.copyto(self.inputs[0].host, input_data.ravel())
        trt_outputs = do_inference_v2(
            self.context, self.bindings, self.inputs, self.outputs, self.stream
        )
        output_name = self.engine.get_tensor_name(1)
        output_shape = self.engine.get_tensor_shape(output_name)
        return trt_outputs[0].reshape(output_shape)


class TRTYOLOEngine(TRTEngine):
    def __init__(self, engine_path, cuda_stream=None):
        super().__init__(engine_path, cuda_stream)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        print(f"  YOLO INT8 [{self.input_height}x{self.input_width}]")
        self.names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 7: 'truck',
            15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            24: 'backpack', 25: 'umbrella', 28: 'suitcase', 36: 'skateboard'
        }

    def predict(self, image):
        original_shape = image.shape[:2]
        image_resized = cv2.resize(image, (self.input_width, self.input_height))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_norm = (image_rgb.astype(np.float32) / 255.0).transpose((2, 0, 1))

        np.copyto(self.inputs[0].host, image_norm.ravel())
        trt_outputs = do_inference_v2(
            self.context, self.bindings, self.inputs, self.outputs, self.stream
        )

        output_name = self.engine.get_tensor_name(1)
        output_shape = self.engine.get_tensor_shape(output_name)
        output_data = trt_outputs[0].reshape(output_shape)

        results = self.post_process(output_data, original_shape)
        return [results]

    def post_process(self, output, original_shape):
        output = np.squeeze(output).T
        boxes, scores, class_ids = [], [], []
        scale_x = original_shape[1] / self.input_width
        scale_y = original_shape[0] / self.input_height

        for detection in output:
            box, class_scores = detection[:4], detection[4:]
            class_id = np.argmax(class_scores)
            max_score = class_scores[class_id]

            if max_score > self.conf_threshold:
                x_center, y_center, width, height = box
                boxes.append([
                    x_center * scale_x, y_center * scale_y,
                    width * scale_x, height * scale_y
                ])
                scores.append(float(max_score))
                class_ids.append(int(class_id))

        final_boxes, final_scores, final_class_ids = [], [], []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                final_boxes = [boxes[i] for i in indices]
                final_scores = [scores[i] for i in indices]
                final_class_ids = [class_ids[i] for i in indices]

        class MockResults:
            def __init__(self, boxes, scores, class_ids):
                self.boxes = self.MockBoxes(boxes, scores, class_ids)

            class MockBoxes:
                def __init__(self, boxes, scores, class_ids):
                    self.xywh = torch.tensor(boxes) if boxes else torch.empty(0, 4)
                    self.cls = torch.tensor(class_ids) if class_ids else torch.empty(0)

                def tolist(self):
                    return self.xywh.tolist() if len(self.xywh) > 0 else []

        return MockResults(final_boxes, final_scores, final_class_ids)


class UltralyticsYOLODetector:
    """YOLO TensorRT 엔진이 없을 때 `.pt` 가중치로 동작하는 fallback 검출기."""

    def __init__(self, model_path: str | Path):
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        print(f"  YOLO PT fallback [{self.model_path.name}]")

    def predict(self, image):
        return self.model.predict(source=image, verbose=False)


# =============================================================================
# Final Processor
# =============================================================================
class FinalProcessor:
    """
    Production-ready rail detection processor.

    Combines:
    - Phase 4: Polynomial tracking for stable rail detection
    - Phase 6: BEV for direction classification only
    - Danger zone detection with YOLO overlap checking
    - Alert panel and mini BEV visualization
    """

    def __init__(
        self,
        model_seg,
        model_det,
        image_size: List[int] = [512, 896],
        config_path: str = str(DEFAULT_TRACKER_CONFIG),
        bev_config_path: str = str(DEFAULT_BEV_CONFIG)
    ):
        """Initialize the final processor."""
        self.model_seg = model_seg
        self.model_det = model_det
        self.image_size = image_size
        self.bev_config_path = str(bev_config_path)

        # Image transform for segmentation
        self.transform_img = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Load configurations
        self.config = self._load_config(config_path)
        self.bev_config = self._load_config(bev_config_path)

        # Components (initialized on first frame)
        self.simple_tracker: Optional[SimpleRailTracker] = None
        self.bev_transformer: Optional[BEVTransformer] = None
        self.bev_analyzer: Optional[BEVPathAnalyzer] = None
        self.danger_detector: Optional[DangerZoneDetector] = None
        self.alert_panel: Optional[AlertPanel] = None
        self.mini_bev: Optional[MiniBEVRenderer] = None

        self.tracker_config_path = str(config_path)

        # Caching
        self.frame_counter = 0
        self.cached_seg_mask = None
        self.cached_det_results = None
        self.cached_bev_result = None

        # Cache intervals
        perf_cfg = self.config.get('performance', {})
        self.seg_cache_interval = perf_cfg.get('segmentation_cache_interval', 3)
        self.det_cache_interval = perf_cfg.get('detection_cache_interval', 1)
        self.bev_cache_interval = 5  # BEV direction doesn't change fast (increased for performance)

        print(f"  Cache: Seg={self.seg_cache_interval}f, Det={self.det_cache_interval}f, BEV={self.bev_cache_interval}f")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _init_components(self, height: int, width: int):
        """Initialize all components for given frame size."""
        # Simple Rail Tracker (Phase 4 - Direct Midpoint Calculation)
        if self.simple_tracker is None:
            try:
                self.simple_tracker = SimpleRailTracker(
                    height, width, alpha=0.3  # EMA smoothing
                )
                print(f"  Simple Rail Tracker [{height}x{width}] (alpha=0.3)")
            except Exception as e:
                print(f"  Tracker init failed: {e}")

        # BEV transformer (Phase 6 - direction only)
        if self.bev_transformer is None:
            bev_cfg = self.bev_config.get('bev_transform', {})
            if bev_cfg:
                src_pts = bev_cfg.get('source_points', {})
                out_size = bev_cfg.get('output_size', {})

                config = BEVConfig(
                    src_top_left=tuple(src_pts.get('top_left', [480, 486])),
                    src_top_right=tuple(src_pts.get('top_right', [1440, 486])),
                    src_bottom_right=tuple(src_pts.get('bottom_right', [width, height])),
                    src_bottom_left=tuple(src_pts.get('bottom_left', [0, height])),
                    bev_width=out_size.get('width', 400),
                    bev_height=out_size.get('height', 600)
                )
                self.bev_transformer = BEVTransformer(config=config)
            else:
                config = create_default_bev_config(width, height)
                self.bev_transformer = BEVTransformer(config=config)
            print(f"  BEV Transformer [{self.bev_transformer.bev_size[0]}x{self.bev_transformer.bev_size[1]}]")

        # BEV analyzer
        if self.bev_analyzer is None:
            self.bev_analyzer = BEVPathAnalyzer(config_path=self.bev_config_path)
            print(f"  BEV Path Analyzer")

        # Danger zone detector
        if self.danger_detector is None:
            self.danger_detector = DangerZoneDetector(config=self.config)
            print(f"  Danger Zone Detector")

        # Alert panel (top-left)
        if self.alert_panel is None:
            self.alert_panel = AlertPanel(
                position=(10, 10),
                size=(300, 180),
                opacity=0.9
            )
            print(f"  Alert Panel")

        # Mini BEV (bottom-right)
        if self.mini_bev is None:
            self.mini_bev = MiniBEVRenderer(
                position="bottom-right",
                size=(240, 360),
                margin=20
            )
            print(f"  Mini BEV Renderer")

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame.

        Returns dict with all results for rendering.
        """
        height, width = frame.shape[:2]
        self.frame_counter += 1

        # Initialize on first frame
        if self.simple_tracker is None:
            self._init_components(height, width)

        result = {
            'success': False,
            'frame_number': self.frame_counter,
            'height': height,
            'width': width
        }

        # === Segmentation (with caching) ===
        if (self.frame_counter - 1) % self.seg_cache_interval == 0 or self.cached_seg_mask is None:
            image_tr = self.transform_img(image=frame)['image'].unsqueeze(0)
            output = self.model_seg.infer(image_tr.numpy().astype(np.float32))
            id_map = np.argmax(
                F.softmax(torch.from_numpy(output), dim=1).cpu().detach().numpy().squeeze(),
                axis=0
            ).astype(np.uint8)
            id_map = image_morpho(id_map)
            id_map = cv2.resize(id_map, (width, height), interpolation=cv2.INTER_NEAREST)
            self.cached_seg_mask = id_map
        else:
            id_map = self.cached_seg_mask

        # Class filtering: Keep only classes 4 (Rail Track) and 9 (Rail Road)
        # All other classes are set to 0 (background) for improved performance
        id_map = np.where((id_map == 4) | (id_map == 9), id_map, 0).astype(np.uint8)

        result['seg_mask'] = id_map

        # === Detection (with caching) ===
        if (self.frame_counter - 1) % self.det_cache_interval == 0 or self.cached_det_results is None:
            detection_results = self.model_det.predict(frame)
            self.cached_det_results = detection_results
        else:
            detection_results = self.cached_det_results

        result['detection_results'] = detection_results

        # === Phase 4: Rail Centerline Extraction (Direct Midpoint) ===
        track_result = self._extract_rail_centerline(id_map, height, width)
        result['track'] = track_result

        # === Danger Zone Detection ===
        if track_result['valid'] and self.danger_detector:
            # Generate hazard zones from tracked rail
            center_pts = np.array(track_result['center_points'])
            width_pts = np.array(track_result['width_profile'])

            zones = self.danger_detector.generate_hazard_zones(
                center_pts, width_pts, height, width
            )
            result['hazard_zones'] = zones

            # Check YOLO detections against zones
            detections = convert_yolo_results_to_detections(detection_results)
            overlap_result = self.danger_detector.check_overlaps(detections, zones)

            result['danger_status'] = overlap_result
        else:
            result['hazard_zones'] = {}
            result['danger_status'] = {
                'max_severity': DangerLevel.SAFE,
                'zone_counts': {DangerLevel.RED: 0, DangerLevel.ORANGE: 0, DangerLevel.YELLOW: 0},
                'hazards': []
            }

        # === Phase 6: BEV Direction (with caching) ===
        if (self.frame_counter - 1) % self.bev_cache_interval == 0 or self.cached_bev_result is None:
            bev_result = self._analyze_bev(frame, id_map)
            self.cached_bev_result = bev_result
        else:
            bev_result = self.cached_bev_result

        result['bev'] = bev_result
        result['success'] = True

        return result

    def _extract_rail_centerline(self, seg_mask: np.ndarray, height: int, width: int) -> Dict:
        """
        Extract rail centerline using direct midpoint calculation.

        No polynomial fitting - just calculate midpoints at each y level.
        This preserves curves naturally and is more robust.
        """
        track_result = {
            'valid': False,
            'center_points': [],
            'left_edge': None,
            'right_edge': None,
            'width_profile': np.array([])
        }

        if self.simple_tracker is None:
            return track_result

        try:
            # 1. Generate y_levels using get_clues (20 samples for smoother curves)
            y_levels = get_clues(seg_mask, 20)
            if not y_levels:
                return track_result

            # 2. Find edges at each y level
            edges_dict = find_edges(seg_mask, y_levels, values=[4, 9])
            if not edges_dict:
                return track_result

            # 3. Identify ego track (closest to center)
            ego_edges = identify_ego_track(edges_dict, width)
            if not ego_edges:
                return track_result

            # 4. Extract left/right edge arrays
            left_edge, right_edge = find_rail_sides(seg_mask, ego_edges)
            if len(left_edge) < 2 or len(right_edge) < 2:
                return track_result

            # Convert to numpy arrays
            left_edge = np.array(left_edge)
            right_edge = np.array(right_edge)

            # 5. Direct midpoint calculation (SimpleRailTracker)
            center_pts, width_pts = self.simple_tracker.update(left_edge, right_edge)

            # 6. Valid if at least 2 points
            if len(center_pts) >= 2:
                track_result['valid'] = True
                track_result['center_points'] = center_pts
                track_result['left_edge'] = left_edge
                track_result['right_edge'] = right_edge
                track_result['width_profile'] = width_pts

        except Exception as e:
            pass

        return track_result

    def _analyze_bev(self, frame: np.ndarray, seg_mask: np.ndarray) -> Dict:
        """Perform BEV analysis for direction detection."""
        bev_result = {
            'bev_image': None,
            'bev_mask': None,
            'paths': [],
            'ego_path': None,
            'junction_type': JunctionType.NONE,
            'ego_direction': PathDirection.UNKNOWN,
            'ego_angle': 0.0,
            'num_paths': 0
        }

        if self.bev_transformer is None or self.bev_analyzer is None:
            return bev_result

        try:
            bev_mask = self.bev_transformer.warp_mask_to_bev(seg_mask)
            bev_image = self.bev_transformer.warp_to_bev(frame)

            analysis = self.bev_analyzer.analyze_frame(
                bev_mask, self.bev_transformer.config.bev_width
            )

            bev_result['bev_image'] = bev_image
            bev_result['bev_mask'] = bev_mask
            bev_result['paths'] = analysis['paths']
            bev_result['ego_path'] = analysis['ego_path']
            bev_result['junction_type'] = analysis['junction_type']
            bev_result['ego_direction'] = analysis['ego_direction']
            bev_result['ego_angle'] = analysis['ego_angle']
            bev_result['num_paths'] = analysis['num_paths']

        except Exception:
            pass

        return bev_result

    def render_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Render all visualizations on the frame.

        Args:
            frame: Original frame
            result: Processing results

        Returns:
            Rendered frame with all overlays
        """
        output = frame.copy()
        height, width = frame.shape[:2]

        track_result = result.get('track', {})
        bev_result = result.get('bev', {})
        danger_status = result.get('danger_status', {})
        hazard_zones = result.get('hazard_zones', {})

        # === 0. Draw SegFormer Rail Mask Overlay ===
        seg_mask = result.get('seg_mask')
        if seg_mask is not None:
            # Create rail overlay (classes 4=Rail Track, 9=Rail Road)
            rail_mask = ((seg_mask == 4) | (seg_mask == 9)).astype(np.uint8)
            if np.any(rail_mask):
                # Create colored overlay for rail region
                overlay = output.copy()
                overlay[rail_mask == 1] = (200, 200, 50)  # Cyan-ish for rail
                output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

        # === 1. Draw hazard zones ===
        if hazard_zones and self.danger_detector:
            output = self.danger_detector.draw_zones(output, hazard_zones)

        # === 2. Draw rail center line ===
        if track_result.get('valid'):
            output = self._draw_rail_overlay(output, track_result)

        # === 3. Draw ALL YOLO detections ===
        hazards = danger_status.get('hazards', [])
        hazard_bboxes = set()

        # First, draw hazards (objects in danger zones) with zone colors
        if hazards and self.danger_detector:
            output = self.danger_detector.draw_hazards(output, hazards)
            hazard_bboxes = {h.bbox_xyxy for h in hazards}

        # Then, draw safe objects (not in danger zones) with blue color
        # Note: convert_yolo_results_to_detections already filters by ALLOWED_HAZARD_CLASSES
        detection_results = result.get('detection_results', [])
        if detection_results:
            detections = convert_yolo_results_to_detections(detection_results)
            for det in detections:
                bbox = det.get('bbox_xyxy')
                if bbox is None:
                    continue
                # Skip if already drawn as hazard
                if tuple(bbox) in hazard_bboxes:
                    continue
                # Double-check class filter (in case detection was passed directly)
                class_id = det.get('class_id')
                if class_id is not None and class_id not in ALLOWED_HAZARD_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                # Safe object: blue box
                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 100, 0), 2)
                label = f"{det.get('class_name', 'obj')}"
                cv2.putText(output, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # === 4. Alert Panel (top-left) ===
        if self.alert_panel:
            severity = danger_status.get('max_severity', DangerLevel.SAFE)
            direction = bev_result.get('ego_direction', PathDirection.UNKNOWN)
            zone_counts = danger_status.get('zone_counts', {})
            angle = bev_result.get('ego_angle', 0.0)

            output = self.alert_panel.render(
                output, severity, direction, zone_counts, angle
            )

        # === 5. Mini BEV (bottom-right) ===
        if self.mini_bev:
            bev_image = bev_result.get('bev_image')
            bev_mask = bev_result.get('bev_mask')
            paths = bev_result.get('paths', [])
            ego_path = bev_result.get('ego_path')
            direction = bev_result.get('ego_direction', PathDirection.UNKNOWN)
            angle = bev_result.get('ego_angle', 0.0)
            junction = bev_result.get('junction_type', JunctionType.NONE)

            # Transform centerline to BEV coordinates
            bev_centerline = None
            bev_size = None
            if track_result.get('valid') and self.bev_transformer:
                center_pts = track_result.get('center_points')
                if center_pts is not None and len(center_pts) >= 2:
                    center_pts_array = np.array(center_pts, dtype=np.float32)
                    bev_centerline = self.bev_transformer.warp_points_to_bev(center_pts_array)
                    bev_size = self.bev_transformer.bev_size

            output = self.mini_bev.render(
                output, bev_image, bev_mask,
                paths, ego_path, direction, angle, junction,
                centerline_pts=bev_centerline,
                bev_size=bev_size
            )

        return output

    def _draw_rail_overlay(self, frame: np.ndarray, track_result: Dict) -> np.ndarray:
        """Draw rail center line."""
        output = frame.copy()
        center_pts = track_result.get('center_points')

        # Draw center line (yellow-green for visibility)
        if center_pts is not None and len(center_pts) >= 2:
            pts = np.array(center_pts, dtype=np.int32)
            # Main line (thick, green-yellow)
            cv2.polylines(output, [pts], False, (0, 255, 128), 4)
            # Outline for contrast
            cv2.polylines(output, [pts], False, (0, 0, 0), 6)
            cv2.polylines(output, [pts], False, (0, 255, 128), 3)

        return output


# =============================================================================
# Main Application
# =============================================================================
def run_video_mode(processor: FinalProcessor, video_dir: str, fullscreen: bool = False):
    """Run in video mode with file selection."""
    print("\n" + "=" * 60)
    print("Video Mode")
    print("Keys: 'q'=quit, 'SPACE'=pause, 'n'=next, 'r'=restart")
    print("=" * 60)

    window_name = 'RailSafeNet Final'
    if fullscreen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    single_video_mode = Path(video_dir).is_file()

    while True:
        video_path = select_video_file(video_dir)
        if video_path is None:
            break

        print(f"\n{os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open: {video_path}")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{fps} FPS, {total_frames} frames")

        fps_deque = deque(maxlen=30)
        frame_count = 0
        paused = False
        processor.frame_counter = 0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                start_time = time.time()

                try:
                    result = processor.process_frame(frame)

                    if result['success']:
                        vis_frame = processor.render_frame(frame, result)

                        # FPS calculation
                        frame_fps = 1 / (time.time() - start_time)
                        fps_deque.append(frame_fps)
                        avg_fps = sum(fps_deque) / len(fps_deque)

                        # FPS overlay
                        info = f"Frame: {frame_count}/{total_frames} | FPS: {avg_fps:.1f}"
                        cv2.putText(vis_frame, info, (10, vis_frame.shape[0] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(vis_frame, info, (10, vis_frame.shape[0] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Resize for display if not fullscreen
                        if not fullscreen:
                            display_h, display_w = vis_frame.shape[:2]
                            vis_frame = cv2.resize(vis_frame, (display_w // 2, display_h // 2))

                        cv2.imshow(window_name, vis_frame)
                    else:
                        display_frame = frame
                        if not fullscreen:
                            display_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                        cv2.imshow(window_name, display_frame)

                except Exception as e:
                    print(f"Frame {frame_count} error: {e}")
                    import traceback
                    traceback.print_exc()
                    display_frame = frame
                    if not fullscreen:
                        display_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                    cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            elif key == ord('n'):
                break
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                processor.frame_counter = 0
                fps_deque.clear()

        cap.release()
        if single_video_mode:
            break

    cv2.destroyAllWindows()


def run_video_save_mode(processor: FinalProcessor, video_path: str, output_path: str, start_time: float = 0):
    """Process video and save to output file (no display)."""
    print("\n" + "=" * 60)
    print("Video Save Mode")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Start time: {start_time}s")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames ({duration:.1f}s)")

    # Calculate start frame
    start_frame = int(start_time * fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Skipping to frame {start_frame} ({start_time}s)")

    frames_to_process = total_frames - start_frame
    print(f"Frames to process: {frames_to_process}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Cannot create output video: {output_path}")
        cap.release()
        return

    fps_deque = deque(maxlen=30)
    frame_count = 0
    processor.frame_counter = 0

    print("\nProcessing... (Press Ctrl+C to stop)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start_proc_time = time.time()

            try:
                result = processor.process_frame(frame)

                if result['success']:
                    vis_frame = processor.render_frame(frame, result)

                    # FPS calculation
                    frame_fps = 1 / (time.time() - start_proc_time)
                    fps_deque.append(frame_fps)
                    avg_fps = sum(fps_deque) / len(fps_deque)

                    # Progress info overlay
                    progress = (frame_count / frames_to_process) * 100
                    info = f"Frame: {frame_count}/{frames_to_process} ({progress:.1f}%) | FPS: {avg_fps:.1f}"
                    cv2.putText(vis_frame, info, (10, vis_frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(vis_frame, info, (10, vis_frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    out.write(vis_frame)
                else:
                    out.write(frame)

                # Progress output
                if frame_count % 100 == 0:
                    progress = (frame_count / frames_to_process) * 100
                    print(f"  Progress: {frame_count}/{frames_to_process} ({progress:.1f}%) - {avg_fps:.1f} FPS")

            except Exception as e:
                print(f"Frame {frame_count} error: {e}")
                out.write(frame)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    cap.release()
    out.release()

    print(f"\nDone! Processed {frame_count} frames")
    print(f"Output saved to: {output_path}")


def run_camera_mode(processor: FinalProcessor, camera_id: int = 0, fullscreen: bool = True):
    """Run in camera mode for real-time processing."""
    print("\n" + "=" * 60)
    print("Camera Mode")
    print("Keys: 'q'=quit, 's'=screenshot")
    print("=" * 60)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    window_name = 'RailSafeNet Final - Camera'
    if fullscreen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps_deque = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        start_time = time.time()

        try:
            result = processor.process_frame(frame)

            if result['success']:
                vis_frame = processor.render_frame(frame, result)

                # FPS calculation
                frame_fps = 1 / (time.time() - start_time)
                fps_deque.append(frame_fps)
                avg_fps = sum(fps_deque) / len(fps_deque)

                # FPS overlay
                cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, vis_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(window_name, vis_frame)
            else:
                cv2.imshow(window_name, frame)

        except Exception as e:
            print(f"Processing error: {e}")
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{timestamp}.png", vis_frame)
            print(f"Screenshot saved: screenshot_{timestamp}.png")

    cap.release()
    cv2.destroyAllWindows()


def run_calibration(video_path: str, config_path: str = str(DEFAULT_BEV_CONFIG)):
    """Run interactive BEV calibration."""
    print("=" * 60)
    print("BEV Calibration Mode")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Cannot read frame")
        return

    print("Click 2 top corners: TL, TR (bottom corners fixed at image edge)")
    print("Press 's' to save, 'r' to reset, 'q' to quit")

    calibrator = BEVCalibrator(frame, bev_width=800, bev_height=600, display_scale=0.5)
    transformer = calibrator.run(save_path=config_path)

    if transformer:
        print(f"Saved to {config_path}")
    else:
        print("Cancelled")


def run_vp_calibration(video_path: str, config_path: str = str(DEFAULT_TRACKER_CONFIG)):
    """
    Run interactive vanishing point calibration.

    Allows user to click on screen to set vanishing point for
    perspective-correct hazard zone generation.
    """
    print("=" * 60)
    print("VP (Vanishing Point) Calibration Mode")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Cannot read frame")
        return

    print(f"\nFrame size: {frame.shape[1]}x{frame.shape[0]}")
    print("Click to set vanishing point where rail tracks converge")

    calibrator = VanishingPointCalibrator(frame, display_scale=0.5)
    vp = calibrator.run(save_path=config_path)

    if vp:
        print(f"\nVP calibration complete: ({vp[0]}, {vp[1]})")
        print(f"  Saved to {config_path}")
        print("\nRun video mode to see the new hazard zones:")
        print("  python videoAssessor.py --backend engine --mode video")
    else:
        print("\nCalibration cancelled")


def build_parser() -> argparse.ArgumentParser:
    """engine backend 전용 CLI parser를 만든다."""
    parser = argparse.ArgumentParser(description="videoAssessor engine backend")
    parser.add_argument("--mode", choices=["video", "camera"], default="video", help="입력 모드")
    parser.add_argument("--video", type=str, default=None, help="직접 지정할 비디오 경로")
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 번호")
    parser.add_argument("--config", type=str, default=str(DEFAULT_TRACKER_CONFIG), help="선로 추적 설정 파일")
    parser.add_argument("--bev-config", type=str, default=str(DEFAULT_BEV_CONFIG), help="BEV 설정 파일")
    parser.add_argument("--fullscreen", action="store_true", help="전체 화면 표시")
    parser.add_argument("--output", type=str, default=None, help="화면 표시 없이 결과 비디오 저장")
    parser.add_argument("--start-time", type=float, default=0.0, help="비디오 시작 시각(초)")
    parser.add_argument("--calibrate", action="store_true", help="BEV calibration 실행")
    parser.add_argument("--calibrate-vp", action="store_true", help="소실점 calibration 실행")
    parser.add_argument("--check-only", action="store_true", help="실행 준비 상태만 점검")
    parser.add_argument("--seg-model", type=str, default=None, help="SegFormer TensorRT engine 경로 override")
    parser.add_argument("--det-model", type=str, default=None, help="YOLO engine 또는 .pt 경로 override")
    return parser


def load_runtime_models(seg_model: str | None = None, det_model: str | None = None):
    """현재 저장소 모델 자산 기준으로 SegFormer/YOLO 런타임을 로드한다."""
    runtime_paths = build_runtime_paths()
    seg_path = Path(seg_model) if seg_model else runtime_paths["seg_engine"]
    det_path = Path(det_model) if det_model else (runtime_paths["yolo_engine"] or runtime_paths["yolo_pt"])

    if seg_path is None:
        raise FileNotFoundError("SegFormer TensorRT engine 파일을 찾지 못했습니다.")
    if det_path is None:
        raise FileNotFoundError("YOLO engine 또는 `.pt` 모델 파일을 찾지 못했습니다.")

    model_seg = TRTSegmentationEngine(str(seg_path))
    if det_path.suffix == ".engine":
        model_det = TRTYOLOEngine(str(det_path))
    else:
        model_det = UltralyticsYOLODetector(str(det_path))
    return model_seg, model_det


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    if args.check_only:
        return run_check_only()

    video_dir = os.environ.get("VIDEO_DIR", str(DEFAULT_VIDEO_DIR))

    if args.calibrate:
        video_path = args.video if args.video else select_video_file(video_dir)
        if video_path:
            run_calibration(video_path, args.bev_config)
        return 0

    if args.calibrate_vp:
        video_path = args.video if args.video else select_video_file(video_dir)
        if video_path:
            run_vp_calibration(video_path, args.config)
        return 0

    print("=" * 70)
    print("videoAssessor engine backend")
    print("=" * 70)

    print("\n모델 로딩 중...")
    try:
        model_seg, model_det = load_runtime_models(args.seg_model, args.det_model)
    except Exception as exc:
        print(f"런타임 모델 로딩 실패: {exc}")
        return 1

    print("\n프로세서 초기화 중...")
    processor = FinalProcessor(
        model_seg,
        model_det,
        image_size=[512, 896],
        config_path=args.config,
        bev_config_path=args.bev_config,
    )

    print("\n준비 완료")

    if args.mode == "video":
        if args.output:
            video_path = args.video if args.video else select_video_file(video_dir)
            if video_path:
                run_video_save_mode(processor, video_path, args.output, args.start_time)
            return 0

        if args.video:
            if Path(args.video).exists():
                run_video_mode(processor, args.video, args.fullscreen)
                return 0
            print(f"비디오 파일을 찾지 못했습니다: {args.video}")
            return 1

        run_video_mode(processor, video_dir, args.fullscreen)
        return 0

    run_camera_mode(processor, args.camera, args.fullscreen)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("Closed")
