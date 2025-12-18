#!/usr/bin/env python3
"""
Video Distance Assessor - Phase 4: Temporal Tracking with INT8
Phase 3 (INT8 + Cache) + Phase 4 (Ego-Track Temporal Tracking)

Phase 4 Features:
- INT8 quantization + intelligent caching (35-38 FPS)
- Ego-track temporal tracking with Kalman filter
- Rail width profile learning (150-300 frames)
- Track continuity through occlusions (up to 30 frames)

Usage:
    # Phase 4 mode (tracking enabled)
    python3 videoAssessor_phase4_tracking.py --enable-tracking

    # Phase 3 mode (tracking disabled, baseline)
    python3 videoAssessor_phase4_tracking.py
"""

import cv2
import os

# Fix DISPLAY environment variable for GUI display
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':1'
    print(f"✓ Set DISPLAY={os.environ['DISPLAY']}")

import time
import numpy as np
import glob
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import deque
import threading
import argparse

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Phase 3 imports
from src.utils.cuda_utils import measure_parallel_overlap

# Phase 4 imports - TRACKING
from src.rail_detection.ego_tracker import EgoTracker
from src.rail_detection.width_profile import RailWidthProfileLearner

# 기존 helper functions
from scripts.metrics_filtered_cls import image_morpho

# ============================================================================
# CONFIGURATION
# ============================================================================
SEG_CACHE_INTERVAL = 3  # SegFormer를 3 프레임마다 1번 실행
DET_CACHE_INTERVAL = 2  # YOLO를 2 프레임마다 1번 실행

# Global flag (set by command line)
TRACKING_ENABLED = False

# -------------------------------------------------------------------
# TensorRT Engine Classes (from Phase 3)
# -------------------------------------------------------------------
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def allocate_buffers(engine, stream=None):
    inputs, outputs, bindings = [], [], []
    if stream is None:
        stream = cuda.Stream()
    for i in range(engine.num_bindings):
        binding_name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(binding_name)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding_name))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding_name):
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
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        if self.engine.has_implicit_batch_dimension == False:
            input_shape = self.engine.get_binding_shape(0)
            self.context.set_binding_shape(0, input_shape)

        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, cuda_stream)
        input_binding_name = self.engine.get_binding_name(0)
        self.input_shape = self.engine.get_binding_shape(input_binding_name)

class TRTSegmentationEngine(TRTEngine):
    def __init__(self, engine_path, cuda_stream=None):
        super().__init__(engine_path, cuda_stream)
        self.expected_height = self.input_shape[2]
        self.expected_width = self.input_shape[3]
        print(f"SegFormer INT8 TRT engine loaded. Input: [{self.expected_height}, {self.expected_width}]")

    def infer(self, input_data):
        np.copyto(self.inputs[0].host, input_data.ravel())
        trt_outputs = do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        output_name = self.engine.get_binding_name(1)
        output_shape = self.context.get_binding_shape(self.engine.get_binding_index(output_name))
        return trt_outputs[0].reshape(output_shape)

class TRTYOLOEngine(TRTEngine):
    def __init__(self, engine_path, cuda_stream=None):
        super().__init__(engine_path, cuda_stream)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        print(f"YOLO INT8 TRT engine loaded. Input: [{self.input_height}, {self.input_width}]")
        self.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 7: 'truck',
                      15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                      24: 'backpack', 25: 'umbrella', 28: 'suitcase', 36: 'skateboard'}

    def predict(self, image):
        original_shape = image.shape[:2]
        image_resized = cv2.resize(image, (self.input_width, self.input_height))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_norm = (image_rgb.astype(np.float32) / 255.0).transpose((2, 0, 1))

        np.copyto(self.inputs[0].host, image_norm.ravel())
        trt_outputs = do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)

        output_name = self.engine.get_binding_name(1)
        output_shape = self.context.get_binding_shape(self.engine.get_binding_index(output_name))
        output_data = trt_outputs[0].reshape(output_shape)

        results = self.post_process(output_data, original_shape)
        return [results]

    def post_process(self, output, original_shape):
        output = np.squeeze(output).T
        boxes, scores, class_ids = [], [], []
        scale_x, scale_y = original_shape[1] / self.input_width, original_shape[0] / self.input_height

        for detection in output:
            box, class_scores = detection[:4], detection[4:]
            class_id = np.argmax(class_scores)
            max_score = class_scores[class_id]

            if max_score > self.conf_threshold:
                x_center, y_center, width, height = box
                boxes.append([x_center * scale_x, y_center * scale_y, width * scale_x, height * scale_y])
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
                def tolist(self): return self.xywh.tolist() if len(self.xywh) > 0 else []

        return MockResults(final_boxes, final_scores, final_class_ids)

# -------------------------------------------------------------------
# Phase 3 Cache Executor (from phase3_cache_int8.py)
# -------------------------------------------------------------------
class CachedInferenceResult:
    """캐싱된 추론 결과"""
    def __init__(self):
        self.segmentation_mask = None
        self.detection_results = None
        self.segmentation_time_ms = 0.0
        self.detection_time_ms = 0.0
        self.total_time_ms = 0.0
        self.seg_from_cache = False
        self.det_from_cache = False
        self.success = False
        self.error_message = None

class CachedExecutor:
    """Phase 3: Intelligent caching executor with INT8 engines"""
    def __init__(self, model_seg, model_det, image_size=[512, 896]):
        self.model_seg = model_seg
        self.model_det = model_det
        self.image_size = image_size

        # Cache storage
        self.cached_seg_mask = None
        self.cached_det_results = None
        self.frame_counter = 0

        print(f"✓ Cached Executor initialized (INT8 Engines)")
        print(f"  - SegFormer INT8 cache interval: every {SEG_CACHE_INTERVAL} frames")
        print(f"  - YOLO INT8 cache interval: every {DET_CACHE_INTERVAL} frames")

    def process_frame_cached(self, frame):
        """캐시를 활용한 프레임 처리 (INT8)"""
        result = CachedInferenceResult()
        self.frame_counter += 1

        try:
            total_start = time.time()

            # 1. Segmentation (캐시 체크)
            if self.frame_counter % SEG_CACHE_INTERVAL == 1 or self.cached_seg_mask is None:
                seg_start = time.time()
                try:
                    image_norm = self.load_frame(frame, self.image_size)
                    output = self.model_seg.infer(image_norm.numpy().astype(np.float32))
                    id_map = np.argmax(F.softmax(torch.from_numpy(output), dim=1).cpu().detach().numpy().squeeze(), axis=0).astype(np.uint8)
                    id_map = image_morpho(id_map)
                    id_map = cv2.resize(id_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    self.cached_seg_mask = id_map
                    result.segmentation_mask = id_map
                    result.segmentation_time_ms = (time.time() - seg_start) * 1000
                    result.seg_from_cache = False

                except Exception as e:
                    print(f"⚠ Segmentation error: {e}")
                    result.success = False
                    result.error_message = f"Segmentation: {e}"
                    return result
            else:
                result.segmentation_mask = self.cached_seg_mask
                result.segmentation_time_ms = 0.0
                result.seg_from_cache = True

            # 2. Detection (캐시 체크)
            if self.frame_counter % DET_CACHE_INTERVAL == 1 or self.cached_det_results is None:
                det_start = time.time()
                try:
                    results = self.model_det.predict(frame)
                    self.cached_det_results = results
                    result.detection_results = results
                    result.detection_time_ms = (time.time() - det_start) * 1000
                    result.det_from_cache = False

                except Exception as e:
                    print(f"⚠ Detection error: {e}")
                    result.success = False
                    result.error_message = f"Detection: {e}"
                    return result
            else:
                result.detection_results = self.cached_det_results
                result.detection_time_ms = 0.0
                result.det_from_cache = True

            result.total_time_ms = (time.time() - total_start) * 1000
            result.success = True
            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

    def load_frame(self, frame, input_size=[512, 896]):
        """프레임 전처리"""
        transform_img = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        image_tr = transform_img(image=frame)['image'].unsqueeze(0)
        return image_tr

# -------------------------------------------------------------------
# Helper Functions (from videoAssessor.py)
# -------------------------------------------------------------------
from videoAssessor import (
    find_extreme_y_values, bresenham_line, find_edges, identify_ego_track,
    find_rail_sides, interpolate_end_points, find_dist_from_edges, find_zone_border,
    get_clues, border_handler, manage_detections, get_bounding_box_points,
    classify_detections, draw_results, select_video_file
)

# -------------------------------------------------------------------
# Phase 4: Tracking Integration
# -------------------------------------------------------------------
def process_frame_with_tracking(frame, cached_executor, ego_tracker, width_learner, target_distances):
    """
    Phase 4: Cache + Tracking을 활용한 프레임 처리

    핵심 차이점:
    1. Phase 3: 매 프레임 독립적으로 ego-track 선택 (jitter 발생)
    2. Phase 4: Tracker를 통해 temporal continuity 유지 (smooth tracking)
    """
    try:
        # Step 1: Phase 3 cached inference
        cache_result = cached_executor.process_frame_cached(frame)

        if not cache_result.success:
            return None, [], cache_result, None

        segmentation_mask = cache_result.segmentation_mask
        detection_results = cache_result.detection_results

        # Step 2: Extract ego-track edges from segmentation
        clues = get_clues(segmentation_mask, 6)
        edges = find_edges(segmentation_mask, clues, min_width=max(1, int(segmentation_mask.shape[1]*0.002)))

        # Phase 4 TRACKING: Use previous track center for consistent selection
        previous_track_center = None
        if ego_tracker.current_track is not None:
            # Calculate previous track center from tracked edges
            prev_left_x = np.mean(ego_tracker.current_track.left_edge[:, 0])
            prev_right_x = np.mean(ego_tracker.current_track.right_edge[:, 0])
            previous_track_center = (prev_left_x + prev_right_x) / 2.0

        ego_edges_raw = identify_ego_track(edges, segmentation_mask.shape[1], previous_track_center)

        # Step 3: Phase 4 TRACKING - 핵심 부분!
        track_state = None
        tracking_time_start = time.time()

        if ego_edges_raw and len(ego_edges_raw) == 2:
            left_edge_list, right_edge_list = ego_edges_raw

            # Convert to numpy arrays
            if len(left_edge_list) >= 3 and len(right_edge_list) >= 3:
                left_edge_np = np.array(left_edge_list, dtype=np.float32)
                right_edge_np = np.array(right_edge_list, dtype=np.float32)

                # Update width profile learner (calibration phase)
                if not width_learner.is_calibrated():
                    width_learner.add_measurement(left_edge_np, right_edge_np)

                    # Check if calibration just completed
                    if width_learner.is_calibrated():
                        width_profile = width_learner.get_profile()
                        ego_tracker.set_width_profile(width_profile)
                        print(f"✅ Width profile calibrated at frame {cached_executor.frame_counter}")

                # Update ego tracker with detection
                track_state = ego_tracker.update(
                    frame_id=cached_executor.frame_counter,
                    detection=(left_edge_np, right_edge_np)
                )
            else:
                # Detection too short - use tracker prediction only
                track_state = ego_tracker.update(
                    frame_id=cached_executor.frame_counter,
                    detection=None
                )
        else:
            # No ego-track detected - use tracker prediction
            track_state = ego_tracker.update(
                frame_id=cached_executor.frame_counter,
                detection=None
            )

        tracking_time_ms = (time.time() - tracking_time_start) * 1000

        # Step 4: Use tracked edges (if available) for border calculation
        if track_state is not None:
            # Convert tracked edges back to list format for border_handler
            tracked_left = [(int(p[0]), int(p[1])) for p in track_state.left_edge]
            tracked_right = [(int(p[0]), int(p[1])) for p in track_state.right_edge]
            ego_edges_for_border = (tracked_left, tracked_right)
        else:
            # Fallback to raw detection
            ego_edges_for_border = ego_edges_raw

        # Step 5: Border calculation (using tracked or raw edges)
        borders = border_handler(segmentation_mask, ego_edges_for_border, target_distances)

        # Step 6: Classification
        boxes_moving, boxes_stationary = manage_detections(detection_results, cached_executor.model_det)
        classification = []
        if borders and len(borders) > 0:
            classification = classify_detections(boxes_moving, boxes_stationary, borders,
                                                frame.shape, segmentation_mask.shape)

        # Add tracking time to cache result
        cache_result.tracking_time_ms = tracking_time_ms

        return borders, classification, cache_result, track_state

    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        import traceback
        traceback.print_exc()
        return None, [], CachedInferenceResult(), None

def draw_tracking_overlay(frame, track_state, width_calibrated):
    """
    Phase 4 tracking 시각화

    - Green edges: DETECTED (segmentation으로 감지됨)
    - Yellow edges: PREDICTED (Kalman filter로 예측됨)
    """
    if track_state is None:
        return frame

    overlay = frame.copy()

    # Color based on tracking status
    if track_state.is_predicted:
        color = (0, 255, 255)  # Yellow - PREDICTED
        status = "PREDICTED"
    else:
        color = (0, 255, 0)    # Green - DETECTED
        status = "DETECTED"

    # Draw left edge
    left_pts = track_state.left_edge.astype(np.int32)
    for i in range(len(left_pts) - 1):
        cv2.line(overlay, tuple(left_pts[i]), tuple(left_pts[i+1]), color, 3)

    # Draw right edge
    right_pts = track_state.right_edge.astype(np.int32)
    for i in range(len(right_pts) - 1):
        cv2.line(overlay, tuple(right_pts[i]), tuple(right_pts[i+1]), color, 3)

    # Status text at bottom
    y_pos = frame.shape[0] - 60
    track_info = f"Track ID:{track_state.track_id} | {status} | Lost:{track_state.frames_since_detection}f"
    cv2.putText(overlay, track_info, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Width profile status
    y_pos += 30
    profile_status = "CALIBRATED" if width_calibrated else "CALIBRATING"
    profile_color = (0, 255, 0) if width_calibrated else (0, 165, 255)
    cv2.putText(overlay, f"Width Profile: {profile_status}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, profile_color, 2)

    # Blend
    alpha = 0.7
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return result

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    global TRACKING_ENABLED

    # Parse arguments
    parser = argparse.ArgumentParser(description='Phase 4: Temporal Tracking')
    parser.add_argument('--enable-tracking', action='store_true',
                       help='Enable Phase 4 temporal tracking')
    parser.add_argument('--video', type=str, help='Path to video file')
    args = parser.parse_args()

    TRACKING_ENABLED = args.enable_tracking

    # Paths
    video_dir = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/crop"
    seg_engine_path = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961_896x512_int8.engine"
    det_engine_path = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/yolo/yolov8s_896x512_int8.engine"

    mode = "Phase 4 TRACKING" if TRACKING_ENABLED else "Phase 3 CACHE"
    print(f"🚀 Video Distance Assessor - {mode}")
    print("="*70)
    print(f"⚡ INT8 Quantized Engines")
    print(f"⚡ Cache: SegFormer every {SEG_CACHE_INTERVAL} frames, YOLO every {DET_CACHE_INTERVAL} frames")
    if TRACKING_ENABLED:
        print(f"⚡ Tracking: Kalman filter + width profile (target: 95%+ continuity)")
    print("="*70)

    # Load models
    print("🔥 Loading INT8 TensorRT engines...")
    try:
        model_seg = TRTSegmentationEngine(seg_engine_path)
        model_det = TRTYOLOEngine(det_engine_path)
        print("✅ Engines loaded")
    except Exception as e:
        print(f"❌ Failed to load engines: {e}")
        return

    # Initialize executor
    cached_executor = CachedExecutor(model_seg, model_det, image_size=[512, 896])

    # Phase 4: Initialize tracking components
    if TRACKING_ENABLED:
        ego_tracker = EgoTracker(max_frames_lost=30, min_confidence=0.3, image_height=1080)
        width_learner = RailWidthProfileLearner(
            num_y_levels=20,
            min_calibration_frames=150,
            max_calibration_frames=300,
            image_height=1080,
            width_tolerance=0.10  # ±10% (stricter than default ±20%)
        )
        print("✅ Tracking components initialized")
        print("  - Width tolerance: ±10% (strict mode for stability)")
    else:
        ego_tracker = None
        width_learner = None

    target_distances = [80, 500, 1000]  # mm - red, orange, yellow

    while True:
        # Select video
        if args.video:
            video_path = args.video
            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                return
        else:
            video_path = select_video_file(video_dir)
            
        if video_path is None:
            break

        print(f"📹 Processing: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📊 {fps} FPS, {total_frames} frames")
        print("Keys: 'q'=quit, 'SPACE'=pause, 'n'=next, 'r'=restart")

        # Monitoring
        fps_deque = deque(maxlen=30)
        cache_stats = {'seg_hits': 0, 'seg_miss': 0, 'det_hits': 0, 'det_miss': 0}
        frame_count = 0
        paused = False

        # Reset
        cached_executor.frame_counter = 0
        if TRACKING_ENABLED:
            ego_tracker.reset()
            width_learner.reset()

        print(f"Starting {mode} mode...")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                start_time = time.time()

                try:
                    # Process frame
                    if TRACKING_ENABLED:
                        borders, classification, cache_result, track_state = process_frame_with_tracking(
                            frame, cached_executor, ego_tracker, width_learner, target_distances
                        )
                    else:
                        # Phase 3 fallback (no tracking)
                        from videoAssessor import get_clues, find_edges, identify_ego_track, border_handler
                        cache_result = cached_executor.process_frame_cached(frame)

                        if cache_result.success:
                            clues = get_clues(cache_result.segmentation_mask, 6)
                            edges = find_edges(cache_result.segmentation_mask, clues,
                                             min_width=max(1, int(cache_result.segmentation_mask.shape[1]*0.002)))
                            ego_edges = identify_ego_track(edges, cache_result.segmentation_mask.shape[1])
                            borders = border_handler(cache_result.segmentation_mask, ego_edges, target_distances)

                            boxes_moving, boxes_stationary = manage_detections(
                                cache_result.detection_results, cached_executor.model_det)
                            classification = []
                            if borders and len(borders) > 0:
                                classification = classify_detections(
                                    boxes_moving, boxes_stationary, borders,
                                    frame.shape, cache_result.segmentation_mask.shape)
                            track_state = None
                        else:
                            borders, classification, track_state = None, [], None

                    # Update cache stats
                    if cache_result.success:
                        cache_stats['seg_hits' if cache_result.seg_from_cache else 'seg_miss'] += 1
                        cache_stats['det_hits' if cache_result.det_from_cache else 'det_miss'] += 1

                    # Visualization
                    processed = frame.copy()

                    # Draw detection boxes
                    if cache_result.success and cache_result.detection_results:
                        det_result = cache_result.detection_results[0]
                        if hasattr(det_result, 'boxes'):
                            for box_xywh, cls_id in zip(det_result.boxes.xywh.tolist(),
                                                        det_result.boxes.cls.tolist()):
                                x_c, y_c, w, h = box_xywh
                                x1, y1 = int(x_c - w/2), int(y_c - h/2)
                                x2, y2 = int(x_c + w/2), int(y_c + h/2)
                                cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                label = model_det.names.get(int(cls_id), f'cls_{int(cls_id)}')
                                cv2.putText(processed, label, (x1, y1-5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Draw borders & classification
                    if borders and len(borders) > 0:
                        processed = draw_results(processed, borders, classification, model_det.names)

                    # Phase 4: Draw tracking overlay
                    if TRACKING_ENABLED and track_state:
                        processed = draw_tracking_overlay(
                            processed, track_state, width_learner.is_calibrated())

                    # FPS
                    frame_fps = 1 / (time.time() - start_time)
                    fps_deque.append(frame_fps)
                    avg_fps = sum(fps_deque) / len(fps_deque)

                    # Info display
                    info = f"Frame: {frame_count}/{total_frames} | FPS: {avg_fps:.1f} | {mode}"
                    cv2.putText(processed, info, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Cache stats
                    if cache_result.success:
                        seg_rate = cache_stats['seg_hits'] / (cache_stats['seg_hits'] + cache_stats['seg_miss']) * 100
                        det_rate = cache_stats['det_hits'] / (cache_stats['det_hits'] + cache_stats['det_miss']) * 100

                        cache_text = f"INT8 | Seg:{seg_rate:.0f}% Det:{det_rate:.0f}%"
                        if TRACKING_ENABLED and hasattr(cache_result, 'tracking_time_ms'):
                            cache_text += f" | Track:{cache_result.tracking_time_ms:.1f}ms"
                        cv2.putText(processed, cache_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    cv2.imshow(f'RailSafeNet - {mode}', processed)

                except Exception as e:
                    print(f"Frame {frame_count} error: {e}")
                    cv2.imshow(f'RailSafeNet - {mode}', frame)

            # Key handling
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
                cached_executor.frame_counter = 0
                if TRACKING_ENABLED:
                    ego_tracker.reset()
                    width_learner.reset()
                fps_deque.clear()
                cache_stats = {'seg_hits': 0, 'seg_miss': 0, 'det_hits': 0, 'det_miss': 0}

        # Summary
        if frame_count > 0:
            print(f"\n📊 Summary:")
            print(f"   Frames: {frame_count}/{total_frames}")
            print(f"   Average FPS: {sum(fps_deque)/len(fps_deque) if fps_deque else 0:.2f}")

            if TRACKING_ENABLED:
                stats = ego_tracker.get_statistics()
                print(f"   Tracking:")
                print(f"     - Continuity: {stats.get('continuity_rate', 0)*100:.1f}%")
                print(f"     - Detected: {stats.get('detected_frames', 0)} frames")
                print(f"     - Predicted: {stats.get('predicted_frames', 0)} frames")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
