#!/usr/bin/env python3
"""
Video Distance Assessor - Phase 3 Cache INT8: Ultimate FPS with INT8 Quantization
SegFormer 캐싱 + YOLO 캐싱 + INT8 엔진으로 극대화된 FPS

Optimization Stack:
- Method 1+3: Intelligent caching (SegFormer 3f, YOLO 2f)
- INT8 Quantization: 30-50% additional speedup vs FP16
- Expected: 40-50+ FPS (vs 12-13 FPS baseline = 3-4x improvement!)

INT8 Engine Paths:
- SegFormer: segformer_b3_transfer_best_0.7961_896x512_int8.engine
- YOLO: yolov8s_896x512_int8.engine

Usage:
    python3 videoAssessor_phase3_cache_int8.py

    조절 가능한 파라미터:
    - SEG_CACHE_INTERVAL: SegFormer 실행 주기 (기본 3)
    - DET_CACHE_INTERVAL: YOLO 실행 주기 (기본 2)
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

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Phase 3 imports
from src.utils.cuda_utils import measure_parallel_overlap

# 기존 helper functions
from scripts.metrics_filtered_cls import image_morpho

# ============================================================================
# CACHE CONFIGURATION - 여기서 캐싱 주기 조절 가능
# ============================================================================
SEG_CACHE_INTERVAL = 3  # SegFormer를 3 프레임마다 1번 실행 (1, 2, 3이면 높을수록 빠름)
DET_CACHE_INTERVAL = 2  # YOLO를 2 프레임마다 1번 실행

# -------------------------------------------------------------------
# TensorRT Engine Classes
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
# Phase 3 Cache: Intelligent Caching Executor
# -------------------------------------------------------------------
class CachedInferenceResult:
    """캐싱된 추론 결과를 저장하는 클래스"""
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
    """
    Intelligent caching executor with INT8 engines
    - SegFormer INT8: SEG_CACHE_INTERVAL 프레임마다 실행
    - YOLO INT8: DET_CACHE_INTERVAL 프레임마다 실행
    """
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
        """
        캐시를 활용한 프레임 처리 (INT8)
        """
        result = CachedInferenceResult()
        self.frame_counter += 1

        try:
            total_start = time.time()

            # 1. Segmentation (캐시 체크)
            if self.frame_counter % SEG_CACHE_INTERVAL == 1 or self.cached_seg_mask is None:
                # 실제 추론 실행
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
                # 캐시에서 가져오기
                result.segmentation_mask = self.cached_seg_mask
                result.segmentation_time_ms = 0.0
                result.seg_from_cache = True

            # 2. Detection (캐시 체크)
            if self.frame_counter % DET_CACHE_INTERVAL == 1 or self.cached_det_results is None:
                # 실제 추론 실행
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
                # 캐시에서 가져오기
                result.detection_results = self.cached_det_results
                result.detection_time_ms = 0.0
                result.det_from_cache = True

            # Total time
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
# Main Processing Function
# -------------------------------------------------------------------
def process_frame_cached(frame, cached_executor, target_distances):
    """
    Phase 3 Cache INT8: 캐싱을 활용한 프레임 처리
    """
    try:
        # 캐시 활용 추론
        result = cached_executor.process_frame_cached(frame)

        if not result.success:
            print(f"⚠️ Cached inference failed: {result.error_message}")
            return None, [], result

        segmentation_mask = result.segmentation_mask
        detection_results = result.detection_results

        # Border calculation
        clues = get_clues(segmentation_mask, 6)
        edges = find_edges(segmentation_mask, clues, min_width=max(1, int(segmentation_mask.shape[1]*0.002)))
        ego_edges = identify_ego_track(edges, segmentation_mask.shape[1])
        borders = border_handler(segmentation_mask, ego_edges, target_distances)

        # Classification
        boxes_moving, boxes_stationary = manage_detections(detection_results, cached_executor.model_det)
        classification = []
        if borders and len(borders) > 0:
            classification = classify_detections(boxes_moving, boxes_stationary, borders,
                                                frame.shape, segmentation_mask.shape)

        return borders, classification, result

    except Exception as e:
        print(f"Frame processing error: {e}")
        return None, [], CachedInferenceResult()

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    # Paths - INT8 ENGINES
    video_dir = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/crop"
    seg_engine_path = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961_896x512_int8.engine"
    det_engine_path = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/yolo/yolov8s_896x512_int8.engine"

    print("🚀 Video Distance Assessor - Phase 3 CACHE + INT8 (Ultimate FPS)")
    print("="*70)
    print(f"⚡ INT8 Quantized Engines")
    print(f"⚡ SegFormer INT8 cache: 1 execution per {SEG_CACHE_INTERVAL} frames")
    print(f"⚡ YOLO INT8 cache: 1 execution per {DET_CACHE_INTERVAL} frames")
    print(f"⚡ Expected: 40-50+ FPS (3-4x improvement vs baseline)")
    print("="*70)

    # Load INT8 models
    print("🔥 Loading INT8 TensorRT engines...")
    try:
        model_seg = TRTSegmentationEngine(seg_engine_path)
        print("✅ SegFormer INT8 TensorRT engine loaded")
    except Exception as e:
        print(f"❌ Failed to load SegFormer INT8 engine: {e}")
        print(f"   Expected path: {seg_engine_path}")
        return

    try:
        model_det = TRTYOLOEngine(det_engine_path)
        print("✅ YOLO INT8 TensorRT engine loaded")
    except Exception as e:
        print(f"❌ Failed to load YOLO INT8 engine: {e}")
        print(f"   Expected path: {det_engine_path}")
        return

    # Initialize cached executor
    cached_executor = CachedExecutor(model_seg, model_det, image_size=[512, 896])

    # Parameters
    target_distances = [80, 400, 1000]  # mm - red, orange, yellow

    while True:
        # Select video
        video_path = select_video_file(video_dir)
        if video_path is None:
            break

        print(f"📹 Processing video: {os.path.basename(video_path)}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            continue

        # Video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"📊 Video: {fps} FPS, {total_frames} frames, {duration:.1f} sec")
        print("Keys: 'q'=quit, 'SPACE'=pause/resume, 'n'=next video, 'r'=restart")

        # Monitoring
        fps_deque = deque(maxlen=30)
        seg_times = deque(maxlen=30)
        det_times = deque(maxlen=30)
        cache_stats = {'seg_hits': 0, 'seg_miss': 0, 'det_hits': 0, 'det_miss': 0}
        frame_count = 0
        paused = False

        # Reset frame counter
        cached_executor.frame_counter = 0

        # Warm-up
        print("Warming up INT8 engines...")
        ret, warm_frame = cap.read()
        if ret:
            try:
                _ = process_frame_cached(warm_frame, cached_executor, target_distances)
                print("Warm-up complete")
            except Exception as e:
                print(f"Warm-up failed: {e}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cached_executor.frame_counter = 0

        print("Starting video processing with INT8 engines...")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break

                frame_count += 1
                start_time = time.time()

                try:
                    # Phase 3 cached processing (INT8)
                    borders, classification, cache_result = process_frame_cached(
                        frame, cached_executor, target_distances
                    )

                    # Update cache stats
                    if cache_result.success:
                        if cache_result.seg_from_cache:
                            cache_stats['seg_hits'] += 1
                        else:
                            cache_stats['seg_miss'] += 1

                        if cache_result.det_from_cache:
                            cache_stats['det_hits'] += 1
                        else:
                            cache_stats['det_miss'] += 1

                    # OPTIMIZED: Skip segmentation mask overlay for maximum FPS
                    processed_frame = frame.copy()

                    # Draw detection boxes (lightweight visualization)
                    if cache_result.success and cache_result.detection_results:
                        det_result = cache_result.detection_results[0]
                        if hasattr(det_result, 'boxes') and hasattr(det_result.boxes, 'xywh'):
                            boxes_xywh = det_result.boxes.xywh.tolist()
                            boxes_cls = det_result.boxes.cls.tolist()

                            for box_xywh, cls_id in zip(boxes_xywh, boxes_cls):
                                x_center, y_center, width, height = box_xywh
                                x1 = int(x_center - width / 2)
                                y1 = int(y_center - height / 2)
                                x2 = int(x_center + width / 2)
                                y2 = int(y_center + height / 2)

                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                cls_name = model_det.names.get(int(cls_id), f'class_{int(cls_id)}')
                                label = f'{cls_name}'
                                cv2.putText(processed_frame, label, (x1, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Draw borders and classification (hazard zones)
                    if borders and len(borders) > 0:
                        processed_frame = draw_results(processed_frame, borders, classification, model_det.names)

                    # Calculate FPS
                    end_time = time.time()
                    frame_fps = 1 / (end_time - start_time)
                    fps_deque.append(frame_fps)
                    avg_fps = sum(fps_deque) / len(fps_deque)

                    # Store times
                    if cache_result.success:
                        seg_times.append(cache_result.segmentation_time_ms)
                        det_times.append(cache_result.detection_time_ms)

                    # Info display
                    info_text = f"Frame: {frame_count}/{total_frames} | FPS: {avg_fps:.1f} | Progress: {frame_count/total_frames*100:.1f}%"
                    cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Cache statistics
                    if cache_result.success:
                        seg_cache_rate = cache_stats['seg_hits'] / (cache_stats['seg_hits'] + cache_stats['seg_miss']) * 100 if (cache_stats['seg_hits'] + cache_stats['seg_miss']) > 0 else 0
                        det_cache_rate = cache_stats['det_hits'] / (cache_stats['det_hits'] + cache_stats['det_miss']) * 100 if (cache_stats['det_hits'] + cache_stats['det_miss']) > 0 else 0

                        cache_text = f"INT8 CACHED | Seg:{seg_cache_rate:.0f}% Det:{det_cache_rate:.0f}% | Seg:{cache_result.segmentation_time_ms:.1f}ms Det:{cache_result.detection_time_ms:.1f}ms"
                        cv2.putText(processed_frame, cache_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        status_text = f"[Seg:{'CACHE' if cache_result.seg_from_cache else 'INFER'}] [Det:{'CACHE' if cache_result.det_from_cache else 'INFER'}]"
                        cv2.putText(processed_frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Performance indicator
                    perf_text = f"⚡ INT8 + CACHE MODE - Seg:{SEG_CACHE_INTERVAL}f Det:{DET_CACHE_INTERVAL}f"
                    cv2.putText(processed_frame, perf_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Distance info
                    distance_text = f"Distances: {target_distances[0]}mm(RED) | {target_distances[1]}mm(ORG) | {target_distances[2]}mm(YEL)"
                    cv2.putText(processed_frame, distance_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # Pause indicator
                    if paused:
                        cv2.putText(processed_frame, "PAUSED - Press SPACE to resume",
                                   (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Display
                    cv2.imshow('RailSafeNet Phase 3 - INT8 CACHE', processed_frame)

                except Exception as e:
                    print(f"Frame {frame_count} processing failed: {e}")
                    cv2.imshow('RailSafeNet Phase 3 - INT8 CACHE', frame)

            # Key handling
            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key == ord('q'):
                print("🛑 Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord('n'):
                print("⏭️ Next video...")
                break
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                cached_executor.frame_counter = 0
                fps_deque.clear()
                seg_times.clear()
                det_times.clear()
                cache_stats = {'seg_hits': 0, 'seg_miss': 0, 'det_hits': 0, 'det_miss': 0}
                print("🔄 Video restarted")

        # Summary
        if frame_count > 0:
            print(f"\n📊 INT8 Processing Summary:")
            print(f"   Processed: {frame_count}/{total_frames} frames")
            print(f"   Average FPS: {sum(fps_deque)/len(fps_deque) if fps_deque else 0:.2f}")
            print(f"   Cache Statistics:")
            print(f"     - SegFormer INT8 cache hit: {cache_stats['seg_hits']}/{cache_stats['seg_hits']+cache_stats['seg_miss']} ({cache_stats['seg_hits']/(cache_stats['seg_hits']+cache_stats['seg_miss'])*100:.1f}%)")
            print(f"     - YOLO INT8 cache hit: {cache_stats['det_hits']}/{cache_stats['det_hits']+cache_stats['det_miss']} ({cache_stats['det_hits']/(cache_stats['det_hits']+cache_stats['det_miss'])*100:.1f}%)")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("🎬 Application closed")
