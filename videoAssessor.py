#!/usr/bin/env python3
"""
Video Distance Assessor
영상에 TheDistanceAssessor 결과를 실시간으로 오버레이하는 스크립트
"""

import cv2
import os
import time
import numpy as np
import argparse
import glob
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import deque

# TensorRT 관련 import
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TheDistanceAssessor 관련 함수들
from scripts.metrics_filtered_cls import image_morpho

# -------------------------------------------------------------------
# TensorRT Engine 클래스 (TheDistanceAssessor_3_engine.py에서 가져옴)
# -------------------------------------------------------------------
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
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
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        if self.engine.has_implicit_batch_dimension == False:
            input_shape = self.engine.get_binding_shape(0)
            self.context.set_binding_shape(0, input_shape)

        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        input_binding_name = self.engine.get_binding_name(0)
        self.input_shape = self.engine.get_binding_shape(input_binding_name)

class TRTSegmentationEngine(TRTEngine):
    def __init__(self, engine_path):
        super().__init__(engine_path)
        self.expected_height = self.input_shape[2]
        self.expected_width = self.input_shape[3]
        print(f"Segmentation TRT engine loaded. Expected input: [{self.expected_height}, {self.expected_width}]")

    def infer(self, input_data):
        np.copyto(self.inputs[0].host, input_data.ravel())
        trt_outputs = do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        output_name = self.engine.get_binding_name(1)
        output_shape = self.context.get_binding_shape(self.engine.get_binding_index(output_name))
        return trt_outputs[0].reshape(output_shape)

class TRTYOLOEngine(TRTEngine):
    def __init__(self, engine_path):
        super().__init__(engine_path)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        print(f"YOLO TRT engine loaded. Using input size: [{self.input_height}, {self.input_width}]")
        self.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 7: 'truck', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 24: 'backpack', 25: 'umbrella', 28: 'suitcase', 36: 'skateboard'}
    
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
# TheDistanceAssessor 헬퍼 함수들
# -------------------------------------------------------------------
def find_extreme_y_values(arr, values=[4, 9]):
    mask = np.isin(arr, values)
    rows_with_values = np.any(mask, axis=1)
    y_indices = np.nonzero(rows_with_values)[0]
    if y_indices.size == 0: return None, None
    return y_indices[0], y_indices[-1]

def bresenham_line(x0, y0, x1, y1):
    line = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        line.append((x0, y0))
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy
    return line

def find_edges(image, y_levels, values=[4, 9], min_width=19):
    edges_dict = {}
    for y in y_levels:
        if y >= image.shape[0]: continue
        row = image[y, :]
        mask = np.isin(row, values).astype(int)
        diff = np.diff(np.pad(mask, (1, 1), 'constant'))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1
        filtered_edges = [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= min_width and s != 0 and e != image.shape[1]-1]
        if filtered_edges:
            edges_dict[y] = filtered_edges
    return edges_dict

def identify_ego_track(edges_dict, image_width, previous_track_center=None):
    """
    Identify ego-track from detected edges.

    Args:
        edges_dict: Dictionary of detected edges per y-level
        image_width: Width of image
        previous_track_center: Optional x-coordinate of previous frame's track center
                              (for temporal consistency in tracking mode)

    Returns:
        Dictionary of ego-track edges per y-level
    """
    ego_edges_dict, last_ego_track_center = {}, None
    sorted_y_levels = sorted(edges_dict.keys(), reverse=True)

    # Determine reference center for initial selection
    if previous_track_center is not None:
        # Phase 4 TRACKING: Use previous track position for consistency
        reference_center = previous_track_center
    else:
        # Phase 3 baseline: Use image center
        reference_center = image_width / 2

    if sorted_y_levels:
        first_y = sorted_y_levels[0]
        tracks_at_first_y = edges_dict.get(first_y, [])
        if tracks_at_first_y:
            # Select track closest to reference center (previous track or image center)
            closest_track = min(tracks_at_first_y,
                              key=lambda track: abs(((track[0] + track[1]) / 2) - reference_center))
            ego_edges_dict[first_y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2

    for y in sorted_y_levels[1:]:
        if last_ego_track_center is None: break
        tracks_at_y = edges_dict.get(y, [])
        if tracks_at_y:
            closest_track = min(tracks_at_y,
                              key=lambda track: abs(((track[0] + track[1]) / 2) - last_ego_track_center))
            ego_edges_dict[y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2
    return ego_edges_dict

def find_rails(arr, y_levels, values=[4, 9], min_width=5):
    for y in y_levels:
        if y >= arr.shape[0]: continue
        row = arr[y, :]
        mask = np.isin(row, values).astype(int)
        diff = np.diff(np.pad(mask, (1, 1), 'constant'))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1
        filtered_edges = [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= min_width and s != 0 and e != arr.shape[1]-1]
    return filtered_edges

def find_rail_sides(img, edges_dict):
    left_border, right_border = [], []
    for y, xs in edges_dict.items():
        rails = find_rails(img, [y], values=[4, 9], min_width=5)
        left_border_actual, right_border_actual = [min(xs)[0], y], [max(xs)[1], y]
        for zone in rails:
            if abs(zone[1] - left_border_actual[0]) < y * 0.04: left_border_actual[0] = zone[0]
            if abs(zone[0] - right_border_actual[0]) < y * 0.04: right_border_actual[0] = zone[1]
        left_border.append(left_border_actual)
        right_border.append(right_border_actual)
    
    # 간단한 이상치 제거
    if len(left_border) > 2:
        left_border = np.array(left_border)
        left_border = left_border[left_border[:, 1] != left_border[:, 1].max()]
    if len(right_border) > 2:
        right_border = np.array(right_border)
        right_border = right_border[right_border[:, 1] != right_border[:, 1].max()]
    
    return left_border, right_border

def interpolate_end_points(end_points_dict):
    if len(end_points_dict) < 2: return []
    line_arr = []
    ys = list(end_points_dict.keys())
    xs = list(end_points_dict.values())
    for i in range(len(ys) - 1):
        y1, y2 = ys[i], ys[i + 1]
        x1, x2 = xs[i], xs[i + 1]
        line = np.array(bresenham_line(x1, y1, x2, y2))
        if np.any(line[:, 0] < 0): line = line[line[:, 0] > 0]
        line_arr.extend(list(line))
    return line_arr

def find_dist_from_edges(edges_dict, left_border, right_border, real_life_width_mm, real_life_target_mm):
    diffs_width = {k: max(e - s for s, e in v) for k, v in edges_dict.items() if v}
    scale_factors = {k: real_life_width_mm / v for k, v in diffs_width.items() if v > 0}
    target_distances_px = {k: int(real_life_target_mm / v) for k, v in scale_factors.items() if v > 0}

    # Border width constraint: Limit to 1.5x rail width to prevent invading adjacent tracks
    # This prevents danger zones from extending too far beyond the ego-track
    for k in target_distances_px.keys():
        rail_width_px = diffs_width.get(k, 100)  # Detected rail width in pixels
        max_border_extension = int(rail_width_px * 1.5)  # Max 1.5x rail width
        target_distances_px[k] = min(target_distances_px[k], max_border_extension)

    end_points_left, end_points_right = {}, {}
    for point in left_border:
        y = point[1]
        if y in target_distances_px:
            end_points_left[y] = point[0] - target_distances_px[y]
    for point in right_border:
        y = point[1]
        if y in target_distances_px:
            end_points_right[y] = point[0] + target_distances_px[y]
    return end_points_left, end_points_right

def find_zone_border(id_map, edges, irl_width_mm=1435, irl_target_mm=1000):
    left_border_pts, right_border_pts = find_rail_sides(id_map, edges)
    end_points_left, end_points_right = find_dist_from_edges(edges, left_border_pts, right_border_pts, irl_width_mm, irl_target_mm)
    border_l = interpolate_end_points(end_points_left)
    border_r = interpolate_end_points(end_points_right)
    return [border_l, border_r]

def get_clues(segmentation_mask, number_of_clues):
    """선로 감지 범위를 이미지 하단 45%로 제한"""
    height = segmentation_mask.shape[0]
    # 이미지 하단 45%까지만 탐지 (상단 55% 제외)
    start_y = int(height * 0.55)  # 55% 지점부터 시작
    limited_mask = segmentation_mask[start_y:, :]

    lowest, highest = find_extreme_y_values(limited_mask)
    if lowest is not None and highest is not None and highest > lowest:
        # 실제 이미지 좌표로 변환 (하단 45% 기준이므로 start_y 더함)
        actual_lowest = lowest + start_y
        actual_highest = highest + start_y
        
        clue_step = int((actual_highest - actual_lowest) / (number_of_clues + 1))
        if clue_step == 0: clue_step = 1
        return [actual_highest - (i * clue_step) for i in range(number_of_clues)] + [actual_lowest]
    return []

def border_handler(id_map, edges, target_distances):
    borders = []
    for target in target_distances:
        borders.append(find_zone_border(id_map, edges, irl_target_mm=target))
    return borders

def manage_detections(results, model):
    if not results or not results[0].boxes: return {}, {}
    bbox, cls = results[0].boxes.xywh.tolist(), results[0].boxes.cls.tolist()
    accepted_moving = {0, 1, 2, 3, 7, 15, 16, 17, 18, 19}
    accepted_stationary = {24, 25, 28, 36}
    boxes_moving, boxes_stationary = {}, {}
    for xywh, clss in zip(bbox, cls):
        if clss in accepted_moving: boxes_moving.setdefault(clss, []).append(xywh)
        if clss in accepted_stationary: boxes_stationary.setdefault(clss, []).append(xywh)
    return boxes_moving, boxes_stationary

def get_bounding_box_points(cx, cy, w, h):
    corners = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    points = []
    for i in range(4):
        p1, p2 = corners[i], corners[(i + 1) % 4]
        points.extend([p1, ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)])
    return points

def classify_detections(boxes_moving, boxes_stationary, borders, img_dims, output_dims):
    if not boxes_moving and not boxes_stationary: return []
    img_h, img_w, _ = img_dims
    scale_w, scale_h = output_dims[1] / img_w, output_dims[0] / img_h
    
    colors = ["red", "orange", "yellow", "green", "blue"]  # 안쪽부터 red, orange, yellow
    boxes_info = []
    
    for item, coords in boxes_moving.items():
        for coord in coords:
            x, y, w, h = coord[0] * scale_w, coord[1] * scale_h, coord[2] * scale_w, coord[3] * scale_h
            points_to_test = get_bounding_box_points(x, y, w, h)
            criticality, color = -1, colors[3]
            
            for i, border in enumerate(borders):
                border_l, border_r = np.array(border[0]), np.array(border[1])
                if border_l.size > 0 and border_r.size > 0:
                    all_points = np.vstack([border_l, border_r[::-1]])
                    if cv2.pointPolygonTest(all_points.astype(np.int32), (x, y), False) >= 0:
                        criticality = i
                        color = colors[i]
                        break
            
            boxes_info.append([item, criticality, color, [x, y], [w, h], 1])
    
    for item, coords in boxes_stationary.items():
        for coord in coords:
            x, y, w, h = coord[0] * scale_w, coord[1] * scale_h, coord[2] * scale_w, coord[3] * scale_h
            points_to_test = get_bounding_box_points(x, y, w, h)
            criticality, color = -1, colors[3]
            
            for i, border in enumerate(borders):
                border_l, border_r = np.array(border[0]), np.array(border[1])
                if border_l.size > 0 and border_r.size > 0:
                    all_points = np.vstack([border_l, border_r[::-1]])
                    if cv2.pointPolygonTest(all_points.astype(np.int32), (x, y), False) >= 0:
                        criticality = i
                        color = colors[4]  # stationary objects are always blue
                        break
            
            boxes_info.append([item, criticality, color, [x, y], [w, h], 0])
    
    return boxes_info

# -------------------------------------------------------------------
# 순차 처리로 변경 (CUDA 컨텍스트 충돌 방지) - 최적화 버전
# -------------------------------------------------------------------
def process_frame_sequential(frame, model_seg, model_det, target_distances, image_size=[512, 896]):
    """순차 처리로 프레임 처리 - CUDA 컨텍스트 안전하고 최적화됨"""
    try:
        # 1. 세그멘테이션
        segmentation_mask = segment_frame(model_seg, frame, image_size)
        if segmentation_mask is None:
            return None, []
        
        # 2. Border calculation - 최적화: 적은 clues
        clues = get_clues(segmentation_mask, 6)  # 6개로 줄임
        edges = find_edges(segmentation_mask, clues, min_width=max(1, int(segmentation_mask.shape[1]*0.002)))
        ego_edges = identify_ego_track(edges, segmentation_mask.shape[1])
        borders = border_handler(segmentation_mask, ego_edges, target_distances)
        
        # 3. 객체 탐지
        results = model_det.predict(frame)
        boxes_moving, boxes_stationary = manage_detections(results, model_det)
        
        # 4. Classification - 간단하게
        classification = []
        if borders and len(borders) > 0:
            classification = classify_detections(boxes_moving, boxes_stationary, borders, frame.shape, segmentation_mask.shape)
        
        return borders, classification
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return None, []
def load_frame(frame, input_size=[512, 896]):
    transform_img = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image_tr = transform_img(image=frame)['image'].unsqueeze(0)
    return image_tr

def segment_frame(model_seg, frame, image_size):
    image_norm = load_frame(frame, image_size)
    if image_norm is None: return None

    output = model_seg.infer(image_norm.numpy().astype(np.float32))
    id_map = np.argmax(F.softmax(torch.from_numpy(output), dim=1).cpu().detach().numpy().squeeze(), axis=0).astype(np.uint8)
    id_map = image_morpho(id_map)
    id_map = cv2.resize(id_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return id_map



def draw_results(frame, borders, classification, names):
    """프레임에 결과를 그려서 반환 - 최적화된 버전"""
    # 위험 구역 색상 (안쪽부터 red, orange, yellow)
    colors_bgr = [(0, 0, 255), (0, 165, 255), (0, 255, 255)]  # Red, Orange, Yellow
    alpha = 0.4
    
    # 위험 구역 오버레이 - 간단하고 빠른 방식
    if borders and len(borders) > 0:
        overlay = frame.copy()
        
        # 역순으로 그려서 안쪽이 위에 오도록 (Yellow -> Orange -> Red)
        for i in reversed(range(len(borders))):
            border = borders[i]
            if border and len(border) >= 2:
                border_l, border_r = np.array(border[0]), np.array(border[1])
                if border_l.size > 0 and border_r.size > 0:
                    all_points = np.vstack([border_l, border_r[::-1]]).astype(np.int32)
                    cv2.fillPoly(overlay, [all_points], colors_bgr[i])
        
        # 한 번에 오버레이 적용
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 경계선은 정순으로 그리기 (가장 안쪽 경계선이 제일 위에)
        for i in range(len(borders)):
            border = borders[i]
            if border and len(border) >= 2:
                border_l, border_r = np.array(border[0]), np.array(border[1])
                if border_l.size > 0:
                    cv2.polylines(frame, [border_l.astype(np.int32)], isClosed=False, 
                                color=colors_bgr[i], thickness=2)
                if border_r.size > 0:
                    cv2.polylines(frame, [border_r.astype(np.int32)], isClosed=False, 
                                color=colors_bgr[i], thickness=2)
    
    # 객체 탐지 결과 그리기
    if classification and len(classification) > 0:
        color_map = {
            "red": (0, 0, 255),
            "orange": (0, 165, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0)
        }
        
        for box in classification:
            name = names.get(box[0], f'class_{box[0]}')
            color = color_map.get(box[2], (0, 255, 0))
            cx, cy, w, h = box[3][0], box[3][1], box[4][0], box[4][1]
            
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f'{name}'
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def natural_sort_key(path):
    """자연스러운 숫자 정렬을 위한 키 함수"""
    import re
    basename = os.path.basename(path)
    # 숫자 부분 추출 (tram123.mp4 -> 123)
    match = re.search(r'tram(\d+)', basename)
    if match:
        return int(match.group(1))
    return 0

def select_video_file(video_dir):
    """비디오 파일 선택 함수"""
    video_files = sorted(glob.glob(os.path.join(video_dir, "tram*.mp4")), key=natural_sort_key)
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return None
    
    print("Available video files:")
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        print(f"{i+1}: {video_name}")
    
    while True:
        try:
            choice = input(f"Select video (1-{len(video_files)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(video_files):
                return video_files[idx]
            else:
                print(f"Please enter a number between 1 and {len(video_files)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

# -------------------------------------------------------------------
# 메인 함수
# -------------------------------------------------------------------
def main():
    # 경로 설정 - 환경변수 또는 상대경로 사용
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.environ.get("VIDEO_DIR", os.path.join(script_dir, "assets", "crop"))
    seg_engine_path = os.environ.get("SEGFORMER_ENGINE",
        os.path.join(script_dir, "assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine"))
    det_engine_path = os.environ.get("YOLO_ENGINE",
        os.path.join(script_dir, "assets/models_pretrained/yolo/yolov8s_896x512.engine"))
    
    print("🚀 Video Distance Assessor")
    print("="*50)
    
    # 모델 로드
    print("🔥 Loading TensorRT engines...")
    try:
        model_seg = TRTSegmentationEngine(seg_engine_path)
        print("✅ SegFormer TensorRT engine loaded")
    except Exception as e:
        print(f"❌ Failed to load SegFormer engine: {e}")
        return
    
    try:
        model_det = TRTYOLOEngine(det_engine_path)
        print("✅ YOLO TensorRT engine loaded")
    except Exception as e:
        print(f"❌ Failed to load YOLO engine: {e}")
        return
    
    # 파라미터 설정
    target_distances = [80, 400, 1000]  # mm - 안쪽부터 red, orange, yellow
    image_size = [512, 896]
    
    while True:
        # 비디오 파일 선택
        video_path = select_video_file(video_dir)
        if video_path is None:
            break
        
        print(f"📹 Processing video: {os.path.basename(video_path)}")
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            continue
        
        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📊 Video info: {fps} FPS, {total_frames} frames, {duration:.1f} seconds")
        print("Press 'q' to quit, 'SPACE' to pause/resume, 'n' for next video")
        
        # 성능 모니터링 설정
        fps_deque = deque(maxlen=30)
        frame_count = 0
        paused = False
        
        # Warm-up
        print("Warming up engines...")
        ret, warm_frame = cap.read()
        if ret:
            try:
                _ = process_frame_sequential(warm_frame, model_seg, model_det, target_distances, image_size)
                print("Warm-up complete")
            except Exception as e:
                print(f"Warm-up failed: {e}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("Starting video processing...")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                frame_count += 1
                start_time = time.time()
                
                try:
                    # 순차 처리로 CUDA 컨텍스트 충돌 방지
                    borders, classification = process_frame_sequential(frame, model_seg, model_det, target_distances, image_size)
                    
                    # 디버그: 결과 확인
                    has_borders = borders is not None and len(borders) > 0
                    has_classification = classification is not None and len(classification) > 0
                    
                    # 결과 그리기 - borders가 None이거나 비어있어도 그리기 시도
                    processed_frame = frame.copy()
                    if has_borders or has_classification:
                        processed_frame = draw_results(processed_frame, borders if has_borders else [], 
                                                      classification if has_classification else [], model_det.names)
                    
                    # FPS 계산
                    end_time = time.time()
                    frame_fps = 1 / (end_time - start_time)
                    fps_deque.append(frame_fps)
                    avg_fps = sum(fps_deque) / len(fps_deque)
                    
                    # 정보 표시
                    info_text = f"Frame: {frame_count}/{total_frames} | FPS: {avg_fps:.1f} | Progress: {frame_count/total_frames*100:.1f}%"
                    cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 거리 정보 표시 (안쪽부터 red, orange, yellow)
                    distance_text = f"Distances: {target_distances[0]}mm(RED) | {target_distances[1]}mm(ORG) | {target_distances[2]}mm(YEL)"
                    cv2.putText(processed_frame, distance_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # 최적화 정보 및 디버그 표시
                    opt_text = f"Sequential Processing | Lower 2/5 | Borders: {len(borders) if borders else 0} | Objects: {len(classification) if classification else 0}"
                    cv2.putText(processed_frame, opt_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 일시정지 상태 표시
                    if paused:
                        cv2.putText(processed_frame, "PAUSED - Press SPACE to resume", 
                                   (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 화면 출력
                    cv2.imshow('RailSafeNet Video Distance Assessor', processed_frame)
                    
                except Exception as e:
                    print(f"Frame processing failed: {e}")
                    cv2.imshow('RailSafeNet Video Distance Assessor', frame)
            else:
                # 일시정지 중
                cv2.waitKey(30)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):  # 스페이스바로 일시정지/재생
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord('n'):  # 다음 비디오
                print("⏭️ Next video...")
                break
            elif key == ord('r'):  # 재시작
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                fps_deque.clear()
                print("🔄 Video restarted")
        
        # 성능 요약
        if frame_count > 0:
            print(f"\n📊 Processing Summary:")
            print(f"   Processed frames: {frame_count}/{total_frames}")
            print(f"   Average FPS: {sum(fps_deque)/len(fps_deque) if fps_deque else 0:.2f}")
        
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