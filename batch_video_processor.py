#!/usr/bin/env python3
"""
Batch Video Processor for RailSafeNet
135개의 tram 영상을 모두 처리하여 결과 저장
"""

import cv2
import os
import time
import numpy as np
import glob
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import deque
from tqdm import tqdm

# TensorRT 관련 import
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TheDistanceAssessor 관련 함수들
from scripts.metrics_filtered_cls import image_morpho

# video_distance_assessor.py에서 가져온 모든 클래스와 함수들
# (TRTEngine, TRTSegmentationEngine, TRTYOLOEngine 등)

# -------------------------------------------------------------------
# TensorRT Engine 클래스
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

# [모든 헬퍼 함수들을 여기에 포함 - find_extreme_y_values, bresenham_line 등]
# video_distance_assessor.py의 모든 함수를 그대로 복사

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

def get_clues(segmentation_mask, number_of_clues):
    """선로 감지 범위를 이미지 하단 2/5로 제한"""
    height = segmentation_mask.shape[0]
    limited_mask = segmentation_mask[height*3//5:, :]
    
    lowest, highest = find_extreme_y_values(limited_mask)
    if lowest is not None and highest is not None and highest > lowest:
        actual_lowest = lowest + height*3//5
        actual_highest = highest + height*3//5
        
        clue_step = int((actual_highest - actual_lowest) / (number_of_clues + 1))
        if clue_step == 0: clue_step = 1
        return [actual_highest - (i * clue_step) for i in range(number_of_clues)] + [actual_lowest]
    return []

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

def identify_ego_track(edges_dict, image_width):
    ego_edges_dict, last_ego_track_center = {}, None
    sorted_y_levels = sorted(edges_dict.keys(), reverse=True)
    image_center_x = image_width / 2
    
    if sorted_y_levels:
        first_y = sorted_y_levels[0]
        tracks_at_first_y = edges_dict.get(first_y, [])
        if tracks_at_first_y:
            closest_track = min(tracks_at_first_y, key=lambda track: abs(((track[0] + track[1]) / 2) - image_center_x))
            ego_edges_dict[first_y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2
    
    for y in sorted_y_levels[1:]:
        if last_ego_track_center is None: break
        tracks_at_y = edges_dict.get(y, [])
        if tracks_at_y:
            closest_track = min(tracks_at_y, key=lambda track: abs(((track[0] + track[1]) / 2) - last_ego_track_center))
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
    """바운딩 박스의 8개 포인트 반환 (더 정확한 판정을 위해)"""
    corners = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    points = []
    for i in range(4):
        p1, p2 = corners[i], corners[(i + 1) % 4]
        points.extend([p1, ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)])
    # 중앙점과 하단 중앙점 추가 (더 정확한 판정)
    points.append((cx, cy))  # 중심점
    points.append((cx, cy + h/2))  # 하단 중심점
    return points

def classify_detections(boxes_moving, boxes_stationary, borders, img_dims, output_dims):
    if not boxes_moving and not boxes_stationary: return []
    img_h, img_w, _ = img_dims
    scale_w, scale_h = output_dims[1] / img_w, output_dims[0] / img_h
    
    colors = ["red", "orange", "yellow", "green", "blue"]
    boxes_info = []
    
    for item, coords in boxes_moving.items():
        for coord in coords:
            x, y, w, h = coord[0] * scale_w, coord[1] * scale_h, coord[2] * scale_w, coord[3] * scale_h
            points_to_test = get_bounding_box_points(x, y, w, h)
            criticality, color = -1, colors[3]
            
            # 여러 포인트 중 하나라도 위험 영역 안에 있으면 해당 색상으로 변경
            for i, border in enumerate(borders):
                border_l, border_r = np.array(border[0]), np.array(border[1])
                if border_l.size > 0 and border_r.size > 0:
                    all_points = np.vstack([border_l, border_r[::-1]])
                    # 여러 포인트 중 2개 이상이 안에 있으면 해당 영역으로 판정
                    inside_count = sum(1 for p in points_to_test if cv2.pointPolygonTest(all_points.astype(np.int32), p, False) >= 0)
                    if inside_count >= 2:  # 2개 이상의 포인트가 안에 있어야 판정
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
                    inside_count = sum(1 for p in points_to_test if cv2.pointPolygonTest(all_points.astype(np.int32), p, False) >= 0)
                    if inside_count >= 2:
                        criticality = i
                        color = colors[4]  # stationary는 blue
                        break
            
            boxes_info.append([item, criticality, color, [x, y], [w, h], 0])
    
    return boxes_info

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

def process_frame_sequential(frame, model_seg, model_det, target_distances, image_size=[512, 896]):
    try:
        segmentation_mask = segment_frame(model_seg, frame, image_size)
        if segmentation_mask is None:
            return None, []
        
        clues = get_clues(segmentation_mask, 6)
        edges = find_edges(segmentation_mask, clues, min_width=max(1, int(segmentation_mask.shape[1]*0.002)))
        ego_edges = identify_ego_track(edges, segmentation_mask.shape[1])
        borders = border_handler(segmentation_mask, ego_edges, target_distances)
        
        results = model_det.predict(frame)
        boxes_moving, boxes_stationary = manage_detections(results, model_det)
        
        classification = []
        if borders and len(borders) > 0:
            classification = classify_detections(boxes_moving, boxes_stationary, borders, frame.shape, segmentation_mask.shape)
        
        return borders, classification
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return None, []

def draw_results(frame, borders, classification, names):
    colors_bgr = [(0, 0, 255), (0, 165, 255), (0, 255, 255)]
    alpha = 0.4
    
    if borders and len(borders) > 0:
        overlay = frame.copy()
        
        for i in reversed(range(len(borders))):
            border = borders[i]
            if border and len(border) >= 2:
                border_l, border_r = np.array(border[0]), np.array(border[1])
                if border_l.size > 0 and border_r.size > 0:
                    all_points = np.vstack([border_l, border_r[::-1]]).astype(np.int32)
                    cv2.fillPoly(overlay, [all_points], colors_bgr[i])
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
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

# -------------------------------------------------------------------
# 배치 처리 메인 함수
# -------------------------------------------------------------------
def process_video_file(video_path, output_path, model_seg, model_det, target_distances, image_size):
    """하나의 비디오 파일을 처리하여 저장"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open: {video_path}")
        return False
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # VideoWriter 설정 - MJPEG 코덱 사용 (가장 호환성 좋음, 인코더 설치 불필요)
    # .avi 확장자로 변경
    output_path = output_path.replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # VideoWriter가 제대로 열렸는지 확인
    if not out.isOpened():
        print(f"Failed to create VideoWriter, aborting this file")
        cap.release()
        return False
    
    frame_count = 0
    
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            try:
                # 프레임 처리
                borders, classification = process_frame_sequential(frame, model_seg, model_det, target_distances, image_size)
                
                # 결과 그리기
                if borders is not None:
                    processed_frame = draw_results(frame.copy(), borders, classification, model_det.names)
                else:
                    processed_frame = frame
                
                # 정보 표시
                info_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                distance_text = f"Distances: {target_distances[0]}mm(RED) | {target_distances[1]}mm(ORG) | {target_distances[2]}mm(YEL)"
                cv2.putText(processed_frame, distance_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 비디오에 쓰기
                out.write(processed_frame)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                out.write(frame)
            
            pbar.update(1)
    
    cap.release()
    out.release()
    
    return True

def main():
    # 경로 설정
    video_dir = "/home/mmc-server4/RailSafeNet/assets/crop"
    output_dir = "/home/mmc-server4/RailSafeNet/assets/crop/results"
    seg_engine_path = "/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine"
    det_engine_path = "/home/mmc-server4/RailSafeNet/assets/models_pretrained/yolo/yolov8s_896x512.engine"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("RailSafeNet Batch Video Processor")
    print("=" * 60)
    
    # 모델 로드
    print("\nLoading TensorRT engines...")
    try:
        model_seg = TRTSegmentationEngine(seg_engine_path)
        print("✓ SegFormer engine loaded")
    except Exception as e:
        print(f"✗ Failed to load SegFormer engine: {e}")
        return
    
    try:
        model_det = TRTYOLOEngine(det_engine_path)
        print("✓ YOLO engine loaded")
    except Exception as e:
        print(f"✗ Failed to load YOLO engine: {e}")
        return
    
    # 파라미터 설정
    target_distances = [80, 400, 1000]  # mm
    image_size = [512, 896]
    
    # 비디오 파일 목록 - 숫자 순서대로 정렬
    video_files = glob.glob(os.path.join(video_dir, "tram*.mp4"))
    
    # 숫자 기준으로 정렬 (tram0.mp4 -> 0, tram1.mp4 -> 1, ...)
    def extract_number(filename):
        import re
        match = re.search(r'tram(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    video_files = sorted(video_files, key=extract_number)
    
    if not video_files:
        print(f"\nNo video files found in {video_dir}")
        return
    
    print(f"\nFound {len(video_files)} video files")
    print(f"Output directory: {output_dir}")
    print(f"Target distances: {target_distances} mm")
    print("\nStarting batch processing...")
    print("=" * 60)
    
    # 통계
    success_count = 0
    fail_count = 0
    total_time = 0
    
    # 각 비디오 처리
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        output_name = video_name.replace('.mp4', '.avi')  # .avi로 저장
        output_path = os.path.join(output_dir, f"processed_{output_name}")
        
        # 이미 처리된 파일은 스킵
        if os.path.exists(output_path):
            print(f"[{i}/{len(video_files)}] Skipping (already exists): {video_name}")
            success_count += 1
            continue
        
        print(f"\n[{i}/{len(video_files)}] Processing: {video_name}")
        
        start_time = time.time()
        
        try:
            success = process_video_file(video_path, output_path, model_seg, model_det, target_distances, image_size)
            
            if success:
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                success_count += 1
                print(f"✓ Completed in {elapsed_time:.1f}s")
            else:
                fail_count += 1
                print(f"✗ Failed to process")
        
        except Exception as e:
            fail_count += 1
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 통계
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {fail_count}")
    if success_count > 0:
        print(f"Average processing time: {total_time/success_count:.1f}s per video")
        print(f"Total processing time: {total_time/60:.1f} minutes")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()