"""TensorRT 기반 철도 위험 구역 추론 파이프라인.

이 모듈은 입력 영상을 받아 다음 단계를 순차적으로 수행한다.
1. SegFormer TensorRT 엔진으로 세그멘테이션 마스크를 생성한다.
2. 클래스 ID 4, 9를 선로 후보로 간주해 주행 선로 경계를 추정한다.
3. 추정된 선로 폭을 실제 궤간(1435 mm)과 연결해 거리 구역 경계를 만든다.
4. YOLO TensorRT 엔진으로 객체를 검출하고 위험 구역과의 겹침으로 위험도를 분류한다.
5. 위험 구역 오버레이와 검출 결과를 시각화한다.

하드코딩된 임계값과 외부 절대경로는 기존 동작 보존을 위해 유지한다.
정확한 실험 근거가 확인되지 않은 값은 요약 문서에서 별도로 정리한다.
"""

import cv2
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.path as mplPath
import matplotlib.patches as patches
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from src.common.metrics_filtered_cls import image_morpho


# --- 🚀 TensorRT 추론을 위한 라이브러리 임포트 ---
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # CUDA context 초기화

# --- 📁 모델 경로 수정: .onnx -> .engine ---
PATH_jpgs = '/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val'
PATH_model_seg = '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine'
PATH_model_det = '/home/mmc-server4/RailSafeNet/assets/models_pretrained/yolo/yolov8s_896x512.engine'
PATH_base = 'RailNet_DT/assets/pilsen_railway_dataset/'
eda_path = '/home/mmc-server4/RailSafeNet/assets/pilsen_railway_dataset/eda_table.table.json'
data_json = json.load(open(eda_path, 'r'))

# --- 🚀 TensorRT API 현대화: DeprecationWarning 해결 ---
class HostDeviceMem(object):
    """TensorRT 입출력 버퍼의 host/device 메모리를 함께 보관한다."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self): return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self): return self.__str__()

def allocate_buffers(engine):
    """엔진의 모든 binding에 대해 pagelocked host 메모리와 device 메모리를 할당한다.

    TensorRT 엔진에 동적 shape가 포함되어 있어도 현재 binding shape 기준으로 버퍼를 잡는다.
    shape가 기대와 다르면 이후 `set_binding_shape` 호출이 선행되어야 한다.
    """

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
    """비동기 CUDA 복사와 TensorRT 실행을 한 번에 수행한다.

    입력 버퍼를 device로 복사하고 `execute_async_v2`를 호출한 뒤,
    출력 버퍼를 다시 host로 복사한다. stream 동기화가 끝난 시점의 host 데이터를 반환한다.
    """

    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

class TRTEngine:
    """TensorRT 엔진 로딩과 공통 실행 컨텍스트를 담당하는 기반 클래스.

    입력 binding 0을 기준으로 shape를 읽으며, explicit batch 엔진이면 현재 shape를
    그대로 execution context에 반영한다. 엔진 파일이 대상 GPU에서 빌드되지 않았거나
    TensorRT 버전이 맞지 않으면 deserialize 단계에서 실패할 수 있다.
    """

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        # Set context for dynamic shape models
        if self.engine.has_implicit_batch_dimension == False:
            input_shape = self.engine.get_binding_shape(0)
            self.context.set_binding_shape(0, input_shape)

        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        
        input_binding_name = self.engine.get_binding_name(0)
        self.input_shape = self.engine.get_binding_shape(input_binding_name)

class TRTSegmentationEngine(TRTEngine):
    """세그멘테이션용 TensorRT 래퍼.

    입력은 NCHW float32 텐서이고 출력은 클래스 logits 텐서라고 가정한다.
    출력 shape는 엔진 metadata를 기준으로 읽어오기 때문에 export 시점과 runtime 시점의
    binding 순서가 달라지면 결과 해석이 깨질 수 있다.
    """

    def __init__(self, engine_path):
        super().__init__(engine_path)
        self.expected_height = self.input_shape[2]
        self.expected_width = self.input_shape[3]
        print(f"Segmentation TRT engine loaded. Expected input: [{self.expected_height}, {self.expected_width}]")

    def infer(self, input_data):
        """전처리된 입력 텐서 1개를 받아 세그멘테이션 logits를 반환한다."""

        np.copyto(self.inputs[0].host, input_data.ravel())
        trt_outputs = do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        output_name = self.engine.get_binding_name(1)
        output_shape = self.context.get_binding_shape(self.engine.get_binding_index(output_name))
        return trt_outputs[0].reshape(output_shape)

class TRTYOLOEngine(TRTEngine):
    """YOLO 검출용 TensorRT 래퍼.

    YOLO 출력은 `(cx, cy, w, h, class_scores...)` 형식으로 가정한다.
    `predict` 단계에서 입력 해상도와 원본 해상도 사이 스케일 차이를 복원하며,
    `post_process` 단계에서 NMS를 적용해 ultralytics 유사 결과 객체로 감싼다.
    """

    def __init__(self, engine_path):
        super().__init__(engine_path)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        print(f"YOLO TRT engine loaded. Using input size: [{self.input_height}, {self.input_width}]")
        self.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle'}
    
    def predict(self, image):
        """원본 이미지를 엔진 입력 크기로 맞춘 뒤 검출 결과를 반환한다."""

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
        """YOLO 원시 출력을 confidence 필터링과 NMS를 거쳐 해석한다.

        여기서는 TensorRT 엔진의 출력 형식을 단순화해 가정하고 있으므로,
        export된 모델의 실제 출력 순서가 다르면 박스 해석이 어긋날 수 있다.
        """

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

def load_segmentation_engine(engine_path): return TRTSegmentationEngine(engine_path)
def load_yolo_engine(engine_path): return TRTYOLOEngine(engine_path)

# (이하 헬퍼 함수들은 TheDistanceAssessor_3_onnx.py와 동일)
def find_extreme_y_values(arr, values=[4, 9]):
    """선로 후보 클래스가 처음/마지막으로 나타나는 y 좌표를 찾는다.

    기본값 `4`, `9`는 기존 파이프라인에서 선로 및 침목/노면 계열로 취급하던 클래스다.
    최종 클래스 매핑 근거는 저장소에 명확히 남아 있지 않아 별도 검증이 필요하다.
    """

    mask = np.isin(arr, values)
    rows_with_values = np.any(mask, axis=1)
    y_indices = np.nonzero(rows_with_values)[0]
    if y_indices.size == 0: return None, None
    return y_indices[0], y_indices[-1]

def bresenham_line(x0, y0, x1, y1):
    """두 점 사이를 정수 격자 좌표로 연결한다.

    위험 구역 경계를 폴리곤으로 닫기 위해 사용하며, 부동소수점 보간 대신
    픽셀 단위 연결을 유지해 시각화와 포함 판정의 일관성을 맞춘다.
    """

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

# --- 🚀 무한 루프 방지: 함수 수정 ---
def interpolate_end_points(end_points_dict, flags):
    """거리 표식의 끝점을 선분으로 보간해 경계 후보를 만든다.

    `flags`는 급격한 x 이동이 발생한 구간 인덱스로, 해당 구간은 보간에서 제외한다.
    점이 2개 미만이면 안정적으로 선분을 만들 수 없으므로 빈 리스트를 반환한다.
    """

    # 점이 2개 미만이면 보간할 수 없으므로 빈 리스트 반환
    if len(end_points_dict) < 2:
        return []
        
    line_arr = []
    ys = list(end_points_dict.keys())
    xs = list(end_points_dict.values())
    
    if flags and len(flags) == 1: pass
    elif flags and np.all(np.diff(flags) == 1): flags = [flags[0]]
    
    for i in range(len(ys) - 1):
        if i in flags: continue
        y1, y2 = ys[i], ys[i + 1]
        x1, x2 = xs[i], xs[i + 1]
        line = np.array(bresenham_line(x1, y1, x2, y2))
        if np.any(line[:, 0] < 0): line = line[line[:, 0] > 0]
        line_arr.extend(list(line))
        
    return line_arr

# ... (나머지 모든 헬퍼 함수는 그대로 유지) ...
def find_nearest_pairs(arr1, arr2):
    """두 점 집합 사이에서 가장 가까운 항목끼리 순차적으로 짝지어 비교한다."""

    arr1_np, arr2_np = np.array(arr1), np.array(arr2)
    base_array, compare_array = (arr1_np, arr2_np) if len(arr1_np) < len(arr2_np) else (arr2_np, arr1_np)
    paired_base, paired_compare = [], []
    paired_mask = np.zeros(len(compare_array), dtype=bool)
    for item in base_array:
        distances = np.linalg.norm(compare_array - item, axis=1)
        nearest_index = np.argmin(distances)
        paired_base.append(item)
        paired_compare.append(compare_array[nearest_index])
        paired_mask[nearest_index] = True
        if paired_mask.all(): break
    paired_base, paired_compare = np.array(paired_base), compare_array[paired_mask]
    return (paired_base, paired_compare) if len(arr1_np) < len(arr2_np) else (paired_compare, paired_base)

def filter_crossings(image, edges_dict):
    """가까이 붙은 선로 폭 후보를 위/아래 행의 기울기 단서로 병합한다.

    간격 50px 이내의 인접 edge는 분기기나 가드레일 영향으로 과분할된 경우로 보고
    추가 검사한다. 위아래 샘플 행에서 x 이동량이 30px 이상 벌어지면 같은 선로로
    이어지는 경우라고 가정해 하나의 edge로 병합한다.
    """

    filtered_edges = {}
    for key, values in edges_dict.items():
        merged = [values[0]]
        for start, end in values[1:]:
            if start - merged[-1][1] < 50:
                key_up, key_down = max(0, key - 10), min(image.shape[0] - 1, key + 10)
                if key_up == 0: key_up = key + 20
                if key_down == image.shape[0] - 1: key_down = key - 20
                edges_to_test_slope1 = robust_edges(image, [key_up], values=[4, 9], min_width=19)
                edges_to_test_slope2 = robust_edges(image, [key_down], values=[4, 9], min_width=19)
                values1, edges_to_test_slope1 = find_nearest_pairs(values, edges_to_test_slope1)
                values2, edges_to_test_slope2 = find_nearest_pairs(values, edges_to_test_slope2)
                differences_y = []
                for i, value in enumerate(values1):
                    if start in value:
                        idx = list(value).index(start)
                        try: differences_y.append(abs(start - edges_to_test_slope1[i][idx]))
                        except: pass
                    if merged[-1][1] in value:
                        idx = list(value).index(merged[-1][1])
                        try: differences_y.append(abs(merged[-1][1] - edges_to_test_slope1[i][idx]))
                        except: pass
                for i, value in enumerate(values2):
                    if start in value:
                        idx = list(value).index(start)
                        try: differences_y.append(abs(start - edges_to_test_slope2[i][idx]))
                        except: pass
                    if merged[-1][1] in value:
                        idx = list(value).index(merged[-1][1])
                        try: differences_y.append(abs(merged[-1][1] - edges_to_test_slope2[i][idx]))
                        except: pass
                if any(element > 30 for element in differences_y):
                    merged[-1] = (merged[-1][0], end)
                else: merged.append((start, end))
            else: merged.append((start, end))
        filtered_edges[key] = merged
    return filtered_edges

def robust_edges(image, y_levels, values=[4, 9], min_width=19):
    """단일 y 레벨에서 최소 폭 조건을 만족하는 선로 후보 구간만 추린다.

    `min_width=19`는 작은 잡음 영역을 제외하기 위한 휴리스틱이다.
    화면 양 끝에 닿는 구간은 절단/배경 노이즈 가능성이 높아 제외한다.
    """

    for y in y_levels:
        row = image[y, :]
        mask = np.isin(row, values).astype(int)
        diff = np.diff(np.pad(mask, (1, 1), 'constant'))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1
        filtered_edges = [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= min_width and s != 0 and e != 1919]
    return filtered_edges

def find_edges(image, y_levels, values=[4, 9], min_width=19):
    """여러 y 레벨에서 선로 폭 후보를 찾고 가드레일 영향을 보정한다.

    클래스 ID 1 주변 구간을 참고해 좌우 경계를 조금 확장한다. 이는 선로 인접 구조물이
    세그멘테이션 상에서 rail body와 붙어 나오는 경우를 완화하려는 보수적 보정이다.
    """

    edges_dict = {}
    for y in y_levels:
        row = image[y, :]
        mask = np.isin(row, values).astype(int)
        diff = np.diff(np.pad(mask, (1, 1), 'constant'))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1
        filtered_edges = [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= min_width and s != 0 and e != 1919]
        edges_with_guard_rails = []
        for edge in filtered_edges:
            cutout_left, cutout_right = image[y, edge[0] - 50:edge[0]][::-1], image[y, edge[1]:edge[1] + 50]
            not_ones_left, not_ones_right = np.where(cutout_left != 1)[0], np.where(cutout_right != 1)[0]
            if len(not_ones_left) > 0 and not_ones_left[0] > 0: edge = (edge[0] - (not_ones_left[0] - 1),) + edge[1:]
            if len(not_ones_right) > 0 and not_ones_right[0] > 0: edge = (edge[0], edge[1] - (not_ones_right[0] - 1)) + edge[2:]
            edges_with_guard_rails.append(edge)
        if edges_with_guard_rails: edges_dict[y] = edges_with_guard_rails
    return filter_crossings(image, {k: v for k, v in edges_dict.items() if v})

def identify_ego_track(edges_dict, image_width):
    """여러 선로 후보 중 주행 선로로 볼 하나의 중심 경로를 선택한다.

    가장 아래 행에서 영상 중앙에 가장 가까운 선로를 ego track으로 시작하고,
    위쪽 행으로 올라가며 직전 중심에 가장 가까운 선로를 이어 붙인다.
    분기 구간에서는 실제 주행 의도와 다를 수 있다는 한계가 있다.
    """

    ego_edges_dict, last_ego_track_center = {}, None
    sorted_y_levels = sorted(edges_dict.keys(), reverse=True)
    image_center_x = image_width / 2
    if sorted_y_levels:
        first_y = sorted_y_levels[0]
        tracks_at_first_y = edges_dict.get(first_y)
        if tracks_at_first_y:
            closest_track = min(tracks_at_first_y, key=lambda track: abs(((track[0] + track[1]) / 2) - image_center_x))
            ego_edges_dict[first_y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2
    for y in sorted_y_levels[1:]:
        if last_ego_track_center is None: break
        tracks_at_y = edges_dict.get(y)
        if tracks_at_y:
            closest_track = min(tracks_at_y, key=lambda track: abs(((track[0] + track[1]) / 2) - last_ego_track_center))
            ego_edges_dict[y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2
    return ego_edges_dict

def find_rails(arr, y_levels, values=[4, 9], min_width=5):
    """보더 보정용 세부 rail strip을 찾는다.

    `find_edges`보다 작은 `min_width=5`를 사용해 실제 레일 측면 가까이의 얇은 띠도
    놓치지 않도록 한다.
    """

    for y in y_levels:
        row = arr[y, :]
        mask = np.isin(row, values).astype(int)
        diff = np.diff(np.pad(mask, (1, 1), 'constant'))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1
        filtered_edges = [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= min_width and s != 0 and e != 1919]
    return filtered_edges

def find_rail_sides(img, edges_dict):
    """ego track의 좌우 실제 레일 측면을 추정한다.

    먼저 edge의 좌/우 끝점을 초기 border로 잡고, 같은 y에서 더 세밀한 rail strip을
    찾아 경계를 안쪽/바깥쪽으로 보정한다. 허용 오차는 `y * 0.04`로 두어 원근 때문에
    아래쪽에서 더 넓은 편차를 허용한다.
    """

    left_border, right_border = [], []
    for y, xs in edges_dict.items():
        rails = find_rails(img, [y], values=[4, 9], min_width=5)
        left_border_actual, right_border_actual = [min(xs)[0], y], [max(xs)[1], y]
        for zone in rails:
            if abs(zone[1] - left_border_actual[0]) < y * 0.04: left_border_actual[0] = zone[0]
            if abs(zone[0] - right_border_actual[0]) < y * 0.04: right_border_actual[0] = zone[1]
        left_border.append(left_border_actual)
        right_border.append(right_border_actual)
    left_border, flags_l, _ = robust_rail_sides(left_border)
    right_border, flags_r, _ = robust_rail_sides(right_border)
    return left_border, right_border, flags_l, flags_r

def robust_rail_sides(border, threshold=7):
    """경계점 시퀀스에서 급격한 점프를 제거해 부드러운 rail side를 만든다.

    x 변화량의 중앙값 대비 `threshold=7`배를 넘는 구간을 이상치로 간주한다.
    이상치가 연속되면 첫 구간만 대표로 남겨 재귀적으로 분할 정제한다.
    """

    border = np.array(border)
    if border.size > 0:
        border = border[border[:, 1] != 1079]
        if border.size < 2: return border, [], []
        steps_x = np.diff(border[:, 0])
        median_step = np.median(np.abs(steps_x))
        threshold_step = np.abs(threshold * np.abs(median_step))
        treshold_overcommings = abs(steps_x) > abs(threshold_step)
        flags = []
        if not np.any(treshold_overcommings): return border, flags, []
        else:
            overcommings_indices = [i for i, e in enumerate(treshold_overcommings) if e]
            if overcommings_indices and np.all(np.diff(overcommings_indices) == 1): overcommings_indices = [overcommings_indices[0]]
            filtered_border, previously_deleted = border, []
            for i in overcommings_indices:
                for item in previously_deleted:
                    if item[0] < i: i -= item[1]
                first_part, second_part = filtered_border[:i + 1], filtered_border[i + 1:]
                if len(second_part) < 2: filtered_border, previously_deleted = first_part, previously_deleted + [[i, len(second_part)]]
                elif len(first_part) < 2: filtered_border, previously_deleted = second_part, previously_deleted + [[i, len(first_part)]]
                else:
                    first_b, _, deleted_first = robust_rail_sides(first_part)
                    second_b, _, _ = robust_rail_sides(second_part)
                    filtered_border = np.concatenate((first_b, second_b), axis=0)
                    if deleted_first:
                        for deleted_item in deleted_first:
                            if deleted_item[0] <= i: i -= deleted_item[1]
                    flags.append(i)
            return filtered_border, flags, previously_deleted
    return border, [], []

def find_dist_from_edges(id_map, image, edges_dict, left_border, right_border, real_life_width_mm, real_life_target_mm):
    """영상 속 선로 폭을 실제 궤간과 연결해 목표 거리만큼 바깥 경계를 계산한다.

    선로 폭 픽셀값과 실제 궤간(기본 1435 mm)의 비율로 y별 scale factor를 만든 뒤,
    `real_life_target_mm`에 해당하는 픽셀 길이를 좌우로 확장한다.
    레일 폭 추정이 불안정한 행에서는 거리 경계도 함께 불안정해질 수 있다.
    """

    diffs_width = {k: max(e - s for s, e in v) for k, v in edges_dict.items() if v}
    scale_factors = {k: real_life_width_mm / v for k, v in diffs_width.items() if v > 0}
    target_distances_px = {k: int(real_life_target_mm / v) for k, v in scale_factors.items() if v > 0}
    end_points_left, region_levels_left = {}, []
    for point in left_border:
        y = point[1]
        if y in target_distances_px:
            min_edge = point[0]
            left_mark_start = min_edge - target_distances_px[y]
            end_points_left[y] = left_mark_start
            if left_mark_start < min_edge:
                y_vals = np.arange(left_mark_start, min_edge)
                region_levels_left.append(np.column_stack((np.full_like(y_vals, y), y_vals)))
    end_points_right, region_levels_right = {}, []
    for point in right_border:
        y = point[1]
        if y in target_distances_px:
            max_edge = point[0]
            right_mark_end = min(id_map.shape[1], max_edge + target_distances_px[y])
            if right_mark_end > max_edge:
                end_points_right[y] = right_mark_end
                y_vals = np.arange(max_edge, right_mark_end)
                region_levels_right.append(np.column_stack((np.full_like(y_vals, y), y_vals)))
    return id_map, end_points_left, end_points_right, region_levels_left, region_levels_right

def extrapolate_line(pixels, image, min_y=None, extr_pixels=10):
    """최근 경계점의 선형 추세를 이용해 위쪽 또는 아래쪽으로 경계를 연장한다.

    경계점이 충분하지 않으면 빈 결과를 반환한다. `extr_pixels=10`은 직전 10개 점으로
    기울기를 추정하겠다는 휴리스틱이며, 곡률이 큰 장면에서는 오차가 커질 수 있다.
    """

    if len(pixels) < extr_pixels: return []
    recent_pixels = np.array(pixels[-extr_pixels:])
    X, y = recent_pixels[:, 0].reshape(-1, 1), recent_pixels[:, 1]
    model = LinearRegression().fit(X, y)
    extrapolate = lambda x: model.coef_[0] * x + model.intercept_
    x_diffs = [pixels[-i][0] - pixels[-(i+1)][0] for i in range(1, extr_pixels-1)]
    y_diffs = [pixels[-i][1] - pixels[-(i+1)][1] for i in range(1, extr_pixels-1)]
    x_diff, y_diff = x_diffs[np.argmax(np.abs(x_diffs))], y_diffs[np.argmax(np.abs(y_diffs))]
    dx, dy = (1 if x_diff >= 0 else -1, 0) if abs(int(x_diff)) >= abs(int(y_diff)) else (0, 1 if y_diff >= 0 else -1)
    new_pixels, (x, y) = [], pixels[-1]
    min_y = min_y if min_y is not None else image.shape[0] - 1
    while 0 <= x < image.shape[1] and min_y <= y < image.shape[0]:
        if dx != 0: x += dx; y = int(extrapolate(x))
        elif dy != 0: y += dy; x = int(x)
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]: new_pixels.append((x, y))
        else: break
    return new_pixels

def extrapolate_borders(dist_marked_id_map, border_l, border_r, lowest_y):
    """좌우 거리 경계를 낮은 y 영역까지 외삽해 닫힌 위험 구역을 만들기 쉽게 한다."""

    border_extrapolation_l2 = extrapolate_line(border_l[::-1], dist_marked_id_map, lowest_y)
    border_extrapolation_r2 = extrapolate_line(border_r[::-1], dist_marked_id_map, lowest_y)
    border_l = border_extrapolation_l2[::-1] + border_l
    border_r = border_extrapolation_r2[::-1] + border_r
    return border_l, border_r

def find_zone_border(id_map, image, edges, irl_width_mm=1435, irl_target_mm=1000, lowest_y = 0):
    """목표 거리 하나에 대한 좌우 위험 구역 경계를 생성한다."""

    left_border, right_border, flags_l, flags_r = find_rail_sides(id_map, edges)
    dist_marked_id_map, end_points_left, end_points_right, left_region, right_region = find_dist_from_edges(id_map, image, edges, left_border, right_border, irl_width_mm, irl_target_mm)
    border_l = interpolate_end_points(end_points_left, flags_l)
    border_r = interpolate_end_points(end_points_right, flags_r)
    border_l, border_r = extrapolate_borders(dist_marked_id_map, border_l, border_r, lowest_y)
    return [border_l, border_r],[left_region, right_region]

def get_clues(segmentation_mask, number_of_clues):
    """선로가 존재하는 y 범위에서 대표 샘플 행을 고른다.

    전 구간을 모두 탐색하면 비용이 커지므로, 상하 extremum 사이를 균등 간격으로
    나눈 샘플 행만 사용한다.
    """

    lowest, highest = find_extreme_y_values(segmentation_mask)
    if lowest is not None and highest is not None and highest > lowest:
        clue_step = int((highest - lowest) / (number_of_clues + 1))
        if clue_step == 0: clue_step = 1
        return [highest - (i * clue_step) for i in range(number_of_clues)] + [lowest]
    return []

def border_handler(id_map, image, edges, target_distances):
    """여러 목표 거리값에 대해 위험 구역 경계를 반복 생성한다.

    기본 거리 목록 `[100, 400, 1000]`은 도메인 기준이 코드에만 남아 있으며,
    정확한 운영 의미는 별도 확인이 필요하다.
    """

    lowest, _ = find_extreme_y_values(id_map)
    borders, regions = [], []
    for target in target_distances:
        borders_regions = find_zone_border(id_map, image, edges, irl_target_mm=target, lowest_y = lowest)
        borders.append(borders_regions[0])
        regions.append(borders_regions[1])
    return borders, id_map, regions

def load(filename, PATH_jpgs, input_size=[512, 896], dataset_type='rs19val', item=None):
    """입력 이미지를 읽고 SegFormer 입력 텐서와 시각화용 원본 이미지를 함께 반환한다.

    반환 이미지 하나는 모델 입력 크기 기준 정규화 텐서이고, 다른 하나는 후속 시각화를
    위해 1920x1080으로 맞춘 BGR 이미지다.
    """

    transform_img = A.Compose([A.Resize(height=input_size[0], width=input_size[1]), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
    image_path = os.path.join(PATH_jpgs, item[1][1]["path"] if dataset_type == 'pilsen' else filename)
    image_in = cv2.imread(image_path)
    if image_in is None: return None, None, None
    image_tr = transform_img(image=image_in)['image'].unsqueeze(0)
    return image_tr, image_in, cv2.resize(image_in, (1920, 1080))

def segment(model_seg, image_size, filename, PATH_jpgs, dataset_type, item=None):
    """세그멘테이션 추론과 후처리를 수행한다.

    TensorRT 출력 logits에 softmax/argmax를 적용해 클래스 맵을 만들고,
    `image_morpho`로 소규모 잡음을 줄인 뒤 검출/시각화 좌표계에 맞게 1920x1080으로 확대한다.
    """

    image_norm, image_orig, image_resized = load(filename, PATH_jpgs, image_size, dataset_type, item)
    if image_norm is None: return None, None

    output = model_seg.infer(image_norm.numpy().astype(np.float32))
    id_map = np.argmax(F.softmax(torch.from_numpy(output), dim=1).cpu().detach().numpy().squeeze(), axis=0).astype(np.uint8)
    id_map = image_morpho(id_map)
    id_map = cv2.resize(id_map, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    return id_map, image_resized

def detect(model_det, filename_img, PATH_jpgs):
    """YOLO 검출 모델을 실행할 원본 이미지를 읽고 예측 결과를 반환한다."""

    image = cv2.imread(os.path.join(PATH_jpgs, filename_img))
    if image is None: return None, None, None
    return model_det.predict(image), model_det, image

def manage_detections(results, model):
    """검출 객체를 이동체와 비이동체로 분리한다.

    `accepted_moving`과 `accepted_stationary`는 경고 색상을 다르게 주기 위한 휴리스틱 집합이다.
    이 분류는 COCO class ID 일부만 사용하며, 현재 저장소에는 선정 근거가 명확히 남아 있지 않다.
    """

    if not results or not results[0].boxes: return {}, {}
    bbox, cls = results[0].boxes.xywh.tolist(), results[0].boxes.cls.tolist()
    accepted_moving = {0, 1, 2, 3, 7, 15, 16, 17, 18, 19}
    accepted_stationary = {24, 25, 28, 36}
    boxes_moving, boxes_stationary = {}, {}
    for xywh, clss in zip(bbox, cls):
        if clss in accepted_moving: boxes_moving.setdefault(clss, []).append(xywh)
        if clss in accepted_stationary: boxes_stationary.setdefault(clss, []).append(xywh)
    return boxes_moving, boxes_stationary

def compute_detection_borders(borders, output_dims=[1080,1920]):
    """좌우 경계만 있는 위험 구역을 다각형 판정에 필요한 닫힌 경계로 변환한다.

    한쪽 경계가 비면 화면 좌우 끝으로 보수적으로 대체해 포함 판정이 완전히 깨지지 않도록 한다.
    """

    det_height, det_width = output_dims[0] - 1, output_dims[1] - 1
    for i, border in enumerate(borders):
        border_l = np.array(border[0]) if border[0] else np.array([[0, 0], [0, 0]])
        border_r = np.array(border[1]) if border[1] else np.array([[0, 0], [0, 0]])
        endpoints_l, endpoints_r = [border_l[0], border_l[-1]], [border_r[0], border_r[-1]]
        if np.array_equal(endpoints_l, [[0,0],[0,0]]): endpoints_l = [[0, endpoints_r[0][1]], [0, endpoints_r[1][1]]]
        if np.array_equal(endpoints_r, [[0,0],[0,0]]): endpoints_r = [[det_width, endpoints_l[0][1]], [det_width, endpoints_l[1][1]]]
        borders[i].append(bresenham_line(endpoints_l[0][0], endpoints_l[0][1], endpoints_r[0][0], endpoints_r[0][1]))
        borders[i].append(bresenham_line(endpoints_l[1][0], endpoints_l[1][1], endpoints_r[1][0], endpoints_r[1][1]))
    return borders

def get_bounding_box_points(cx, cy, w, h):
    """박스 네 모서리와 각 변의 중점을 만들어 위험 구역 포함 판정용 샘플 점을 만든다."""

    corners = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    points = []
    for i in range(4):
        p1, p2 = corners[i], corners[(i + 1) % 4]
        points.extend([p1, ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)])
    return points

def classify_detections(boxes, borders, img_dims, output_dims=[1080,1920]):
    """검출 박스를 위험 구역과 비교해 색상/위험도 레벨을 정한다.

    입력 박스는 검출 원본 이미지 좌표계에 있으므로, 세그멘테이션 출력 좌표계로 스케일을 맞춘다.
    가장 안쪽 위험 구역에 한 점이라도 포함되면 그 구역 색상을 채택하고 이후 검사는 중단한다.
    """

    if not boxes: return []
    img_h, img_w, _ = img_dims
    scale_w, scale_h = output_dims[1] / img_w, output_dims[0] / img_h
    
    colors = ["red", "orange", "yellow", "green", "blue"]
    
    borders = compute_detection_borders(borders, output_dims)
    boxes_info = []
    
    for is_moving, box_dict in boxes.items():
        for item, coords in box_dict.items():
            for coord in coords:
                x, y, w, h = coord[0] * scale_w, coord[1] * scale_h, coord[2] * scale_w, coord[3] * scale_h
                points_to_test = get_bounding_box_points(x, y, w, h)
                criticality, color = -1, colors[3] # 기본값: Green
                
                for i, border in enumerate(borders):
                    border_nonempty = [np.array(arr) for arr in border if np.array(arr).size > 0]
                    if border_nonempty:
                        instance_border_path = mplPath.Path(np.vstack(border_nonempty))
                        if any(instance_border_path.contains_point(p) for p in points_to_test):
                            criticality = i
                            color = colors[i] if is_moving else colors[4] # is_moving: Red/Orange/Yellow, not moving: Blue
                            break # 가장 안쪽 지역에서 감지되면 더 검사할 필요 없이 중단
                            
                boxes_info.append([item, criticality, color, [x, y], [w, h], is_moving])
                
    return boxes_info

def create_danger_zone_mask(borders, image_shape):
    """시각화용 위험 구역 마스크를 생성한다.

    경계점이 3개 미만이면 유효한 폴리곤을 만들 수 없으므로 해당 구역 마스크는 비워 둔다.
    """

    masks = []
    for border in borders:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        all_points = []
        for side in border:
            if isinstance(side, np.ndarray) and side.size > 0: all_points.extend(side.tolist())
            elif isinstance(side, list) and side: all_points.extend(side)
        if len(all_points) >= 3:
            cv2.fillPoly(mask, [np.array(all_points, dtype=np.int32)], 255)
        masks.append(mask)
    return masks

# TheDistanceAssessor_3_engine.py 파일에서 아래 함수를 찾아 전체를 교체하세요.

def show_result(classification, names, borders, image, file_index):
    """위험 구역 오버레이와 검출 박스를 한 장의 결과 화면으로 표시한다.

    빨강/주황/노랑은 거리 구역, 파랑은 정지 물체 강조에 사용한다.
    결과 저장이 아니라 즉시 표시 중심이므로 headless 환경에서는 동작 제약이 있다.
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(16, 16 * (image_rgb.shape[0] / image_rgb.shape[1])), dpi=100)
    plt.imshow(image_rgb)
    
    danger_masks = create_danger_zone_mask(borders, image_rgb.shape)
    
    # --- 🚀 수정: 색상 순서 변경 (Red, Orange, Yellow 순으로) ---
    colors_rgba = [(1, 0, 0, 0.2), (1, 0.5, 0, 0.2), (1, 1, 0, 0.2)] # Red, Orange, Yellow
    colors_border = ['red', 'orange', 'yellow']

    for i, mask in enumerate(danger_masks):
        if np.any(mask):
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
            colored_mask[mask > 0] = colors_rgba[i % len(colors_rgba)]
            plt.imshow(colored_mask)

    if classification:
        for box in classification:
            name, color, (cx, cy), (w, h) = names[box[0]], box[2], box[3], box[4]
            x, y = cx - w / 2, cy - h / 2
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax = plt.gca()
            ax.add_patch(rect)
            plt.text(x, y - 10, name, color='white', fontsize=10, bbox=dict(facecolor=color, alpha=0.8, pad=1))

    for i, border in enumerate(borders):
        for side in border:
            side = np.array(side)
            if side.size > 0:
                plt.plot(side[:, 0], side[:, 1], '-', color=colors_border[i], linewidth=1.5, alpha=0.9)

    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    print(f'Frame {file_index} processed successfully.')
    

def run(model_seg, model_det, image_size, filepath_img, PATH_jpgs, dataset_type, target_distances, file_index, item=None, num_ys=15):
    """단일 이미지에 대한 전체 추론 파이프라인을 실행한다.

    세그멘테이션 실패 시 해당 파일은 건너뛰고, 검출 실패 시에는 위험 구역만 표시한다.
    `num_ys=15`는 선로 edge를 추정할 샘플 행 수를 의미한다.
    """

    segmentation_mask, image = segment(model_seg, image_size, filepath_img, PATH_jpgs, dataset_type, item=item)
    if segmentation_mask is None:
        print(f'Skipping file (could not load): {filepath_img}')
        return
    print(f'File: {filepath_img}')
    
    clues = get_clues(segmentation_mask, num_ys)
    edges = find_edges(segmentation_mask, clues)
    ego_edges = identify_ego_track(edges, segmentation_mask.shape[1])
    
    borders, _, _ = border_handler(segmentation_mask, image, ego_edges, target_distances)
    
    results, model, image_det = detect(model_det, filepath_img, PATH_jpgs)
    classification = []
    if results is not None and model is not None:
        boxes_moving, boxes_stationary = manage_detections(results, model)
        classification = classify_detections({1: boxes_moving, 0: boxes_stationary}, borders, image_det.shape, output_dims=segmentation_mask.shape)
    
    show_result(classification, model.names if model else {}, borders, image, file_index)

if __name__ == "__main__":
    data_type = 'railsem19'
    image_size = [512, 896]
    target_distances = [100, 400, 1000]
    num_ys = 15
    model_type = "segformer"
    
    try:
        model_seg = load_segmentation_engine(PATH_model_seg)
        model_det = load_yolo_engine(PATH_model_det)
        
        file_index = 0
        if data_type == 'pilsen':
            for item in enumerate(data_json["data"]):
                filepath_img = item[1][1]["path"]
                run(model_seg, model_det, image_size, filepath_img, PATH_base, data_type, target_distances, file_index, item=item, num_ys=num_ys)
                file_index += 1
        elif data_type == 'railsem19':
            for filename_img in sorted(os.listdir(PATH_jpgs)):
                run(model_seg, model_det, image_size, filename_img, PATH_jpgs, data_type, target_distances, file_index, item=None, num_ys=num_ys)
                file_index += 1
        else: # testdata
            PATH_jpgs = 'Grafika/Video_export/frames'
            for filename_img in sorted(os.listdir(PATH_jpgs)):
                run(model_seg, model_det, image_size, filename_img, PATH_jpgs, data_type, target_distances, file_index, item=None, num_ys=num_ys)
                file_index += 1 

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
