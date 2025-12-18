import sys
import os
import cv2
import time
import numpy as np
import glob

# Add current directory to sys.path so that imports in core_logic work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now we can import from core_logic which imports from videoAssessor, src, scripts, etc.
try:
    from core_logic import TRTSegmentationEngine, TRTYOLOEngine, CachedExecutor, process_frame_cached, CachedInferenceResult
    from videoAssessor import draw_results
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    # Fallback for development if paths are messy
    sys.path.append(os.path.join(current_dir, '..'))
    from backend.core_logic import TRTSegmentationEngine, TRTYOLOEngine, CachedExecutor, process_frame_cached, CachedInferenceResult
    from backend.videoAssessor import draw_results

class RailSafeEngine:
    def __init__(self):
        self.initialized = False
        self.cap = None
        self.video_files = []
        self.current_video_idx = 0
        
        # Model Paths (Default to Docker paths, fallback to local)
        self.assets_dir = os.environ.get("ASSETS_DIR", "/app/assets")
        if not os.path.exists(self.assets_dir):
             # Fallback to local dev path
             self.assets_dir = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets"

        self.seg_engine_path = os.path.join(self.assets_dir, "models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961_896x512_int8.engine")
        self.det_engine_path = os.path.join(self.assets_dir, "models_pretrained/yolo/yolov8s_896x512_int8.engine")
        self.video_dir = os.path.join(self.assets_dir, "crop")

        self.model_seg = None
        self.model_det = None
        self.cached_executor = None
        self.target_distances = [80, 400, 1000]
        
        self.frame_counter = 0
        self.fps_deque = []
        self.cache_stats = {'seg_hits': 0, 'seg_miss': 0, 'det_hits': 0, 'det_miss': 0}

        self.init_models()
        self.load_playlist()
        self.open_next_video()

    def init_models(self):
        print("Initializing Engines...")
        try:
            if os.path.exists(self.seg_engine_path):
                self.model_seg = TRTSegmentationEngine(self.seg_engine_path)
                print("SegFormer Loaded")
            else:
                print(f"SegFormer Engine not found at {self.seg_engine_path}")

            if os.path.exists(self.det_engine_path):
                self.model_det = TRTYOLOEngine(self.det_engine_path)
                print("YOLO Loaded")
            else:
                print(f"YOLO Engine not found at {self.det_engine_path}")
            
            if self.model_seg and self.model_det:
                self.cached_executor = CachedExecutor(self.model_seg, self.model_det, image_size=[512, 896])
                self.initialized = True
        except Exception as e:
            print(f"Failed to initialize models: {e}")

    def load_playlist(self):
        if os.path.exists(self.video_dir):
            self.video_files = sorted(glob.glob(os.path.join(self.video_dir, "tram*.mp4")))
            print(f"Found {len(self.video_files)} videos")
        else:
            print(f"Video dir not found: {self.video_dir}")

    def open_next_video(self):
        if not self.video_files:
            return False
        
        video_path = self.video_files[self.current_video_idx]
        self.cap = cv2.VideoCapture(video_path)
        self.current_video_idx = (self.current_video_idx + 1) % len(self.video_files)
        self.frame_counter = 0
        if self.cached_executor:
            self.cached_executor.frame_counter = 0
        return True

    def process_next_frame(self):
        if not self.initialized or not self.cap:
            return None, {}

        ret, frame = self.cap.read()
        if not ret:
            # Loop video or go to next
            self.open_next_video()
            ret, frame = self.cap.read()
            if not ret: return None, {}

        self.frame_counter += 1
        start_time = time.time()
        
        info = {'status': 'SAFE', 'fps': 0, 'speed': 0, 'seg_cache': 0, 'det_cache': 0}

        try:
            # Inference
            borders, classification, cache_result = process_frame_cached(
                frame, self.cached_executor, self.target_distances
            )

            # Update Stats
            if cache_result.success:
                if cache_result.seg_from_cache: self.cache_stats['seg_hits'] += 1
                else: self.cache_stats['seg_miss'] += 1
                if cache_result.det_from_cache: self.cache_stats['det_hits'] += 1
                else: self.cache_stats['det_miss'] += 1

            # Draw
            processed_frame = frame.copy()
            
            # Draw Boxes
            if cache_result.success and cache_result.detection_results:
                det_result = cache_result.detection_results[0]
                if hasattr(det_result, 'boxes') and hasattr(det_result.boxes, 'xywh'):
                    boxes_xywh = det_result.boxes.xywh.tolist()
                    boxes_cls = det_result.boxes.cls.tolist()
                    for box_xywh, cls_id in zip(boxes_xywh, boxes_cls):
                         x_center, y_center, width, height = box_xywh
                         x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
                         x2, y2 = int(x_center + width / 2), int(y_center + height / 2)
                         cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Draw Borders & Classification
            if borders and len(borders) > 0:
                processed_frame = draw_results(processed_frame, borders, classification, self.model_det.names)
                
                # Determine Status
                # Check if any object is in RED zone
                status = 'SAFE'
                for obj in classification:
                    # obj format: [item, criticality, color, [x, y], [w, h], is_moving]
                    # criticality: 0=RED, 1=ORANGE, 2=YELLOW
                    crit = obj[1]
                    if crit == 0:
                        status = 'DANGER'
                        break
                    elif crit == 1 and status != 'DANGER':
                        status = 'WARNING'
                    elif crit == 2 and status not in ['DANGER', 'WARNING']:
                        status = 'CAUTION'
                info['status'] = status

            # FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            info['fps'] = fps
            
            # Cache Rates
            total_seg = self.cache_stats['seg_hits'] + self.cache_stats['seg_miss']
            total_det = self.cache_stats['det_hits'] + self.cache_stats['det_miss']
            info['seg_cache'] = (self.cache_stats['seg_hits'] / total_seg * 100) if total_seg > 0 else 0
            info['det_cache'] = (self.cache_stats['det_hits'] / total_det * 100) if total_det > 0 else 0

            return processed_frame, info

        except Exception as e:
            print(f"Processing Error: {e}")
            return frame, info
