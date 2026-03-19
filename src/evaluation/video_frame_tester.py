"""비디오 프레임 단위로 선로 분할과 위험 구역을 점검하는 수동 평가 스크립트."""

import cv2
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.path as mplPath
import matplotlib.patches as patches
from ultralyticsplus import YOLO
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

# Video frame testing with enhanced RailSafeNet

# Configuration
PATH_model_seg_best = '/home/mmc-server4/RailSafeNet/models/segformer_b3_transfer_best_rail_0.7791.pth'
PATH_model_det = '/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/yolo/yolov8s.pt'
VIDEO_DIR = '/home/mmc-server4/RailSafeNet/assets/crop/'

def load_models():
    """세그멘테이션 모델과 YOLO 모델을 프레임 평가용으로 로드한다."""
    print("📥 Loading models...")

    # Load segmentation model
    model_seg = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3",
        num_labels=13,
        ignore_mismatched_sizes=True
    )
    state_dict = torch.load(PATH_model_seg_best, map_location='cpu')
    model_seg.load_state_dict(state_dict, strict=False)
    model_seg.eval()

    # Load YOLO model
    model_yolo = YOLO(PATH_model_det)
    model_yolo.overrides['conf'] = 0.25
    model_yolo.overrides['iou'] = 0.45
    model_yolo.overrides['agnostic_nms'] = False
    model_yolo.overrides['max_det'] = 1000

    print("✅ Models loaded successfully")
    return model_seg, model_yolo

def extract_frames(video_path, max_frames=10, skip_frames=30):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for sampling
        if frame_count % skip_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"📹 Extracted {len(frames)} frames from video")
    return frames

def preprocess_frame(frame, image_size=(1024, 1024)):
    """프레임을 SegFormer 입력 형식으로 정규화한다."""
    # Resize to model input size
    frame_resized = cv2.resize(frame, image_size)

    # Normalize for SegFormer
    frame_norm = frame_resized.astype(np.float32) / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        frame_norm[:, :, i] = (frame_norm[:, :, i] - mean[i]) / std[i]

    # Convert to tensor
    frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).unsqueeze(0)

    return frame_tensor

def process_frame_segmentation(model, frame_tensor):
    """Process frame through segmentation model"""
    with torch.no_grad():
        outputs = model(frame_tensor)
        logits = outputs.logits

        # Resize to original resolution
        logits = F.interpolate(
            logits, size=(1080, 1920),
            mode='bilinear', align_corners=False
        )

        predictions = torch.argmax(logits, dim=1)

    return predictions.squeeze().cpu().numpy()

def detect_rails_in_frame(prediction, values=[1, 4, 9]):
    """프레임 마스크에서 rail 관련 클래스가 나타나는 y 범위를 찾는다."""
    mask = np.isin(prediction, values)
    rows_with_rails = np.any(mask, axis=1)
    y_indices = np.nonzero(rows_with_rails)[0]

    if y_indices.size == 0:
        return None, 0
    else:
        rail_pixels = np.sum(mask)
        return (y_indices[0], y_indices[-1]), rail_pixels

def find_rail_edges_frame(prediction, y_min, y_max, values=[1, 4, 9]):
    """Find rail edges in frame"""
    y_step = max(1, (y_max - y_min) // 20)
    y_levels = list(range(y_min, y_max + 1, y_step))
    if y_max not in y_levels:
        y_levels.append(y_max)

    edges_dict = {}
    for y in y_levels:
        if y >= prediction.shape[0]:
            continue

        row = prediction[y, :]
        mask = np.isin(row, values).astype(int)

        # Find contiguous segments
        padded_mask = np.pad(mask, (1, 1), 'constant', constant_values=0)
        diff = np.diff(padded_mask)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        # Filter segments
        filtered_edges = []
        for start, end in zip(starts, ends):
            width = end - start + 1
            if width >= 3 and start > 10 and end < prediction.shape[1] - 10:
                filtered_edges.append((start, end))

        if filtered_edges:
            edges_dict[y] = filtered_edges

    return edges_dict

def create_danger_zone_frame(prediction, rail_extent, values=[1, 4, 9]):
    """프레임 단위 위험 구역을 선형 경계로 근사한다.

    속도 우선의 수동 검토용 구현이므로, 본 추론 파이프라인의 정밀 거리 구역 계산과
    동일한 결과를 보장하지 않는다.
    """
    if not rail_extent:
        return None

    y_min, y_max = rail_extent
    edges_dict = find_rail_edges_frame(prediction, y_min, y_max, values)

    if not edges_dict or len(edges_dict) < 3:
        return None

    # Collect boundary points (simplified approach for video frames)
    left_points = []
    right_points = []

    for y, edges in edges_dict.items():
        # Use center track (first edge if multiple)
        start, end = edges[0]
        left_points.append([y, start])
        right_points.append([y, end])

    if len(left_points) < 3:
        return None

    # Fit linear boundaries for speed
    left_points = np.array(left_points)
    right_points = np.array(right_points)

    left_model = LinearRegression().fit(left_points[:, 0].reshape(-1, 1), left_points[:, 1])
    right_model = LinearRegression().fit(right_points[:, 0].reshape(-1, 1), right_points[:, 1])

    # Generate danger zone
    y_values = np.arange(y_min, y_max + 1)
    left_x = left_model.predict(y_values.reshape(-1, 1))
    right_x = right_model.predict(y_values.reshape(-1, 1))

    left_boundary = [(x, y) for y, x in zip(y_values, left_x)]
    right_boundary = [(x, y) for y, x in zip(y_values, right_x)]

    danger_zone = left_boundary + right_boundary[::-1]
    return danger_zone

def process_yolo_frame(model_yolo, frame):
    """Process YOLO detection on frame"""
    results = model_yolo(frame)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = model_yolo.names[cls]

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': class_name,
                    'class_id': cls
                })

    return detections

def check_danger_intersections(detections, danger_zone):
    """검출 박스 중심이 위험 구역 내부에 들어오는지 검사한다."""
    if not danger_zone:
        return []

    path = mplPath.Path(danger_zone)
    danger_detections = []

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if path.contains_point((center_x, center_y)):
            detection['in_danger_zone'] = True
            danger_detections.append(detection)
        else:
            detection['in_danger_zone'] = False

    return danger_detections

def visualize_frame_result(frame, prediction, danger_zone, detections, frame_idx, video_name, save_path):
    """Visualize frame result"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original frame
    axes[0, 0].imshow(frame)
    axes[0, 0].set_title(f'Original Frame {frame_idx} - {video_name}')
    axes[0, 0].axis('off')

    # Segmentation
    axes[0, 1].imshow(prediction, cmap='tab20')
    axes[0, 1].set_title('Rail Segmentation')
    axes[0, 1].axis('off')

    # YOLO detections
    axes[1, 0].imshow(frame)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = 'red' if det.get('in_danger_zone', False) else 'green'
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor=color, facecolor='none')
        axes[1, 0].add_patch(rect)
        axes[1, 0].text(x1, y1-5, f"{det['class']}: {det['confidence']:.2f}",
                       color=color, fontweight='bold', fontsize=8)
    axes[1, 0].set_title(f'Object Detection ({len(detections)} objects)')
    axes[1, 0].axis('off')

    # Combined result
    axes[1, 1].imshow(frame)
    if danger_zone:
        danger_x = [p[0] for p in danger_zone]
        danger_y = [p[1] for p in danger_zone]
        axes[1, 1].plot(danger_x, danger_y, 'r-', linewidth=2, alpha=0.8)
        axes[1, 1].fill(danger_x, danger_y, color='red', alpha=0.2)

    # Highlight danger detections
    danger_count = 0
    for det in detections:
        if det.get('in_danger_zone', False):
            danger_count += 1
            x1, y1, x2, y2 = det['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=3, edgecolor='red', facecolor='none')
            axes[1, 1].add_patch(rect)

    axes[1, 1].set_title(f'Safety Assessment ({danger_count} in danger zone)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def process_video_frames(video_name, model_seg, model_yolo, save_dir, max_frames=5):
    """Process frames from a video"""
    video_path = os.path.join(VIDEO_DIR, video_name)

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None

    print(f"\n🎬 Processing video: {video_name}")

    # Extract frames
    frames = extract_frames(video_path, max_frames=max_frames, skip_frames=30)

    if not frames:
        print(f"❌ No frames extracted from {video_name}")
        return None

    results = []

    for frame_idx, frame in enumerate(frames):
        start_time = time.time()

        # Preprocess frame
        frame_tensor = preprocess_frame(frame)

        # Segmentation
        prediction = process_frame_segmentation(model_seg, frame_tensor)

        # Rail detection
        rail_extent, rail_pixels = detect_rails_in_frame(prediction)

        # Create danger zone
        danger_zone = create_danger_zone_frame(prediction, rail_extent) if rail_extent else None

        # YOLO detection
        detections = process_yolo_frame(model_yolo, frame)

        # Check danger intersections
        danger_detections = check_danger_intersections(detections, danger_zone)

        # Visualize
        save_path = os.path.join(save_dir, f"{video_name.replace('.mp4', '')}_frame_{frame_idx:02d}.jpg")
        visualize_frame_result(frame, prediction, danger_zone, detections, frame_idx, video_name, save_path)

        processing_time = time.time() - start_time

        result = {
            'video': video_name,
            'frame': frame_idx,
            'rail_detected': rail_extent is not None,
            'rail_pixels': rail_pixels if rail_extent else 0,
            'danger_zone_created': danger_zone is not None,
            'total_detections': len(detections),
            'danger_detections': len(danger_detections),
            'danger_objects': [det['class'] for det in danger_detections],
            'processing_time': processing_time
        }

        results.append(result)

        status = "✅" if danger_zone else "⚠️ "
        print(f"  {status} Frame {frame_idx}: Rail={rail_extent is not None}, Zone={danger_zone is not None}, Objects={len(detections)}, Danger={len(danger_detections)}, Time={processing_time:.2f}s")

    return results

def main():
    """Main video frame testing"""
    print("🎬 RailSafeNet Video Frame Testing")
    print("=" * 50)

    # Create results directory
    save_dir = "/home/mmc-server4/RailSafeNet/video_results"
    os.makedirs(save_dir, exist_ok=True)

    # Load models
    model_seg, model_yolo = load_models()

    # Test videos (select a few representative ones)
    test_videos = ['tram0.mp4', 'tram1.mp4', 'tram10.mp4', 'tram25.mp4', 'tram50.mp4']

    all_results = []
    total_frames = 0
    successful_zones = 0
    total_processing_time = 0

    for video_name in test_videos:
        try:
            results = process_video_frames(video_name, model_seg, model_yolo, save_dir, max_frames=5)

            if results:
                all_results.extend(results)
                total_frames += len(results)
                successful_zones += sum(1 for r in results if r['danger_zone_created'])
                total_processing_time += sum(r['processing_time'] for r in results)

        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")

    # Summary
    print(f"\n📊 VIDEO FRAME TESTING SUMMARY")
    print("=" * 50)
    print(f"🎬 Videos processed: {len(test_videos)}")
    print(f"🖼️  Total frames: {total_frames}")
    print(f"✅ Successful danger zones: {successful_zones}/{total_frames}")
    print(f"📈 Success rate: {successful_zones/total_frames*100:.1f}%")
    print(f"⚡ Average processing time: {total_processing_time/total_frames:.2f}s per frame")
    print(f"🎯 Estimated FPS: {1/(total_processing_time/total_frames):.1f}")

    # Detailed results
    print(f"\n📋 DETAILED RESULTS")
    print("-" * 50)
    for result in all_results:
        status = "✅" if result['danger_zone_created'] else "❌"
        danger_info = f", {result['danger_detections']} danger" if result['danger_detections'] > 0 else ""
        print(f"{status} {result['video']} Frame {result['frame']}: Rail={result['rail_detected']}, Zone={result['danger_zone_created']}, Objects={result['total_detections']}{danger_info}")

    print(f"\n💾 Results saved in: {save_dir}")

    return all_results

if __name__ == "__main__":
    main()
