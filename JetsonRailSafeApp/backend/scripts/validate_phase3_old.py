#!/usr/bin/env python3
"""
Phase 3 Validation Script: Parallel Processing Performance
Measures speedup, GPU utilization, and parallel overlap
"""

import sys
import os

# Fix DISPLAY environment variable for GUI display (if needed by dependencies)
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':1'

import time
import cv2
import numpy as np
from collections import deque
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cuda_utils import measure_parallel_overlap
from src.rail_detection.parallel_engine import ParallelExecutor, SequentialExecutor


# ============================================================================
# Configuration
# ============================================================================

TEST_VIDEOS = [
    "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/crop/tram1.mp4",  # Changed from tram0 (intro video)
    "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/crop/tram10.mp4",
    "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/crop/tram25.mp4",
]

SEGMENTATION_ENGINE = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine"
DETECTION_ENGINE = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/yolo/yolov8s_896x512.engine"

# Target metrics (from spec.md)
TARGET_SPEEDUP = 1.5
TARGET_OVERLAP = 0.70  # 70%
TARGET_GPU_UTIL = 0.85  # 85%

# Test configuration
NUM_FRAMES_TO_TEST = 300  # Test first 5 seconds at 30 FPS


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_frame(frame, target_size=(512, 896)):
    """Preprocess frame for SegFormer inference (returns torch tensor)"""
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform_img = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image_tr = transform_img(image=frame)['image'].unsqueeze(0)
    return image_tr


# ============================================================================
# Sequential Baseline Test
# ============================================================================

def test_sequential(video_path, num_frames=NUM_FRAMES_TO_TEST):
    """Test sequential (non-parallel) processing"""
    print(f"\n{'='*70}")
    print(f"Sequential Baseline: {Path(video_path).name}")
    print(f"{'='*70}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        return None

    # Load models for this test (independent instance)
    print("  Loading models for sequential test...")
    from videoAssessor_phase3 import TRTSegmentationEngine, TRTYOLOEngine
    model_seg = TRTSegmentationEngine(SEGMENTATION_ENGINE)
    model_det = TRTYOLOEngine(DETECTION_ENGINE)

    executor = SequentialExecutor(model_seg, model_det)

    frame_count = 0
    total_seg_time = 0.0
    total_det_time = 0.0
    total_time = 0.0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        image_norm = preprocess_frame(frame)

        # Process sequentially
        seg_result, det_result, timing = executor.process_frame_sequential(frame, image_norm)

        if timing['success']:
            total_seg_time += timing['seg_time_ms']
            total_det_time += timing['det_time_ms']
            total_time += timing['total_time_ms']
            frame_count += 1

        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{num_frames} frames...")

    cap.release()

    # Calculate averages
    avg_seg_time = total_seg_time / frame_count
    avg_det_time = total_det_time / frame_count
    avg_total_time = total_time / frame_count
    fps = 1000.0 / avg_total_time

    results = {
        'frames': frame_count,
        'avg_seg_time_ms': avg_seg_time,
        'avg_det_time_ms': avg_det_time,
        'avg_total_time_ms': avg_total_time,
        'fps': fps
    }

    print(f"\n{'Sequential Results':-^70}")
    print(f"  Frames processed:    {frame_count}")
    print(f"  Avg Seg time:        {avg_seg_time:.2f} ms")
    print(f"  Avg Det time:        {avg_det_time:.2f} ms")
    print(f"  Avg Total time:      {avg_total_time:.2f} ms")
    print(f"  FPS:                 {fps:.2f}")

    return results


# ============================================================================
# Parallel Test
# ============================================================================

def test_parallel(video_path, num_frames=NUM_FRAMES_TO_TEST):
    """Test parallel processing with independent CUDA streams"""
    print(f"\n{'='*70}")
    print(f"Parallel Processing: {Path(video_path).name}")
    print(f"{'='*70}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        return None

    # Load models for this test (independent instance)
    print("  Loading models for parallel test...")
    from videoAssessor_phase3 import TRTSegmentationEngine, TRTYOLOEngine
    model_seg = TRTSegmentationEngine(SEGMENTATION_ENGINE)
    model_det = TRTYOLOEngine(DETECTION_ENGINE)

    # Create executor (CPU threading + TensorRT GPU optimization)
    executor = ParallelExecutor(model_seg, model_det)

    frame_count = 0
    total_seg_time = 0.0
    total_det_time = 0.0
    total_time = 0.0
    total_overlap = 0.0
    total_speedup = 0.0
    truly_parallel_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        image_norm = preprocess_frame(frame)

        # Process in parallel
        seg_result, det_result, timing = executor.process_frame_parallel(frame, image_norm)

        if timing['success']:
            total_seg_time += timing['seg_time_ms']
            total_det_time += timing['det_time_ms']
            total_time += timing['total_time_ms']

            # Accumulate overlap metrics
            if 'overlap_ratio' in timing:
                total_overlap += timing['overlap_ratio']
                total_speedup += timing['speedup']
                if timing['is_truly_parallel']:
                    truly_parallel_count += 1

            frame_count += 1

        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{num_frames} frames...")

    cap.release()

    # Calculate averages
    avg_seg_time = total_seg_time / frame_count
    avg_det_time = total_det_time / frame_count
    avg_total_time = total_time / frame_count
    avg_overlap_ratio = total_overlap / frame_count
    avg_speedup = total_speedup / frame_count
    truly_parallel_pct = (truly_parallel_count / frame_count) * 100
    fps = 1000.0 / avg_total_time

    results = {
        'frames': frame_count,
        'avg_seg_time_ms': avg_seg_time,
        'avg_det_time_ms': avg_det_time,
        'avg_total_time_ms': avg_total_time,
        'avg_overlap_ratio': avg_overlap_ratio,
        'avg_speedup': avg_speedup,
        'truly_parallel_pct': truly_parallel_pct,
        'fps': fps
    }

    print(f"\n{'Parallel Results':-^70}")
    print(f"  Frames processed:    {frame_count}")
    print(f"  Avg Seg time:        {avg_seg_time:.2f} ms")
    print(f"  Avg Det time:        {avg_det_time:.2f} ms")
    print(f"  Avg Total time:      {avg_total_time:.2f} ms")
    print(f"  Avg Overlap ratio:   {avg_overlap_ratio:.2%}")
    print(f"  Avg Speedup:         {avg_speedup:.2f}x")
    print(f"  Truly parallel:      {truly_parallel_pct:.1f}%")
    print(f"  FPS:                 {fps:.2f}")

    return results


# ============================================================================
# Comparison and Validation
# ============================================================================

def compare_and_validate(seq_results, par_results, video_name):
    """Compare sequential vs. parallel and validate against targets"""
    print(f"\n{'='*70}")
    print(f"Comparison: {video_name}")
    print(f"{'='*70}")

    if not seq_results or not par_results:
        print("✗ Missing results - cannot compare")
        return False

    # Calculate speedup
    speedup = seq_results['avg_total_time_ms'] / par_results['avg_total_time_ms']
    overlap_ratio = par_results['avg_overlap_ratio']

    print(f"\n  Sequential total:    {seq_results['avg_total_time_ms']:.2f} ms ({seq_results['fps']:.2f} FPS)")
    print(f"  Parallel total:      {par_results['avg_total_time_ms']:.2f} ms ({par_results['fps']:.2f} FPS)")
    print(f"  Speedup:             {speedup:.2f}x")
    print(f"  Overlap ratio:       {overlap_ratio:.2%}")

    # Validate against targets
    print(f"\n{'Validation':-^70}")

    speedup_pass = speedup >= TARGET_SPEEDUP
    overlap_pass = overlap_ratio >= TARGET_OVERLAP
    parallel_pass = par_results['truly_parallel_pct'] >= 70.0

    print(f"  Speedup ≥ {TARGET_SPEEDUP}x:        {'✓ PASS' if speedup_pass else '✗ FAIL'} ({speedup:.2f}x)")
    print(f"  Overlap ≥ {TARGET_OVERLAP:.0%}:         {'✓ PASS' if overlap_pass else '✗ FAIL'} ({overlap_ratio:.1%})")
    print(f"  Truly parallel ≥ 70%:  {'✓ PASS' if parallel_pass else '✗ FAIL'} ({par_results['truly_parallel_pct']:.1f}%)")

    all_pass = speedup_pass and overlap_pass and parallel_pass

    if all_pass:
        print(f"\n  {'✓ ALL CHECKS PASSED':^70}")
    else:
        print(f"\n  {'✗ SOME CHECKS FAILED':^70}")

    return all_pass


# ============================================================================
# Main
# ============================================================================

def main():
    """Run Phase 3 validation on all test videos"""
    print("\n" + "="*70)
    print("Phase 3 Validation: Parallel Processing Performance")
    print("="*70)

    # Check if videos exist
    available_videos = [v for v in TEST_VIDEOS if os.path.exists(v)]

    if not available_videos:
        print("\n✗ No test videos found!")
        print("  Expected videos:")
        for v in TEST_VIDEOS:
            print(f"    {v}")
        return 1

    print(f"\n  Found {len(available_videos)} test video(s)")
    for v in available_videos:
        print(f"    {Path(v).name}")

    # Run tests on each video (models loaded independently in each test)
    all_passed = True

    for video_path in available_videos:
        video_name = Path(video_path).name

        try:
            # Sequential baseline (loads its own models)
            seq_results = test_sequential(video_path)

            # Parallel test (loads its own models)
            par_results = test_parallel(video_path)

            # Compare and validate
            passed = compare_and_validate(seq_results, par_results, video_name)

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"\n✗ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Final summary
    print(f"\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")

    if all_passed:
        print("\n  ✓ Phase 3 validation PASSED on all test videos")
        print("  Ready to proceed to Phase 4")
        return 0
    else:
        print("\n  ✗ Phase 3 validation FAILED on one or more videos")
        print("  Review results and optimize before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
