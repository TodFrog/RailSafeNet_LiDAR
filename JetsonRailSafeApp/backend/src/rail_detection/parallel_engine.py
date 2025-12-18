#!/usr/bin/env python3
"""
Parallel Inference Engine for Rail Detection
Phase 3: Parallel Processing with Independent CUDA Streams

This module coordinates parallel execution of SegFormer (segmentation) and
YOLO (detection) using independent CUDA streams and Python threading.
"""

import threading
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
import cv2

from src.utils.cuda_utils import measure_parallel_overlap


class ParallelExecutor:
    """
    Coordinates parallel inference of segmentation and detection models.

    Uses independent CUDA streams and Python threading to achieve true
    parallel GPU execution. Falls back to sequential mode on error.
    """

    def __init__(
        self,
        model_seg,
        model_det,
        stream_manager=None,
        enable_profiling: bool = False
    ):
        """
        Initialize parallel executor.

        Args:
            model_seg: SegFormer inference engine
            model_det: YOLO inference engine
            stream_manager: DEPRECATED - kept for compatibility
            enable_profiling: DEPRECATED - kept for compatibility
        """
        self.model_seg = model_seg
        self.model_det = model_det

        # Results storage (shared between threads)
        self.seg_result = None
        self.det_result = None
        self.seg_error = None
        self.det_error = None

        # Timing (for real-time monitoring)
        self.seg_time_ms = 0.0
        self.det_time_ms = 0.0
        self.total_time_ms = 0.0

        # Statistics
        self.total_frames = 0
        self.parallel_successes = 0
        self.parallel_failures = 0

        print("✓ ParallelExecutor initialized (CPU threading)")

    def process_frame_parallel(
        self,
        frame: np.ndarray,
        image_norm: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Any], Dict[str, float]]:
        """
        Process frame with parallel inference.

        Args:
            frame: Original BGR frame (for YOLO)
            image_norm: Normalized frame (for SegFormer)

        Returns:
            Tuple of (segmentation_mask, detection_results, timing_info)
        """
        self.total_frames += 1

        # Reset results
        self.seg_result = None
        self.det_result = None
        self.seg_error = None
        self.det_error = None

        start_time = time.time()

        # Create threads for parallel execution (CPU-side)
        thread_seg = threading.Thread(target=self._run_segmentation, args=(image_norm,))
        thread_det = threading.Thread(target=self._run_detection, args=(frame,))

        # Start both threads
        thread_seg.start()
        thread_det.start()

        # Wait for both to complete
        thread_seg.join()
        thread_det.join()

        self.total_time_ms = (time.time() - start_time) * 1000

        # Check for errors
        if self.seg_error or self.det_error:
            self.parallel_failures += 1
            print(f"⚠ Parallel execution failed: seg={self.seg_error}, det={self.det_error}")
            return None, None, self._get_timing_info(success=False)

        self.parallel_successes += 1

        return self.seg_result, self.det_result, self._get_timing_info(success=True)

    def _run_segmentation(self, image_norm: np.ndarray):
        """Run segmentation inference (called in separate thread)"""
        try:
            seg_start = time.time()

            # Run inference (TensorRT manages GPU internally)
            output = self.model_seg.infer(image_norm)

            # Post-process
            self.seg_result = self._post_process_segmentation(output)

            self.seg_time_ms = (time.time() - seg_start) * 1000

        except Exception as e:
            self.seg_error = str(e)
            print(f"⚠ Segmentation error: {e}")

    def _run_detection(self, frame: np.ndarray):
        """Run detection inference (called in separate thread)"""
        try:
            det_start = time.time()

            # Run inference (TensorRT manages GPU internally)
            results = self.model_det.predict(frame)
            self.det_result = results[0] if results else None

            self.det_time_ms = (time.time() - det_start) * 1000

        except Exception as e:
            self.det_error = str(e)
            print(f"⚠ Detection error: {e}")

    def _post_process_segmentation(self, output: np.ndarray) -> np.ndarray:
        """
        Post-process segmentation output.

        Args:
            output: Raw model output (C, H, W)

        Returns:
            Binary mask (H, W) with rail pixels = 255
        """
        # Get class predictions
        preds = np.argmax(output, axis=0)  # (H, W)

        # Create binary mask (class 1 = rail)
        rail_mask = (preds == 1).astype(np.uint8) * 255

        return rail_mask

    def _get_timing_info(self, success: bool) -> Dict[str, float]:
        """Get timing information for this frame"""
        timing = {
            'seg_time_ms': self.seg_time_ms,
            'det_time_ms': self.det_time_ms,
            'total_time_ms': self.total_time_ms,
            'success': success
        }

        if success and self.seg_time_ms > 0 and self.det_time_ms > 0:
            # Calculate overlap metrics
            overlap_metrics = measure_parallel_overlap(
                self.seg_time_ms,
                self.det_time_ms,
                self.total_time_ms
            )
            timing.update(overlap_metrics)

        return timing

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with frame counts, success rate, average times
        """
        success_rate = self.parallel_successes / max(self.total_frames, 1)

        return {
            'total_frames': self.total_frames,
            'parallel_successes': self.parallel_successes,
            'parallel_failures': self.parallel_failures,
            'success_rate': success_rate
        }

    def reset_statistics(self):
        """Reset all statistics counters"""
        self.total_frames = 0
        self.parallel_successes = 0
        self.parallel_failures = 0


class SequentialExecutor:
    """
    Fallback sequential executor (for comparison or when parallel fails).

    Executes segmentation and detection sequentially on the default stream.
    """

    def __init__(self, model_seg, model_det):
        """
        Initialize sequential executor.

        Args:
            model_seg: SegFormer inference engine
            model_det: YOLO inference engine
        """
        self.model_seg = model_seg
        self.model_det = model_det

        # Statistics
        self.total_frames = 0
        self.total_time_ms = 0.0

        print("✓ SequentialExecutor initialized (fallback mode)")

    def process_frame_sequential(
        self,
        frame: np.ndarray,
        image_norm: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Any], Dict[str, float]]:
        """
        Process frame with sequential inference.

        Args:
            frame: Original BGR frame (for YOLO)
            image_norm: Normalized frame (for SegFormer)

        Returns:
            Tuple of (segmentation_mask, detection_results, timing_info)
        """
        self.total_frames += 1
        start_time = time.time()

        # Segmentation first
        seg_start = time.time()
        try:
            # Convert torch tensor to numpy if needed
            if hasattr(image_norm, 'numpy'):
                image_norm_np = image_norm.numpy().astype(np.float32)
            else:
                image_norm_np = image_norm
            output = self.model_seg.infer(image_norm_np)
            seg_result = self._post_process_segmentation(output)
            seg_time_ms = (time.time() - seg_start) * 1000
        except Exception as e:
            print(f"⚠ Segmentation error: {e}")
            return None, None, {'seg_time_ms': 0, 'det_time_ms': 0, 'total_time_ms': 0, 'success': False}

        # Detection second
        det_start = time.time()
        try:
            results = self.model_det.predict(frame)  # TensorRT engine doesn't support verbose parameter
            det_result = results[0] if results else None
            det_time_ms = (time.time() - det_start) * 1000
        except Exception as e:
            print(f"⚠ Detection error: {e}")
            return seg_result, None, {'seg_time_ms': seg_time_ms, 'det_time_ms': 0, 'total_time_ms': 0, 'success': False}

        total_time_ms = (time.time() - start_time) * 1000
        self.total_time_ms += total_time_ms

        timing_info = {
            'seg_time_ms': seg_time_ms,
            'det_time_ms': det_time_ms,
            'total_time_ms': total_time_ms,
            'success': True
        }

        return seg_result, det_result, timing_info

    def _post_process_segmentation(self, output: np.ndarray) -> np.ndarray:
        """Post-process segmentation output"""
        preds = np.argmax(output, axis=0)
        rail_mask = (preds == 1).astype(np.uint8) * 255
        return rail_mask

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        avg_time_ms = self.total_time_ms / max(self.total_frames, 1)

        return {
            'total_frames': self.total_frames,
            'total_time_ms': self.total_time_ms,
            'average_time_ms': avg_time_ms
        }
