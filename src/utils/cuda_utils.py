#!/usr/bin/env python3
"""
CUDA Stream Management Utilities for Parallel Inference
Phase 3: Parallel Processing
"""

import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA
import numpy as np
from typing import Optional, List
import threading
import time


class CUDAStreamManager:
    """
    Manages independent CUDA streams for parallel inference.

    Creates and manages two CUDA streams (one for SegFormer, one for YOLO)
    to enable true parallel execution on GPU.
    """

    def __init__(self, num_streams: int = 2):
        """
        Initialize CUDA stream manager.

        Args:
            num_streams: Number of streams to create (default: 2 for SegFormer + YOLO)
        """
        self.num_streams = num_streams
        self.streams = []
        self.stream_locks = []
        self.is_busy = []

        # Create CUDA streams
        for i in range(num_streams):
            stream = cuda.Stream()
            self.streams.append(stream)
            self.stream_locks.append(threading.Lock())
            self.is_busy.append(False)

        # Statistics
        self.total_dispatches = 0
        self.total_synchronizations = 0
        self.failed_dispatches = 0
        self.sync_times = []

        print(f"✓ CUDAStreamManager initialized with {num_streams} streams")

    def get_stream(self, stream_id: int) -> cuda.Stream:
        """
        Get CUDA stream by ID.

        Args:
            stream_id: Stream identifier (0 or 1)

        Returns:
            PyCUDA Stream object

        Raises:
            ValueError: If stream_id is invalid
        """
        if stream_id < 0 or stream_id >= self.num_streams:
            raise ValueError(f"Invalid stream_id: {stream_id}. Must be 0-{self.num_streams-1}")

        return self.streams[stream_id]

    def mark_busy(self, stream_id: int):
        """Mark stream as busy (inference in progress)"""
        with self.stream_locks[stream_id]:
            if self.is_busy[stream_id]:
                raise RuntimeError(f"Stream {stream_id} is already busy! Double-dispatch detected.")
            self.is_busy[stream_id] = True
            self.total_dispatches += 1

    def mark_idle(self, stream_id: int):
        """Mark stream as idle (inference complete)"""
        with self.stream_locks[stream_id]:
            self.is_busy[stream_id] = False

    def is_stream_busy(self, stream_id: int) -> bool:
        """Check if stream is currently executing"""
        with self.stream_locks[stream_id]:
            return self.is_busy[stream_id]

    def synchronize_stream(self, stream_id: int) -> float:
        """
        Wait for specific stream to complete.

        Args:
            stream_id: Stream to synchronize

        Returns:
            Synchronization time in milliseconds
        """
        start_time = time.time()

        try:
            self.streams[stream_id].synchronize()
            self.mark_idle(stream_id)
            self.total_synchronizations += 1
        except cuda.Error as e:
            self.failed_dispatches += 1
            raise RuntimeError(f"CUDA synchronization failed for stream {stream_id}: {e}")

        sync_time_ms = (time.time() - start_time) * 1000
        self.sync_times.append(sync_time_ms)

        return sync_time_ms

    def synchronize_all(self) -> float:
        """
        Wait for all streams to complete.

        Returns:
            Total synchronization time in milliseconds
        """
        start_time = time.time()

        for stream_id in range(self.num_streams):
            if self.is_stream_busy(stream_id):
                self.synchronize_stream(stream_id)

        total_sync_time_ms = (time.time() - start_time) * 1000
        return total_sync_time_ms

    def get_statistics(self) -> dict:
        """
        Get CUDA stream statistics.

        Returns:
            Dictionary with dispatch counts, sync times, success rate
        """
        avg_sync_time = np.mean(self.sync_times) if self.sync_times else 0.0
        success_rate = (self.total_dispatches - self.failed_dispatches) / max(self.total_dispatches, 1)

        return {
            'total_dispatches': self.total_dispatches,
            'total_synchronizations': self.total_synchronizations,
            'failed_dispatches': self.failed_dispatches,
            'success_rate': success_rate,
            'average_sync_time_ms': avg_sync_time,
            'num_active_streams': sum(self.is_busy)
        }

    def reset_statistics(self):
        """Reset all statistics counters"""
        self.total_dispatches = 0
        self.total_synchronizations = 0
        self.failed_dispatches = 0
        self.sync_times = []

    def __del__(self):
        """Cleanup: synchronize all streams before destruction"""
        try:
            self.synchronize_all()
        except:
            pass


class CUDAEvent:
    """
    Wrapper for CUDA events to measure timing.
    """

    def __init__(self):
        """Initialize CUDA event pair for timing"""
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()

    def record_start(self, stream: cuda.Stream):
        """Record start event on stream"""
        self.start_event.record(stream)

    def record_end(self, stream: cuda.Stream):
        """Record end event on stream"""
        self.end_event.record(stream)

    def synchronize(self):
        """Wait for end event to complete"""
        self.end_event.synchronize()

    def elapsed_time_ms(self) -> float:
        """
        Get elapsed time between start and end events.

        Returns:
            Elapsed time in milliseconds
        """
        self.synchronize()
        return self.start_event.time_till(self.end_event)


def measure_parallel_overlap(seg_time_ms: float, det_time_ms: float, total_time_ms: float) -> dict:
    """
    Measure actual parallel overlap from timing measurements.

    Args:
        seg_time_ms: Segmentation inference time
        det_time_ms: Detection inference time
        total_time_ms: Total parallel execution time

    Returns:
        Dictionary with overlap metrics
    """
    # Sequential baseline: sum of both times
    sequential_time_ms = seg_time_ms + det_time_ms

    # Ideal parallel time: max of both
    ideal_parallel_time_ms = max(seg_time_ms, det_time_ms)

    # Actual overlap: how much time was saved
    overlap_ms = sequential_time_ms - total_time_ms

    # Speedup: sequential / parallel
    speedup = sequential_time_ms / total_time_ms if total_time_ms > 0 else 1.0

    # Overlap ratio: what percentage of the shorter task overlapped
    min_time = min(seg_time_ms, det_time_ms)
    overlap_ratio = overlap_ms / min_time if min_time > 0 else 0.0

    # Is truly parallel: overlap ratio > 70%
    is_truly_parallel = overlap_ratio > 0.7

    return {
        'sequential_time_ms': sequential_time_ms,
        'ideal_parallel_time_ms': ideal_parallel_time_ms,
        'actual_parallel_time_ms': total_time_ms,
        'overlap_ms': overlap_ms,
        'speedup': speedup,
        'overlap_ratio': overlap_ratio,
        'is_truly_parallel': is_truly_parallel
    }
