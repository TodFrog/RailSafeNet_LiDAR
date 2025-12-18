#!/usr/bin/env python3
"""
Unit tests for Phase 3 Parallel Processing Engine
Tests CUDAStreamManager, ParallelExecutor, and timing utilities
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch

# Import modules under test
from src.utils.cuda_utils import measure_parallel_overlap
from src.rail_detection.parallel_engine import ParallelExecutor, SequentialExecutor


# ============================================================================
# Test: Parallel Overlap Measurement
# ============================================================================

@pytest.mark.unit
@pytest.mark.phase3
def test_measure_parallel_overlap_perfect():
    """Test overlap measurement with perfect parallelism"""
    seg_time = 15.0
    det_time = 10.0
    total_time = 15.0  # Perfect overlap (max of both)

    metrics = measure_parallel_overlap(seg_time, det_time, total_time)

    assert metrics['sequential_time_ms'] == 25.0  # 15 + 10
    assert metrics['ideal_parallel_time_ms'] == 15.0  # max(15, 10)
    assert metrics['actual_parallel_time_ms'] == 15.0
    assert metrics['overlap_ms'] == 10.0  # 25 - 15
    assert metrics['speedup'] == pytest.approx(1.67, rel=0.01)  # 25/15
    assert metrics['overlap_ratio'] == 1.0  # 10/10 (shorter task fully overlapped)
    assert metrics['is_truly_parallel'] is True  # overlap > 70%


@pytest.mark.unit
@pytest.mark.phase3
def test_measure_parallel_overlap_partial():
    """Test overlap measurement with partial parallelism"""
    seg_time = 15.0
    det_time = 10.0
    total_time = 20.0  # Some overlap, but not perfect

    metrics = measure_parallel_overlap(seg_time, det_time, total_time)

    assert metrics['sequential_time_ms'] == 25.0
    assert metrics['overlap_ms'] == 5.0  # 25 - 20
    assert metrics['speedup'] == pytest.approx(1.25, rel=0.01)  # 25/20
    assert metrics['overlap_ratio'] == 0.5  # 5/10 (50% of shorter task overlapped)
    assert metrics['is_truly_parallel'] is False  # overlap < 70%


@pytest.mark.unit
@pytest.mark.phase3
def test_measure_parallel_overlap_no_parallelism():
    """Test overlap measurement with no parallelism (sequential)"""
    seg_time = 15.0
    det_time = 10.0
    total_time = 25.0  # No overlap (sequential execution)

    metrics = measure_parallel_overlap(seg_time, det_time, total_time)

    assert metrics['sequential_time_ms'] == 25.0
    assert metrics['overlap_ms'] == 0.0
    assert metrics['speedup'] == 1.0  # No speedup
    assert metrics['overlap_ratio'] == 0.0
    assert metrics['is_truly_parallel'] is False


# ============================================================================
# Test: CUDA Stream Manager (Mock-based)
# ============================================================================

@pytest.mark.unit
@pytest.mark.phase3
@pytest.mark.skipif(not pytest.importorskip("pycuda", reason="PyCUDA not available"),
                   reason="Requires CUDA")
def test_cuda_stream_manager_initialization():
    """Test CUDAStreamManager initialization"""
    from src.utils.cuda_utils import CUDAStreamManager

    manager = CUDAStreamManager(num_streams=2)

    assert len(manager.streams) == 2
    assert len(manager.stream_locks) == 2
    assert len(manager.is_busy) == 2
    assert manager.total_dispatches == 0
    assert manager.total_synchronizations == 0


@pytest.mark.unit
@pytest.mark.phase3
@pytest.mark.skipif(not pytest.importorskip("pycuda", reason="PyCUDA not available"),
                   reason="Requires CUDA")
def test_cuda_stream_manager_get_stream():
    """Test getting streams by ID"""
    from src.utils.cuda_utils import CUDAStreamManager

    manager = CUDAStreamManager(num_streams=2)

    stream0 = manager.get_stream(0)
    stream1 = manager.get_stream(1)

    assert stream0 is not None
    assert stream1 is not None
    assert stream0 != stream1  # Different streams


@pytest.mark.unit
@pytest.mark.phase3
@pytest.mark.skipif(not pytest.importorskip("pycuda", reason="PyCUDA not available"),
                   reason="Requires CUDA")
def test_cuda_stream_manager_mark_busy():
    """Test marking streams as busy/idle"""
    from src.utils.cuda_utils import CUDAStreamManager

    manager = CUDAStreamManager(num_streams=2)

    # Initially idle
    assert manager.is_stream_busy(0) is False

    # Mark busy
    manager.mark_busy(0)
    assert manager.is_stream_busy(0) is True
    assert manager.total_dispatches == 1

    # Mark idle
    manager.mark_idle(0)
    assert manager.is_stream_busy(0) is False


# ============================================================================
# Test: ParallelExecutor (Mock-based)
# ============================================================================

@pytest.mark.unit
@pytest.mark.phase3
def test_parallel_executor_initialization():
    """Test ParallelExecutor initialization with mock models"""
    mock_seg_model = Mock()
    mock_det_model = Mock()
    mock_stream_manager = Mock()

    executor = ParallelExecutor(
        model_seg=mock_seg_model,
        model_det=mock_det_model,
        stream_manager=mock_stream_manager,
        enable_profiling=False  # Disable profiling for unit test
    )

    assert executor.model_seg == mock_seg_model
    assert executor.model_det == mock_det_model
    assert executor.total_frames == 0
    assert executor.parallel_successes == 0
    assert executor.parallel_failures == 0


@pytest.mark.unit
@pytest.mark.phase3
def test_parallel_executor_process_frame_success():
    """Test successful parallel frame processing"""
    # Create mock models
    mock_seg_model = Mock()
    mock_det_model = Mock()
    mock_stream_manager = Mock()

    # Setup mock returns
    mock_seg_output = np.random.rand(19, 512, 896).astype(np.float32)
    mock_seg_output[1, :, :] = 2.0  # Rail class has highest probability
    mock_seg_model.infer.return_value = mock_seg_output

    mock_det_results = [Mock()]
    mock_det_model.predict.return_value = mock_det_results

    mock_stream_manager.get_stream.side_effect = [Mock(), Mock()]
    mock_stream_manager.synchronize_all.return_value = 1.5

    # Create executor
    executor = ParallelExecutor(
        model_seg=mock_seg_model,
        model_det=mock_det_model,
        stream_manager=mock_stream_manager,
        enable_profiling=False
    )

    # Process frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    image_norm = np.random.rand(3, 512, 896).astype(np.float32)

    seg_result, det_result, timing_info = executor.process_frame_parallel(frame, image_norm)

    # Verify results
    assert seg_result is not None
    assert det_result is not None
    assert timing_info['success'] is True
    assert executor.total_frames == 1
    assert executor.parallel_successes == 1
    assert executor.parallel_failures == 0


@pytest.mark.unit
@pytest.mark.phase3
def test_parallel_executor_statistics():
    """Test executor statistics collection"""
    mock_seg_model = Mock()
    mock_det_model = Mock()
    mock_stream_manager = Mock()
    mock_stream_manager.get_statistics.return_value = {'total_dispatches': 10}

    executor = ParallelExecutor(
        model_seg=mock_seg_model,
        model_det=mock_det_model,
        stream_manager=mock_stream_manager,
        enable_profiling=False
    )

    # Simulate some executions
    executor.total_frames = 10
    executor.parallel_successes = 9
    executor.parallel_failures = 1

    stats = executor.get_statistics()

    assert stats['total_frames'] == 10
    assert stats['parallel_successes'] == 9
    assert stats['parallel_failures'] == 1
    assert stats['success_rate'] == 0.9
    assert 'stream_stats' in stats


# ============================================================================
# Test: SequentialExecutor (Baseline for comparison)
# ============================================================================

@pytest.mark.unit
@pytest.mark.phase3
def test_sequential_executor_initialization():
    """Test SequentialExecutor initialization"""
    mock_seg_model = Mock()
    mock_det_model = Mock()

    executor = SequentialExecutor(
        model_seg=mock_seg_model,
        model_det=mock_det_model
    )

    assert executor.model_seg == mock_seg_model
    assert executor.model_det == mock_det_model
    assert executor.total_frames == 0
    assert executor.total_time_ms == 0.0


@pytest.mark.unit
@pytest.mark.phase3
def test_sequential_executor_process_frame():
    """Test sequential frame processing"""
    # Create mock models
    mock_seg_model = Mock()
    mock_det_model = Mock()

    # Setup mock returns
    mock_seg_output = np.random.rand(19, 512, 896).astype(np.float32)
    mock_seg_output[1, :, :] = 2.0  # Rail class
    mock_seg_model.infer.return_value = mock_seg_output

    mock_det_results = [Mock()]
    mock_det_model.predict.return_value = mock_det_results

    # Create executor
    executor = SequentialExecutor(
        model_seg=mock_seg_model,
        model_det=mock_det_model
    )

    # Process frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    image_norm = np.random.rand(3, 512, 896).astype(np.float32)

    seg_result, det_result, timing_info = executor.process_frame_sequential(frame, image_norm)

    # Verify results
    assert seg_result is not None
    assert det_result is not None
    assert timing_info['success'] is True
    assert timing_info['seg_time_ms'] > 0
    assert timing_info['det_time_ms'] > 0
    assert timing_info['total_time_ms'] > 0
    assert executor.total_frames == 1


@pytest.mark.unit
@pytest.mark.phase3
def test_sequential_executor_statistics():
    """Test sequential executor statistics"""
    mock_seg_model = Mock()
    mock_det_model = Mock()

    executor = SequentialExecutor(
        model_seg=mock_seg_model,
        model_det=mock_det_model
    )

    # Simulate executions
    executor.total_frames = 5
    executor.total_time_ms = 125.0  # 25ms average

    stats = executor.get_statistics()

    assert stats['total_frames'] == 5
    assert stats['total_time_ms'] == 125.0
    assert stats['average_time_ms'] == 25.0


# ============================================================================
# Test: Post-processing
# ============================================================================

@pytest.mark.unit
@pytest.mark.phase3
def test_segmentation_post_processing():
    """Test segmentation output post-processing"""
    mock_stream_manager = Mock()

    executor = ParallelExecutor(
        model_seg=Mock(),
        model_det=Mock(),
        stream_manager=mock_stream_manager,
        enable_profiling=False
    )

    # Create mock output with class 1 (rail) having highest probability
    output = np.zeros((19, 512, 896), dtype=np.float32)
    output[1, 200:400, 300:600] = 2.0  # Rail region

    mask = executor._post_process_segmentation(output)

    # Verify mask
    assert mask.shape == (512, 896)
    assert mask.dtype == np.uint8
    assert np.max(mask) == 255  # Rail pixels
    assert np.min(mask) == 0    # Non-rail pixels
    assert np.sum(mask == 255) > 0  # Has rail pixels
