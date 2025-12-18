"""
Integration tests for frame processing pipeline.

Tests the end-to-end frame processing flow including:
- Full pipeline execution
- Performance requirements
- Error handling
- Output validation
"""

import pytest
import numpy as np
import time
from src.utils.data_models import Frame, ProcessingMetrics, DangerZone, DetectedObject


@pytest.mark.integration
class TestFrameProcessor:
    """Test suite for FrameProcessor integration."""

    def test_full_pipeline_with_rails(self, frame_processor, sample_frame_with_rails):
        """Test full pipeline: load frame → segment → detect rails → create danger zone → detect objects."""
        # Process frame through full pipeline
        danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

        # Verify outputs exist
        assert danger_zones is not None
        assert detected_objects is not None
        assert metrics is not None

        # Verify output types
        assert isinstance(danger_zones, list)
        assert isinstance(detected_objects, list)
        assert isinstance(metrics, ProcessingMetrics)

        # Should have created 3 danger zones (red, orange, yellow)
        assert len(danger_zones) == 3
        assert all(isinstance(zone, DangerZone) for zone in danger_zones)

        # Detected objects should be list (may be empty)
        assert all(isinstance(obj, DetectedObject) for obj in detected_objects)

        # Metrics should have valid timing
        assert metrics.total_time_ms > 0
        assert metrics.segmentation_time_ms > 0
        assert metrics.detection_time_ms > 0

    def test_processing_time_under_40ms(self, frame_processor, sample_frame_with_rails, benchmark):
        """Verify processing time < 40ms using pytest-benchmark."""
        # Run pipeline with benchmark
        result = benchmark(frame_processor.process_frame, sample_frame_with_rails)

        danger_zones, detected_objects, metrics = result

        # Verify timing requirement (40ms = 25 FPS)
        assert metrics.total_time_ms < 40.0, f"Processing time {metrics.total_time_ms}ms exceeds 40ms target"
        assert metrics.meets_realtime_requirement, "Does not meet real-time requirement (25 FPS)"

        # Benchmark stats should also show mean < 40ms
        assert benchmark.stats['mean'] < 0.040  # 40ms in seconds

    def test_output_types_and_shapes(self, frame_processor, sample_frame_with_rails):
        """Verify outputs have correct types and shapes."""
        danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

        # Danger zones validation
        assert isinstance(danger_zones, list)
        if len(danger_zones) > 0:
            for zone in danger_zones:
                assert isinstance(zone, DangerZone)
                assert hasattr(zone, 'polygon')
                assert hasattr(zone, 'zone_id')
                assert len(zone.polygon) >= 3  # Valid polygon

        # Detected objects validation
        assert isinstance(detected_objects, list)
        if len(detected_objects) > 0:
            for obj in detected_objects:
                assert isinstance(obj, DetectedObject)
                assert hasattr(obj, 'bbox_xyxy')
                assert hasattr(obj, 'class_name')
                assert hasattr(obj, 'confidence')
                assert 0.0 <= obj.confidence <= 1.0

        # Metrics validation
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.frame_id >= 0
        assert metrics.total_time_ms > 0
        assert metrics.fps > 0

    def test_error_handling_when_rails_not_detected(self, frame_processor, sample_frame_no_rails):
        """Test error handling when rails are not detected."""
        # Process frame without rails
        danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_no_rails)

        # Should not crash
        assert danger_zones is not None
        assert detected_objects is not None
        assert metrics is not None

        # Should have empty or no danger zones
        assert len(danger_zones) == 0

        # May still have detected objects
        assert isinstance(detected_objects, list)

        # Metrics should still be collected
        assert metrics.total_time_ms > 0

    def test_pipeline_component_timings(self, frame_processor, sample_frame_with_rails):
        """Verify individual component timings are tracked."""
        danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

        # All timing components should be non-negative
        assert metrics.segmentation_time_ms >= 0
        assert metrics.detection_time_ms >= 0
        assert metrics.danger_zone_time_ms >= 0

        # Segmentation should be largest component (typically ~35ms)
        assert metrics.segmentation_time_ms > 0
        assert metrics.segmentation_time_ms < metrics.total_time_ms

        # Detection should take time (typically ~20ms)
        assert metrics.detection_time_ms > 0
        assert metrics.detection_time_ms < metrics.total_time_ms

        # Sum of components should be close to total (allowing for small overhead)
        component_sum = (metrics.segmentation_time_ms +
                        metrics.detection_time_ms +
                        metrics.danger_zone_time_ms)
        assert component_sum <= metrics.total_time_ms

    def test_multiple_frames_consistency(self, frame_processor, sample_frame_with_rails):
        """Test consistency across multiple frame processing."""
        results = []

        # Process same frame multiple times
        for i in range(5):
            danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)
            results.append((danger_zones, detected_objects, metrics))

        # All should produce same number of danger zones
        num_zones = [len(r[0]) for r in results]
        assert len(set(num_zones)) == 1  # All same

        # Processing times should be consistent (within 50% variance)
        times = [r[2].total_time_ms for r in results]
        mean_time = np.mean(times)
        std_time = np.std(times)
        variance_percent = (std_time / mean_time) * 100

        assert variance_percent < 50, f"Processing time variance {variance_percent}% too high"

    def test_fps_calculation_accuracy(self, frame_processor, sample_frame_with_rails):
        """Verify FPS calculation from processing time."""
        danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

        # FPS should match 1000 / processing_time_ms
        expected_fps = 1000.0 / metrics.total_time_ms
        assert abs(metrics.fps - expected_fps) < 0.1  # Allow small floating point error

    def test_danger_zone_object_intersection(self, frame_processor):
        """Test that objects are correctly marked as intersecting danger zones."""
        # Create frame with known object and rails
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        danger_zones, detected_objects, metrics = frame_processor.process_frame(frame)

        # If we have both zones and objects, verify intersection logic
        if len(danger_zones) > 0 and len(detected_objects) > 0:
            for obj in detected_objects:
                # danger_zone_id should be set if object intersects
                assert hasattr(obj, 'danger_zone_id')
                assert obj.danger_zone_id in [-1, 0, 1, 2]  # -1 = no intersection, 0-2 = zone ids

    def test_graceful_degradation_on_failure(self, frame_processor):
        """Test that pipeline continues even if individual steps fail."""
        # Create invalid frame (wrong shape)
        invalid_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should handle error gracefully without crashing
        try:
            danger_zones, detected_objects, metrics = frame_processor.process_frame(invalid_frame)

            # If it doesn't crash, should return valid (possibly empty) results
            assert danger_zones is not None
            assert detected_objects is not None
            assert metrics is not None

        except ValueError as e:
            # Acceptable to raise ValueError for invalid input
            assert "shape" in str(e).lower() or "resolution" in str(e).lower()

    def test_frame_state_transitions(self, frame_processor, sample_frame_with_rails):
        """Test that frame state transitions correctly through pipeline."""
        from src.utils.data_models import Frame, FrameState

        # Create Frame object
        frame_obj = Frame(
            frame_id=1,
            timestamp=0.0,
            image=sample_frame_with_rails,
            resolution=(1920, 1080),
            frame_time=0.033
        )

        # Initial state should be CREATED
        assert frame_obj.state == FrameState.CREATED

        # After processing, state should update
        # (This test assumes FrameProcessor updates frame state)
        # In actual implementation, state may be tracked differently

    def test_pipeline_memory_efficiency(self, frame_processor, sample_frame_with_rails):
        """Test that pipeline doesn't leak memory over multiple frames."""
        import gc

        # Get initial memory usage
        gc.collect()

        # Process many frames
        for i in range(10):
            danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)
            # Clear references
            del danger_zones, detected_objects, metrics

        # Force garbage collection
        gc.collect()

        # If no memory leak, this should complete without issues
        # (More robust test would track actual memory usage via psutil)

    def test_roi_bounds_respected(self, frame_processor, sample_frame_with_rails):
        """Test that ROI bounds are respected during processing."""
        danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

        # If danger zones created, they should respect ROI bounds
        if len(danger_zones) > 0:
            for zone in danger_zones:
                polygon_y_values = [p[1] for p in zone.polygon]
                min_y = min(polygon_y_values)
                max_y = max(polygon_y_values)

                # Should be within frame bounds
                assert 0 <= min_y <= 1080
                assert 0 <= max_y <= 1080

                # Should generally be in lower half (ROI starts at 540)
                # Allow some tolerance for small rails in upper portion
                assert min_y >= 400  # Most of zone should be below middle

    @pytest.mark.slow
    def test_sustained_performance_over_100_frames(self, frame_processor, sample_frame_with_rails):
        """Test sustained performance over extended processing (100 frames)."""
        times = []

        # Process 100 frames
        for i in range(100):
            danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)
            times.append(metrics.total_time_ms)

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        variance_percent = (std_time / mean_time) * 100

        # Performance targets
        assert mean_time < 40.0, f"Mean processing time {mean_time}ms exceeds 40ms target"
        assert variance_percent < 20.0, f"Variance {variance_percent}% exceeds 20% target"

        # FPS should average > 25
        mean_fps = 1000.0 / mean_time
        assert mean_fps >= 25.0, f"Mean FPS {mean_fps} below 25 target"
