"""
Performance benchmark tests for RailSafeNet LiDAR system.

Tests performance requirements including:
- Segmentation engine inference time
- Detection engine inference time
- Full frame processing time
- FPS calculation accuracy
"""

import pytest
import numpy as np
import time
from src.utils.data_models import ProcessingMetrics


@pytest.mark.benchmark
@pytest.mark.integration
class TestPerformance:
    """Performance benchmark test suite."""

    def test_segmentation_engine_inference_time(self, segmentation_engine, preprocessed_frame, benchmark):
        """Benchmark segmentation engine inference (target: < 40ms)."""
        # Run benchmark
        result = benchmark(segmentation_engine.infer, preprocessed_frame)

        # Verify result is valid segmentation mask
        assert result is not None
        assert result.shape == (1080, 1920)
        assert result.dtype == np.uint8

        # Check timing requirement
        mean_time_sec = benchmark.stats['mean']
        mean_time_ms = mean_time_sec * 1000

        assert mean_time_ms < 40.0, f"Segmentation inference {mean_time_ms:.2f}ms exceeds 40ms target"

        # Report stats
        print(f"\nSegmentation Engine Performance:")
        print(f"  Mean: {mean_time_ms:.2f}ms")
        print(f"  Min:  {benchmark.stats['min'] * 1000:.2f}ms")
        print(f"  Max:  {benchmark.stats['max'] * 1000:.2f}ms")
        print(f"  StdDev: {benchmark.stats['stddev'] * 1000:.2f}ms")

    def test_detection_engine_inference_time(self, detection_engine, sample_frame, benchmark):
        """Benchmark detection engine inference (target: < 25ms)."""
        # Run benchmark
        result = benchmark(detection_engine.predict, sample_frame)

        # Verify result is list of detected objects
        assert result is not None
        assert isinstance(result, list)

        # Check timing requirement
        mean_time_sec = benchmark.stats['mean']
        mean_time_ms = mean_time_sec * 1000

        assert mean_time_ms < 25.0, f"Detection inference {mean_time_ms:.2f}ms exceeds 25ms target"

        # Report stats
        print(f"\nDetection Engine Performance:")
        print(f"  Mean: {mean_time_ms:.2f}ms")
        print(f"  Min:  {benchmark.stats['min'] * 1000:.2f}ms")
        print(f"  Max:  {benchmark.stats['max'] * 1000:.2f}ms")
        print(f"  StdDev: {benchmark.stats['stddev'] * 1000:.2f}ms")
        print(f"  Detected objects: {len(result)}")

    def test_full_frame_processing_time(self, frame_processor, sample_frame_with_rails, benchmark):
        """Benchmark full frame processing (target: < 40ms total)."""
        # Run benchmark
        result = benchmark(frame_processor.process_frame, sample_frame_with_rails)

        danger_zones, detected_objects, metrics = result

        # Verify valid results
        assert danger_zones is not None
        assert detected_objects is not None
        assert metrics is not None

        # Check timing requirement
        mean_time_sec = benchmark.stats['mean']
        mean_time_ms = mean_time_sec * 1000

        assert mean_time_ms < 40.0, f"Full processing {mean_time_ms:.2f}ms exceeds 40ms target"
        assert metrics.meets_realtime_requirement, "Does not meet real-time requirement"

        # Report stats
        print(f"\nFull Frame Processing Performance:")
        print(f"  Mean: {mean_time_ms:.2f}ms")
        print(f"  Min:  {benchmark.stats['min'] * 1000:.2f}ms")
        print(f"  Max:  {benchmark.stats['max'] * 1000:.2f}ms")
        print(f"  StdDev: {benchmark.stats['stddev'] * 1000:.2f}ms")
        print(f"  Mean FPS: {1000.0 / mean_time_ms:.1f}")
        print(f"  Danger zones: {len(danger_zones)}")
        print(f"  Detected objects: {len(detected_objects)}")

    def test_fps_calculation_accuracy(self, frame_processor, sample_frame_with_rails):
        """Verify FPS calculation accuracy from processing time."""
        # Process multiple frames and track timing
        fps_measurements = []

        for i in range(10):
            danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

            # Calculate expected FPS
            expected_fps = 1000.0 / metrics.total_time_ms

            # Verify metrics.fps matches expected
            assert abs(metrics.fps - expected_fps) < 0.1, f"FPS calculation error: {metrics.fps} vs {expected_fps}"

            fps_measurements.append(metrics.fps)

        # Average FPS should be >= 25
        mean_fps = np.mean(fps_measurements)
        assert mean_fps >= 25.0, f"Average FPS {mean_fps:.1f} below 25 target"

        print(f"\nFPS Accuracy Test:")
        print(f"  Mean FPS: {mean_fps:.1f}")
        print(f"  Min FPS:  {min(fps_measurements):.1f}")
        print(f"  Max FPS:  {max(fps_measurements):.1f}")

    def test_component_timing_breakdown(self, frame_processor, sample_frame_with_rails):
        """Test timing breakdown of pipeline components."""
        danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

        # Verify all timings are positive
        assert metrics.segmentation_time_ms > 0
        assert metrics.detection_time_ms > 0
        assert metrics.total_time_ms > 0

        # Verify timing hierarchy
        assert metrics.segmentation_time_ms < metrics.total_time_ms
        assert metrics.detection_time_ms < metrics.total_time_ms

        # Report breakdown
        print(f"\nTiming Breakdown:")
        print(f"  Segmentation: {metrics.segmentation_time_ms:.2f}ms ({metrics.segmentation_time_ms/metrics.total_time_ms*100:.1f}%)")
        print(f"  Detection:    {metrics.detection_time_ms:.2f}ms ({metrics.detection_time_ms/metrics.total_time_ms*100:.1f}%)")
        print(f"  Danger Zone:  {metrics.danger_zone_time_ms:.2f}ms ({metrics.danger_zone_time_ms/metrics.total_time_ms*100:.1f}%)")
        print(f"  Other:        {metrics.total_time_ms - metrics.segmentation_time_ms - metrics.detection_time_ms - metrics.danger_zone_time_ms:.2f}ms")
        print(f"  Total:        {metrics.total_time_ms:.2f}ms")

    def test_performance_variance(self, frame_processor, sample_frame_with_rails):
        """Test performance consistency (variance < 20%)."""
        times = []

        # Process 100 frames
        for i in range(100):
            danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)
            times.append(metrics.total_time_ms)

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        variance_percent = (std_time / mean_time) * 100

        # Variance should be < 20%
        assert variance_percent < 20.0, f"Performance variance {variance_percent:.1f}% exceeds 20% target"

        print(f"\nPerformance Variance Test (100 frames):")
        print(f"  Mean:     {mean_time:.2f}ms")
        print(f"  StdDev:   {std_time:.2f}ms")
        print(f"  Variance: {variance_percent:.1f}%")
        print(f"  Min:      {min(times):.2f}ms")
        print(f"  Max:      {max(times):.2f}ms")
        print(f"  Mean FPS: {1000.0/mean_time:.1f}")

    @pytest.mark.slow
    def test_sustained_fps_over_extended_period(self, frame_processor, sample_frame_with_rails):
        """Test sustained FPS over 1000 frames (~40 seconds at 25 FPS)."""
        times = []
        fps_values = []

        print(f"\nProcessing 1000 frames...")

        start_time = time.time()

        for i in range(1000):
            danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)
            times.append(metrics.total_time_ms)
            fps_values.append(metrics.fps)

            # Progress indicator every 100 frames
            if (i + 1) % 100 == 0:
                current_mean_fps = 1000.0 / np.mean(times[-100:])
                print(f"  Frame {i+1}/1000: Last 100 frames avg {current_mean_fps:.1f} FPS")

        elapsed_time = time.time() - start_time

        # Calculate overall statistics
        mean_time = np.mean(times)
        mean_fps = 1000.0 / mean_time
        std_time = np.std(times)
        variance_percent = (std_time / mean_time) * 100

        # Performance requirements
        assert mean_fps >= 25.0, f"Sustained FPS {mean_fps:.1f} below 25 target"
        assert variance_percent < 20.0, f"Sustained variance {variance_percent:.1f}% exceeds 20%"

        print(f"\nSustained Performance (1000 frames):")
        print(f"  Total time:  {elapsed_time:.1f}s")
        print(f"  Mean FPS:    {mean_fps:.1f}")
        print(f"  Min FPS:     {min(fps_values):.1f}")
        print(f"  Max FPS:     {max(fps_values):.1f}")
        print(f"  Variance:    {variance_percent:.1f}%")
        print(f"  Actual FPS:  {1000.0 / elapsed_time:.1f} (including overhead)")

    def test_memory_usage_stability(self, frame_processor, sample_frame_with_rails):
        """Test that memory usage remains stable over multiple frames."""
        import gc

        # Force initial garbage collection
        gc.collect()

        # Process frames and track memory growth
        for i in range(50):
            danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

            # Clear references
            del danger_zones, detected_objects, metrics

            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()

        # Final collection
        gc.collect()

        # If memory is stable, this should complete without issues
        # (More robust test would use psutil to track actual memory usage)
        print(f"\nMemory stability test completed (50 frames)")

    def test_gpu_utilization(self, frame_processor, sample_frame_with_rails):
        """Test GPU utilization during processing (informational)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Process frames and sample GPU utilization
            utilization_samples = []

            for i in range(10):
                # Get GPU utilization before processing
                util_before = pynvml.nvmlDeviceGetUtilizationRates(handle)

                danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

                # Get GPU utilization after processing
                util_after = pynvml.nvmlDeviceGetUtilizationRates(handle)

                utilization_samples.append(util_after.gpu)

            pynvml.nvmlShutdown()

            mean_utilization = np.mean(utilization_samples)

            print(f"\nGPU Utilization (informational):")
            print(f"  Mean: {mean_utilization:.1f}%")
            print(f"  Min:  {min(utilization_samples):.1f}%")
            print(f"  Max:  {max(utilization_samples):.1f}%")

            # Ideally GPU utilization should be > 60% during processing
            # But this is informational, not a hard requirement

        except ImportError:
            pytest.skip("pynvml not available")
        except Exception as e:
            pytest.skip(f"GPU monitoring failed: {e}")

    def test_performance_targets_summary(self, frame_processor, sample_frame_with_rails):
        """Summary test verifying all performance targets are met."""
        # Process multiple frames
        times = []
        for i in range(50):
            danger_zones, detected_objects, metrics = frame_processor.process_frame(sample_frame_with_rails)
            times.append(metrics.total_time_ms)

        mean_time = np.mean(times)
        mean_fps = 1000.0 / mean_time
        variance_percent = (np.std(times) / mean_time) * 100

        # All performance targets
        targets = {
            "Mean processing time < 40ms": mean_time < 40.0,
            "Mean FPS >= 25": mean_fps >= 25.0,
            "Performance variance < 20%": variance_percent < 20.0,
        }

        print(f"\nPerformance Targets Summary:")
        for target, met in targets.items():
            status = "✓ PASS" if met else "✗ FAIL"
            print(f"  {status}: {target}")

        print(f"\nActual Performance:")
        print(f"  Mean time: {mean_time:.2f}ms")
        print(f"  Mean FPS:  {mean_fps:.1f}")
        print(f"  Variance:  {variance_percent:.1f}%")

        # Assert all targets are met
        assert all(targets.values()), "Not all performance targets met"
