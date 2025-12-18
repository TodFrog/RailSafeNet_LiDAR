#!/usr/bin/env python3
"""
Phase 2 Validation Script

This script validates that all Phase 2 (Foundational) components are properly implemented
and ready for User Story implementation.

Usage:
    python scripts/validate_phase2.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_imports():
    """Validate that all Phase 2 modules can be imported."""
    print("=" * 60)
    print("Phase 2 Validation - Import Tests")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Data Models
    print("\n[Test 1] Data Models Import...")
    try:
        from src.utils.data_models import (
            Frame, FrameState,
            SegmentationResult,
            RailExtent,
            RailTrack,
            DetectedObject,
            DangerZone,
            ProcessingMetrics,
            VanishingPoint,
            TrackingStateData
        )
        print("  ✓ All data model entities imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 2: Geometry Utilities
    print("\n[Test 2] Geometry Utilities Import...")
    try:
        from src.utils.geometry import (
            bresenham_line,
            interpolate_boundary,
            is_simple_polygon,
            compute_convergence_angle,
            find_nearest_pairs,
            compute_line_intersection,
            point_in_polygon
        )
        print("  ✓ All geometry functions imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 3: Configuration
    print("\n[Test 3] Configuration Import...")
    try:
        from src.utils.config import RailDetectionConfig
        print("  ✓ Configuration class imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 4: TensorRT Engines
    print("\n[Test 4] TensorRT Engine Wrappers Import...")
    try:
        from src.rail_detection.segmentation import (
            SegmentationEngine,
            HostDeviceMem,
            allocate_buffers,
            do_inference_v2
        )
        from src.rail_detection.detection import DetectionEngine
        print("  ✓ TensorRT engine wrappers imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def validate_data_models():
    """Validate data model functionality."""
    print("\n" + "=" * 60)
    print("Phase 2 Validation - Data Model Tests")
    print("=" * 60)

    import numpy as np
    from src.utils.data_models import (
        Frame, FrameState,
        SegmentationResult,
        RailExtent,
        RailTrack,
        DetectedObject,
        DangerZone,
        ProcessingMetrics
    )

    tests_passed = 0
    tests_failed = 0

    # Test 1: Frame Creation
    print("\n[Test 1] Frame Entity Creation...")
    try:
        frame_data = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=1,
            timestamp=0.0,
            image=frame_data,
            resolution=(1920, 1080),
            frame_time=0.033
        )
        assert frame.state == FrameState.CREATED
        print(f"  ✓ Frame created successfully (ID: {frame.frame_id})")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 2: SegmentationResult with validation
    print("\n[Test 2] SegmentationResult Entity Creation...")
    try:
        seg_mask = np.random.randint(0, 13, (1080, 1920), dtype=np.uint8)
        seg_result = SegmentationResult(
            frame_id=1,
            segmentation_mask=seg_mask,
            class_labels=list(range(13)),
            rail_classes=[4, 9],
            inference_time_ms=35.5,
            roi_bounds=(540, 0, 1080, 1920)
        )
        assert hasattr(seg_result, 'has_rails')
        print(f"  ✓ SegmentationResult created (inference: {seg_result.inference_time_ms}ms)")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 3: RailTrack with boundaries
    print("\n[Test 3] RailTrack Entity Creation...")
    try:
        left_boundary = [(900, 1080), (910, 800), (920, 600)]
        right_boundary = [(1000, 1080), (1010, 800), (1020, 600)]
        track = RailTrack(
            track_id=0,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            convergence_angle=5.0,
            is_ego_track=True,
            confidence=0.95
        )
        avg_width = track.average_width
        print(f"  ✓ RailTrack created (avg width: {avg_width:.1f} pixels)")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 4: DetectedObject classification
    print("\n[Test 4] DetectedObject Entity Creation...")
    try:
        detection = DetectedObject(
            object_id=0,
            class_id=0,  # person
            class_name="person",
            bbox_xywh=(960, 540, 100, 200),
            bbox_xyxy=(910, 440, 1010, 640),
            confidence=0.85,
            is_moving=True,
            danger_zone_id=0,
            criticality=2
        )
        print(f"  ✓ DetectedObject created ({detection.class_name}, conf: {detection.confidence:.2f})")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 5: ProcessingMetrics with FPS calculation
    print("\n[Test 5] ProcessingMetrics Entity Creation...")
    try:
        metrics = ProcessingMetrics(
            frame_id=1,
            total_time_ms=38.5,
            segmentation_time_ms=35.0,
            detection_time_ms=22.0
        )
        fps = metrics.fps
        meets_target = metrics.meets_realtime_requirement
        print(f"  ✓ ProcessingMetrics created (FPS: {fps:.1f}, meets target: {meets_target})")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def validate_geometry():
    """Validate geometry utilities."""
    print("\n" + "=" * 60)
    print("Phase 2 Validation - Geometry Utility Tests")
    print("=" * 60)

    import numpy as np
    from src.utils.geometry import (
        bresenham_line,
        interpolate_boundary,
        is_simple_polygon,
        point_in_polygon
    )

    tests_passed = 0
    tests_failed = 0

    # Test 1: Bresenham Line
    print("\n[Test 1] Bresenham Line Algorithm...")
    try:
        line = bresenham_line(0, 0, 10, 5)
        assert len(line) > 0
        assert line[0] == (0, 0)
        assert line[-1] == (10, 5)
        print(f"  ✓ Bresenham line generated ({len(line)} points)")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 2: Boundary Interpolation
    print("\n[Test 2] Boundary Interpolation...")
    try:
        points = [(0, 0), (10, 10), (20, 20)]
        interpolated = interpolate_boundary(points, gaps=[])
        assert len(interpolated) >= len(points)
        print(f"  ✓ Boundary interpolated ({len(interpolated)} points)")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 3: Simple Polygon Check
    print("\n[Test 3] Simple Polygon Validation...")
    try:
        valid_polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert is_simple_polygon(valid_polygon) == True

        invalid_polygon = [(0, 0), (0, 0)]
        assert is_simple_polygon(invalid_polygon) == False
        print(f"  ✓ Polygon validation working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    # Test 4: Point in Polygon
    print("\n[Test 4] Point in Polygon Test...")
    try:
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((5, 5), polygon) == True
        assert point_in_polygon((15, 15), polygon) == False
        print(f"  ✓ Point-in-polygon algorithm working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def validate_configuration():
    """Validate configuration management."""
    print("\n" + "=" * 60)
    print("Phase 2 Validation - Configuration Tests")
    print("=" * 60)

    from src.utils.config import RailDetectionConfig

    tests_passed = 0
    tests_failed = 0

    # Test 1: Default Configuration
    print("\n[Test 1] Default Configuration Creation...")
    try:
        config = RailDetectionConfig.default()
        assert config.enable_tracking == True
        assert config.enable_vp_filtering == True
        assert config.roi_height_fraction == 0.5
        assert config.target_fps == 25.0
        print(f"  ✓ Default configuration created")
        print(f"    - Target FPS: {config.target_fps}")
        print(f"    - ROI height: {config.roi_height_fraction}")
        print(f"    - Track width: {config.track_width_mm}mm")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def validate_project_structure():
    """Validate project directory structure."""
    print("\n" + "=" * 60)
    print("Phase 2 Validation - Project Structure")
    print("=" * 60)

    required_dirs = [
        "src",
        "src/rail_detection",
        "src/processing",
        "src/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
    ]

    required_files = [
        "src/__init__.py",
        "src/rail_detection/__init__.py",
        "src/processing/__init__.py",
        "src/utils/__init__.py",
        "src/utils/data_models.py",
        "src/utils/geometry.py",
        "src/utils/config.py",
        "src/rail_detection/segmentation.py",
        "src/rail_detection/detection.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "pytest.ini",
        "requirements.txt",
    ]

    tests_passed = 0
    tests_failed = 0

    print("\n[Test 1] Required Directories...")
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"  ✓ {dir_path}")
            tests_passed += 1
        else:
            print(f"  ✗ {dir_path} - NOT FOUND")
            tests_failed += 1

    print("\n[Test 2] Required Files...")
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            size = full_path.stat().st_size
            print(f"  ✓ {file_path} ({size} bytes)")
            tests_passed += 1
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            tests_failed += 1

    return tests_passed, tests_failed


def main():
    """Run all Phase 2 validation tests."""
    print("\n" + "=" * 60)
    print("RAILSAFENET LIDAR - PHASE 2 VALIDATION")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print()

    total_passed = 0
    total_failed = 0

    # Run all validation tests
    try:
        passed, failed = validate_project_structure()
        total_passed += passed
        total_failed += failed
    except Exception as e:
        print(f"\n✗ Structure validation failed: {e}")
        total_failed += 1

    try:
        passed, failed = validate_imports()
        total_passed += passed
        total_failed += failed
    except Exception as e:
        print(f"\n✗ Import validation failed: {e}")
        total_failed += 1

    try:
        passed, failed = validate_data_models()
        total_passed += passed
        total_failed += failed
    except Exception as e:
        print(f"\n✗ Data model validation failed: {e}")
        total_failed += 1

    try:
        passed, failed = validate_geometry()
        total_passed += passed
        total_failed += failed
    except Exception as e:
        print(f"\n✗ Geometry validation failed: {e}")
        total_failed += 1

    try:
        passed, failed = validate_configuration()
        total_passed += passed
        total_failed += failed
    except Exception as e:
        print(f"\n✗ Configuration validation failed: {e}")
        total_failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {total_passed}")
    print(f"Tests Failed: {total_failed}")
    print(f"Total Tests:  {total_passed + total_failed}")
    print()

    if total_failed == 0:
        print("🎉 Phase 2 validation PASSED - Ready for User Story implementation!")
        print()
        return 0
    else:
        print("⚠️  Phase 2 validation FAILED - Please fix errors before proceeding")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
