"""
Unit tests for danger zone computation module.

Tests the danger zone creation functionality including:
- Creating danger zones from rail tracks
- Polygon continuity validation
- Object intersection detection
"""

import pytest
import numpy as np
from src.utils.data_models import RailTrack, DangerZone, DetectedObject


@pytest.mark.unit
class TestDangerZoneComputation:
    """Test suite for danger zone computation functions."""

    def test_create_danger_zone_with_valid_rail_extent(self):
        """Test creating danger zone with valid rail extent -> returns valid polygon."""
        from src.rail_detection.danger_zone import DangerZoneComputer

        # Create rail track
        left_boundary = [(900, 1080), (910, 900), (920, 700), (930, 540)]
        right_boundary = [(1000, 1080), (1010, 900), (1020, 700), (1030, 540)]

        rail_track = RailTrack(
            track_id=0,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            convergence_angle=5.0,
            is_ego_track=True,
            confidence=0.95
        )

        # Create danger zone computer
        dz_computer = DangerZoneComputer(
            track_width_mm=1435,
            danger_distances_mm=[100, 400, 1000]
        )

        # Compute zones
        zones = dz_computer.compute_zones(rail_track, frame_shape=(1080, 1920))

        # Should create 3 zones (red, orange, yellow)
        assert len(zones) == 3
        assert all(isinstance(zone, DangerZone) for zone in zones)

        # Verify zone ordering (innermost to outermost)
        assert zones[0].zone_id == 0  # Red (innermost)
        assert zones[1].zone_id == 1  # Orange
        assert zones[2].zone_id == 2  # Yellow (outermost)

        # Verify distance thresholds
        assert zones[0].distance_threshold_mm == 100
        assert zones[1].distance_threshold_mm == 400
        assert zones[2].distance_threshold_mm == 1000

    def test_polygon_continuity_no_gaps(self):
        """Test that generated polygons have no gaps."""
        from src.rail_detection.danger_zone import DangerZoneComputer

        left_boundary = [(900, 1080), (910, 800), (920, 600)]
        right_boundary = [(1000, 1080), (1010, 800), (1020, 600)]

        rail_track = RailTrack(
            track_id=0,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            convergence_angle=3.0,
            is_ego_track=True,
            confidence=0.90
        )

        dz_computer = DangerZoneComputer(
            track_width_mm=1435,
            danger_distances_mm=[100, 400, 1000]
        )

        zones = dz_computer.compute_zones(rail_track, frame_shape=(1080, 1920))

        # Check polygon continuity for each zone
        for zone in zones:
            polygon = zone.polygon
            assert len(polygon) >= 3  # Minimum valid polygon

            # Verify no consecutive duplicate points (gaps)
            for i in range(len(polygon) - 1):
                p1 = polygon[i]
                p2 = polygon[i + 1]
                # Points should be different
                assert p1 != p2, f"Duplicate consecutive points at index {i}: {p1}"

            # Verify polygon is continuous (forms closed loop)
            assert zone.is_continuous

    def test_polygon_bounds_match_rail_extent(self):
        """Test that polygon bounds match rail extent."""
        from src.rail_detection.danger_zone import DangerZoneComputer

        left_boundary = [(900, 1080), (910, 900), (920, 700)]
        right_boundary = [(1000, 1080), (1010, 900), (1020, 700)]

        rail_track = RailTrack(
            track_id=0,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            convergence_angle=5.0,
            is_ego_track=True,
            confidence=0.95
        )

        dz_computer = DangerZoneComputer(
            track_width_mm=1435,
            danger_distances_mm=[100, 400, 1000]
        )

        zones = dz_computer.compute_zones(rail_track, frame_shape=(1080, 1920))

        # For innermost zone (red), bounds should be close to rail boundaries
        red_zone = zones[0]
        polygon_y_values = [p[1] for p in red_zone.polygon]

        # Y extent should match or be close to rail extent
        min_y = min(polygon_y_values)
        max_y = max(polygon_y_values)

        rail_min_y = min(p[1] for p in left_boundary + right_boundary)
        rail_max_y = max(p[1] for p in left_boundary + right_boundary)

        # Allow some tolerance for zone expansion
        assert min_y <= rail_min_y + 50  # Zone should not start much higher
        assert max_y >= rail_max_y - 50  # Zone should extend to bottom

    def test_create_danger_zone_with_insufficient_rail_extent(self):
        """Test with insufficient rail extent (< 3 levels) -> returns None or empty list."""
        from src.rail_detection.danger_zone import DangerZoneComputer

        # Create rail track with only 2 points (insufficient)
        left_boundary = [(900, 1080), (910, 1000)]
        right_boundary = [(1000, 1080), (1010, 1000)]

        rail_track = RailTrack(
            track_id=0,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            convergence_angle=5.0,
            is_ego_track=True,
            confidence=0.95
        )

        dz_computer = DangerZoneComputer(
            track_width_mm=1435,
            danger_distances_mm=[100, 400, 1000]
        )

        zones = dz_computer.compute_zones(rail_track, frame_shape=(1080, 1920))

        # Should return empty list or None for insufficient extent
        assert zones is None or len(zones) == 0

    def test_check_intersection_object_inside_zone(self):
        """Test object intersection detection - object inside danger zone."""
        from src.rail_detection.danger_zone import DangerZoneComputer

        # Create simple rectangular danger zone
        polygon = [(800, 600), (1100, 600), (1100, 1000), (800, 1000)]
        danger_zone = DangerZone(
            zone_id=0,
            polygon=polygon,
            track_id=0,
            distance_threshold_mm=100,
            color_code=(255, 0, 0),
            area_pixels=90000
        )

        # Create detected object inside zone
        detected_obj = DetectedObject(
            object_id=1,
            class_id=0,  # person
            class_name="person",
            bbox_xywh=(950, 800, 100, 200),  # Center at (950, 800)
            bbox_xyxy=(900, 700, 1000, 900),
            confidence=0.85,
            is_moving=True
        )

        dz_computer = DangerZoneComputer(
            track_width_mm=1435,
            danger_distances_mm=[100, 400, 1000]
        )

        # Check intersection
        intersects = dz_computer.check_intersection(detected_obj, [danger_zone])

        # Object center is inside zone
        assert intersects == 0  # Returns zone_id of intersecting zone

    def test_check_intersection_object_outside_zone(self):
        """Test object intersection detection - object outside danger zone."""
        from src.rail_detection.danger_zone import DangerZoneComputer

        # Create simple rectangular danger zone
        polygon = [(800, 600), (1100, 600), (1100, 1000), (800, 1000)]
        danger_zone = DangerZone(
            zone_id=0,
            polygon=polygon,
            track_id=0,
            distance_threshold_mm=100,
            color_code=(255, 0, 0),
            area_pixels=90000
        )

        # Create detected object outside zone
        detected_obj = DetectedObject(
            object_id=1,
            class_id=0,  # person
            class_name="person",
            bbox_xywh=(200, 300, 100, 200),  # Center at (200, 300) - far from zone
            bbox_xyxy=(150, 200, 250, 400),
            confidence=0.85,
            is_moving=True
        )

        dz_computer = DangerZoneComputer(
            track_width_mm=1435,
            danger_distances_mm=[100, 400, 1000]
        )

        # Check intersection
        intersects = dz_computer.check_intersection(detected_obj, [danger_zone])

        # Object is outside zone
        assert intersects == -1  # No intersection

    def test_danger_zone_color_codes(self):
        """Test correct color assignment for danger zones."""
        from src.rail_detection.danger_zone import DangerZoneComputer

        left_boundary = [(900, 1080), (910, 800), (920, 600)]
        right_boundary = [(1000, 1080), (1010, 800), (1020, 600)]

        rail_track = RailTrack(
            track_id=0,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            convergence_angle=5.0,
            is_ego_track=True,
            confidence=0.95
        )

        dz_computer = DangerZoneComputer(
            track_width_mm=1435,
            danger_distances_mm=[100, 400, 1000]
        )

        zones = dz_computer.compute_zones(rail_track, frame_shape=(1080, 1920))

        # Verify color codes (BGR format typically used in OpenCV)
        # Red zone (innermost)
        assert zones[0].color_code == (0, 0, 255) or zones[0].color_code == (255, 0, 0)

        # Orange zone
        assert zones[1].color_code == (0, 165, 255) or zones[1].color_code == (255, 165, 0)

        # Yellow zone
        assert zones[2].color_code == (0, 255, 255) or zones[2].color_code == (255, 255, 0)

    def test_danger_zone_area_calculation(self):
        """Test that danger zone area is calculated correctly."""
        from src.rail_detection.danger_zone import DangerZoneComputer

        left_boundary = [(900, 1080), (910, 800), (920, 600)]
        right_boundary = [(1000, 1080), (1010, 800), (1020, 600)]

        rail_track = RailTrack(
            track_id=0,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            convergence_angle=5.0,
            is_ego_track=True,
            confidence=0.95
        )

        dz_computer = DangerZoneComputer(
            track_width_mm=1435,
            danger_distances_mm=[100, 400, 1000]
        )

        zones = dz_computer.compute_zones(rail_track, frame_shape=(1080, 1920))

        # All zones should have positive area
        for zone in zones:
            assert zone.area_pixels > 0

        # Outer zones should have larger area than inner zones
        assert zones[2].area_pixels > zones[1].area_pixels
        assert zones[1].area_pixels > zones[0].area_pixels

    def test_multiple_zones_intersection_priority(self):
        """Test that object in multiple zones returns highest priority (innermost)."""
        from src.rail_detection.danger_zone import DangerZoneComputer

        # Create overlapping zones for testing
        zone_red = DangerZone(
            zone_id=0,
            polygon=[(900, 700), (1000, 700), (1000, 900), (900, 900)],
            track_id=0,
            distance_threshold_mm=100,
            color_code=(0, 0, 255),
            area_pixels=20000
        )

        zone_orange = DangerZone(
            zone_id=1,
            polygon=[(850, 650), (1050, 650), (1050, 950), (850, 950)],
            track_id=0,
            distance_threshold_mm=400,
            color_code=(0, 165, 255),
            area_pixels=60000
        )

        # Object in both zones
        detected_obj = DetectedObject(
            object_id=1,
            class_id=0,
            class_name="person",
            bbox_xywh=(950, 800, 50, 100),
            bbox_xyxy=(925, 750, 975, 850),
            confidence=0.85,
            is_moving=True
        )

        dz_computer = DangerZoneComputer(
            track_width_mm=1435,
            danger_distances_mm=[100, 400, 1000]
        )

        # Check intersection - should return innermost zone (highest priority)
        intersects = dz_computer.check_intersection(detected_obj, [zone_red, zone_orange])

        # Should return red zone (zone_id=0) as it has higher priority
        assert intersects == 0
