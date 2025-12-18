"""
Unit tests for basic rail detection module.

Tests the core rail detection functionality including:
- Detecting rails from segmentation masks
- Extracting rail extents and boundaries
- ROI processing
"""

import pytest
import numpy as np
from src.utils.data_models import RailExtent, SegmentationResult


@pytest.mark.unit
class TestBasicRailDetection:
    """Test suite for basic rail detection functions."""

    def test_detect_rails_with_valid_mask(self, mock_segmentation_mask_with_rails):
        """Test detecting rails from segmentation mask containing rails (class 4, 9)."""
        from src.rail_detection.basic_detector import detect_rails_in_frame

        # Mock mask has rails at class 4 and 9
        result = detect_rails_in_frame(mock_segmentation_mask_with_rails)

        # Should detect rails
        assert result is not None
        assert isinstance(result, RailExtent)
        assert result.y_min >= 0
        assert result.y_max <= 1079
        assert result.y_min < result.y_max
        assert result.vertical_span > 0

    def test_detect_rails_without_rails(self):
        """Test detecting rails from mask without rails -> returns None."""
        from src.rail_detection.basic_detector import detect_rails_in_frame

        # Create mask with no rail pixels (all class 0)
        mask_no_rails = np.zeros((1080, 1920), dtype=np.uint8)

        result = detect_rails_in_frame(mask_no_rails)

        # Should return None when no rails detected
        assert result is None

    def test_detect_rails_with_insufficient_extent(self):
        """Test that insufficient rail extent (< 50 pixels) returns None."""
        from src.rail_detection.basic_detector import detect_rails_in_frame

        # Create mask with very small rail section (< 50 pixels vertical)
        mask_small_rails = np.zeros((1080, 1920), dtype=np.uint8)
        mask_small_rails[1000:1020, 900:1000] = 4  # Only 20 pixels high

        result = detect_rails_in_frame(mask_small_rails)

        # Should return None due to insufficient extent
        assert result is None

    def test_rail_pixel_counting_accuracy(self, mock_segmentation_mask_with_rails):
        """Test accurate counting of rail pixels in segmentation mask."""
        from src.rail_detection.basic_detector import count_rail_pixels

        # Count pixels with class 4 or 9
        rail_pixel_count = count_rail_pixels(mock_segmentation_mask_with_rails, rail_classes=[4, 9])

        # Verify count is positive and matches expected range
        assert rail_pixel_count > 0
        assert rail_pixel_count <= (1080 * 1920)  # Can't exceed total pixels

        # Count should match manual calculation from fixture
        expected_count = np.sum(np.isin(mock_segmentation_mask_with_rails, [4, 9]))
        assert rail_pixel_count == expected_count

    def test_y_min_y_max_extraction(self, mock_segmentation_mask_with_rails):
        """Test correct extraction of y_min and y_max from rail mask."""
        from src.rail_detection.basic_detector import extract_rail_extent

        y_min, y_max = extract_rail_extent(mock_segmentation_mask_with_rails, rail_classes=[4, 9])

        # Verify y_min and y_max are within valid range
        assert 0 <= y_min < 1080
        assert 0 < y_max <= 1080
        assert y_min < y_max

        # Verify y_min is the first row with rail pixels
        rail_mask = np.isin(mock_segmentation_mask_with_rails, [4, 9])
        rows_with_rails = np.any(rail_mask, axis=1)
        expected_y_min = np.where(rows_with_rails)[0][0]
        expected_y_max = np.where(rows_with_rails)[0][-1] + 1

        assert y_min == expected_y_min
        assert y_max == expected_y_max

    def test_roi_extension_to_half_frame(self):
        """Test that ROI is extended to 1/2 frame height (540 pixels)."""
        from src.rail_detection.basic_detector import get_roi_bounds

        # Get ROI with new 1/2 frame height setting
        y_roi_start, y_roi_end = get_roi_bounds(frame_height=1080, roi_fraction=0.5)

        # Should start at 1/2 frame height
        assert y_roi_start == int(1080 * 0.5)
        assert y_roi_start == 540
        assert y_roi_end == 1080

    def test_find_rail_edges_frame(self, mock_segmentation_mask_with_rails):
        """Test edge detection for rail boundaries."""
        from src.rail_detection.basic_detector import find_rail_edges_frame

        # Find edges at specific y level
        y_level = 800
        left_edge, right_edge = find_rail_edges_frame(mock_segmentation_mask_with_rails, y_level, rail_classes=[4, 9])

        # Should find valid edges
        assert left_edge is not None
        assert right_edge is not None
        assert isinstance(left_edge, int)
        assert isinstance(right_edge, int)
        assert 0 <= left_edge < 1920
        assert 0 <= right_edge < 1920
        assert left_edge < right_edge  # Left should be before right

    def test_find_rail_edges_no_rails_at_level(self):
        """Test edge detection returns None when no rails at y level."""
        from src.rail_detection.basic_detector import find_rail_edges_frame

        # Create mask with rails only at bottom
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        mask[1000:1080, 900:1000] = 4

        # Try to find edges where no rails exist
        y_level = 100  # No rails here
        left_edge, right_edge = find_rail_edges_frame(mask, y_level, rail_classes=[4, 9])

        assert left_edge is None
        assert right_edge is None

    def test_detect_rails_with_segmentation_result(self):
        """Test integration with SegmentationResult data model."""
        from src.rail_detection.basic_detector import detect_rails_from_segmentation_result

        # Create SegmentationResult
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        mask[600:1080, 900:920] = 4  # Left rail
        mask[600:1080, 1000:1020] = 4  # Right rail

        seg_result = SegmentationResult(
            frame_id=1,
            segmentation_mask=mask,
            class_labels=list(range(13)),
            rail_classes=[4, 9],
            inference_time_ms=35.0,
            roi_bounds=(540, 0, 1080, 1920)
        )

        rail_extent = detect_rails_from_segmentation_result(seg_result)

        assert rail_extent is not None
        assert rail_extent.y_min >= 600
        assert rail_extent.y_max <= 1080
        assert rail_extent.has_rails

    def test_rail_extent_validation(self):
        """Test RailExtent validation requirements."""
        # Valid rail extent
        valid_extent = RailExtent(
            y_min=600,
            y_max=1080,
            vertical_span=480,
            edges_by_row={}
        )
        assert valid_extent.y_min == 600
        assert valid_extent.y_max == 1080
        assert valid_extent.vertical_span >= 50

        # Invalid: vertical_span < 50
        with pytest.raises(ValueError, match="vertical_span"):
            invalid_extent = RailExtent(
                y_min=1000,
                y_max=1030,
                vertical_span=30,  # Too small
                edges_by_row={}
            )

        # Invalid: y_min >= y_max
        with pytest.raises(ValueError, match="y_min.*y_max"):
            invalid_extent = RailExtent(
                y_min=800,
                y_max=700,  # Invalid order
                vertical_span=100,
                edges_by_row={}
            )
