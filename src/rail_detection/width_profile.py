#!/usr/bin/env python3
"""
Rail Width Profile Learner for Phase 4 Temporal Tracking

This module implements perspective-corrected rail width profile learning
that adapts over 150-300 frames to establish baseline width expectations.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

from src.utils.data_models import RailWidthProfile

logger = logging.getLogger(__name__)


class RailWidthProfileLearner:
    """
    Learns perspective-corrected rail width profile over initial frames.

    The rail width varies with perspective (wider at bottom, narrower at top).
    This learner establishes a baseline profile by collecting width measurements
    at different vertical levels over 150-300 frames.

    Attributes:
        num_y_levels: Number of vertical sampling levels (default: 20)
        min_calibration_frames: Minimum frames needed for calibration (default: 150)
        max_calibration_frames: Maximum frames to collect (default: 300)
        width_tolerance: Validation tolerance as fraction (default: 0.20 for ±20%)
    """

    def __init__(
        self,
        num_y_levels: int = 20,
        min_calibration_frames: int = 150,
        max_calibration_frames: int = 300,
        width_tolerance: float = 0.20,
        image_height: int = 512
    ):
        """
        Initialize the width profile learner.

        Args:
            num_y_levels: Number of vertical levels to sample
            min_calibration_frames: Minimum frames before calibration is complete
            max_calibration_frames: Maximum frames to collect data
            width_tolerance: Tolerance for width validation (±20% default)
            image_height: Height of input images for y-level calculation
        """
        self.num_y_levels = num_y_levels
        self.min_calibration_frames = min_calibration_frames
        self.max_calibration_frames = max_calibration_frames
        self.width_tolerance = width_tolerance
        self.image_height = image_height

        # Define y-levels to sample (from 20% to 90% of image height)
        # Top 10% and bottom 10% excluded for stability
        self.y_levels = np.linspace(
            int(image_height * 0.2),
            int(image_height * 0.9),
            num_y_levels
        ).astype(np.int32)

        # Storage for width measurements at each level
        # measurements[i] = list of widths measured at y_levels[i]
        self.measurements: List[List[float]] = [[] for _ in range(num_y_levels)]

        # Counter for frames processed
        self.frames_collected = 0

        # Cached profile (None until calibration complete)
        self._profile: Optional[RailWidthProfile] = None

        logger.info(
            f"RailWidthProfileLearner initialized: "
            f"{num_y_levels} levels, "
            f"calibration in {min_calibration_frames}-{max_calibration_frames} frames"
        )

    def add_measurement(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray
    ) -> bool:
        """
        Add width measurements from detected rail edges.

        Args:
            left_edge: (N, 2) array of [x, y] points for left rail edge
            right_edge: (M, 2) array of [x, y] points for right rail edge

        Returns:
            True if measurement was successfully added, False otherwise
        """
        if self.frames_collected >= self.max_calibration_frames:
            return False

        # Validate inputs
        if left_edge.shape[0] < 2 or right_edge.shape[0] < 2:
            logger.debug("Insufficient edge points for width measurement")
            return False

        if left_edge.shape[1] != 2 or right_edge.shape[1] != 2:
            logger.warning(f"Invalid edge shape: left={left_edge.shape}, right={right_edge.shape}")
            return False

        # For each y-level, find corresponding width
        for i, y_target in enumerate(self.y_levels):
            width = self._measure_width_at_y(left_edge, right_edge, y_target)
            if width is not None:
                self.measurements[i].append(width)

        self.frames_collected += 1

        # Invalidate cached profile if we're still collecting
        if self._profile is not None and self.frames_collected < self.max_calibration_frames:
            self._profile = None

        # Log progress every 50 frames
        if self.frames_collected % 50 == 0:
            logger.info(
                f"Width profile learning: {self.frames_collected}/{self.max_calibration_frames} frames"
            )

        return True

    def _measure_width_at_y(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray,
        y_target: int,
        tolerance: int = 5
    ) -> Optional[float]:
        """
        Measure rail width at a specific y-coordinate.

        Args:
            left_edge: (N, 2) array of [x, y] points
            right_edge: (M, 2) array of [x, y] points
            y_target: Target y-coordinate
            tolerance: Pixel tolerance for y matching

        Returns:
            Width in pixels, or None if no valid measurement
        """
        # Find points near y_target on both edges
        left_y = left_edge[:, 1]
        right_y = right_edge[:, 1]

        left_mask = np.abs(left_y - y_target) <= tolerance
        right_mask = np.abs(right_y - y_target) <= tolerance

        if not (left_mask.any() and right_mask.any()):
            return None

        # Get average x-coordinate for matching points
        left_x_mean = np.mean(left_edge[left_mask, 0])
        right_x_mean = np.mean(right_edge[right_mask, 0])

        # Width is difference in x-coordinates
        width = abs(right_x_mean - left_x_mean)

        # Perspective-aware width bounds (y-dependent)
        # Lower y (top) = narrower, Higher y (bottom) = wider
        min_width, max_width = self._get_perspective_width_bounds(y_target)

        if width < min_width or width > max_width:
            return None

        return width

    def _get_perspective_width_bounds(self, y: float) -> tuple:
        """
        Get perspective-aware width bounds based on y-coordinate.

        For 512px height (resized):
        - Top (y=256): 40-100px (narrow, far away)
        - Middle (y=384): 60-150px
        - Bottom (y=512): 80-200px (wide, close)

        Args:
            y: Y-coordinate (vertical position in image)

        Returns:
            Tuple of (min_width, max_width) in pixels
        """
        # Image height from config
        image_height = float(self.image_height)

        # Normalize y to [0, 1] where 0=top, 1=bottom
        y_norm = np.clip(y / image_height, 0.0, 1.0)

        # Linear interpolation for min/max width
        # Top (y_norm=0.5, half way): min=40, max=100
        # Bottom (y_norm=1.0): min=80, max=200
        min_width = 40 + (80 - 40) * (y_norm - 0.5) * 2  # 40 at top, 80 at bottom
        max_width = 100 + (200 - 100) * (y_norm - 0.5) * 2  # 100 at top, 200 at bottom

        # Clamp to reasonable values
        min_width = max(30, min(min_width, 80))
        max_width = max(100, min(max_width, 220))

        return float(min_width), float(max_width)

    def get_profile(self) -> Optional[RailWidthProfile]:
        """
        Get the current width profile.

        Returns:
            RailWidthProfile if enough data collected, None otherwise
        """
        if self._profile is not None:
            return self._profile

        if self.frames_collected < self.min_calibration_frames:
            logger.debug(
                f"Insufficient frames for calibration: "
                f"{self.frames_collected}/{self.min_calibration_frames}"
            )
            return None

        # Build profile from collected measurements
        self._profile = self._build_profile()
        return self._profile

    def _build_profile(self) -> RailWidthProfile:
        """
        Build RailWidthProfile from collected measurements.

        Returns:
            Constructed RailWidthProfile
        """
        widths = np.zeros(self.num_y_levels, dtype=np.float32)
        variances = np.zeros(self.num_y_levels, dtype=np.float32)

        for i in range(self.num_y_levels):
            measurements = self.measurements[i]

            if len(measurements) == 0:
                # No measurements at this level - interpolate later
                widths[i] = np.nan
                variances[i] = np.nan
            else:
                # Use median for robustness against outliers
                widths[i] = np.median(measurements)
                variances[i] = np.var(measurements)

        # Interpolate missing values
        widths = self._interpolate_missing(widths)
        variances = self._interpolate_missing(variances)

        # Determine if calibrated
        is_calibrated = self.frames_collected >= self.min_calibration_frames

        profile = RailWidthProfile(
            y_levels=self.y_levels.astype(np.float32),
            widths=widths,
            variances=variances,
            num_samples=self.frames_collected,
            is_calibrated=is_calibrated
        )

        if is_calibrated:
            logger.info(
                f"✅ Width profile calibration complete: "
                f"{self.frames_collected} frames, "
                f"width range: {widths.min():.1f}-{widths.max():.1f} pixels"
            )

        return profile

    def _interpolate_missing(self, values: np.ndarray) -> np.ndarray:
        """
        Interpolate NaN values in array.

        Args:
            values: Array potentially containing NaN values

        Returns:
            Array with NaNs interpolated
        """
        if not np.isnan(values).any():
            return values

        # Find valid (non-NaN) indices
        valid_mask = ~np.isnan(values)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            # No valid values - return zeros
            logger.warning("No valid width measurements collected!")
            return np.zeros_like(values)

        if len(valid_indices) == 1:
            # Only one valid value - fill all with it
            return np.full_like(values, values[valid_indices[0]])

        # Interpolate
        all_indices = np.arange(len(values))
        interpolated = np.interp(
            all_indices,
            valid_indices,
            values[valid_mask]
        )

        return interpolated

    def is_calibrated(self) -> bool:
        """
        Check if profile is calibrated (enough frames collected).

        Returns:
            True if calibrated, False otherwise
        """
        return self.frames_collected >= self.min_calibration_frames

    def reset(self):
        """Reset the learner to initial state."""
        self.measurements = [[] for _ in range(self.num_y_levels)]
        self.frames_collected = 0
        self._profile = None
        logger.info("Width profile learner reset")

    def get_statistics(self) -> dict:
        """
        Get current learning statistics.

        Returns:
            Dictionary with statistics
        """
        total_measurements = sum(len(m) for m in self.measurements)
        measurements_per_level = [len(m) for m in self.measurements]

        return {
            'frames_collected': self.frames_collected,
            'total_measurements': total_measurements,
            'avg_measurements_per_level': total_measurements / self.num_y_levels if total_measurements > 0 else 0,
            'min_measurements_per_level': min(measurements_per_level) if measurements_per_level else 0,
            'max_measurements_per_level': max(measurements_per_level) if measurements_per_level else 0,
            'is_calibrated': self.is_calibrated(),
            'calibration_progress': min(1.0, self.frames_collected / self.min_calibration_frames)
        }
