#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Rail Tracker - Direct Midpoint Calculation
==================================================
직접 중점 계산 기반 Rail Tracker.
Polynomial fitting 없이 각 y 레벨의 중점과 폭을 직접 계산.

Features:
1. 각 y 레벨에서 left_x, right_x 매칭
2. 중점 = (left_x + right_x) / 2
3. 폭 = |right_x - left_x|
4. EMA smoothing for temporal continuity

Author: RailSafeNet Team
Date: 2025-01-06
"""

import numpy as np
from typing import List, Tuple, Optional


class SimpleRailTracker:
    """
    직접 중점 계산 기반 Rail Tracker.

    Polynomial fitting 없이 각 y 레벨의 중점과 폭을 직접 계산합니다.
    곡선 구간에서도 곡선 그대로 표시됩니다.
    """

    def __init__(self, height: int, width: int, alpha: float = 0.3):
        """
        Initialize the simple rail tracker.

        Args:
            height: Image height in pixels
            width: Image width in pixels
            alpha: EMA smoothing factor (0.0-1.0)
                   Higher = faster response, less smoothing
                   Lower = smoother, slower response
        """
        self.height = height
        self.img_width = width
        self.alpha = alpha

        # Previous frame data for EMA smoothing
        self.prev_centers: Optional[np.ndarray] = None
        self.prev_widths: Optional[np.ndarray] = None
        self.prev_y_levels: Optional[np.ndarray] = None

        # Statistics
        self.frame_count = 0

    def update(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Update tracker with detected rail edges.

        Args:
            left_edge: (N, 2) array of left edge points [(x, y), ...]
            right_edge: (N, 2) array of right edge points [(x, y), ...]

        Returns:
            center_points: List of (x, y) center points
            width_profile: Array of widths at each center point
        """
        self.frame_count += 1

        # Validate input
        if left_edge is None or right_edge is None:
            return [], np.array([])

        if len(left_edge) < 2 or len(right_edge) < 2:
            return [], np.array([])

        # Build lookup table for right edge by y-coordinate
        right_lookup = {}
        for pt in right_edge:
            y = int(pt[1])
            if y not in right_lookup:
                right_lookup[y] = pt[0]

        # Calculate centers and widths
        centers = []
        widths = []
        y_levels = []

        for left_pt in left_edge:
            left_x = left_pt[0]
            y = int(left_pt[1])

            # Find matching right_x at same y or nearby y
            right_x = None
            for offset in range(5):  # Search within ±4 pixels
                if y + offset in right_lookup:
                    right_x = right_lookup[y + offset]
                    break
                if y - offset in right_lookup:
                    right_x = right_lookup[y - offset]
                    break

            if right_x is not None:
                center_x = (left_x + right_x) / 2.0
                width = abs(right_x - left_x)

                # Sanity check: reasonable width (50-500 pixels)
                if 30 < width < 600:
                    centers.append(center_x)
                    widths.append(width)
                    y_levels.append(y)

        if len(centers) < 2:
            return [], np.array([])

        # Convert to numpy arrays
        centers_arr = np.array(centers, dtype=np.float32)
        widths_arr = np.array(widths, dtype=np.float32)
        y_levels_arr = np.array(y_levels, dtype=np.float32)

        # Apply EMA smoothing if previous data exists with same length
        if (self.prev_centers is not None and
            len(self.prev_centers) == len(centers_arr) and
            np.allclose(self.prev_y_levels, y_levels_arr, atol=5)):
            # Same y-levels, apply smoothing
            centers_arr = self.alpha * centers_arr + (1 - self.alpha) * self.prev_centers
            widths_arr = self.alpha * widths_arr + (1 - self.alpha) * self.prev_widths

        # Store for next frame
        self.prev_centers = centers_arr.copy()
        self.prev_widths = widths_arr.copy()
        self.prev_y_levels = y_levels_arr.copy()

        # Create output as list of tuples
        center_points = [(int(cx), int(y)) for cx, y in zip(centers_arr, y_levels_arr)]

        return center_points, widths_arr

    def reset(self):
        """Reset tracker state."""
        self.prev_centers = None
        self.prev_widths = None
        self.prev_y_levels = None
        self.frame_count = 0

    def get_statistics(self) -> dict:
        """Get tracking statistics."""
        return {
            "frame_count": self.frame_count,
            "has_prev_data": self.prev_centers is not None,
            "prev_points_count": len(self.prev_centers) if self.prev_centers is not None else 0
        }
