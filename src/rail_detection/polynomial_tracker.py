#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polynomial Rail Tracker with Temporal Smoothing
================================================
Replaces Phase 4 Kalman-based tracking with polynomial fitting approach.

Key Features:
1. 2nd-degree polynomial curve fitting (x = a*y^2 + b*y + c)
2. Temporal smoothing using Exponential Moving Average (EMA)
3. Straight-line locking for direct track sections
4. YAML-based configuration for easy parameter tuning

Based on video_center_smooth_v2.py with improvements.
"""

import numpy as np
import yaml
from typing import Optional, Tuple, List, Dict
from pathlib import Path


class PolynomialRailTracker:
    """
    Polynomial-based rail tracker with temporal smoothing and straight-line locking.

    This tracker fits a 2nd-degree polynomial to detected rail edges and applies
    temporal smoothing to reduce jitter and maintain continuous tracks.
    """

    def __init__(self, height: int, width: int, config_path: Optional[str] = None):
        """
        Initialize the polynomial rail tracker.

        Args:
            height: Image height in pixels
            width: Image width in pixels
            config_path: Path to YAML configuration file (optional)
        """
        self.height = height
        self.img_width = width

        # Load configuration
        self.config = self._load_config(config_path)

        # Extract configuration parameters
        track_cfg = self.config['tracking_region']
        self.start_y = height - track_cfg['bottom_offset_px']
        self.end_y = int(height * track_cfg['top_percentage'])
        self.scan_step = track_cfg['scan_step_px']

        # Polynomial fitting parameters
        poly_cfg = self.config['polynomial_fitting']
        self.min_measurements = poly_cfg['min_measurements']
        self.weight_bias = poly_cfg['weight_bias']

        # Temporal smoothing parameters
        smooth_cfg = self.config['temporal_smoothing']
        self.alpha = smooth_cfg['alpha']

        # Straight-line locking parameters
        straight_cfg = self.config['straight_line_locking']
        self.straight_threshold = straight_cfg['curvature_threshold']
        self.reduction_factor = straight_cfg['reduction_factor']

        # Width constraints
        width_cfg = self.config['width_constraints']
        self.min_width = width_cfg['min_width_px']
        self.max_width = width_cfg['max_width_px']
        self.clamp_min = width_cfg['clamp_min_px']
        self.clamp_max = width_cfg['clamp_max_px']

        # Edge selection
        edge_cfg = self.config['edge_selection']
        self.search_offset = edge_cfg['search_offset_range']

        # Performance
        perf_cfg = self.config['performance']
        self.output_step = perf_cfg['output_step_px']

        # Advanced
        adv_cfg = self.config['advanced']
        self.fit_width_poly = adv_cfg['fit_width_polynomial']
        self.reduce_width_in_straight = adv_cfg['reduce_width_curvature_in_straight']

        # For polynomial fitting, normalize y to [-1, 1]
        self.y_mid = (self.start_y + self.end_y) / 2.0
        self.y_scale = (self.start_y - self.end_y) / 2.0

        # Temporal smoothing: store previous coefficients
        self.prev_coeffs_center = None  # [c, b, a]
        self.prev_coeffs_width = None   # [c_w, b_w, a_w]

        # Current mode
        self.mode = "Unknown"

        # Statistics
        self.frame_count = 0
        self.straight_frames = 0
        self.curved_frames = 0

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default path
            config_path = Path(__file__).parent.parent.parent / "config" / "rail_tracker_config.yaml"

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize y coordinates to [-1, 1] range for numerical stability."""
        return (y - self.y_mid) / self.y_scale

    def _denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized y back to pixel coordinates."""
        return y_norm * self.y_scale + self.y_mid

    def _apply_temporal_smoothing(
        self,
        curr_coeffs: np.ndarray,
        prev_coeffs: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply Exponential Moving Average (EMA) for temporal smoothing.

        Formula: New = alpha * Current + (1 - alpha) * Previous

        Args:
            curr_coeffs: Current frame coefficients
            prev_coeffs: Previous frame coefficients (None for first frame)

        Returns:
            Smoothed coefficients
        """
        if prev_coeffs is None:
            return curr_coeffs

        smoothed = self.alpha * curr_coeffs + (1.0 - self.alpha) * prev_coeffs
        return smoothed

    def _apply_straight_line_locking(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Force curvature coefficient 'a' to near-zero if rail is straight.

        Args:
            coeffs: [c, b, a] polynomial coefficients

        Returns:
            Modified coefficients with reduced 'a' if straight
        """
        c, b, a = coeffs

        if abs(a) < self.straight_threshold:
            # This is a straight line, reduce curvature aggressively
            a = a * self.reduction_factor
            self.mode = "Straight"
            self.straight_frames += 1
        else:
            self.mode = "Curved"
            self.curved_frames += 1

        return np.array([c, b, a], dtype=np.float32)

    def process_frame(
        self,
        edges_dict: dict
    ) -> Tuple[
        Optional[np.ndarray],  # left_edge (smooth)
        Optional[np.ndarray],  # right_edge (smooth)
        List[Tuple[int, int]],  # raw_center_points
        str  # mode ("Straight" or "Curved")
    ]:
        """
        Extract rail measurements and fit smooth polynomial curve with temporal smoothing.

        Args:
            edges_dict: Dictionary of detected edges {y: [(left_x, right_x), ...]}

        Returns:
            Tuple of (left_edge, right_edge, raw_center_points, mode)
            - left_edge: (N, 2) array of smooth left edge points
            - right_edge: (N, 2) array of smooth right edge points
            - raw_center_points: List of raw detected center points
            - mode: "Straight" or "Curved"
        """
        self.frame_count += 1

        # 1. Extract measurements from edges_dict
        measurements = []  # [(y, center_x, width)]
        raw_points = []

        # Scan from bottom to top
        for y in range(self.start_y, self.end_y, -self.scan_step):
            found_y = -1

            # Look for edges around this Y
            for offset in range(self.search_offset):
                if (y + offset) in edges_dict:
                    found_y = y + offset
                    break
                if (y - offset) in edges_dict:
                    found_y = y - offset
                    break

            if found_y != -1:
                candidates = edges_dict[found_y]

                # Find candidate closest to image center (or previous center)
                if len(measurements) == 0:
                    best_pair = min(
                        candidates,
                        key=lambda p: abs((p[0] + p[1]) / 2 - self.img_width / 2)
                    )
                else:
                    prev_center = measurements[-1][1]
                    best_pair = min(
                        candidates,
                        key=lambda p: abs((p[0] + p[1]) / 2 - prev_center)
                    )

                center_x = (best_pair[0] + best_pair[1]) / 2.0
                width = abs(best_pair[1] - best_pair[0])

                # Sanity check: reasonable width
                if self.min_width < width < self.max_width:
                    measurements.append((y, center_x, width))
                    raw_points.append((int(center_x), y))

        if len(measurements) < self.min_measurements:
            return None, None, [], self.mode

        # 2. Fit polynomial to center line: x = a*y^2 + b*y + c
        y_samples = np.array([m[0] for m in measurements], dtype=np.float32)
        x_samples = np.array([m[1] for m in measurements], dtype=np.float32)
        w_samples = np.array([m[2] for m in measurements], dtype=np.float32)

        # Normalize y for numerical stability
        y_norm = self._normalize_y(y_samples)

        # Weighted least squares (trust bottom points more)
        weights = 1.0 + self.weight_bias * y_norm
        W = np.diag(weights)

        # Design matrix: [1, y, y^2]
        X = np.column_stack((np.ones_like(y_norm), y_norm, y_norm**2))

        try:
            # Solve: (X^T W X)^-1 X^T W x
            XTW = X.T @ W

            # Current frame coefficients
            curr_coeffs_center = np.linalg.solve(XTW @ X, XTW @ x_samples)

            if self.fit_width_poly:
                curr_coeffs_width = np.linalg.solve(XTW @ X, XTW @ w_samples)
            else:
                # Use constant width (mean)
                curr_coeffs_width = np.array([np.mean(w_samples), 0, 0], dtype=np.float32)

            # Apply temporal smoothing (EMA)
            coeffs_center = self._apply_temporal_smoothing(
                curr_coeffs_center,
                self.prev_coeffs_center
            )
            coeffs_width = self._apply_temporal_smoothing(
                curr_coeffs_width,
                self.prev_coeffs_width
            )

            # Apply straight-line locking
            coeffs_center = self._apply_straight_line_locking(coeffs_center)

            # Also apply to width if in straight mode
            if self.mode == "Straight" and self.reduce_width_in_straight:
                coeffs_width[2] = coeffs_width[2] * self.reduction_factor

            # Update previous coefficients
            self.prev_coeffs_center = coeffs_center.copy()
            self.prev_coeffs_width = coeffs_width.copy()

        except np.linalg.LinAlgError:
            return None, None, raw_points, self.mode

        # 3. Generate smooth curve with dense points
        y_dense = np.arange(
            self.end_y,
            self.start_y + 1,
            self.output_step,
            dtype=np.float32
        )
        y_dense_norm = self._normalize_y(y_dense)

        # Calculate center: x = a*y^2 + b*y + c
        center_dense = (
            coeffs_center[0] +
            coeffs_center[1] * y_dense_norm +
            coeffs_center[2] * y_dense_norm**2
        )

        # Calculate width: w = a_w*y^2 + b_w*y + c_w
        width_dense = (
            coeffs_width[0] +
            coeffs_width[1] * y_dense_norm +
            coeffs_width[2] * y_dense_norm**2
        )

        # Clamp width to reasonable range
        width_dense = np.clip(width_dense, self.clamp_min, self.clamp_max)

        # 4. Create left and right edges
        left_x = center_dense - width_dense / 2.0
        right_x = center_dense + width_dense / 2.0

        left_edge = np.column_stack((left_x, y_dense)).astype(np.float32)
        right_edge = np.column_stack((right_x, y_dense)).astype(np.float32)

        return left_edge, right_edge, raw_points, self.mode

    def get_statistics(self) -> Dict[str, any]:
        """Get tracking statistics."""
        if self.frame_count == 0:
            return {
                "total_frames": 0,
                "straight_frames": 0,
                "curved_frames": 0,
                "straight_percentage": 0.0,
                "curved_percentage": 0.0
            }

        return {
            "total_frames": self.frame_count,
            "straight_frames": self.straight_frames,
            "curved_frames": self.curved_frames,
            "straight_percentage": (self.straight_frames / self.frame_count) * 100,
            "curved_percentage": (self.curved_frames / self.frame_count) * 100
        }

    def reset(self):
        """Reset tracker state."""
        self.prev_coeffs_center = None
        self.prev_coeffs_width = None
        self.mode = "Unknown"
        self.frame_count = 0
        self.straight_frames = 0
        self.curved_frames = 0

    def update(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, str]:
        """
        Update tracker with detected rail edges.

        This method provides a simplified interface for tracking rail edges
        by computing center points and width profile from left/right edges.

        Args:
            left_edge: (N, 2) array of left edge points [(x, y), ...]
            right_edge: (N, 2) array of right edge points [(x, y), ...]

        Returns:
            center_points: List of (x, y) center points for the rail
            width_profile: Array of widths at each center point
            mode: "Straight" or "Curved"
        """
        self.frame_count += 1

        # Validate input
        if left_edge is None or right_edge is None:
            return [], np.array([]), self.mode

        if len(left_edge) < self.min_measurements or len(right_edge) < self.min_measurements:
            return [], np.array([]), self.mode

        # Build measurements from left/right edges
        # Match points by y-coordinate
        measurements = []

        # Create lookup for right edge by y
        right_lookup = {}
        for pt in right_edge:
            y = int(pt[1])
            if y not in right_lookup:
                right_lookup[y] = []
            right_lookup[y].append(pt[0])

        # Match left edge points with closest right edge points
        for left_pt in left_edge:
            left_x, left_y = left_pt[0], int(left_pt[1])

            # Find right point at same or nearby y
            best_right_x = None
            for offset in range(self.search_offset):
                if (left_y + offset) in right_lookup:
                    best_right_x = min(right_lookup[left_y + offset], key=lambda rx: abs(rx - left_x))
                    break
                if (left_y - offset) in right_lookup:
                    best_right_x = min(right_lookup[left_y - offset], key=lambda rx: abs(rx - left_x))
                    break

            if best_right_x is not None:
                center_x = (left_x + best_right_x) / 2.0
                width = abs(best_right_x - left_x)

                # Sanity check: reasonable width
                if self.min_width < width < self.max_width:
                    measurements.append((left_y, center_x, width))

        if len(measurements) < self.min_measurements:
            return [], np.array([]), self.mode

        # Sort by y (descending - bottom to top)
        measurements.sort(key=lambda m: -m[0])

        # Extract arrays
        y_samples = np.array([m[0] for m in measurements], dtype=np.float32)
        x_samples = np.array([m[1] for m in measurements], dtype=np.float32)
        w_samples = np.array([m[2] for m in measurements], dtype=np.float32)

        # Normalize y for numerical stability
        y_norm = self._normalize_y(y_samples)

        # Weighted least squares (trust bottom points more)
        weights = 1.0 + self.weight_bias * y_norm
        W = np.diag(weights)

        # Design matrix: [1, y, y^2]
        X = np.column_stack((np.ones_like(y_norm), y_norm, y_norm**2))

        try:
            # Solve: (X^T W X)^-1 X^T W x
            XTW = X.T @ W

            # Current frame coefficients
            curr_coeffs_center = np.linalg.solve(XTW @ X, XTW @ x_samples)

            if self.fit_width_poly:
                curr_coeffs_width = np.linalg.solve(XTW @ X, XTW @ w_samples)
            else:
                # Use constant width (mean)
                curr_coeffs_width = np.array([np.mean(w_samples), 0, 0], dtype=np.float32)

            # Apply temporal smoothing (EMA)
            coeffs_center = self._apply_temporal_smoothing(
                curr_coeffs_center,
                self.prev_coeffs_center
            )
            coeffs_width = self._apply_temporal_smoothing(
                curr_coeffs_width,
                self.prev_coeffs_width
            )

            # Apply straight-line locking
            coeffs_center = self._apply_straight_line_locking(coeffs_center)

            # Also apply to width if in straight mode
            if self.mode == "Straight" and self.reduce_width_in_straight:
                coeffs_width[2] = coeffs_width[2] * self.reduction_factor

            # Update previous coefficients
            self.prev_coeffs_center = coeffs_center.copy()
            self.prev_coeffs_width = coeffs_width.copy()

        except np.linalg.LinAlgError:
            return [], np.array([]), self.mode

        # Generate smooth curve with dense points
        y_dense = np.arange(
            self.end_y,
            self.start_y + 1,
            self.output_step,
            dtype=np.float32
        )
        y_dense_norm = self._normalize_y(y_dense)

        # Calculate center: x = a*y^2 + b*y + c
        center_dense = (
            coeffs_center[0] +
            coeffs_center[1] * y_dense_norm +
            coeffs_center[2] * y_dense_norm**2
        )

        # Calculate width: w = a_w*y^2 + b_w*y + c_w
        width_dense = (
            coeffs_width[0] +
            coeffs_width[1] * y_dense_norm +
            coeffs_width[2] * y_dense_norm**2
        )

        # Clamp width to reasonable range
        width_dense = np.clip(width_dense, self.clamp_min, self.clamp_max)

        # Create center points list
        center_points = [(int(x), int(y)) for x, y in zip(center_dense, y_dense)]

        return center_points, width_dense, self.mode
