#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV Path Analyzer Module
========================
Phase 6: Analyze rail paths in Bird's Eye View space for
improved ego-path selection and direction classification.

Features:
- Extract rail center lines from BEV segmentation mask
- Compute path direction relative to vehicle heading (Y-axis)
- Classify path direction (STRAIGHT, LEFT, RIGHT)
- Select ego-path (the path vehicle should follow)
- Junction detection and classification

Author: RailSafeNet Team
Date: 2025-12-19
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pathlib import Path
from scipy import ndimage


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization (thinning) without cv2.ximgproc.

    Uses iterative erosion and opening to extract skeleton.

    Args:
        mask: Binary mask (0 or 255)

    Returns:
        Skeleton mask (0 or 255)
    """
    # Normalize to binary (0 or 1)
    img = (mask > 0).astype(np.uint8)

    # Initialize skeleton
    skeleton = np.zeros_like(img)

    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Erode the image
        eroded = cv2.erode(img, kernel)

        # Opening of eroded image
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)

        # Subtract opened from eroded
        temp = cv2.subtract(eroded, opened)

        # Union with skeleton
        skeleton = cv2.bitwise_or(skeleton, temp)

        # Update image for next iteration
        img = eroded.copy()

        # Stop when image is fully eroded
        if cv2.countNonZero(img) == 0:
            break

    return skeleton * 255


class PathDirection(Enum):
    """Direction classification for rail paths."""
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    UNKNOWN = 3


class JunctionType(Enum):
    """Junction type classification."""
    NONE = 0          # No junction (single path)
    Y_SPLIT = 1       # Y-junction (1 path splits into 2)
    MERGE = 2         # Merge (2 paths merge into 1)
    CROSSING = 3      # Crossing (paths intersect)
    PARALLEL = 4      # Parallel tracks


@dataclass
class BEVRailPath:
    """
    Represents a single rail path in BEV space.

    Attributes:
        path_id: Unique identifier for this path
        points: List of (x, y) coordinates along the path center line
        angle: Average angle from Y-axis (degrees), positive = right, negative = left
        curvature: Average curvature of the path
        direction: Classified direction (STRAIGHT, LEFT, RIGHT)
        confidence: Confidence score (0-1) for this path detection
        width: Average path width in pixels
    """
    path_id: int
    points: np.ndarray  # (N, 2) array of (x, y) coordinates
    angle: float = 0.0
    curvature: float = 0.0
    direction: PathDirection = PathDirection.UNKNOWN
    confidence: float = 0.0
    width: float = 0.0
    is_ego_path: bool = False


@dataclass
class PathAnalysisConfig:
    """Configuration for path analysis."""
    # Direction thresholds (degrees from Y-axis)
    straight_threshold: float = 10.0  # |angle| < threshold = STRAIGHT
    turn_threshold: float = 30.0      # |angle| > threshold = definite turn

    # Minimum path length (pixels in BEV)
    min_path_length: int = 50

    # Curvature threshold for straight detection
    curvature_threshold: float = 0.01

    # Rail class IDs in segmentation mask
    rail_class_ids: List[int] = field(default_factory=lambda: [4, 9])

    # Morphology kernel size for cleaning
    morph_kernel_size: int = 5

    # Minimum area for valid path component
    min_area: int = 500

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PathAnalysisConfig':
        """Create config from dictionary."""
        path_cfg = config.get('path_analysis', {})

        return cls(
            straight_threshold=path_cfg.get('straight_threshold', 10.0),
            turn_threshold=path_cfg.get('turn_threshold', 30.0),
            min_path_length=path_cfg.get('min_path_length', 50),
            curvature_threshold=path_cfg.get('curvature_threshold', 0.01),
            rail_class_ids=path_cfg.get('rail_class_ids', [4, 9]),
            morph_kernel_size=path_cfg.get('morph_kernel_size', 5),
            min_area=path_cfg.get('min_area', 500)
        )


class BEVPathAnalyzer:
    """
    Analyze rail paths in Bird's Eye View space.

    Extracts rail center lines, computes directions, and selects
    the ego-path (the path the vehicle should follow).
    """

    def __init__(self, config: Optional[PathAnalysisConfig] = None,
                 config_path: Optional[str] = None):
        """
        Initialize BEV Path Analyzer.

        Args:
            config: PathAnalysisConfig object
            config_path: Path to YAML config file
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            self.config = PathAnalysisConfig()

        # Morphology kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )

        # Path counter
        self._path_id_counter = 0

        print(f"✓ BEV Path Analyzer initialized [angle thresholds: ±{self.config.straight_threshold}°]")

    def _load_config(self, config_path: str) -> PathAnalysisConfig:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            print(f"⚠️ Path analysis config not found: {config_path}, using defaults")
            return PathAnalysisConfig()

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return PathAnalysisConfig.from_dict(config_dict)

    def _get_next_path_id(self) -> int:
        """Get next unique path ID."""
        self._path_id_counter += 1
        return self._path_id_counter

    def extract_rail_mask(self, bev_mask: np.ndarray) -> np.ndarray:
        """
        Extract binary rail mask from BEV segmentation mask.

        Args:
            bev_mask: BEV segmentation mask with class IDs

        Returns:
            Binary mask where rail pixels are 255
        """
        # Create binary mask for rail classes
        rail_mask = np.zeros_like(bev_mask, dtype=np.uint8)
        for class_id in self.config.rail_class_ids:
            rail_mask[bev_mask == class_id] = 255

        # Morphological operations to clean up
        rail_mask = cv2.morphologyEx(rail_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        rail_mask = cv2.morphologyEx(rail_mask, cv2.MORPH_OPEN, self.morph_kernel)

        return rail_mask

    def extract_rail_paths(self, bev_mask: np.ndarray) -> List[BEVRailPath]:
        """
        Extract individual rail paths from BEV segmentation mask.

        Args:
            bev_mask: BEV segmentation mask with class IDs

        Returns:
            List of BEVRailPath objects
        """
        # Get binary rail mask
        rail_mask = self.extract_rail_mask(bev_mask)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            rail_mask, connectivity=8
        )

        paths = []

        # Process each component (skip background label 0)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            # Filter by minimum area
            if area < self.config.min_area:
                continue

            # Create mask for this component
            component_mask = (labels == label).astype(np.uint8) * 255

            # Extract skeleton/center line using thinning
            skeleton = skeletonize(component_mask)

            # Get skeleton points
            skeleton_pts = np.column_stack(np.where(skeleton > 0))  # (y, x) format
            if len(skeleton_pts) < self.config.min_path_length:
                continue

            # Convert to (x, y) format and sort by y (bottom to top in BEV)
            points = skeleton_pts[:, ::-1]  # (x, y)
            sorted_indices = np.argsort(points[:, 1])[::-1]  # Sort by y descending
            points = points[sorted_indices]

            # Create path object
            path = BEVRailPath(
                path_id=self._get_next_path_id(),
                points=points.astype(np.float32)
            )

            # Compute path properties
            self._compute_path_properties(path, component_mask)

            paths.append(path)

        return paths

    def _compute_path_properties(self, path: BEVRailPath, mask: np.ndarray):
        """
        Compute angle, curvature, and direction for a path.

        Args:
            path: BEVRailPath to update
            mask: Binary mask of this path
        """
        points = path.points

        if len(points) < 3:
            path.direction = PathDirection.UNKNOWN
            return

        # Compute angle using linear regression on path points
        # In BEV: Y-axis is forward, X-axis is lateral
        # Angle = atan2(dx, dy) where dy is always positive (forward)

        # Use points from bottom half of path (closer to vehicle, more reliable)
        n_points = len(points)
        bottom_half = points[:n_points // 2] if n_points > 10 else points

        if len(bottom_half) >= 2:
            # Fit line to bottom portion
            x = bottom_half[:, 0]
            y = bottom_half[:, 1]

            # Check if y values have sufficient variation for fitting
            y_range = np.ptp(y)  # peak-to-peak range
            if y_range > 5:  # Need at least 5 pixels of y variation
                try:
                    # Linear fit: x = a*y + b
                    with np.errstate(all='ignore'):  # Suppress polyfit warnings
                        coeffs = np.polyfit(y, x, 1)
                    slope = coeffs[0]  # dx/dy

                    # Angle from Y-axis (forward direction)
                    # Positive angle = path goes right, Negative = path goes left
                    angle_rad = np.arctan(slope)
                    path.angle = np.degrees(angle_rad)
                except (np.linalg.LinAlgError, ValueError):
                    path.angle = 0.0
            else:
                # Not enough y variation, assume straight
                path.angle = 0.0

        # Compute curvature using quadratic fit
        if len(points) >= 5:
            x = points[:, 0]
            y = points[:, 1]

            # Check if y values have sufficient variation for quadratic fitting
            y_range = np.ptp(y)
            if y_range > 10:  # Need at least 10 pixels of y variation for quadratic
                try:
                    # Quadratic fit: x = a*y^2 + b*y + c
                    with np.errstate(all='ignore'):  # Suppress polyfit warnings
                        coeffs = np.polyfit(y, x, 2)
                    # Curvature is related to the quadratic coefficient
                    path.curvature = abs(coeffs[0])
                except (np.linalg.LinAlgError, ValueError):
                    path.curvature = 0.0
            else:
                path.curvature = 0.0

        # Classify direction based on angle
        path.direction = self._classify_direction(path.angle, path.curvature)

        # Estimate width from mask
        path.width = self._estimate_path_width(mask)

        # Confidence based on path length and consistency
        path.confidence = min(1.0, len(points) / 200.0)

    def _classify_direction(self, angle: float, curvature: float) -> PathDirection:
        """
        Classify path direction based on angle and curvature.

        Args:
            angle: Angle from Y-axis in degrees
            curvature: Path curvature

        Returns:
            PathDirection enum
        """
        abs_angle = abs(angle)

        # Check if path is straight (both low angle and low curvature)
        if abs_angle < self.config.straight_threshold:
            if curvature < self.config.curvature_threshold:
                return PathDirection.STRAIGHT

        # Classify by angle direction
        if angle < -self.config.straight_threshold:
            return PathDirection.LEFT
        elif angle > self.config.straight_threshold:
            return PathDirection.RIGHT
        else:
            return PathDirection.STRAIGHT

    def _estimate_path_width(self, mask: np.ndarray) -> float:
        """
        Estimate average path width from mask.

        Args:
            mask: Binary mask of the path

        Returns:
            Average width in pixels
        """
        # Use distance transform to find width at each point
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Get maximum distance along skeleton (half-width)
        skeleton = skeletonize(mask)
        skeleton_distances = dist_transform[skeleton > 0]

        if len(skeleton_distances) > 0:
            return np.mean(skeleton_distances) * 2  # Full width
        return 0.0

    def select_ego_path(self, paths: List[BEVRailPath],
                        bev_width: int) -> Optional[BEVRailPath]:
        """
        Select the ego-path (the path the vehicle should follow).

        Selection criteria:
        1. Prefer STRAIGHT paths
        2. Among straight paths, prefer the one closest to center
        3. If no straight path, prefer the one with smallest angle

        Args:
            paths: List of detected BEVRailPath objects
            bev_width: Width of BEV image (for center calculation)

        Returns:
            Selected ego-path or None if no valid path
        """
        if not paths:
            return None

        if len(paths) == 1:
            paths[0].is_ego_path = True
            return paths[0]

        center_x = bev_width / 2.0

        # Score each path
        scored_paths = []
        for path in paths:
            # Get x-position at bottom of path (closest to vehicle)
            if len(path.points) > 0:
                bottom_x = path.points[0, 0]
            else:
                continue

            # Calculate distance from center
            center_dist = abs(bottom_x - center_x)

            # Score calculation:
            # - Prefer straight paths (direction score)
            # - Prefer paths closer to center (position score)
            # - Prefer smaller absolute angle

            direction_score = 0
            if path.direction == PathDirection.STRAIGHT:
                direction_score = 100
            elif path.direction == PathDirection.LEFT:
                direction_score = 50
            elif path.direction == PathDirection.RIGHT:
                direction_score = 50
            else:
                direction_score = 0

            position_score = max(0, 100 - center_dist)
            angle_score = max(0, 90 - abs(path.angle))

            total_score = direction_score * 2 + position_score + angle_score
            scored_paths.append((total_score, path))

        # Sort by score (descending)
        scored_paths.sort(key=lambda x: x[0], reverse=True)

        # Mark and return best path
        best_path = scored_paths[0][1]
        best_path.is_ego_path = True
        return best_path

    def detect_junction(self, paths: List[BEVRailPath]) -> JunctionType:
        """
        Detect junction type from detected paths.

        Args:
            paths: List of detected BEVRailPath objects

        Returns:
            JunctionType enum
        """
        num_paths = len(paths)

        if num_paths <= 1:
            return JunctionType.NONE

        # Check if paths are parallel
        angles = [p.angle for p in paths]
        angle_diff = max(angles) - min(angles)

        if angle_diff < 15:  # Paths are roughly parallel
            return JunctionType.PARALLEL

        # Check if paths diverge or converge
        # Get bottom points (near vehicle) for each path
        bottom_xs = []
        for p in paths:
            if len(p.points) > 0:
                bottom_xs.append(p.points[0, 0])

        # Get top points (far from vehicle)
        top_xs = []
        for p in paths:
            if len(p.points) > 0:
                top_xs.append(p.points[-1, 0])

        bottom_spread = max(bottom_xs) - min(bottom_xs) if bottom_xs else 0
        top_spread = max(top_xs) - min(top_xs) if top_xs else 0

        if bottom_spread < top_spread:
            # Paths are diverging (Y-split)
            return JunctionType.Y_SPLIT
        elif bottom_spread > top_spread:
            # Paths are converging (merge)
            return JunctionType.MERGE
        else:
            # Paths may be crossing
            return JunctionType.CROSSING

    def analyze_frame(self, bev_mask: np.ndarray,
                      bev_width: int) -> Dict[str, Any]:
        """
        Full analysis of a single BEV frame.

        Args:
            bev_mask: BEV segmentation mask
            bev_width: Width of BEV image

        Returns:
            Dictionary with analysis results
        """
        # Extract paths
        paths = self.extract_rail_paths(bev_mask)

        # Select ego-path
        ego_path = self.select_ego_path(paths, bev_width)

        # Detect junction
        junction = self.detect_junction(paths)

        return {
            'paths': paths,
            'ego_path': ego_path,
            'junction_type': junction,
            'num_paths': len(paths),
            'ego_direction': ego_path.direction if ego_path else PathDirection.UNKNOWN,
            'ego_angle': ego_path.angle if ego_path else 0.0
        }

    def draw_paths_on_bev(self, bev_image: np.ndarray,
                          paths: List[BEVRailPath],
                          ego_path: Optional[BEVRailPath] = None) -> np.ndarray:
        """
        Visualize detected paths on BEV image.

        Args:
            bev_image: BEV image to draw on
            paths: List of detected paths
            ego_path: Selected ego-path (drawn differently)

        Returns:
            Annotated BEV image
        """
        output = bev_image.copy()

        # Color mapping for directions
        direction_colors = {
            PathDirection.STRAIGHT: (0, 255, 0),    # Green
            PathDirection.LEFT: (255, 255, 0),      # Cyan
            PathDirection.RIGHT: (0, 255, 255),     # Yellow
            PathDirection.UNKNOWN: (128, 128, 128)  # Gray
        }

        for path in paths:
            points = path.points.astype(np.int32)
            if len(points) < 2:
                continue

            # Choose color based on direction
            color = direction_colors[path.direction]

            # Draw thicker line for ego-path
            thickness = 4 if path.is_ego_path else 2

            # Draw path line
            for i in range(len(points) - 1):
                pt1 = tuple(points[i])
                pt2 = tuple(points[i + 1])
                cv2.line(output, pt1, pt2, color, thickness)

            # Draw start point (bottom, near vehicle)
            cv2.circle(output, tuple(points[0]), 8, color, -1)

            # Draw path info
            if len(points) > 0:
                text_pos = (int(points[0, 0]) + 10, int(points[0, 1]) - 10)
                info_text = f"{path.direction.name[:1]} {path.angle:.1f}°"
                cv2.putText(output, info_text, text_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw ego-path indicator
        if ego_path is not None and len(ego_path.points) > 0:
            cv2.putText(output, "EGO", (int(ego_path.points[0, 0]) - 20, int(ego_path.points[0, 1]) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return output


def get_imm_prior_from_bev(bev_direction: PathDirection) -> np.ndarray:
    """
    Convert BEV path direction to IMM model probability prior.

    This function is used to integrate BEV analysis with IMM-SVSF filter.
    BEV direction provides geometric prior for IMM mode selection.

    Args:
        bev_direction: PathDirection from BEV analysis

    Returns:
        np.ndarray of shape (3,) with [straight_prob, left_prob, right_prob]
    """
    if bev_direction == PathDirection.STRAIGHT:
        # Strong prior for straight
        return np.array([0.8, 0.1, 0.1], dtype=np.float32)
    elif bev_direction == PathDirection.LEFT:
        # Prior for left turn
        return np.array([0.2, 0.7, 0.1], dtype=np.float32)
    elif bev_direction == PathDirection.RIGHT:
        # Prior for right turn
        return np.array([0.2, 0.1, 0.7], dtype=np.float32)
    else:
        # Uniform prior if unknown
        return np.array([0.34, 0.33, 0.33], dtype=np.float32)


if __name__ == "__main__":
    # Test with synthetic data
    print("BEV Path Analyzer - Test Mode")

    # Create synthetic BEV mask with two diverging paths
    bev_height, bev_width = 600, 400
    test_mask = np.zeros((bev_height, bev_width), dtype=np.uint8)

    # Draw left path (class 4)
    for y in range(bev_height):
        x = int(bev_width * 0.4 - y * 0.1)
        if 0 <= x < bev_width:
            cv2.circle(test_mask, (x, y), 10, 4, -1)

    # Draw right/straight path (class 4)
    for y in range(bev_height):
        x = int(bev_width * 0.5 + y * 0.02)
        if 0 <= x < bev_width:
            cv2.circle(test_mask, (x, y), 10, 4, -1)

    # Create analyzer
    analyzer = BEVPathAnalyzer()

    # Analyze
    result = analyzer.analyze_frame(test_mask, bev_width)

    print(f"Detected {result['num_paths']} paths")
    print(f"Junction type: {result['junction_type'].name}")
    if result['ego_path']:
        print(f"Ego path: Direction={result['ego_direction'].name}, Angle={result['ego_angle']:.1f}°")

    # Visualize
    vis_mask = cv2.cvtColor(test_mask * 20, cv2.COLOR_GRAY2BGR)
    vis_mask = analyzer.draw_paths_on_bev(vis_mask, result['paths'], result['ego_path'])

    cv2.imshow('BEV Path Analysis Test', vis_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
