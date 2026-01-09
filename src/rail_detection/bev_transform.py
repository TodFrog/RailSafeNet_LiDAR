#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bird's Eye View (BEV) Transformation Module
============================================
Phase 6: Convert perspective camera view to top-down bird's eye view
for improved rail path direction analysis.

Features:
- Homography-based perspective transformation
- Point coordinate transformation (camera <-> BEV)
- Interactive calibration tool
- Cached homography matrix for performance

Author: RailSafeNet Team
Date: 2025-12-19
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class InterpolationMethod(Enum):
    """Interpolation methods for warp transformation."""
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    LANCZOS = cv2.INTER_LANCZOS4


@dataclass
class BEVConfig:
    """Configuration for BEV transformation."""
    # Source points in camera image (trapezoid - 4 corners)
    src_top_left: Tuple[int, int]
    src_top_right: Tuple[int, int]
    src_bottom_right: Tuple[int, int]
    src_bottom_left: Tuple[int, int]

    # BEV output dimensions
    bev_width: int
    bev_height: int

    # Interpolation method
    interpolation: InterpolationMethod = InterpolationMethod.LINEAR

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BEVConfig':
        """Create BEVConfig from dictionary (loaded from YAML)."""
        bev_cfg = config.get('bev_transform', {})
        src_pts = bev_cfg.get('source_points', {})
        out_size = bev_cfg.get('output_size', {})

        interp_str = bev_cfg.get('interpolation', 'linear').lower()
        interp_map = {
            'nearest': InterpolationMethod.NEAREST,
            'linear': InterpolationMethod.LINEAR,
            'cubic': InterpolationMethod.CUBIC,
            'lanczos': InterpolationMethod.LANCZOS
        }

        return cls(
            src_top_left=tuple(src_pts.get('top_left', [400, 400])),
            src_top_right=tuple(src_pts.get('top_right', [1520, 400])),
            src_bottom_right=tuple(src_pts.get('bottom_right', [1920, 1080])),
            src_bottom_left=tuple(src_pts.get('bottom_left', [0, 1080])),
            bev_width=out_size.get('width', 400),
            bev_height=out_size.get('height', 600),
            interpolation=interp_map.get(interp_str, InterpolationMethod.LINEAR)
        )


class BEVTransformer:
    """
    Bird's Eye View Transformer.

    Transforms camera images and point coordinates between
    perspective view and top-down bird's eye view using homography.
    """

    def __init__(self, config: Optional[BEVConfig] = None, config_path: Optional[str] = None):
        """
        Initialize BEV Transformer.

        Args:
            config: BEVConfig object with transformation parameters
            config_path: Path to YAML config file (used if config is None)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            # Default config for 1920x1080 camera
            self.config = BEVConfig(
                src_top_left=(400, 400),
                src_top_right=(1520, 400),
                src_bottom_right=(1920, 1080),
                src_bottom_left=(0, 1080),
                bev_width=400,
                bev_height=600
            )

        # Compute homography matrices
        self.H = self._compute_homography()
        self.H_inv = np.linalg.inv(self.H)

        # Cache output size
        self.bev_size = (self.config.bev_width, self.config.bev_height)

        print(f"✓ BEV Transformer initialized [Output: {self.bev_size[0]}x{self.bev_size[1]}]")

    def _load_config(self, config_path: str) -> BEVConfig:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            print(f"⚠️ BEV config not found: {config_path}, using defaults")
            return BEVConfig(
                src_top_left=(400, 400),
                src_top_right=(1520, 400),
                src_bottom_right=(1920, 1080),
                src_bottom_left=(0, 1080),
                bev_width=400,
                bev_height=600
            )

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return BEVConfig.from_dict(config_dict)

    def _compute_homography(self) -> np.ndarray:
        """
        Compute homography matrix from source (camera) to destination (BEV).

        Source: Trapezoid in camera image
        Destination: Rectangle in BEV

        Returns:
            3x3 homography matrix
        """
        # Source points (trapezoid in camera image)
        src_pts = np.float32([
            list(self.config.src_top_left),
            list(self.config.src_top_right),
            list(self.config.src_bottom_right),
            list(self.config.src_bottom_left)
        ])

        # Destination points (rectangle in BEV)
        dst_pts = np.float32([
            [0, 0],                                              # Top-left
            [self.config.bev_width - 1, 0],                      # Top-right
            [self.config.bev_width - 1, self.config.bev_height - 1],  # Bottom-right
            [0, self.config.bev_height - 1]                      # Bottom-left
        ])

        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H

    def warp_to_bev(self, image: np.ndarray) -> np.ndarray:
        """
        Transform camera image to bird's eye view.

        Args:
            image: Input camera image (H x W x C)

        Returns:
            BEV transformed image (bev_height x bev_width x C)
        """
        return cv2.warpPerspective(
            image,
            self.H,
            self.bev_size,
            flags=self.config.interpolation.value
        )

    def warp_mask_to_bev(self, mask: np.ndarray) -> np.ndarray:
        """
        Transform segmentation mask to bird's eye view.

        Uses NEAREST interpolation to preserve class labels.

        Args:
            mask: Input segmentation mask (H x W)

        Returns:
            BEV transformed mask (bev_height x bev_width)
        """
        return cv2.warpPerspective(
            mask,
            self.H,
            self.bev_size,
            flags=cv2.INTER_NEAREST
        )

    def warp_points_to_bev(self, points: np.ndarray) -> np.ndarray:
        """
        Transform point coordinates from camera space to BEV space.

        Args:
            points: Input points (N x 2) in camera coordinates

        Returns:
            Transformed points (N x 2) in BEV coordinates
        """
        if len(points) == 0:
            return points

        # Reshape for perspectiveTransform: (N, 1, 2)
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed.reshape(-1, 2)

    def warp_points_from_bev(self, bev_points: np.ndarray) -> np.ndarray:
        """
        Transform point coordinates from BEV space to camera space.

        Args:
            bev_points: Input points (N x 2) in BEV coordinates

        Returns:
            Transformed points (N x 2) in camera coordinates
        """
        if len(bev_points) == 0:
            return bev_points

        # Reshape for perspectiveTransform: (N, 1, 2)
        pts = bev_points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H_inv)
        return transformed.reshape(-1, 2)

    def warp_from_bev(self, bev_image: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Inverse transform: BEV image back to camera perspective.

        Args:
            bev_image: BEV image (bev_height x bev_width x C)
            original_size: Original camera image size (width, height)

        Returns:
            Image in camera perspective (H x W x C)
        """
        return cv2.warpPerspective(
            bev_image,
            self.H_inv,
            original_size,
            flags=self.config.interpolation.value
        )

    def draw_source_region(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 2) -> np.ndarray:
        """
        Draw the source trapezoid region on camera image.

        Args:
            image: Input camera image
            color: Line color (BGR)
            thickness: Line thickness

        Returns:
            Image with drawn trapezoid
        """
        output = image.copy()
        pts = np.array([
            self.config.src_top_left,
            self.config.src_top_right,
            self.config.src_bottom_right,
            self.config.src_bottom_left
        ], dtype=np.int32)

        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw corner markers
        for i, pt in enumerate(pts):
            cv2.circle(output, tuple(pt), 8, color, -1)
            cv2.putText(output, f"P{i+1}", (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output

    def draw_grid_on_bev(self, bev_image: np.ndarray, spacing: int = 50,
                         color: Tuple[int, int, int] = (100, 100, 100)) -> np.ndarray:
        """
        Draw grid lines on BEV image for reference.

        Args:
            bev_image: BEV image
            spacing: Grid spacing in pixels
            color: Grid line color (BGR)

        Returns:
            BEV image with grid
        """
        output = bev_image.copy()
        h, w = output.shape[:2]

        # Vertical lines
        for x in range(0, w, spacing):
            cv2.line(output, (x, 0), (x, h - 1), color, 1)

        # Horizontal lines
        for y in range(0, h, spacing):
            cv2.line(output, (0, y), (w - 1, y), color, 1)

        # Draw center line (vehicle forward direction)
        center_x = w // 2
        cv2.line(output, (center_x, 0), (center_x, h - 1), (0, 255, 0), 2)

        return output

    def update_source_points(self, top_left: Tuple[int, int], top_right: Tuple[int, int],
                             bottom_right: Tuple[int, int], bottom_left: Tuple[int, int]):
        """
        Update source points and recompute homography.

        Args:
            top_left: New top-left point
            top_right: New top-right point
            bottom_right: New bottom-right point
            bottom_left: New bottom-left point
        """
        self.config.src_top_left = top_left
        self.config.src_top_right = top_right
        self.config.src_bottom_right = bottom_right
        self.config.src_bottom_left = bottom_left

        self.H = self._compute_homography()
        self.H_inv = np.linalg.inv(self.H)

    def get_source_points(self) -> List[Tuple[int, int]]:
        """Get current source points."""
        return [
            self.config.src_top_left,
            self.config.src_top_right,
            self.config.src_bottom_right,
            self.config.src_bottom_left
        ]

    def save_config(self, config_path: str):
        """
        Save current configuration to YAML file.

        Args:
            config_path: Path to save configuration
        """
        config_dict = {
            'bev_transform': {
                'source_points': {
                    'top_left': list(self.config.src_top_left),
                    'top_right': list(self.config.src_top_right),
                    'bottom_right': list(self.config.src_bottom_right),
                    'bottom_left': list(self.config.src_bottom_left)
                },
                'output_size': {
                    'width': self.config.bev_width,
                    'height': self.config.bev_height
                },
                'interpolation': self.config.interpolation.name.lower()
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        print(f"✓ BEV config saved to {config_path}")


class BEVCalibrator:
    """
    Interactive calibration tool for BEV transformation.

    Allows user to select 4 source points by clicking on the image.
    Bottom points are fixed at the image bottom edge for stability.
    """

    def __init__(self, image: np.ndarray, bev_width: int = 400, bev_height: int = 600,
                 display_scale: float = 0.5):
        """
        Initialize calibrator.

        Args:
            image: Sample camera image for calibration
            bev_width: Desired BEV output width
            bev_height: Desired BEV output height
            display_scale: Scale factor for display (0.5 = half size)
        """
        self.image = image.copy()
        self.orig_h, self.orig_w = image.shape[:2]
        self.display_scale = display_scale
        self.bev_width = bev_width
        self.bev_height = bev_height

        # Only 2 points needed: Top-Left, Top-Right (bottom fixed at image edge)
        self.points: List[Tuple[int, int]] = []
        self.point_names = ['Top-Left', 'Top-Right']
        self.window_name = 'BEV Calibration - Click 2 top corners (TL, TR)'
        self.bev_transformer: Optional[BEVTransformer] = None

        # Fixed bottom points at image bottom edge
        self.bottom_left = (0, self.orig_h - 1)
        self.bottom_right = (self.orig_w - 1, self.orig_h - 1)

        # Initialize display image (scaled)
        self.display_image = self._scale_image(image)

    def _scale_image(self, image: np.ndarray) -> np.ndarray:
        """Scale image for display."""
        new_w = int(image.shape[1] * self.display_scale)
        new_h = int(image.shape[0] * self.display_scale)
        return cv2.resize(image, (new_w, new_h))

    def _display_to_orig(self, x: int, y: int) -> Tuple[int, int]:
        """Convert display coordinates to original image coordinates."""
        orig_x = int(x / self.display_scale)
        orig_y = int(y / self.display_scale)
        return (orig_x, orig_y)

    def _orig_to_display(self, x: int, y: int) -> Tuple[int, int]:
        """Convert original image coordinates to display coordinates."""
        disp_x = int(x * self.display_scale)
        disp_y = int(y * self.display_scale)
        return (disp_x, disp_y)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:  # Only 2 points needed (top corners)
                # Convert display coords to original coords
                orig_pt = self._display_to_orig(x, y)
                self.points.append(orig_pt)
                self._update_display()

                if len(self.points) == 2:
                    self._create_transformer()

    def _update_display(self):
        """Update display with current points (scaled for display)."""
        # Work on original size first, then scale
        display_orig = self.image.copy()

        # Draw fixed bottom points (always visible)
        bl_color = (255, 165, 0)  # Orange for fixed points
        br_color = (255, 165, 0)
        cv2.circle(display_orig, self.bottom_left, 8, bl_color, -1)
        cv2.circle(display_orig, self.bottom_right, 8, br_color, -1)
        cv2.putText(display_orig, "BL(fixed)", (self.bottom_left[0] + 10, self.bottom_left[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, bl_color, 2)
        cv2.putText(display_orig, "BR(fixed)", (self.bottom_right[0] - 100, self.bottom_right[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, br_color, 2)

        # Draw user-selected top points
        for i, pt in enumerate(self.points):
            color = (0, 255, 0)
            cv2.circle(display_orig, pt, 8, color, -1)
            cv2.putText(display_orig, f"{i+1}:{self.point_names[i]}",
                       (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw polygon connecting all 4 points (TL, TR, BR, BL)
        if len(self.points) >= 1:
            all_pts = self.points.copy()
            if len(self.points) == 2:
                all_pts.extend([self.bottom_right, self.bottom_left])
            pts = np.array(all_pts, dtype=np.int32)
            cv2.polylines(display_orig, [pts], isClosed=len(self.points) == 2,
                         color=(0, 255, 0), thickness=2)

        # Instructions
        if len(self.points) < 2:
            text = f"Click point {len(self.points)+1}: {self.point_names[len(self.points)]}"
        else:
            text = "Press 's' to save, 'r' to reset, 'q' to quit"
        cv2.putText(display_orig, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Scale for display
        self.display_image = self._scale_image(display_orig)

    def _create_transformer(self):
        """Create BEV transformer with selected points (2 top + 2 fixed bottom)."""
        if len(self.points) == 2:
            config = BEVConfig(
                src_top_left=self.points[0],
                src_top_right=self.points[1],
                src_bottom_right=self.bottom_right,
                src_bottom_left=self.bottom_left,
                bev_width=self.bev_width,
                bev_height=self.bev_height
            )
            self.bev_transformer = BEVTransformer(config=config)

    def run(self, save_path: Optional[str] = None) -> Optional[BEVTransformer]:
        """
        Run interactive calibration.

        Args:
            save_path: Path to save configuration (optional)

        Returns:
            Configured BEVTransformer or None if cancelled
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._update_display()

        while True:
            cv2.imshow(self.window_name, self.display_image)

            # Show BEV preview if transformer is ready (also scaled)
            if self.bev_transformer is not None:
                bev_image = self.bev_transformer.warp_to_bev(self.image)
                bev_image = self.bev_transformer.draw_grid_on_bev(bev_image)
                # Scale BEV preview to match display scale
                bev_scaled = self._scale_image(bev_image)
                cv2.imshow('BEV Preview', bev_scaled)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.points = []
                self.bev_transformer = None
                self._update_display()
                cv2.destroyWindow('BEV Preview')
            elif key == ord('s') and self.bev_transformer is not None:
                if save_path:
                    self.bev_transformer.save_config(save_path)
                break

        cv2.destroyAllWindows()
        return self.bev_transformer


def create_default_bev_config(image_width: int = 1920, image_height: int = 1080,
                               bev_width: int = 400, bev_height: int = 600) -> BEVConfig:
    """
    Create default BEV configuration for given image size.

    Default trapezoid covers the road region in front of the vehicle.

    Args:
        image_width: Camera image width
        image_height: Camera image height
        bev_width: Desired BEV output width
        bev_height: Desired BEV output height

    Returns:
        BEVConfig with reasonable defaults
    """
    # Default trapezoid: narrower at top (horizon), wider at bottom (near vehicle)
    # Top corners at ~40% from sides at 40% height
    # Bottom corners at image edges at bottom
    top_margin = int(image_width * 0.25)
    top_y = int(image_height * 0.45)

    return BEVConfig(
        src_top_left=(top_margin, top_y),
        src_top_right=(image_width - top_margin, top_y),
        src_bottom_right=(image_width, image_height),
        src_bottom_left=(0, image_height),
        bev_width=bev_width,
        bev_height=bev_height
    )


# Module-level convenience functions
def warp_image_to_bev(image: np.ndarray, config: BEVConfig) -> np.ndarray:
    """Convenience function to warp image to BEV."""
    transformer = BEVTransformer(config=config)
    return transformer.warp_to_bev(image)


def warp_points_to_bev(points: np.ndarray, config: BEVConfig) -> np.ndarray:
    """Convenience function to warp points to BEV."""
    transformer = BEVTransformer(config=config)
    return transformer.warp_points_to_bev(points)


if __name__ == "__main__":
    # Test BEV transformation
    import argparse

    parser = argparse.ArgumentParser(description='BEV Transform Test')
    parser.add_argument('--image', type=str, help='Test image path')
    parser.add_argument('--calibrate', action='store_true', help='Run interactive calibration')
    parser.add_argument('--config', type=str, default='config/bev_config.yaml',
                       help='Config file path')
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"❌ Cannot load image: {args.image}")
            exit(1)

        if args.calibrate:
            calibrator = BEVCalibrator(image)
            transformer = calibrator.run(save_path=args.config)
        else:
            # Use default config
            h, w = image.shape[:2]
            config = create_default_bev_config(w, h)
            transformer = BEVTransformer(config=config)

            # Show result
            camera_with_region = transformer.draw_source_region(image)
            bev = transformer.warp_to_bev(image)
            bev_with_grid = transformer.draw_grid_on_bev(bev)

            cv2.imshow('Camera View', camera_with_region)
            cv2.imshow('BEV', bev_with_grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Usage: python bev_transform.py --image <path> [--calibrate] [--config <path>]")
