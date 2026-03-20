#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini BEV Renderer
=================
Picture-in-Picture BEV view for direction visualization.

Provides:
- Compact BEV overlay in corner
- Path visualization with ego-path highlight
- Direction arrow and junction indicator
- Grid overlay for distance reference

Author: RailSafeNet Team
Date: 2025-01-06
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum

try:
    from .bev_path_analyzer import PathDirection, JunctionType
except ImportError:
    class PathDirection(Enum):
        STRAIGHT = 0
        LEFT = 1
        RIGHT = 2
        UNKNOWN = 3

    class JunctionType(Enum):
        NONE = 0
        Y_SPLIT = 1
        MERGE = 2
        CROSSING = 3
        PARALLEL = 4


class MiniBEVRenderer:
    """
    Renders a mini BEV (Bird's Eye View) as a Picture-in-Picture overlay.

    Shows:
    - Rail paths in BEV space
    - Ego path (selected path) highlighted
    - Direction indicator
    - Junction type label
    """

    def __init__(
        self,
        position: str = "bottom-right",
        size: Tuple[int, int] = (240, 360),
        margin: int = 20,
        opacity: float = 0.95
    ):
        """
        Initialize the mini BEV renderer.

        Args:
            position: Corner position ("bottom-right", "bottom-left", "top-right", "top-left")
            size: (width, height) of the mini BEV
            margin: Margin from frame edges
            opacity: Overlay opacity (0-1)
        """
        self.position = position
        self.width, self.height = size
        self.margin = margin
        self.opacity = opacity

        # Colors (BGR)
        self.bg_color = (30, 30, 30)
        self.grid_color = (60, 60, 60)
        self.rail_color = (200, 200, 0)  # Cyan for rail mask
        self.ego_path_color = (0, 255, 0)  # Green for ego path
        self.other_path_color = (100, 100, 100)  # Gray for other paths
        self.center_line_color = (0, 200, 200)  # Yellow for center
        self.direction_colors = {
            PathDirection.STRAIGHT: (0, 255, 0),
            PathDirection.LEFT: (255, 200, 0),
            PathDirection.RIGHT: (0, 200, 255),
            PathDirection.UNKNOWN: (128, 128, 128)
        }

        # Fonts
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        bev_image: Optional[np.ndarray],
        bev_mask: Optional[np.ndarray],
        paths: List = None,
        ego_path: Optional = None,
        direction: PathDirection = PathDirection.UNKNOWN,
        direction_angle: float = 0.0,
        junction_type: JunctionType = JunctionType.NONE,
        centerline_pts: Optional[np.ndarray] = None,
        bev_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Render the mini BEV overlay on the frame.

        Args:
            frame: Main camera frame to overlay on
            bev_image: BEV-transformed image (optional)
            bev_mask: BEV-transformed segmentation mask
            paths: List of detected paths (unused - kept for compatibility)
            ego_path: Selected ego path (unused - kept for compatibility)
            direction: Current direction
            direction_angle: Direction angle in degrees
            junction_type: Detected junction type
            centerline_pts: BEV-transformed centerline points (N, 2) array
            bev_size: Original BEV size (width, height) for scaling

        Returns:
            Frame with mini BEV overlay
        """
        # Need at least some content to display
        if bev_image is None and bev_mask is None and centerline_pts is None:
            return frame

        result = frame.copy()
        frame_h, frame_w = frame.shape[:2]

        # Calculate position
        pos_x, pos_y = self._calculate_position(frame_w, frame_h)

        # Create mini BEV view
        mini_bev = self._create_mini_bev(
            bev_image, bev_mask, paths, ego_path,
            direction, direction_angle, junction_type,
            centerline_pts, bev_size
        )

        # Create overlay region
        overlay = result.copy()

        # Draw border first
        border_thickness = 2
        cv2.rectangle(
            overlay,
            (pos_x - border_thickness, pos_y - border_thickness),
            (pos_x + self.width + border_thickness, pos_y + self.height + border_thickness),
            (255, 255, 255),
            border_thickness
        )

        # Place mini BEV
        overlay[pos_y:pos_y + self.height, pos_x:pos_x + self.width] = mini_bev

        # Blend with opacity
        result = cv2.addWeighted(overlay, self.opacity, result, 1 - self.opacity, 0)

        # Add label (always visible)
        cv2.putText(
            result, "BEV",
            (pos_x + 5, pos_y - 5),
            self.font, 0.5, (255, 255, 255), 1
        )

        return result

    def _calculate_position(self, frame_w: int, frame_h: int) -> Tuple[int, int]:
        """Calculate position based on corner setting."""
        if self.position == "bottom-right":
            x = frame_w - self.width - self.margin
            y = frame_h - self.height - self.margin
        elif self.position == "bottom-left":
            x = self.margin
            y = frame_h - self.height - self.margin
        elif self.position == "top-right":
            x = frame_w - self.width - self.margin
            y = self.margin + 50  # Leave space for FPS display
        elif self.position == "top-left":
            x = self.margin + 320  # Leave space for alert panel
            y = self.margin
        else:
            x = frame_w - self.width - self.margin
            y = frame_h - self.height - self.margin

        return x, y

    def _create_mini_bev(
        self,
        bev_image: Optional[np.ndarray],
        bev_mask: Optional[np.ndarray],
        paths: List,
        ego_path: Optional,
        direction: PathDirection,
        angle: float,
        junction: JunctionType,
        centerline_pts: Optional[np.ndarray] = None,
        bev_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Create the mini BEV visualization.

        Shows:
        - Grid background
        - Centerline (from tracker, transformed to BEV)
        - Direction arrow
        - Info panel

        Does NOT show:
        - Segformer mask overlay (removed for clarity)
        - Paths from BEVPathAnalyzer (using centerline instead)
        """
        # Start with background or BEV image
        if bev_image is not None:
            # Resize BEV image to mini size
            mini = cv2.resize(bev_image, (self.width, self.height))

            # Brighten if too dark
            if np.mean(mini) < 40:
                mini = cv2.convertScaleAbs(mini, alpha=2.0, beta=30)
        else:
            mini = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        # Draw grid
        mini = self._draw_grid(mini)

        # Draw centerline (instead of segformer mask and paths)
        if centerline_pts is not None and len(centerline_pts) >= 2:
            mini = self._draw_centerline(mini, centerline_pts, bev_size)

        # Draw vehicle reference line (from bottom center going up)
        cv2.line(
            mini,
            (self.width // 2, self.height),
            (self.width // 2, self.height - 30),
            self.center_line_color,
            2
        )

        # Draw direction arrow at bottom center
        self._draw_direction_arrow(
            mini,
            self.width // 2,
            self.height - 50,
            direction, angle
        )

        # Draw info panel at bottom
        mini = self._draw_info_panel(mini, direction, angle, junction)

        return mini

    def _draw_centerline(
        self,
        image: np.ndarray,
        centerline_pts: np.ndarray,
        bev_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Draw the rail centerline on the mini BEV.

        Args:
            image: Mini BEV image to draw on
            centerline_pts: BEV-transformed centerline points (N, 2)
            bev_size: Original BEV size (width, height) for scaling

        Returns:
            Image with centerline drawn
        """
        result = image.copy()

        if len(centerline_pts) < 2:
            return result

        # Calculate scale factors
        if bev_size is not None:
            bev_w, bev_h = bev_size
            scale_x = self.width / bev_w
            scale_y = self.height / bev_h
        else:
            # Assume standard 400x600 BEV if not specified
            scale_x = self.width / 400
            scale_y = self.height / 600

        # Scale points to mini BEV size
        scaled_pts = centerline_pts.copy()
        scaled_pts[:, 0] = centerline_pts[:, 0] * scale_x
        scaled_pts[:, 1] = centerline_pts[:, 1] * scale_y
        scaled_pts = scaled_pts.astype(np.int32)

        # Draw centerline with glow effect for visibility
        # Outer glow (thicker, darker)
        cv2.polylines(result, [scaled_pts], False, (0, 100, 100), 5)
        # Inner line (thinner, brighter yellow)
        cv2.polylines(result, [scaled_pts], False, (0, 255, 255), 2)

        # Draw points along the line for reference
        for i in range(0, len(scaled_pts), max(1, len(scaled_pts) // 5)):
            pt = tuple(scaled_pts[i])
            cv2.circle(result, pt, 3, (0, 255, 255), -1)

        return result

    def _draw_grid(self, image: np.ndarray) -> np.ndarray:
        """Draw a reference grid on the BEV."""
        result = image.copy()

        # Vertical lines
        for i in range(1, 4):
            x = i * self.width // 4
            cv2.line(result, (x, 0), (x, self.height), self.grid_color, 1)

        # Horizontal lines
        for i in range(1, 6):
            y = i * self.height // 6
            cv2.line(result, (0, y), (self.width, y), self.grid_color, 1)

        # Center line (stronger)
        cv2.line(
            result,
            (self.width // 2, 0),
            (self.width // 2, self.height),
            (80, 80, 80), 1
        )

        return result

    def _overlay_rail_mask(
        self,
        image: np.ndarray,
        bev_mask: np.ndarray
    ) -> np.ndarray:
        """Overlay rail mask on the image."""
        result = image.copy()

        # Resize mask to mini size
        mask_resized = cv2.resize(
            bev_mask,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST
        )

        # Create rail binary mask (classes 4 and 9)
        rail_binary = ((mask_resized == 4) | (mask_resized == 9)).astype(np.uint8)

        # Create colored overlay
        rail_overlay = np.zeros_like(result)
        rail_overlay[rail_binary > 0] = self.rail_color

        # Blend
        result = cv2.addWeighted(result, 0.7, rail_overlay, 0.3, 0)

        return result

    def _draw_paths(
        self,
        image: np.ndarray,
        paths: List,
        ego_path: Optional,
        bev_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Draw detected paths on the image."""
        result = image.copy()

        if bev_mask is None:
            return result

        # Get scale factors
        if hasattr(bev_mask, 'shape'):
            orig_h, orig_w = bev_mask.shape[:2]
            scale_x = self.width / orig_w
            scale_y = self.height / orig_h
        else:
            scale_x = scale_y = 1.0

        # Draw other paths first (gray)
        for path in paths:
            if path == ego_path:
                continue

            if hasattr(path, 'skeleton_points') and path.skeleton_points is not None:
                points = path.skeleton_points
                if len(points) >= 2:
                    scaled_pts = (points * [scale_x, scale_y]).astype(np.int32)
                    cv2.polylines(
                        result, [scaled_pts],
                        False, self.other_path_color, 2
                    )

        # Draw ego path (green, thick)
        if ego_path is not None:
            if hasattr(ego_path, 'skeleton_points') and ego_path.skeleton_points is not None:
                points = ego_path.skeleton_points
                if len(points) >= 2:
                    scaled_pts = (points * [scale_x, scale_y]).astype(np.int32)
                    cv2.polylines(
                        result, [scaled_pts],
                        False, self.ego_path_color, 3
                    )

        return result

    def _draw_direction_arrow(
        self,
        image: np.ndarray,
        cx: int, cy: int,
        direction: PathDirection,
        angle: float
    ):
        """Draw direction indicator arrow."""
        color = self.direction_colors.get(direction, (200, 200, 200))
        size = 20

        # Arrow based on angle
        angle_rad = np.radians(-angle)

        # Arrow tip
        tip_x = int(cx + size * np.sin(angle_rad))
        tip_y = int(cy - size * np.cos(angle_rad))

        # Arrow base
        base_x = int(cx - size * 0.5 * np.sin(angle_rad))
        base_y = int(cy + size * 0.5 * np.cos(angle_rad))

        # Draw arrow
        cv2.arrowedLine(
            image,
            (base_x, base_y), (tip_x, tip_y),
            color, 2, tipLength=0.4
        )

        # Draw circle at center
        cv2.circle(image, (cx, cy), 4, color, -1)

    def _draw_info_panel(
        self,
        image: np.ndarray,
        direction: PathDirection,
        angle: float,
        junction: JunctionType
    ) -> np.ndarray:
        """Draw info panel at bottom of BEV."""
        result = image.copy()

        # Background panel at bottom
        panel_h = 60
        cv2.rectangle(
            result,
            (0, self.height - panel_h),
            (self.width, self.height),
            (0, 0, 0),
            -1
        )

        # Direction text
        dir_name = direction.name if hasattr(direction, 'name') else str(direction)
        dir_color = self.direction_colors.get(direction, (200, 200, 200))

        cv2.putText(
            result, f"Dir: {dir_name}",
            (5, self.height - 40),
            self.font, 0.45, dir_color, 1
        )

        # Angle
        cv2.putText(
            result, f"Angle: {angle:+.1f}",
            (5, self.height - 20),
            self.font, 0.4, (180, 180, 180), 1
        )

        # Junction type
        junc_name = junction.name if hasattr(junction, 'name') else "NONE"
        junc_color = (255, 255, 0) if junction != JunctionType.NONE else (100, 100, 100)

        cv2.putText(
            result, f"Junc: {junc_name}",
            (self.width // 2, self.height - 20),
            self.font, 0.4, junc_color, 1
        )

        return result


class MiniBEVSimple:
    """
    Simplified mini BEV that just shows direction.
    Use when full BEV is not available.
    """

    def __init__(
        self,
        position: str = "bottom-right",
        size: Tuple[int, int] = (150, 150),
        margin: int = 20
    ):
        self.position = position
        self.width, self.height = size
        self.margin = margin
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        direction: PathDirection,
        angle: float
    ) -> np.ndarray:
        """Render simple direction indicator."""
        result = frame.copy()
        frame_h, frame_w = frame.shape[:2]

        # Position
        if self.position == "bottom-right":
            x = frame_w - self.width - self.margin
            y = frame_h - self.height - self.margin
        else:
            x = self.margin
            y = frame_h - self.height - self.margin

        # Background
        cv2.rectangle(
            result,
            (x, y),
            (x + self.width, y + self.height),
            (30, 30, 30),
            -1
        )
        cv2.rectangle(
            result,
            (x, y),
            (x + self.width, y + self.height),
            (100, 100, 100),
            2
        )

        # Direction arrow
        cx = x + self.width // 2
        cy = y + self.height // 2

        # Large arrow
        size = 40
        angle_rad = np.radians(-angle)

        tip_x = int(cx + size * np.sin(angle_rad))
        tip_y = int(cy - size * np.cos(angle_rad))
        base_x = int(cx - size * 0.5 * np.sin(angle_rad))
        base_y = int(cy + size * 0.5 * np.cos(angle_rad))

        colors = {
            PathDirection.STRAIGHT: (0, 255, 0),
            PathDirection.LEFT: (255, 200, 0),
            PathDirection.RIGHT: (0, 200, 255),
            PathDirection.UNKNOWN: (128, 128, 128)
        }
        color = colors.get(direction, (200, 200, 200))

        cv2.arrowedLine(result, (base_x, base_y), (tip_x, tip_y), color, 4, tipLength=0.3)

        # Direction text
        dir_name = direction.name if hasattr(direction, 'name') else "?"
        cv2.putText(
            result, dir_name,
            (x + 10, y + self.height - 10),
            self.font, 0.5, color, 1
        )

        return result
