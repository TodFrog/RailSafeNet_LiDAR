#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alert Panel
===========
Visual alert panel for displaying danger status and direction.

Provides:
- Status indicator (SAFE/CAUTION/WARNING/DANGER) with color background
- Direction arrow (STRAIGHT/LEFT/RIGHT)
- Zone object counts

Author: RailSafeNet Team
Date: 2025-01-06
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum

# Import from sibling module
try:
    from .danger_zone_detector import DangerLevel
except ImportError:
    from danger_zone_detector import DangerLevel

try:
    from .bev_path_analyzer import PathDirection
except ImportError:
    # Define locally if import fails
    class PathDirection(Enum):
        STRAIGHT = 0
        LEFT = 1
        RIGHT = 2
        UNKNOWN = 3


class AlertPanel:
    """
    Visual alert panel displayed in the top-left corner.

    Shows:
    - Current danger status with color-coded background
    - Direction indicator with arrow
    - Zone object counts (RED/ORANGE/YELLOW)
    """

    def __init__(
        self,
        position: Tuple[int, int] = (10, 10),
        size: Tuple[int, int] = (300, 180),
        opacity: float = 0.85
    ):
        """
        Initialize the alert panel.

        Args:
            position: (x, y) position of top-left corner
            size: (width, height) of the panel
            opacity: Panel background opacity (0-1)
        """
        self.position = position
        self.width, self.height = size
        self.opacity = opacity

        # Colors (BGR)
        self.status_colors = {
            DangerLevel.SAFE: (0, 180, 0),      # Green
            DangerLevel.YELLOW: (0, 200, 200),  # Yellow
            DangerLevel.ORANGE: (0, 140, 255),  # Orange
            DangerLevel.RED: (0, 0, 220)        # Red
        }

        self.direction_colors = {
            PathDirection.STRAIGHT: (0, 255, 0),   # Green
            PathDirection.LEFT: (255, 200, 0),     # Cyan
            PathDirection.RIGHT: (0, 200, 255),    # Yellow
            PathDirection.UNKNOWN: (128, 128, 128) # Gray
        }

        # Background color (dark gray with transparency)
        self.bg_color = (40, 40, 40)

        # Fonts
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX

    def render(
        self,
        frame: np.ndarray,
        severity: DangerLevel,
        direction: PathDirection,
        zone_counts: Dict[DangerLevel, int],
        direction_angle: float = 0.0
    ) -> np.ndarray:
        """
        Render the alert panel on the frame.

        Args:
            frame: Image to draw on
            severity: Current maximum danger level
            direction: Current track direction
            zone_counts: Dict mapping DangerLevel to object count
            direction_angle: Direction angle in degrees

        Returns:
            Frame with alert panel drawn
        """
        result = frame.copy()
        x, y = self.position

        # Create panel background
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + self.width, y + self.height),
            self.bg_color,
            -1
        )

        # Panel border
        cv2.rectangle(
            overlay,
            (x, y),
            (x + self.width, y + self.height),
            (100, 100, 100),
            2
        )

        # Apply opacity
        result = cv2.addWeighted(overlay, self.opacity, result, 1 - self.opacity, 0)

        # === Section 1: Status Indicator ===
        status_y = y + 10
        status_height = 45
        status_color = self.status_colors.get(severity, (128, 128, 128))
        status_text = self._get_status_text(severity)

        # Status background (color-coded)
        cv2.rectangle(
            result,
            (x + 10, status_y),
            (x + self.width - 10, status_y + status_height),
            status_color,
            -1
        )

        # Status text (centered)
        text_size, _ = cv2.getTextSize(status_text, self.font_bold, 1.0, 2)
        text_x = x + (self.width - text_size[0]) // 2
        text_y = status_y + (status_height + text_size[1]) // 2

        # Text shadow
        cv2.putText(
            result, status_text,
            (text_x + 1, text_y + 1),
            self.font_bold, 1.0, (0, 0, 0), 3
        )
        cv2.putText(
            result, status_text,
            (text_x, text_y),
            self.font_bold, 1.0, (255, 255, 255), 2
        )

        # === Section 2: Direction Indicator ===
        dir_y = status_y + status_height + 15
        dir_height = 60

        # Direction background
        cv2.rectangle(
            result,
            (x + 10, dir_y),
            (x + self.width - 10, dir_y + dir_height),
            (60, 60, 60),
            -1
        )

        # Draw arrow
        arrow_center_x = x + 60
        arrow_center_y = dir_y + dir_height // 2
        self._draw_direction_arrow(
            result, arrow_center_x, arrow_center_y,
            direction, direction_angle, size=25
        )

        # Direction text
        dir_text = direction.name if hasattr(direction, 'name') else str(direction)
        dir_color = self.direction_colors.get(direction, (200, 200, 200))

        cv2.putText(
            result, dir_text,
            (x + 100, arrow_center_y + 8),
            self.font, 0.7, dir_color, 2
        )

        # Angle text
        angle_text = f"{direction_angle:+.1f} deg"
        cv2.putText(
            result, angle_text,
            (x + 200, arrow_center_y + 8),
            self.font, 0.5, (180, 180, 180), 1
        )

        # === Section 3: Zone Counts ===
        count_y = dir_y + dir_height + 15

        # Zone count boxes
        box_width = (self.width - 40) // 3
        box_height = 30

        zones = [
            (DangerLevel.RED, "RED"),
            (DangerLevel.ORANGE, "ORG"),
            (DangerLevel.YELLOW, "YLW")
        ]

        for i, (level, label) in enumerate(zones):
            box_x = x + 15 + i * (box_width + 5)
            count = zone_counts.get(level, 0)
            color = self.status_colors.get(level, (128, 128, 128))

            # Box background (darker if count > 0)
            bg = color if count > 0 else (50, 50, 50)
            cv2.rectangle(
                result,
                (box_x, count_y),
                (box_x + box_width, count_y + box_height),
                bg,
                -1
            )

            # Border
            cv2.rectangle(
                result,
                (box_x, count_y),
                (box_x + box_width, count_y + box_height),
                color,
                1
            )

            # Label and count
            text = f"{label}:{count}"
            text_size, _ = cv2.getTextSize(text, self.font, 0.5, 1)
            text_x = box_x + (box_width - text_size[0]) // 2
            text_y = count_y + (box_height + text_size[1]) // 2

            text_color = (255, 255, 255) if count > 0 else (150, 150, 150)
            cv2.putText(
                result, text,
                (text_x, text_y),
                self.font, 0.5, text_color, 1
            )

        return result

    def _draw_direction_arrow(
        self,
        frame: np.ndarray,
        cx: int, cy: int,
        direction: PathDirection,
        angle: float,
        size: int = 25
    ):
        """
        Draw a direction arrow.

        Args:
            frame: Image to draw on
            cx, cy: Arrow center position
            direction: Direction enum
            angle: Rotation angle in degrees
            size: Arrow size in pixels
        """
        color = self.direction_colors.get(direction, (200, 200, 200))

        # Calculate arrow points based on angle
        angle_rad = np.radians(-angle)  # Negate for correct visual direction

        # Arrow tip (forward)
        tip_x = int(cx + size * np.sin(angle_rad))
        tip_y = int(cy - size * np.cos(angle_rad))

        # Arrow base (backward)
        base_x = int(cx - size * 0.4 * np.sin(angle_rad))
        base_y = int(cy + size * 0.4 * np.cos(angle_rad))

        # Arrow wings
        wing_angle = 0.5  # radians
        wing_size = size * 0.4

        left_wing_x = int(tip_x - wing_size * np.sin(angle_rad - wing_angle))
        left_wing_y = int(tip_y + wing_size * np.cos(angle_rad - wing_angle))

        right_wing_x = int(tip_x - wing_size * np.sin(angle_rad + wing_angle))
        right_wing_y = int(tip_y + wing_size * np.cos(angle_rad + wing_angle))

        # Draw arrow body
        cv2.line(frame, (base_x, base_y), (tip_x, tip_y), color, 3)

        # Draw arrow head
        cv2.line(frame, (tip_x, tip_y), (left_wing_x, left_wing_y), color, 3)
        cv2.line(frame, (tip_x, tip_y), (right_wing_x, right_wing_y), color, 3)

        # Draw circle at center
        cv2.circle(frame, (cx, cy), 5, color, -1)

    def _get_status_text(self, severity: DangerLevel) -> str:
        """Get display text for severity level."""
        texts = {
            DangerLevel.SAFE: "SAFE",
            DangerLevel.YELLOW: "CAUTION",
            DangerLevel.ORANGE: "WARNING",
            DangerLevel.RED: "DANGER"
        }
        return texts.get(severity, "UNKNOWN")


class CompactAlertPanel(AlertPanel):
    """
    Compact version of the alert panel for smaller displays.
    Shows only status and direction in a single row.
    """

    def __init__(
        self,
        position: Tuple[int, int] = (10, 10),
        size: Tuple[int, int] = (200, 50),
        opacity: float = 0.85
    ):
        super().__init__(position, size, opacity)

    def render(
        self,
        frame: np.ndarray,
        severity: DangerLevel,
        direction: PathDirection,
        zone_counts: Dict[DangerLevel, int],
        direction_angle: float = 0.0
    ) -> np.ndarray:
        """Render compact panel with status and direction only."""
        result = frame.copy()
        x, y = self.position

        # Background
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + self.width, y + self.height),
            self.bg_color,
            -1
        )
        result = cv2.addWeighted(overlay, self.opacity, result, 1 - self.opacity, 0)

        # Status color bar
        status_color = self.status_colors.get(severity, (128, 128, 128))
        cv2.rectangle(
            result,
            (x, y),
            (x + 10, y + self.height),
            status_color,
            -1
        )

        # Status text
        status_text = self._get_status_text(severity)
        cv2.putText(
            result, status_text,
            (x + 20, y + 32),
            self.font, 0.7, (255, 255, 255), 2
        )

        # Direction arrow (small)
        self._draw_direction_arrow(
            result, x + self.width - 30, y + self.height // 2,
            direction, direction_angle, size=15
        )

        return result
