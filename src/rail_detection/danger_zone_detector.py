#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Danger Zone Detector
====================
Detects overlap between YOLO bounding boxes and hazard zones.

Provides:
- Polygon-based hazard zone generation from rail edges
- Bbox vs polygon overlap detection
- Zone classification (RED/ORANGE/YELLOW)
- Severity and count reporting

Author: RailSafeNet Team
Date: 2025-01-06
"""

import cv2
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path


class DangerLevel(IntEnum):
    """Danger level enumeration (higher = more dangerous)."""
    SAFE = 0
    YELLOW = 1   # Warning zone
    ORANGE = 2   # Caution zone
    RED = 3      # Danger zone


@dataclass
class DetectedHazard:
    """Object detected in a hazard zone."""
    object_id: int
    class_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    danger_level: DangerLevel
    zone_name: str


class VanishingPointCalibrator:
    """
    Interactive vanishing point calibration.

    Allows user to click on screen to set vanishing point location.
    The VP is saved to config file for use in hazard zone generation.

    Usage:
        calibrator = VanishingPointCalibrator(frame, display_scale=0.5)
        vp = calibrator.run(save_path='config/rail_tracker_config.yaml')
    """

    def __init__(self, image: np.ndarray, display_scale: float = 0.5):
        """
        Initialize calibrator.

        Args:
            image: Reference frame for calibration
            display_scale: Scale factor for display (0.5 = half size)
        """
        self.image = image.copy()
        self.display_scale = display_scale
        self.vp_point: Optional[Tuple[int, int]] = None
        self.window_name = 'VP Calibration - Click to set vanishing point'
        self.display_image = None

        # Initial display
        self._update_display()

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click to set VP."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates to original image coordinates
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)
            self.vp_point = (orig_x, orig_y)
            self._update_display()

    def _update_display(self):
        """Update display with VP location and guide lines."""
        display = self.image.copy()
        h, w = display.shape[:2]

        # Draw instructions
        cv2.putText(display, "Click to set vanishing point",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2)
        cv2.putText(display, "'s'=Save  'r'=Reset  'q'=Quit",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (200, 200, 200), 2)

        if self.vp_point:
            vp_x, vp_y = self.vp_point

            # Draw VP point (red circle)
            cv2.circle(display, (vp_x, vp_y), 12, (0, 0, 255), -1)
            cv2.circle(display, (vp_x, vp_y), 14, (255, 255, 255), 2)

            # Display coordinates
            cv2.putText(display, f"VP: ({vp_x}, {vp_y})",
                       (vp_x + 20, vp_y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 2)

            # Draw guide lines from VP to bottom corners (yellow - zone preview)
            cv2.line(display, (vp_x, vp_y), (0, h), (0, 255, 255), 2)
            cv2.line(display, (vp_x, vp_y), (w, h), (0, 255, 255), 2)

            # Draw guide lines to rail zone estimation (green - inner)
            center_bottom = w // 2
            margin = 100  # Approximate rail zone half-width at bottom
            cv2.line(display, (vp_x, vp_y), (center_bottom - margin, h), (0, 255, 0), 1)
            cv2.line(display, (vp_x, vp_y), (center_bottom + margin, h), (0, 255, 0), 1)

            # Draw guide lines to wider zones (orange)
            cv2.line(display, (vp_x, vp_y), (center_bottom - margin * 2, h), (0, 165, 255), 1)
            cv2.line(display, (vp_x, vp_y), (center_bottom + margin * 2, h), (0, 165, 255), 1)

        # Scale for display
        self.display_image = cv2.resize(display, None,
            fx=self.display_scale, fy=self.display_scale)

    def run(self, save_path: str) -> Optional[Tuple[int, int]]:
        """
        Run calibration interface.

        Args:
            save_path: Path to config file for saving VP

        Returns:
            Tuple (vp_x, vp_y) if saved, None if cancelled
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n=== VP Calibration ===")
        print("Click on the image to set vanishing point")
        print("'s' = Save and exit")
        print("'r' = Reset")
        print("'q' = Quit without saving")
        print("=" * 22)

        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Calibration cancelled")
                break
            elif key == ord('r'):
                self.vp_point = None
                self._update_display()
                print("VP reset")
            elif key == ord('s') and self.vp_point:
                self._save_config(save_path)
                break

        cv2.destroyAllWindows()
        return self.vp_point

    def _save_config(self, config_path: str):
        """Save VP coordinates to config file."""
        config_path = Path(config_path)

        # Load existing config or create new
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        # Add/update vanishing_point in hazard_zones section
        if 'hazard_zones' not in config:
            config['hazard_zones'] = {}

        config['hazard_zones']['vanishing_point'] = {
            'x': self.vp_point[0],
            'y': self.vp_point[1]
        }

        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"\n✓ VP saved: ({self.vp_point[0]}, {self.vp_point[1]}) -> {config_path}")


class DangerZoneDetector:
    """
    Detects objects in hazard zones around the rail track.

    Creates perspective-aware hazard zones (Red/Orange/Yellow) from
    rail edge points and checks if YOLO detections overlap with them.

    Zone structure:
    - Red zone: Rail track width + small margin (immediate danger)
    - Orange zone: Expands from red with perspective taper (warning)
    - Yellow zone: Expands from orange with perspective taper (caution)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the detector.

        Args:
            config: Configuration dictionary with zone widths and taper
        """
        # Default configuration
        # Red zone: margin on each side of rail (total = rail_width + 2*margin)
        self.red_margin = 15          # Margin on each side of rail for red zone (px)
        # Orange/Yellow: expansion widths at bottom (will taper toward top)
        self.orange_expansion = 80    # Orange zone expansion at bottom (px)
        self.yellow_expansion = 120   # Yellow zone expansion at bottom (px)

        # Perspective parameters
        self.perspective_power = 0.8  # Power factor for non-linear perspective

        # Zone colors (BGR)
        self.colors = {
            DangerLevel.RED: (0, 0, 255),
            DangerLevel.ORANGE: (0, 165, 255),
            DangerLevel.YELLOW: (0, 255, 255),
            DangerLevel.SAFE: (0, 255, 0)
        }

        # Zone transparency
        self.zone_alpha = 0.3

        # Manual vanishing point (from calibration, takes priority over auto)
        self.manual_vp: Optional[Tuple[float, float]] = None

        # Load from config if provided
        if config:
            hz = config.get('hazard_zones', {})
            self.red_margin = hz.get('red_zone_margin_px', self.red_margin)
            self.orange_expansion = hz.get('orange_zone_expansion_px', self.orange_expansion)
            self.yellow_expansion = hz.get('yellow_zone_expansion_px', self.yellow_expansion)
            self.perspective_power = hz.get('perspective_power', self.perspective_power)
            self.zone_alpha = hz.get('zone_alpha', self.zone_alpha)

            # Load manual VP if saved from calibration
            vp_config = hz.get('vanishing_point')
            if vp_config and 'x' in vp_config and 'y' in vp_config:
                self.manual_vp = (float(vp_config['x']), float(vp_config['y']))
                print(f"  [DangerZone] Manual VP loaded: {self.manual_vp}")

        # Cached zone polygons
        self._zone_polygons: Dict[DangerLevel, np.ndarray] = {}
        self._frame_height = 0
        self._frame_width = 0
        self._vanishing_point: Optional[Tuple[float, float]] = None

    def _estimate_vanishing_point(
        self,
        center_points: np.ndarray,
        frame_height: int,
        frame_width: int
    ) -> Tuple[float, float]:
        """
        Estimate vanishing point from rail centerline.

        Uses linear regression on centerline points to extrapolate
        where the line would converge at y=0 (top of frame).

        Args:
            center_points: Array of (x, y) rail center points
            frame_height: Frame height
            frame_width: Frame width

        Returns:
            (vp_x, vp_y): Vanishing point coordinates
        """
        if len(center_points) < 3:
            # Not enough points, use frame center as default VP
            return (frame_width / 2, 0)

        # Extract x and y coordinates
        xs = center_points[:, 0].astype(np.float64)
        ys = center_points[:, 1].astype(np.float64)

        # Linear regression: x = a*y + b (y as independent variable)
        # This fits a line through the centerline points
        try:
            coeffs = np.polyfit(ys, xs, 1)  # x = coeffs[0]*y + coeffs[1]
            # At y = 0 (top of frame), x = coeffs[1]
            vp_x = coeffs[1]
        except np.RankWarning:
            vp_x = frame_width / 2

        # VP is at top of frame (y = 0)
        vp_y = 0.0

        # Clamp VP x to reasonable range (within frame with some margin)
        vp_x = np.clip(vp_x, frame_width * 0.1, frame_width * 0.9)

        return (float(vp_x), float(vp_y))

    def _compute_zone_boundary_at_y(
        self,
        y: float,
        bottom_x: float,
        vp_x: float,
        vp_y: float,
        bottom_y: float
    ) -> float:
        """
        Compute zone boundary x-coordinate at a given y position.

        Uses linear interpolation between vanishing point and bottom boundary.
        This creates perspective-correct zone boundaries that converge toward VP.

        Args:
            y: Y coordinate to compute boundary for
            bottom_x: X coordinate of boundary at bottom
            vp_x: Vanishing point x coordinate
            vp_y: Vanishing point y coordinate
            bottom_y: Y coordinate of bottom reference point

        Returns:
            X coordinate of boundary at the given y
        """
        # Avoid division by zero
        if abs(bottom_y - vp_y) < 1:
            return bottom_x

        # Linear interpolation: compute t parameter (0 at VP, 1 at bottom)
        t = (y - vp_y) / (bottom_y - vp_y)

        # Interpolate x position
        x = vp_x + (bottom_x - vp_x) * t

        return x

    def generate_hazard_zones(
        self,
        center_points: np.ndarray,
        width_profile: np.ndarray,
        frame_height: int,
        frame_width: int = 1920
    ) -> Dict[DangerLevel, np.ndarray]:
        """
        Generate hazard zone polygons that follow the rail curve with VP perspective.

        Zone boundaries follow the rail centerline curve (center_points) while
        applying VP-based perspective scaling - wider at bottom, narrower at top.

        Zone structure (from center outward at each y level):
        - Red: rail_width/2 + red_margin * scale
        - Orange: red_offset + orange_expansion * scale
        - Yellow: orange_offset + yellow_expansion * scale

        Where scale = (y - vp_y) / (bottom_y - vp_y), creating perspective taper.

        Args:
            center_points: Array of (x, y) rail center points (defines curve)
            width_profile: Array of rail widths at each center point
            frame_height: Frame height
            frame_width: Frame width for VP estimation

        Returns:
            Dictionary mapping DangerLevel to polygon arrays
        """
        if center_points is None or len(center_points) < 2:
            return {}

        if width_profile is None or len(width_profile) < 2:
            return {}

        self._frame_height = frame_height
        self._frame_width = frame_width

        # 1. Use manual VP if set (from calibration), otherwise estimate
        if self.manual_vp is not None:
            vp_x, vp_y = self.manual_vp
        else:
            vp_x, vp_y = self._estimate_vanishing_point(
                center_points, frame_height, frame_width
            )
        self._vanishing_point = (vp_x, vp_y)

        # 2. Find bottom reference point (largest y = closest to camera)
        bottom_idx = np.argmax(center_points[:, 1])
        bottom_y = float(center_points[bottom_idx, 1])

        # 3. Build boundaries for each y position using curve-following + VP perspective
        # Zone boundaries now follow the rail centerline curve while applying
        # VP-based perspective scaling (narrower at top, wider at bottom)
        red_left, red_right = [], []
        orange_left, orange_right = [], []
        yellow_left, yellow_right = [], []

        for i, (center_x, center_y) in enumerate(center_points):
            if i >= len(width_profile):
                break

            cx, cy = float(center_x), float(center_y)

            # Get actual rail half-width at this y level
            rail_half = float(width_profile[i]) / 2

            # VP-based scale factor (1.0 at bottom, 0.0 at VP)
            # This creates perspective taper while following the curve
            if abs(bottom_y - vp_y) > 1:
                scale = (cy - vp_y) / (bottom_y - vp_y)
                scale = max(0.0, min(1.0, scale))  # Clamp to [0, 1]
            else:
                scale = 1.0

            # Calculate zone offsets with VP scaling
            # Red zone: rail width + scaled margin
            red_offset = rail_half + self.red_margin * scale
            # Orange zone: red + scaled expansion
            orange_offset = red_offset + self.orange_expansion * scale
            # Yellow zone: orange + scaled expansion
            yellow_offset = orange_offset + self.yellow_expansion * scale

            # Apply offsets from current center_x (follows the curve)
            # Use round() instead of int() for more accurate polygon coordinates
            red_left.append((round(cx - red_offset), round(cy)))
            red_right.append((round(cx + red_offset), round(cy)))
            orange_left.append((round(cx - orange_offset), round(cy)))
            orange_right.append((round(cx + orange_offset), round(cy)))
            yellow_left.append((round(cx - yellow_offset), round(cy)))
            yellow_right.append((round(cx + yellow_offset), round(cy)))

        # 4. Create polygons (closed shapes: left boundary + reversed right boundary)
        zones = {}

        if len(red_left) >= 2:
            zones[DangerLevel.RED] = np.array(
                red_left + red_right[::-1], dtype=np.int32
            )
        if len(orange_left) >= 2:
            zones[DangerLevel.ORANGE] = np.array(
                orange_left + orange_right[::-1], dtype=np.int32
            )
        if len(yellow_left) >= 2:
            zones[DangerLevel.YELLOW] = np.array(
                yellow_left + yellow_right[::-1], dtype=np.int32
            )

        self._zone_polygons = zones
        return zones

    def check_overlaps(
        self,
        detections: List[Dict],
        zones: Optional[Dict[DangerLevel, np.ndarray]] = None
    ) -> Dict:
        """
        Check which detections overlap with hazard zones.

        Args:
            detections: List of detection dicts with 'bbox_xyxy', 'class_name', etc.
            zones: Optional zone polygons (uses cached if not provided)

        Returns:
            Dict with:
                - max_severity: DangerLevel (highest severity among all detections)
                - zone_counts: {DangerLevel: count}
                - hazards: List[DetectedHazard]
        """
        if zones is None:
            zones = self._zone_polygons

        result = {
            'max_severity': DangerLevel.SAFE,
            'zone_counts': {
                DangerLevel.RED: 0,
                DangerLevel.ORANGE: 0,
                DangerLevel.YELLOW: 0
            },
            'hazards': []
        }

        if not zones or not detections:
            return result

        for i, det in enumerate(detections):
            bbox = det.get('bbox_xyxy')
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            # Check points to test: center and corners
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            bottom_center = ((x1 + x2) // 2, y2)  # Bottom center is most relevant
            corners = [
                (x1, y1), (x2, y1),  # Top corners
                (x1, y2), (x2, y2)   # Bottom corners
            ]

            # Test against zones in priority order (RED first)
            detected_level = DangerLevel.SAFE

            for level in [DangerLevel.RED, DangerLevel.ORANGE, DangerLevel.YELLOW]:
                if level not in zones:
                    continue

                polygon = zones[level]

                # Check if any point is inside the zone polygon
                if self._point_in_polygon(bottom_center, polygon):
                    detected_level = level
                    break

                if self._point_in_polygon(center, polygon):
                    detected_level = level
                    break

                for corner in corners:
                    if self._point_in_polygon(corner, polygon):
                        detected_level = level
                        break

                if detected_level != DangerLevel.SAFE:
                    break

            if detected_level != DangerLevel.SAFE:
                hazard = DetectedHazard(
                    object_id=i,
                    class_name=det.get('class_name', 'unknown'),
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=det.get('confidence', 0.0),
                    danger_level=detected_level,
                    zone_name=detected_level.name
                )
                result['hazards'].append(hazard)
                result['zone_counts'][detected_level] += 1

                if detected_level > result['max_severity']:
                    result['max_severity'] = detected_level

        return result

    def _point_in_polygon(self, point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """
        Check if a point is inside a polygon using cv2.pointPolygonTest.

        Args:
            point: (x, y) tuple
            polygon: Polygon vertices as numpy array

        Returns:
            True if point is inside polygon
        """
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0  # >= 0 means inside or on edge

    def draw_zones(
        self,
        frame: np.ndarray,
        zones: Optional[Dict[DangerLevel, np.ndarray]] = None,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Draw hazard zones on frame as semi-transparent overlays.

        Args:
            frame: Image to draw on
            zones: Zone polygons (uses cached if not provided)
            alpha: Transparency (0-1, uses default if not provided)

        Returns:
            Frame with zones drawn
        """
        if zones is None:
            zones = self._zone_polygons

        if not zones:
            return frame

        if alpha is None:
            alpha = self.zone_alpha

        overlay = frame.copy()

        # Draw zones from outer (yellow) to inner (red)
        for level in [DangerLevel.YELLOW, DangerLevel.ORANGE, DangerLevel.RED]:
            if level in zones:
                color = self.colors[level]
                cv2.fillPoly(overlay, [zones[level]], color)

        # Blend with original
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return result

    def draw_hazards(
        self,
        frame: np.ndarray,
        hazards: List[DetectedHazard],
        draw_bbox: bool = True,
        draw_label: bool = True
    ) -> np.ndarray:
        """
        Draw detected hazards on frame.

        Args:
            frame: Image to draw on
            hazards: List of DetectedHazard objects
            draw_bbox: Whether to draw bounding boxes
            draw_label: Whether to draw labels

        Returns:
            Frame with hazards drawn
        """
        result = frame.copy()

        for hazard in hazards:
            color = self.colors[hazard.danger_level]
            x1, y1, x2, y2 = hazard.bbox_xyxy

            if draw_bbox:
                # Draw thick border for dangerous objects
                thickness = 3 if hazard.danger_level == DangerLevel.RED else 2
                cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            if draw_label:
                # Label with class name and danger level
                label = f"{hazard.class_name} [{hazard.zone_name}]"

                # Background for text
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    result,
                    (x1, y1 - th - 8),
                    (x1 + tw + 4, y1),
                    color,
                    -1
                )

                # Text
                cv2.putText(
                    result, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )

        return result

    def get_status_color(self, severity: DangerLevel) -> Tuple[int, int, int]:
        """Get color for a given severity level."""
        return self.colors.get(severity, (128, 128, 128))

    def get_status_name(self, severity: DangerLevel) -> str:
        """Get display name for a given severity level."""
        names = {
            DangerLevel.SAFE: "SAFE",
            DangerLevel.YELLOW: "CAUTION",
            DangerLevel.ORANGE: "WARNING",
            DangerLevel.RED: "DANGER"
        }
        return names.get(severity, "UNKNOWN")


# Allowed YOLO classes for hazard detection
# Only detect objects that could be hazards on rail tracks
# Excludes: train(6), truck(7), backpack(24), umbrella(25), suitcase(28), skateboard(36)
ALLOWED_HAZARD_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
}


def convert_yolo_results_to_detections(
    yolo_results,
    allowed_classes: Optional[set] = None
) -> List[Dict]:
    """
    Convert YOLO results to detection dictionaries.

    Args:
        yolo_results: YOLO prediction results (list of result objects)
        allowed_classes: Set of allowed class IDs (uses ALLOWED_HAZARD_CLASSES if None)

    Returns:
        List of detection dicts with bbox_xyxy, class_name, confidence
        Only includes detections for allowed classes
    """
    detections = []

    if not yolo_results:
        return detections

    # Use default allowed classes if not specified
    if allowed_classes is None:
        allowed_classes = set(ALLOWED_HAZARD_CLASSES.keys())

    # Handle TRT mock results format
    for result in yolo_results:
        boxes = result.boxes
        if boxes is None:
            continue

        xywh_list = boxes.xywh.tolist() if hasattr(boxes.xywh, 'tolist') else []
        cls_list = boxes.cls.tolist() if hasattr(boxes.cls, 'tolist') else []

        for i, xywh in enumerate(xywh_list):
            if len(xywh) < 4:
                continue

            class_id = int(cls_list[i]) if i < len(cls_list) else 0

            # Filter: Skip classes not in allowed list
            if class_id not in allowed_classes:
                continue

            x_c, y_c, w, h = xywh
            x1 = int(x_c - w / 2)
            y1 = int(y_c - h / 2)
            x2 = int(x_c + w / 2)
            y2 = int(y_c + h / 2)

            # Get class name from allowed classes dict
            class_name = ALLOWED_HAZARD_CLASSES.get(class_id, f'class_{class_id}')

            detections.append({
                'bbox_xyxy': (x1, y1, x2, y2),
                'class_name': class_name,
                'class_id': class_id,
                'confidence': 0.5  # Default if not available
            })

    return detections
