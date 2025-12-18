"""
Geometry utilities for rail detection and danger zone computation.

This module provides geometric operations used throughout the detection pipeline.
"""

from typing import List, Tuple, Optional
import numpy as np
from .data_models import RailTrack, VanishingPoint


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Bresenham's line algorithm for pixel-perfect line drawing.

    Args:
        x0, y0: Start point coordinates
        x1, y1: End point coordinates

    Returns:
        List of (x, y) coordinates along the line

    Example:
        >>> points = bresenham_line(0, 0, 5, 3)
        >>> len(points) > 0
        True
    """
    line = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        line.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return line


def interpolate_boundary(points: List[Tuple[int, int]],
                        gaps: List[int]) -> List[Tuple[int, int]]:
    """
    Interpolate missing points in boundary with gap handling.

    Args:
        points: List of boundary points (x, y)
        gaps: List of indices where gaps occur

    Returns:
        Interpolated boundary points

    Note:
        - If points < 2, returns empty list
        - Uses Bresenham line algorithm for interpolation
        - Skips interpolation at gap indices
    """
    if len(points) < 2:
        return []

    line_arr = []

    # Convert gaps to set for O(1) lookup
    gap_set = set(gaps) if gaps else set()

    for i in range(len(points) - 1):
        if i in gap_set:
            continue

        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        # Generate line segment
        segment = bresenham_line(x1, y1, x2, y2)

        # Filter out negative x coordinates
        segment = [(x, y) for x, y in segment if x >= 0]

        line_arr.extend(segment)

    return line_arr


def is_simple_polygon(polygon: List[Tuple[int, int]]) -> bool:
    """
    Check if polygon has no self-intersections.

    Args:
        polygon: List of (x, y) vertices

    Returns:
        True if polygon is simple (no self-intersections), False otherwise

    Note:
        This is a simplified check. For production, consider more robust algorithms.
    """
    if len(polygon) < 3:
        return False

    # Simplified check: ensure no duplicate consecutive points
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        if p1 == p2:
            return False

    # TODO: Implement full self-intersection check using line segment intersection
    # For now, assume polygon is simple if no duplicate consecutive points
    return True


def compute_convergence_angle(track: RailTrack, vp: VanishingPoint) -> float:
    """
    Compute angle between track direction and vanishing point.

    Args:
        track: Rail track with left/right boundaries
        vp: Vanishing point

    Returns:
        Convergence angle in degrees

    Note:
        - Positive angle means track converges toward VP
        - Angle should typically be < 45 degrees for valid tracks
        - Uses center line of track for angle calculation
    """
    if not track.center_line or len(track.center_line) < 2:
        # Calculate center line from boundaries
        if len(track.left_boundary) < 2:
            return 0.0

        # Use two points from far and near ends
        lx1, ly1 = track.left_boundary[0]
        rx1, ry1 = track.right_boundary[0]
        cx1, cy1 = (lx1 + rx1) / 2, (ly1 + ry1) / 2

        lx2, ly2 = track.left_boundary[-1]
        rx2, ry2 = track.right_boundary[-1]
        cx2, cy2 = (lx2 + rx2) / 2, (ly2 + ry2) / 2
    else:
        # Use existing center line
        cx1, cy1 = track.center_line[0]
        cx2, cy2 = track.center_line[-1]

    # Vector from near to far point of track
    track_vec = np.array([cx1 - cx2, cy1 - cy2])

    # Vector from track center to vanishing point
    track_center_x = (cx1 + cx2) / 2
    track_center_y = (cy1 + cy2) / 2
    vp_vec = np.array([vp.x - track_center_x, vp.y - track_center_y])

    # Calculate angle between vectors
    track_norm = np.linalg.norm(track_vec)
    vp_norm = np.linalg.norm(vp_vec)

    if track_norm == 0 or vp_norm == 0:
        return 0.0

    # Dot product and angle
    cos_angle = np.dot(track_vec, vp_vec) / (track_norm * vp_norm)
    # Clamp to [-1, 1] to avoid numerical errors in arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def find_nearest_pairs(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find nearest point pairs between two arrays.

    Args:
        arr1: First array of points (N x 2)
        arr2: Second array of points (M x 2)

    Returns:
        Tuple of (paired_arr1, paired_arr2) with matched points

    Note:
        Uses the smaller array as the base and finds nearest neighbors in the larger array.
    """
    if len(arr1) == 0 or len(arr2) == 0:
        return np.array([]), np.array([])

    # Use smaller array as base
    if len(arr1) < len(arr2):
        base_array = arr1
        compare_array = arr2
        swap = False
    else:
        base_array = arr2
        compare_array = arr1
        swap = True

    paired_base = []
    paired_compare = []
    paired_mask = np.zeros(len(compare_array), dtype=bool)

    for item in base_array:
        # Calculate distances to all points in compare_array
        distances = np.linalg.norm(compare_array - item, axis=1)
        nearest_index = np.argmin(distances)

        paired_base.append(item)
        paired_compare.append(compare_array[nearest_index])
        paired_mask[nearest_index] = True

        if paired_mask.all():
            break

    paired_base = np.array(paired_base)
    paired_compare = compare_array[paired_mask]

    # Return in original order
    if swap:
        return paired_compare, paired_base
    else:
        return paired_base, paired_compare


def compute_line_intersection(line1: Tuple[Tuple[int, int], Tuple[int, int]],
                              line2: Tuple[Tuple[int, int], Tuple[int, int]]) -> Optional[Tuple[float, float]]:
    """
    Compute intersection point of two lines.

    Args:
        line1: ((x1, y1), (x2, y2)) - First line segment
        line2: ((x3, y3), (x4, y4)) - Second line segment

    Returns:
        (x, y) intersection point, or None if lines are parallel

    Note:
        Uses line equation: (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1)
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        # Lines are parallel
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Compute intersection point
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (float(x), float(y))


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        point: (x, y) point to test
        polygon: List of (x, y) vertices defining polygon

    Returns:
        True if point is inside polygon, False otherwise

    Note:
        Uses ray casting algorithm (count intersections with horizontal ray)
    """
    if len(polygon) < 3:
        return False

    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]

        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x, p1y = p2x, p2y

    return inside
