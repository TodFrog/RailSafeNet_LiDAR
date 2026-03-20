"""
Data models for RailSafeNet LiDAR system.

This module defines all core entities used throughout the rail hazard detection pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np


# Enums for state management

class FrameState(Enum):
    """Frame processing state."""
    CREATED = "created"
    PREPROCESSING = "preprocessing"
    SEGMENTED = "segmented"
    DETECTED = "detected"
    ANALYZED = "analyzed"
    COMPLETED = "completed"
    FAILED = "failed"


class VanishingPointState(Enum):
    """Vanishing point computation state."""
    NOT_COMPUTED = "not_computed"
    COMPUTING = "computing"
    VALID = "valid"
    CACHED = "cached"
    STALE = "stale"
    RECOMPUTING = "recomputing"
    INVALID = "invalid"


class TrackingState(Enum):
    """Rail tracking state."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    PREDICTING = "predicting"
    LOST = "lost"
    REINITIALIZED = "reinitialized"


# Core Entity Classes

@dataclass
class Frame:
    """
    Represents a single video frame with associated metadata.

    Attributes:
        frame_id: Sequential frame number in video
        timestamp: Timestamp in seconds from video start
        image: Raw RGB image data (shape: [1080, 1920, 3])
        resolution: Image dimensions (height, width)
        frame_time: Wall-clock time when frame was captured/processed
        state: Current processing state
    """
    frame_id: int
    timestamp: float
    image: np.ndarray
    resolution: Tuple[int, int]
    frame_time: float
    state: FrameState = FrameState.CREATED

    def __post_init__(self):
        """Validate frame data."""
        if self.image.shape != (1080, 1920, 3):
            raise ValueError(f"Frame must be 1080x1920x3, got {self.image.shape}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"Frame must be uint8, got {self.image.dtype}")
        if self.frame_id < 0:
            raise ValueError(f"Frame ID must be non-negative, got {self.frame_id}")
        if self.timestamp < 0.0:
            raise ValueError(f"Timestamp must be non-negative, got {self.timestamp}")


@dataclass
class SegmentationResult:
    """
    Output of SegFormer semantic segmentation model.

    Attributes:
        frame_id: Reference to parent Frame
        segmentation_mask: Class predictions per pixel (shape: [1080, 1920])
        class_labels: Mapping of class IDs to names
        rail_classes: Classes considered as rail (default: [1, 4, 9])
        inference_time_ms: Segmentation inference duration
        roi_bounds: Region of interest (y_min, y_max, x_min, x_max)
    """
    frame_id: int
    segmentation_mask: np.ndarray
    class_labels: Dict[int, str]
    rail_classes: List[int] = field(default_factory=lambda: [1, 4, 9])
    inference_time_ms: float = 0.0
    roi_bounds: Tuple[int, int, int, int] = (0, 1079, 0, 1919)

    def __post_init__(self):
        """Validate segmentation result."""
        if self.segmentation_mask.shape != (1080, 1920):
            raise ValueError(f"Mask must be 1080x1920, got {self.segmentation_mask.shape}")
        if np.any(self.segmentation_mask >= 13):
            raise ValueError(f"Class IDs must be < 13 for 13-class model")
        if len(self.rail_classes) == 0:
            raise ValueError("Must specify at least one rail class")
        if self.inference_time_ms < 0:
            raise ValueError(f"Inference time must be positive, got {self.inference_time_ms}")

    @property
    def has_rails(self) -> bool:
        """Check if any pixel belongs to rail classes."""
        return np.any(np.isin(self.segmentation_mask, self.rail_classes))

    @property
    def rail_pixel_count(self) -> int:
        """Count total number of rail pixels."""
        return np.sum(np.isin(self.segmentation_mask, self.rail_classes))

    @property
    def rail_coverage_ratio(self) -> float:
        """Calculate rail coverage ratio."""
        return self.rail_pixel_count / (1080 * 1920)


@dataclass
class RailExtent:
    """
    Vertical and horizontal extent of detected rail regions.

    Attributes:
        y_min: Topmost row containing rail pixels
        y_max: Bottommost row containing rail pixels
        vertical_span: y_max - y_min + 1
        edges_by_row: Rail edge segments per y-level
    """
    y_min: int
    y_max: int
    vertical_span: int
    edges_by_row: Dict[int, List[Tuple[int, int]]]

    def __post_init__(self):
        """Validate rail extent."""
        if not (0 <= self.y_min < self.y_max <= 1079):
            raise ValueError(f"Invalid y bounds: {self.y_min}, {self.y_max}")
        if self.vertical_span < 50:
            raise ValueError(f"Minimum rail visibility is 50 pixels, got {self.vertical_span}")

        # Validate edges
        for y, edges in self.edges_by_row.items():
            for start, end in edges:
                if not (0 < start < end < 1920):
                    raise ValueError(f"Invalid edge at y={y}: ({start}, {end})")
                if (end - start) < 3:
                    raise ValueError(f"Minimum edge width is 3 pixels, got {end - start}")

    @property
    def num_tracks(self) -> int:
        """Estimate number of distinct rail tracks."""
        if not self.edges_by_row:
            return 0
        # Average number of edge pairs across rows
        avg_edges = np.mean([len(edges) for edges in self.edges_by_row.values()])
        return int(avg_edges)

    @property
    def is_straight(self) -> bool:
        """Check if edges have nearly constant slopes."""
        # Simplified: check if x-coordinates change linearly with y
        if len(self.edges_by_row) < 2:
            return True
        # TODO: Implement slope variance calculation
        return False

    @property
    def is_curved(self) -> bool:
        """Check if edges have varying slopes."""
        return not self.is_straight


@dataclass
class RailTrack:
    """
    A single identified rail track with geometric properties.

    Attributes:
        track_id: Unique identifier within frame
        left_boundary: Left edge points [(x, y), ...] from far to near
        right_boundary: Right edge points [(x, y), ...] from far to near
        center_line: Computed center points
        width_profile: Track width at each y-level (pixels)
        convergence_angle: Angle toward vanishing point (degrees)
        is_ego_track: True if this is the primary track
        confidence: Detection confidence (0.0-1.0)
        is_predicted: True if track from temporal tracking prediction
    """
    track_id: int
    left_boundary: List[Tuple[int, int]]
    right_boundary: List[Tuple[int, int]]
    center_line: List[Tuple[int, int]] = field(default_factory=list)
    width_profile: List[int] = field(default_factory=list)
    convergence_angle: float = 0.0
    is_ego_track: bool = False
    confidence: float = 1.0
    is_predicted: bool = False

    def __post_init__(self):
        """Validate rail track."""
        if len(self.left_boundary) != len(self.right_boundary):
            raise ValueError("Left and right boundaries must have same length")
        if len(self.left_boundary) < 3:
            raise ValueError(f"Minimum 3 points for valid track, got {len(self.left_boundary)}")

        # Validate left < right
        for (lx, ly), (rx, ry) in zip(self.left_boundary, self.right_boundary):
            if lx >= rx:
                raise ValueError(f"Left boundary must be left of right at y={ly}")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

        if abs(self.convergence_angle) > 45:
            raise ValueError(f"Convergence angle should be < 45 degrees, got {self.convergence_angle}")

    @property
    def average_width(self) -> float:
        """Calculate mean track width."""
        if not self.width_profile:
            # Calculate from boundaries
            widths = [rx - lx for (lx, _), (rx, _) in zip(self.left_boundary, self.right_boundary)]
            return float(np.mean(widths))
        return float(np.mean(self.width_profile))

    @property
    def track_length(self) -> int:
        """Calculate vertical span of track."""
        if not self.left_boundary:
            return 0
        y_coords = [y for _, y in self.left_boundary]
        return max(y_coords) - min(y_coords) + 1

    @property
    def curvature(self) -> float:
        """Measure of track bending (0 = straight, higher = more curved)."""
        # TODO: Implement curvature calculation
        return 0.0


@dataclass
class VanishingPoint:
    """
    Estimated perspective vanishing point for rail tracks.

    Attributes:
        frame_id: Reference frame
        x: X-coordinate in frame
        y: Y-coordinate in frame
        confidence: Estimation confidence (0.0-1.0)
        num_inliers: Number of lines converging to this point (RANSAC)
        last_updated_frame: Frame when VP was last recomputed
        cache_valid: True if cached VP is still valid
        state: Current VP state
    """
    frame_id: int
    x: float
    y: float
    confidence: float
    num_inliers: int
    last_updated_frame: int
    cache_valid: bool = True
    state: VanishingPointState = VanishingPointState.VALID

    def __post_init__(self):
        """Validate vanishing point."""
        # Allow VP slightly outside frame (perspective)
        if not (-500 <= self.x <= 2420):
            raise ValueError(f"VP x-coordinate out of range: {self.x}")
        if not (-500 <= self.y <= 1580):
            raise ValueError(f"VP y-coordinate out of range: {self.y}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.num_inliers < 3:
            raise ValueError(f"Minimum 3 inliers for reliable VP, got {self.num_inliers}")

    @property
    def is_cached(self) -> bool:
        """Check if VP is from cache."""
        return self.frame_id != self.last_updated_frame

    @property
    def cache_age(self) -> int:
        """Get cache age in frames."""
        return self.frame_id - self.last_updated_frame


@dataclass
class TrackingStateData:
    """
    Kalman filter state for temporal rail tracking.

    Attributes:
        state_vector: [left_x, left_y, right_x, right_y, vel_x, vel_y]
        covariance_matrix: State uncertainty (6x6)
        last_measurement: Most recent detected rail edges (4D)
        frames_since_update: Frames since last valid detection
        is_initialized: True after first reliable detection
        confidence: Tracking confidence (0.0-1.0)
        state: Current tracking state
    """
    state_vector: np.ndarray
    covariance_matrix: np.ndarray
    last_measurement: np.ndarray
    frames_since_update: int
    is_initialized: bool
    confidence: float
    state: TrackingState = TrackingState.TRACKING

    def __post_init__(self):
        """Validate tracking state."""
        if self.state_vector.shape != (6,):
            raise ValueError(f"State vector must be shape (6,), got {self.state_vector.shape}")
        if self.covariance_matrix.shape != (6, 6):
            raise ValueError(f"Covariance must be shape (6, 6), got {self.covariance_matrix.shape}")
        if self.frames_since_update < 0:
            raise ValueError(f"Frames since update must be non-negative")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_tracking_valid(self) -> bool:
        """Check if tracking is valid."""
        return self.frames_since_update < 5 and self.confidence > 0.5

    @property
    def predicted_position(self) -> np.ndarray:
        """Get predicted next frame position."""
        # Return position components of state vector
        return self.state_vector[:4]


@dataclass
class DangerZone:
    """
    Polygonal hazard region on rail track.

    Attributes:
        zone_id: Unique identifier (0=innermost, 1=middle, 2=outer)
        polygon: Boundary points defining zone
        track_id: Reference to parent RailTrack
        distance_threshold_mm: Real-world distance from track center
        color_code: Visualization color ("red", "orange", "yellow")
        area_pixels: Total area of zone in pixels
    """
    zone_id: int
    polygon: List[Tuple[int, int]]
    track_id: int
    distance_threshold_mm: int
    color_code: str
    area_pixels: int = 0

    def __post_init__(self):
        """Validate danger zone."""
        if len(self.polygon) < 3:
            raise ValueError(f"Polygon must have >= 3 points, got {len(self.polygon)}")
        if self.zone_id not in [0, 1, 2]:
            raise ValueError(f"Zone ID must be 0, 1, or 2, got {self.zone_id}")
        if self.distance_threshold_mm <= 0:
            raise ValueError(f"Distance threshold must be positive, got {self.distance_threshold_mm}")
        # TODO: Implement self-intersection check

    @property
    def is_continuous(self) -> bool:
        """Check if polygon has no gaps."""
        # TODO: Implement continuity check
        return True

    @property
    def vertical_span(self) -> int:
        """Calculate vertical extent of zone."""
        if not self.polygon:
            return 0
        y_coords = [y for _, y in self.polygon]
        return max(y_coords) - min(y_coords) + 1


@dataclass
class DetectedObject:
    """
    Object detected by YOLO model in the frame.

    Attributes:
        object_id: Unique identifier within frame
        class_id: YOLO class ID
        class_name: Human-readable class name
        bbox_xywh: Bounding box center + size (x, y, width, height)
        bbox_xyxy: Bounding box corners (x1, y1, x2, y2)
        confidence: Detection confidence (0.0-1.0)
        is_moving: True if object class is typically mobile
        danger_zone_id: ID of intersecting DangerZone (None if outside)
        criticality: Danger level (-1=safe, 0=red, 1=orange, 2=yellow)
    """
    object_id: int
    class_id: int
    class_name: str
    bbox_xywh: Tuple[float, float, float, float]
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    is_moving: bool
    danger_zone_id: Optional[int] = None
    criticality: int = -1

    # Classification constants
    ACCEPTED_MOVING = {0, 1, 2, 3, 7, 15, 16, 17, 18, 19}
    ACCEPTED_STATIONARY = {24, 25, 28, 36}

    def __post_init__(self):
        """Validate detected object."""
        if not (0 <= self.class_id < 40):
            raise ValueError(f"Class ID must be in [0, 40), got {self.class_id}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.criticality not in [-1, 0, 1, 2]:
            raise ValueError(f"Criticality must be in [-1, 0, 1, 2], got {self.criticality}")

        # Validate bounding box
        _, _, w, h = self.bbox_xywh
        if w <= 0 or h <= 0:
            raise ValueError(f"Bbox width and height must be positive, got w={w}, h={h}")

    @property
    def is_in_danger(self) -> bool:
        """Check if object is in danger zone."""
        return self.danger_zone_id is not None

    @property
    def area(self) -> int:
        """Calculate bounding box area."""
        _, _, w, h = self.bbox_xywh
        return int(w * h)


@dataclass
class ProcessingMetrics:
    """
    Performance measurements for frame processing.

    Attributes:
        frame_id: Reference to parent Frame
        total_time_ms: End-to-end processing time
        segmentation_time_ms: SegFormer inference time
        detection_time_ms: YOLO inference time
        danger_zone_time_ms: Danger zone computation time
        tracking_time_ms: Temporal tracking update time
        vp_estimation_time_ms: Vanishing point estimation time
        visualization_time_ms: Rendering time
        timestamp: Wall-clock time of measurement
        gpu_memory_mb: GPU memory usage (if available)
    """
    frame_id: int
    total_time_ms: float
    segmentation_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    danger_zone_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    vp_estimation_time_ms: float = 0.0
    visualization_time_ms: float = 0.0
    timestamp: float = 0.0
    gpu_memory_mb: float = 0.0

    def __post_init__(self):
        """Validate processing metrics."""
        if self.total_time_ms < 0:
            raise ValueError(f"Total time must be non-negative, got {self.total_time_ms}")

        # Check component times are non-negative
        for field_name in ['segmentation_time_ms', 'detection_time_ms', 'danger_zone_time_ms',
                          'tracking_time_ms', 'vp_estimation_time_ms', 'visualization_time_ms']:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")

    @property
    def fps(self) -> float:
        """Calculate frames per second."""
        if self.total_time_ms == 0:
            return 0.0
        return 1000.0 / self.total_time_ms

    @property
    def meets_realtime_requirement(self) -> bool:
        """Check if processing time meets 25 FPS target (40ms)."""
        return self.total_time_ms <= 40.0


@dataclass
class SessionMetrics:
    """
    Aggregated statistics for a video processing session.

    Attributes:
        session_id: Unique session identifier
        total_frames: Number of frames processed
        successful_frames: Frames with successful rail detection
        failed_frames: Frames that encountered errors
        average_fps: Mean FPS across all frames
        min_fps: Minimum FPS observed
        max_fps: Maximum FPS observed
        p95_latency_ms: 95th percentile frame processing time
        variance_percent: Standard deviation / mean * 100
        multi_track_frames: Frames with 2+ detected tracks
        junction_frames: Frames with track intersections
    """
    session_id: str
    total_frames: int
    successful_frames: int
    failed_frames: int
    average_fps: float
    min_fps: float
    max_fps: float
    p95_latency_ms: float
    variance_percent: float
    multi_track_frames: int = 0
    junction_frames: int = 0

    def __post_init__(self):
        """Validate session metrics."""
        if self.total_frames != self.successful_frames + self.failed_frames:
            raise ValueError(f"Frame accounting error: {self.total_frames} != {self.successful_frames} + {self.failed_frames}")
        if not (self.min_fps <= self.average_fps <= self.max_fps):
            raise ValueError(f"FPS ordering violated: min={self.min_fps}, avg={self.average_fps}, max={self.max_fps}")
        if self.variance_percent < 0:
            raise ValueError(f"Variance must be non-negative, got {self.variance_percent}")

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_frames == 0:
            return 0.0
        return self.successful_frames / self.total_frames

    @property
    def meets_performance_target(self) -> bool:
        """Check if performance meets targets (25 FPS, 20% variance)."""
        return self.average_fps >= 25.0 and self.variance_percent <= 20.0


# =============================================================================
# Phase 4 Entities: Temporal Tracking
# =============================================================================

@dataclass
class EgoTrackState:
    """
    Temporal state of the Ego-track (primary rail track being followed).

    This class maintains the state of the ego rail track across frames,
    enabling temporal tracking with Kalman filtering for occlusion handling.

    Attributes:
        frame_id: Current frame number
        track_id: Persistent track identifier (stable across frames)
        left_edge: Left boundary points (N, 2) as [(x, y), ...]
        right_edge: Right boundary points (N, 2) as [(x, y), ...]
        center_line: Computed center points (N, 2)
        velocity: Track velocity (vx, vy) in pixels/frame
        confidence: Detection confidence (0.0-1.0)
        frames_since_detection: 0 = detected, >0 = predicted
        is_predicted: True if using Kalman prediction
        kalman_state: Kalman filter state vector (6D: x, y, vx, vy, w, h)
        kalman_covariance: State uncertainty (6x6 matrix)
    """
    frame_id: int
    track_id: int
    left_edge: np.ndarray  # shape: (N, 2)
    right_edge: np.ndarray  # shape: (N, 2)
    center_line: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 1.0
    frames_since_detection: int = 0
    is_predicted: bool = False
    kalman_state: Optional[np.ndarray] = None
    kalman_covariance: Optional[np.ndarray] = None
    polynomial_coeffs: Optional[Tuple[float, float, float]] = None  # (c, b, a) for x = ay^2 + by + c

    def __post_init__(self):
        """Validate ego track state."""
        if len(self.left_edge) != len(self.right_edge):
            raise ValueError(f"Left/right edge length mismatch: {len(self.left_edge)} != {len(self.right_edge)}")
        if len(self.left_edge) < 3:
            raise ValueError(f"Minimum 3 points required, got {len(self.left_edge)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.frames_since_detection < 0:
            raise ValueError(f"Frames since detection must be non-negative")
        if self.is_predicted and self.frames_since_detection == 0:
            raise ValueError("Predicted track must have frames_since_detection > 0")

        # Validate Kalman state if present
        if self.kalman_state is not None:
            if self.kalman_state.shape != (6,):
                raise ValueError(f"Kalman state must be (6,), got {self.kalman_state.shape}")
        if self.kalman_covariance is not None:
            if self.kalman_covariance.shape != (6, 6):
                raise ValueError(f"Kalman covariance must be (6, 6), got {self.kalman_covariance.shape}")

    @property
    def is_tracking_valid(self) -> bool:
        """Check if tracking is valid (recent detection + good confidence)."""
        return self.frames_since_detection < 10 and self.confidence > 0.5

    @property
    def average_width(self) -> float:
        """Calculate mean track width."""
        widths = np.linalg.norm(self.right_edge - self.left_edge, axis=1)
        return float(np.mean(widths))

    @property
    def track_length_pixels(self) -> int:
        """Calculate vertical span of track."""
        if len(self.left_edge) == 0:
            return 0
        y_coords = self.left_edge[:, 1]
        return int(np.ptp(y_coords) + 1)


@dataclass
class RailWidthProfile:
    """
    Learned perspective-corrected rail width profile.

    This profile is learned over 150-300 frames to understand how rail
    width changes with perspective (distance from camera).

    Attributes:
        y_levels: Y-coordinates sampled (e.g., [540, 541, ..., 1079])
        widths: Rail width in pixels at each y-level
        variances: Variance at each y-level (confidence bounds)
        num_samples: Number of frames used to learn profile
        is_calibrated: True after 150+ frames
        min_width: Minimum observed width (farthest point)
        max_width: Maximum observed width (nearest point)
        perspective_slope: Estimated vanishing point slope
    """
    y_levels: np.ndarray  # shape: (M,)
    widths: np.ndarray  # shape: (M,)
    variances: np.ndarray  # shape: (M,)
    num_samples: int = 0
    is_calibrated: bool = False
    min_width: float = 0.0
    max_width: float = 0.0
    perspective_slope: float = 0.0

    def __post_init__(self):
        """Validate rail width profile."""
        if len(self.y_levels) != len(self.widths) or len(self.widths) != len(self.variances):
            raise ValueError(f"Array length mismatch: y_levels={len(self.y_levels)}, widths={len(self.widths)}, variances={len(self.variances)}")
        if self.num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {self.num_samples}")
        if self.is_calibrated and self.num_samples < 150:
            raise ValueError(f"Calibration requires 150+ samples, got {self.num_samples}")
        if len(self.widths) > 0 and self.min_width >= self.max_width:
            # Allow equal only if all widths are same
            if not np.allclose(self.widths, self.widths[0]):
                raise ValueError(f"min_width must be < max_width, got {self.min_width} >= {self.max_width}")

    def get_expected_width(self, y: int) -> float:
        """Get expected width at y-level."""
        if len(self.y_levels) == 0:
            return 0.0
        # Linear interpolation
        idx = np.searchsorted(self.y_levels, y)
        if idx == 0:
            return float(self.widths[0])
        if idx >= len(self.y_levels):
            return float(self.widths[-1])
        # Interpolate
        y1, y2 = self.y_levels[idx-1], self.y_levels[idx]
        w1, w2 = self.widths[idx-1], self.widths[idx]
        ratio = (y - y1) / (y2 - y1)
        return float(w1 + ratio * (w2 - w1))

    def get_width_bounds(self, y: int) -> Tuple[float, float]:
        """Get acceptable width range at y-level (±20%)."""
        expected = self.get_expected_width(y)
        return (expected * 0.8, expected * 1.2)

    def is_width_valid(self, y: int, observed_width: float) -> bool:
        """Check if observed width is within acceptable bounds."""
        if not self.is_calibrated:
            return True  # Accept all widths during calibration
        min_w, max_w = self.get_width_bounds(y)
        return min_w <= observed_width <= max_w


@dataclass
class TrackHistory:
    """
    Circular buffer of recent Ego-track states.

    Maintains the last 30 frames (1 second at 30 FPS) of track history
    for temporal prediction during occlusions.

    Attributes:
        buffer: Circular buffer of EgoTrackState (max 30 frames)
        max_size: Buffer capacity (default: 30)
        current_idx: Current write position
        is_full: True after 30 frames collected
    """
    buffer: List[EgoTrackState] = field(default_factory=list)
    max_size: int = 30
    current_idx: int = 0
    is_full: bool = False

    def __post_init__(self):
        """Validate track history."""
        if len(self.buffer) > self.max_size:
            raise ValueError(f"Buffer size {len(self.buffer)} exceeds max_size {self.max_size}")
        if self.max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {self.max_size}")

    def add(self, state: EgoTrackState):
        """Add new state to history."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(state)
            self.current_idx = len(self.buffer) - 1
            if len(self.buffer) == self.max_size:
                self.is_full = True
        else:
            # Circular buffer: overwrite oldest
            self.current_idx = (self.current_idx + 1) % self.max_size
            self.buffer[self.current_idx] = state

    def get_recent(self, n: int) -> List[EgoTrackState]:
        """Get last n frames."""
        if n <= 0:
            return []
        if n >= len(self.buffer):
            return list(self.buffer)
        # Return last n elements
        if len(self.buffer) < self.max_size:
            return self.buffer[-n:]
        else:
            # Circular buffer: get n elements before current_idx
            indices = [(self.current_idx - i) % self.max_size for i in range(n-1, -1, -1)]
            return [self.buffer[i] for i in indices]

    def get_average_velocity(self) -> Tuple[float, float]:
        """Calculate average velocity over buffer."""
        if len(self.buffer) == 0:
            return (0.0, 0.0)
        velocities = np.array([state.velocity for state in self.buffer])
        return tuple(np.mean(velocities, axis=0))

    def get_average_position(self) -> Tuple[float, float]:
        """Calculate average center position."""
        if len(self.buffer) == 0:
            return (0.0, 0.0)
        # Average center line midpoint
        positions = []
        for state in self.buffer:
            if len(state.center_line) > 0:
                center = np.mean(state.center_line, axis=0)
                positions.append(center)
        if len(positions) == 0:
            return (0.0, 0.0)
        return tuple(np.mean(positions, axis=0))

    @property
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
