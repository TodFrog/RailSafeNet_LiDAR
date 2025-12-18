# Module Interface Contracts

**Feature Branch**: `001-what-why-home`
**Created**: 2025-10-14
**Status**: Phase 1 Design

## Overview

This document defines the public interfaces (APIs) for all new and refactored modules in the enhanced rail hazard detection system. These contracts serve as the specification for implementation and testing.

---

## 1. Rail Detection Module (`src/rail_detection/`)

### 1.1 SegmentationEngine

**Purpose**: Wrapper for TensorRT SegFormer model inference.

```python
class SegmentationEngine:
    """TensorRT-optimized SegFormer semantic segmentation engine."""

    def __init__(self, engine_path: str, image_size: Tuple[int, int] = (512, 896)):
        """
        Initialize segmentation engine.

        Args:
            engine_path: Path to TensorRT engine file
            image_size: Input size for model (height, width)

        Raises:
            FileNotFoundError: If engine file does not exist
            RuntimeError: If TensorRT engine cannot be loaded
        """
        ...

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run segmentation inference on image.

        Args:
            image: Preprocessed image tensor (shape: [1, 3, H, W], dtype: float32)

        Returns:
            Segmentation mask (shape: [1080, 1920], dtype: uint8)
            Class IDs from 0-12

        Raises:
            ValueError: If image shape is incorrect
            RuntimeError: If inference fails
        """
        ...

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape (batch, channels, height, width)."""
        ...

    def close(self):
        """Release GPU resources."""
        ...
```

**Contract Guarantees**:
- Output mask always matches target resolution (1080x1920)
- Class IDs are in range [0, 12]
- Inference time < 40ms on Nvidia Titan RTX

---

### 1.2 DetectionEngine

**Purpose**: Wrapper for TensorRT YOLO model inference.

```python
class DetectionEngine:
    """TensorRT-optimized YOLO object detection engine."""

    def __init__(self, engine_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize detection engine.

        Args:
            engine_path: Path to TensorRT engine file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Raises:
            FileNotFoundError: If engine file does not exist
            RuntimeError: If TensorRT engine cannot be loaded
        """
        ...

    def predict(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Run object detection on image.

        Args:
            image: RGB image (shape: [H, W, 3], dtype: uint8)

        Returns:
            List of DetectedObject with bounding boxes, classes, confidences

        Raises:
            ValueError: If image format is incorrect
            RuntimeError: If inference fails
        """
        ...

    @property
    def class_names(self) -> Dict[int, str]:
        """Get mapping of class IDs to human-readable names."""
        ...

    def close(self):
        """Release GPU resources."""
        ...
```

**Contract Guarantees**:
- All returned objects have confidence >= conf_threshold
- NMS already applied (no overlapping boxes for same class)
- Inference time < 25ms on Nvidia Titan RTX

---

### 1.3 RailTracker

**Purpose**: Temporal continuity tracking using Kalman filtering.

```python
class RailTracker:
    """Kalman filter-based rail track temporal tracker."""

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 1.0):
        """
        Initialize rail tracker.

        Args:
            process_noise: Process noise covariance (motion uncertainty)
            measurement_noise: Measurement noise covariance (detection uncertainty)
        """
        ...

    def update(self, detected_track: Optional[RailTrack], frame_time: float) -> RailTrack:
        """
        Update tracker with new detection or predict if missing.

        Args:
            detected_track: Detected rail track (None if detection failed)
            frame_time: Timestamp for this frame

        Returns:
            Updated or predicted RailTrack with tracking confidence

        Behavior:
            - If detected_track provided: Update Kalman filter, return corrected track
            - If detected_track is None: Predict position, return predicted track
            - Confidence decreases with consecutive missing detections
        """
        ...

    def reset(self):
        """Reset tracker state (call when track is lost)."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if tracker has received initial measurement."""
        ...

    @property
    def frames_since_update(self) -> int:
        """Get number of frames since last valid detection."""
        ...

    @property
    def tracking_confidence(self) -> float:
        """Get current tracking confidence (0.0-1.0)."""
        ...
```

**Contract Guarantees**:
- Tracking confidence degrades linearly: 1.0 → 0.0 over 10 frames without detection
- Predicted tracks are marked with `is_predicted=True` flag
- Tracker automatically resets if confidence < 0.3

---

### 1.4 VanishingPointEstimator

**Purpose**: Estimate perspective vanishing point for rail filtering.

```python
class VanishingPointEstimator:
    """Vanishing point estimation using Hough transform and RANSAC."""

    def __init__(self, cache_frames: int = 10, angle_threshold: float = 10.0):
        """
        Initialize vanishing point estimator.

        Args:
            cache_frames: Number of frames to cache VP estimate
            angle_threshold: Max angle deviation for track filtering (degrees)
        """
        ...

    def estimate(self, segmentation_mask: np.ndarray, frame_id: int) -> VanishingPoint:
        """
        Estimate vanishing point from segmentation mask.

        Args:
            segmentation_mask: Rail segmentation (shape: [1080, 1920])
            frame_id: Current frame number (for cache management)

        Returns:
            VanishingPoint with (x, y) coordinates and confidence

        Behavior:
            - If cached VP is valid (age < cache_frames): return cached
            - Otherwise: compute new VP using Hough + RANSAC
            - If estimation fails: return previous VP with reduced confidence
        """
        ...

    def filter_tracks(self, rail_tracks: List[RailTrack], vp: VanishingPoint) -> List[RailTrack]:
        """
        Filter rail tracks by convergence to vanishing point.

        Args:
            rail_tracks: All detected rail tracks
            vp: Vanishing point for filtering

        Returns:
            Subset of tracks that converge toward VP within angle_threshold

        Behavior:
            - Compute convergence angle for each track
            - Keep tracks with angle < angle_threshold
            - Always keep at least 1 track (ego track, closest to center)
        """
        ...

    def invalidate_cache(self):
        """Force recomputation on next estimate() call."""
        ...
```

**Contract Guarantees**:
- VP estimation time < 8ms
- Cached VP returned in < 0.1ms
- At least 1 track always passes filtering (ego track)

---

### 1.5 DangerZoneComputer

**Purpose**: Compute danger zones from rail tracks.

```python
class DangerZoneComputer:
    """Danger zone polygon computation for rail tracks."""

    def __init__(self,
                 track_width_mm: int = 1435,
                 danger_distances_mm: List[int] = [100, 400, 1000]):
        """
        Initialize danger zone computer.

        Args:
            track_width_mm: Standard rail track width (e.g., 1435mm for standard gauge)
            danger_distances_mm: Distances from track center for zone boundaries
        """
        ...

    def compute_zones(self, rail_track: RailTrack, frame_shape: Tuple[int, int]) -> List[DangerZone]:
        """
        Compute danger zones for a rail track.

        Args:
            rail_track: Rail track with left/right boundaries
            frame_shape: Frame dimensions (height, width)

        Returns:
            List of DangerZone polygons (innermost to outermost)

        Raises:
            ValueError: If rail_track boundaries are invalid
        """
        ...

    def check_intersection(self, obj: DetectedObject, zones: List[DangerZone]) -> Optional[int]:
        """
        Check if detected object intersects any danger zone.

        Args:
            obj: Detected object with bounding box
            zones: List of danger zones (sorted by priority)

        Returns:
            Zone ID of highest-priority intersecting zone, or None if no intersection

        Behavior:
            - Test zones in order (innermost first)
            - Use center point of bounding box for intersection test
            - Return immediately upon first intersection (highest priority)
        """
        ...
```

**Contract Guarantees**:
- Danger zone polygons are simple (no self-intersections)
- Zones are continuous (no gaps) from y_min to y_max
- Computation time < 15ms per track

---

## 2. Processing Module (`src/processing/`)

### 2.1 FrameProcessor

**Purpose**: Orchestrate full frame processing pipeline.

```python
class FrameProcessor:
    """Enhanced frame processing pipeline with temporal tracking and VP filtering."""

    def __init__(self,
                 segmentation_engine: SegmentationEngine,
                 detection_engine: DetectionEngine,
                 enable_tracking: bool = True,
                 enable_vp_filtering: bool = True):
        """
        Initialize frame processor.

        Args:
            segmentation_engine: Segmentation model wrapper
            detection_engine: Object detection model wrapper
            enable_tracking: Enable temporal continuity tracking
            enable_vp_filtering: Enable vanishing point filtering
        """
        ...

    def process_frame(self, frame: Frame) -> Tuple[List[DangerZone], List[DetectedObject], ProcessingMetrics]:
        """
        Process a single frame through full pipeline.

        Args:
            frame: Input frame with image data

        Returns:
            Tuple of (danger_zones, detected_objects, metrics)

        Pipeline Steps:
            1. Segmentation inference
            2. Rail extent detection
            3. Rail track identification
            4. Temporal tracking update (if enabled)
            5. Vanishing point estimation and filtering (if enabled)
            6. Danger zone computation
            7. Object detection inference
            8. Danger zone intersection check
            9. Metrics collection

        Raises:
            RuntimeError: If critical pipeline step fails
        """
        ...

    def reset(self):
        """Reset temporal state (trackers, VP cache)."""
        ...

    @property
    def performance_stats(self) -> Dict[str, float]:
        """Get average timings for each pipeline component."""
        ...
```

**Contract Guarantees**:
- Total processing time target: < 40ms per frame
- Graceful degradation: If VP/tracking fails, continue with basic detection
- All exceptions are caught and logged, pipeline never crashes mid-stream

---

### 2.2 VideoProcessor

**Purpose**: Process video streams with session management.

```python
class VideoProcessor:
    """Video stream processor with session metrics."""

    def __init__(self, frame_processor: FrameProcessor, max_frames: Optional[int] = None):
        """
        Initialize video processor.

        Args:
            frame_processor: Frame processor instance
            max_frames: Maximum frames to process (None = unlimited)
        """
        ...

    def process_video(self, video_path: str,
                      output_dir: Optional[str] = None,
                      visualize: bool = True) -> SessionMetrics:
        """
        Process entire video file.

        Args:
            video_path: Path to input video
            output_dir: Directory for output visualizations (None = no output)
            visualize: Generate visualization images

        Returns:
            SessionMetrics with aggregated statistics

        Behavior:
            - Open video with cv2.VideoCapture
            - Process frames sequentially
            - Track performance metrics
            - Optionally save visualizations
            - Handle errors gracefully (log and continue)

        Raises:
            FileNotFoundError: If video file does not exist
        """
        ...

    def process_stream(self,
                       frame_generator: Callable[[], Optional[np.ndarray]],
                       callback: Optional[Callable[[Frame, List[DangerZone], List[DetectedObject]], None]] = None) -> SessionMetrics:
        """
        Process live video stream.

        Args:
            frame_generator: Function that yields frames (None = end of stream)
            callback: Optional callback for each processed frame

        Returns:
            SessionMetrics when stream ends

        Behavior:
            - Call frame_generator() to get frames
            - Process each frame through pipeline
            - Call callback() with results
            - Continue until frame_generator returns None
        """
        ...
```

**Contract Guarantees**:
- Video processing never crashes on bad frames (skip and log)
- Session metrics always returned, even if processing stopped early
- Memory efficient: No accumulation of processed frames

---

### 2.3 PerformanceMonitor

**Purpose**: Track and report performance metrics.

```python
class PerformanceMonitor:
    """Real-time performance monitoring and reporting."""

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of frames for rolling statistics
        """
        ...

    def record(self, metrics: ProcessingMetrics):
        """Record metrics for a processed frame."""
        ...

    def get_current_stats(self) -> Dict[str, float]:
        """
        Get current performance statistics.

        Returns:
            Dictionary with:
                - average_fps: Mean FPS over window
                - min_fps: Minimum FPS in window
                - max_fps: Maximum FPS in window
                - p95_latency_ms: 95th percentile latency
                - variance_percent: Frame time variance
        """
        ...

    def is_meeting_targets(self) -> bool:
        """
        Check if performance meets targets.

        Returns:
            True if average_fps >= 25 and variance_percent <= 20
        """
        ...

    def generate_report(self) -> str:
        """Generate human-readable performance report."""
        ...
```

**Contract Guarantees**:
- Statistics computed incrementally (no full re-scan)
- Rolling window maintains constant memory usage
- Report generation < 1ms

---

## 3. Utils Module (`src/utils/`)

### 3.1 Geometry Utilities

```python
def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Bresenham's line algorithm for pixel-perfect line drawing."""
    ...

def interpolate_boundary(points: List[Tuple[int, int]],
                         gaps: List[int]) -> List[Tuple[int, int]]:
    """Interpolate missing points in boundary with gap handling."""
    ...

def is_simple_polygon(polygon: List[Tuple[int, int]]) -> bool:
    """Check if polygon has no self-intersections."""
    ...

def compute_convergence_angle(track: RailTrack, vp: VanishingPoint) -> float:
    """Compute angle between track direction and vanishing point."""
    ...
```

---

### 3.2 Visualization Utilities

```python
def visualize_frame(frame: Frame,
                    segmentation: np.ndarray,
                    danger_zones: List[DangerZone],
                    detections: List[DetectedObject],
                    save_path: Optional[str] = None) -> np.ndarray:
    """
    Generate visualization of frame processing results.

    Args:
        frame: Original frame
        segmentation: Segmentation mask
        danger_zones: Computed danger zones
        detections: Detected objects
        save_path: Optional path to save image

    Returns:
        Visualization image (RGB, 1920x1080)
    """
    ...

def create_summary_grid(frames: List[Frame],
                        results: List[Tuple[List[DangerZone], List[DetectedObject]]],
                        grid_size: Tuple[int, int] = (2, 3)) -> np.ndarray:
    """Create grid visualization of multiple frame results."""
    ...
```

---

## 4. Configuration Management

```python
@dataclass
class RailDetectionConfig:
    """Configuration for rail detection system."""

    # Model paths
    segmentation_engine_path: str
    detection_engine_path: str

    # Processing parameters
    enable_tracking: bool = True
    enable_vp_filtering: bool = True
    roi_height_fraction: float = 0.5  # ROI extends to 1/2 of frame height

    # Tracking parameters
    tracking_process_noise: float = 0.1
    tracking_measurement_noise: float = 1.0

    # Vanishing point parameters
    vp_cache_frames: int = 10
    vp_angle_threshold: float = 10.0

    # Danger zone parameters
    track_width_mm: int = 1435
    danger_distances_mm: List[int] = field(default_factory=lambda: [100, 400, 1000])

    # Performance parameters
    target_fps: float = 25.0
    max_variance_percent: float = 20.0

    @classmethod
    def from_file(cls, config_path: str) -> 'RailDetectionConfig':
        """Load configuration from JSON/YAML file."""
        ...

    def to_file(self, config_path: str):
        """Save configuration to JSON/YAML file."""
        ...
```

---

## Contract Testing Requirements

Each interface must have corresponding contract tests:

1. **Input Validation Tests**: Verify all ValueError and RuntimeError cases
2. **Output Format Tests**: Verify return types and shapes
3. **Performance Tests**: Verify timing guarantees with pytest-benchmark
4. **Boundary Tests**: Test edge cases (empty inputs, maximum sizes)
5. **Integration Tests**: Verify interfaces work together correctly

Example:
```python
# tests/unit/test_segmentation_engine.py

def test_segmentation_engine_output_shape(segmentation_engine, sample_frame):
    """Contract: Output mask must be 1080x1920."""
    result = segmentation_engine.infer(sample_frame)
    assert result.shape == (1080, 1920)

def test_segmentation_engine_performance(segmentation_engine, sample_frame, benchmark):
    """Contract: Inference must complete in < 40ms."""
    result = benchmark(segmentation_engine.infer, sample_frame)
    assert benchmark.stats['mean'] < 0.040  # 40ms
```
