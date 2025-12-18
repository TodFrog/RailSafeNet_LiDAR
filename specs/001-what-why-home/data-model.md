# Data Model: Enhanced Rail Hazard Detection - 3-Phase Optimization

**Feature Branch**: `001-what-why-home`
**Created**: 2025-10-14
**Revised**: 2025-10-16 (Updated for parallel processing, temporal tracking, IMM filter)
**Status**: Phase 2 Complete - Ready for Phase 3

## Overview

This document defines core data entities for all three implementation phases:
- **Phase 2 (Complete)**: Base entities for frame processing and detection
- **Phase 3**: Parallel inference entities (CUDA streams, synchronization)
- **Phase 4**: Temporal tracking entities (Ego-track state, width profile)
- **Phase 5**: IMM filter entities (path hypotheses, junction detection)

---

## Phase 2 Entities (Existing - Complete)

### 1. Frame

**Description**: Single video frame with metadata

**Fields**:
- `frame_id: int` - Sequential frame number
- `timestamp: float` - Timestamp in seconds from video start
- `image: np.ndarray` - RGB image data (shape: [1080, 1920, 3])
- `resolution: Tuple[int, int]` - (height, width)
- `frame_time: float` - Wall-clock processing time

**Validation**:
- `image.shape == (1080, 1920, 3)` - Full HD RGB
- `image.dtype == np.uint8` - 8-bit color
- `frame_id >= 0`, `timestamp >= 0.0`

---

### 2. SegmentationResult

**Description**: SegFormer model output

**Fields**:
- `frame_id: int` - Parent frame reference
- `segmentation_mask: np.ndarray` - Class per pixel (shape: [1080, 1920], dtype: uint8)
- `class_labels: Dict[int, str]` - Class ID to name mapping
- `rail_classes: List[int]` - Rail class IDs (default: [1, 4, 9])
- `inference_time_ms: float` - Inference duration
- `roi_bounds: Tuple[int, int, int, int]` - ROI (y_min, y_max, x_min, x_max)

**Derived Properties**:
- `has_rails: bool` - True if any pixel in rail_classes
- `rail_pixel_count: int` - Total rail pixels
- `rail_coverage_ratio: float` - rail_pixel_count / (1080 * 1920)

---

### 3. DetectedObject

**Description**: YOLO detection result

**Fields**:
- `object_id: int` - Unique ID within frame
- `class_id: int` - YOLO class ID (0-39)
- `class_name: str` - Human-readable name
- `bbox_xywh: Tuple[float, float, float, float]` - Center + size
- `bbox_xyxy: Tuple[int, int, int, int]` - Corner coordinates
- `confidence: float` - Detection confidence (0.0-1.0)
- `is_moving: bool` - True if typically mobile class
- `danger_zone_id: Optional[int]` - Intersecting danger zone (None if outside)
- `criticality: int` - Danger level (-1=safe, 0=red, 1=orange, 2=yellow)

**Validation**:
- `0 <= class_id < 40` - Valid YOLO class
- `0.0 <= confidence <= 1.0`
- `criticality in [-1, 0, 1, 2]`

**Classification Constants**:
```python
ACCEPTED_MOVING = {0, 1, 2, 3, 7, 15, 16, 17, 18, 19}  # person, vehicles, animals
ACCEPTED_STATIONARY = {24, 25, 28, 36}  # backpack, umbrella, suitcase, skateboard
```

---

### 4. DangerZone

**Description**: Polygonal hazard region on rails

**Fields**:
- `zone_id: int` - ID (0=innermost/red, 1=middle/orange, 2=outer/yellow)
- `polygon: List[Tuple[int, int]]` - Boundary points
- `track_id: int` - Parent rail track reference
- `distance_threshold_mm: int` - Real-world distance from track center (80, 400, 1000)
- `color_code: str` - Visualization color ("red", "orange", "yellow")
- `area_pixels: int` - Total area

**Validation**:
- `len(polygon) >= 3` - Valid polygon
- `zone_id in [0, 1, 2]` - Valid zone level
- No self-intersections

---

### 5. ProcessingMetrics

**Description**: Performance measurements per frame

**Fields**:
- `frame_id: int`
- `total_time_ms: float` - End-to-end processing time
- `segmentation_time_ms: float` - SegFormer inference time
- `detection_time_ms: float` - YOLO inference time
- `danger_zone_time_ms: float` - Danger zone computation time
- `tracking_time_ms: float` - Temporal tracking update time (Phase 4+)
- `timestamp: float` - Wall-clock time
- `gpu_memory_mb: float` - GPU memory usage (if available)

**Derived Properties**:
- `fps: float` - 1000.0 / total_time_ms
- `meets_realtime_requirement: bool` - total_time_ms <= 40.0 (ideal), <= 50.0 (acceptable)

---

## Phase 3 Entities (Parallel Processing)

### 6. InferenceStream

**Description**: Wraps a CUDA stream for independent model execution

**Fields**:
- `stream_id: int` - Unique stream identifier (0 or 1)
- `cuda_stream: pycuda.driver.Stream` - PyCUDA stream object
- `device_mem: pycuda.driver.DeviceAllocation` - Allocated GPU memory
- `host_mem: np.ndarray` - Page-locked host memory
- `bindings: List[int]` - TensorRT binding addresses
- `is_busy: bool` - True if inference in progress
- `last_sync_time: float` - Last synchronization timestamp

**Validation**:
- `stream_id in [0, 1]` - Only two streams (SegFormer, YOLO)
- `is_busy` state must be managed correctly (no double-dispatch)

**Lifecycle**:
```
IDLE → dispatch_inference() → BUSY → synchronize() → IDLE
```

---

### 7. ParallelInferenceResult

**Description**: Combined result from parallel SegFormer + YOLO execution

**Fields**:
- `frame_id: int`
- `segmentation_result: SegmentationResult` - From Stream 0
- `detection_results: List[DetectedObject]` - From Stream 1
- `segmentation_time_ms: float` - Stream 0 time
- `detection_time_ms: float` - Stream 1 time
- `parallel_overlap_ms: float` - Actual overlap time (measure of true parallelism)
- `synchronization_overhead_ms: float` - Time spent waiting for synchronization
- `success: bool` - True if both inferences succeeded
- `error_message: Optional[str]` - Error details if failed

**Derived Properties**:
- `effective_speedup: float` - (seg_time + det_time) / max(seg_time, det_time)
- `is_truly_parallel: bool` - parallel_overlap_ms > 0.7 * min(seg_time, det_time)

**Validation**:
- If `success == True`, both segmentation_result and detection_results must be valid
- `parallel_overlap_ms >= 0` - Cannot be negative

---

### 8. CUDAStreamState

**Description**: State tracking for CUDA stream management

**Fields**:
- `num_active_streams: int` - Currently active streams (0, 1, or 2)
- `total_dispatches: int` - Lifetime dispatch count
- `total_synchronizations: int` - Lifetime sync count
- `failed_dispatches: int` - Failed dispatch count (CUDA errors)
- `average_sync_time_ms: float` - Average synchronization overhead
- `cuda_context_valid: bool` - True if CUDA context is healthy

**Derived Properties**:
- `success_rate: float` - (total_dispatches - failed_dispatches) / total_dispatches
- `average_parallelism: float` - Estimated from sync times vs. individual inference times

**Validation**:
- `0 <= num_active_streams <= 2`
- `total_synchronizations <= total_dispatches` - Can't sync more than dispatch

---

## Phase 4 Entities (Temporal Tracking)

### 9. EgoTrackState

**Description**: Temporal state of the Ego-track (primary rail track being followed)

**Fields**:
- `frame_id: int` - Current frame
- `track_id: int` - Persistent track identifier (remains stable across frames)
- `left_edge: np.ndarray` - Left boundary points (shape: [N, 2], dtype: int)
- `right_edge: np.ndarray` - Right boundary points (shape: [N, 2], dtype: int)
- `center_line: np.ndarray` - Computed center points (shape: [N, 2])
- `velocity: Tuple[float, float]` - (vx, vy) in pixels/frame
- `confidence: float` - Detection confidence (0.0-1.0)
- `frames_since_detection: int` - 0 = detected this frame, >0 = predicted
- `is_predicted: bool` - True if using Kalman prediction (not detected)
- `kalman_state: np.ndarray` - Kalman filter state vector (6D: x, y, vx, vy, w, h)
- `kalman_covariance: np.ndarray` - State uncertainty (6x6 matrix)

**Validation**:
- `len(left_edge) == len(right_edge) >= 3` - Minimum 3 points
- `0.0 <= confidence <= 1.0`
- `frames_since_detection >= 0`
- If `is_predicted == True`, then `frames_since_detection > 0`

**Derived Properties**:
- `is_tracking_valid: bool` - True if frames_since_detection < 10 and confidence > 0.5
- `average_width: float` - Mean distance between left and right edges
- `track_length_pixels: int` - Vertical span of track

**State Transitions**:
```
UNINITIALIZED → DETECTED → TRACKING ⇄ PREDICTED (occlusion)
                              ↓
                           LOST (frames_since_detection > 30)
```

---

### 10. RailWidthProfile

**Description**: Learned perspective-corrected rail width profile

**Fields**:
- `y_levels: np.ndarray` - Y-coordinates (e.g., [540, 541, ..., 1079])
- `widths: np.ndarray` - Rail width in pixels at each y-level
- `variances: np.ndarray` - Variance for each y-level (confidence bounds)
- `num_samples: int` - Number of frames used to learn profile
- `is_calibrated: bool` - True after 150+ frames (5-10 seconds)
- `min_width: float` - Minimum observed width (at farthest y)
- `max_width: float` - Maximum observed width (at nearest y)
- `perspective_slope: float` - Estimated vanishing point slope

**Validation**:
- `len(y_levels) == len(widths) == len(variances)`
- `num_samples >= 0`
- `is_calibrated == True` only if `num_samples >= 150`
- `min_width < max_width` - Perspective should make near rails wider

**Derived Properties**:
- `get_expected_width(y: int) -> float` - Returns expected width at y-level
- `get_width_bounds(y: int) -> Tuple[float, float]` - Returns (min, max) acceptable width at y
- `is_width_valid(y: int, observed_width: float) -> bool` - Checks if within ±20%

**Usage**:
```python
# During detection
for y, detected_width in detected_rails:
    if not profile.is_width_valid(y, detected_width):
        # Reject as false positive
        continue
```

---

### 11. TrackHistory

**Description**: Circular buffer of recent Ego-track states

**Fields**:
- `buffer: deque[EgoTrackState]` - Circular buffer (max 30 frames = 1 second at 30 FPS)
- `max_size: int` - Buffer capacity (default: 30)
- `current_idx: int` - Current write position
- `is_full: bool` - True after 30 frames

**Validation**:
- `len(buffer) <= max_size`
- All entries in buffer must have sequential frame_ids (no gaps)

**Derived Properties**:
- `get_recent(n: int) -> List[EgoTrackState]` - Returns last n frames
- `get_average_velocity() -> Tuple[float, float]` - Average velocity over buffer
- `get_average_position() -> Tuple[float, float]` - Average center position

**Usage**:
```python
# Predict during occlusion
if current_detection is None:
    avg_vel = history.get_average_velocity()
    last_state = history.buffer[-1]
    predicted_position = last_state.center + avg_vel
```

---

## Phase 5 Entities (IMM Filter)

### 12. PathHypothesis

**Description**: Single Kalman filter tracking one possible junction path

**Fields**:
- `path_id: int` - Unique path identifier (0, 1, 2)
- `kalman_filter: cv2.KalmanFilter` - OpenCV Kalman filter instance
- `state: np.ndarray` - State vector (6D: left_x, left_y, right_x, right_y, vx, vy)
- `covariance: np.ndarray` - State uncertainty (6x6 matrix)
- `probability: float` - Path probability (0.0-1.0, sum=1.0 across all paths)
- `measurement_likelihood: float` - Likelihood of recent measurement given this path
- `divergence_angle: float` - Angle relative to main track heading (degrees)
- `frames_active: int` - Number of frames this hypothesis has been active

**Validation**:
- `0 <= path_id <= 2` - Maximum 3 paths
- `0.0 <= probability <= 1.0`
- `measurement_likelihood >= 0.0`

**Derived Properties**:
- `is_dominant: bool` - True if probability > 0.9
- `predicted_position: np.ndarray` - Next frame prediction

---

### 13. IMMState

**Description**: Collection of active path hypotheses at junction

**Fields**:
- `frame_id: int` - Current frame
- `hypotheses: List[PathHypothesis]` - Active path hypotheses (2-3 paths)
- `num_paths: int` - Number of active paths
- `mode: str` - "INACTIVE", "JUNCTION_ENTRY", "TRACKING", "JUNCTION_EXIT"
- `frames_since_activation: int` - Frames since IMM mode enabled
- `dominant_path_id: Optional[int]` - Path with highest probability (if any >0.9)
- `weighted_output: np.ndarray` - Probability-weighted average Ego-track position

**Validation**:
- `2 <= num_paths <= 3` when mode != "INACTIVE"
- `len(hypotheses) == num_paths`
- Sum of probabilities across hypotheses == 1.0 (within tolerance 0.001)

**Derived Properties**:
- `should_collapse: bool` - True if one path >0.9 for 10+ frames
- `is_ambiguous: bool` - True if all paths have probability 0.2-0.5 (no clear winner)

**State Transitions**:
```
INACTIVE → JUNCTION_ENTRY (2+ paths detected)
         → TRACKING (maintain multiple hypotheses)
         → JUNCTION_EXIT (one path >0.9 for 10+ frames)
         → INACTIVE (collapse to single path)
```

---

### 14. JunctionEvent

**Description**: Detected junction entry/exit event

**Fields**:
- `frame_id: int` - Frame where junction detected
- `event_type: str` - "JUNCTION_ENTRY", "JUNCTION_EXIT"
- `num_paths: int` - Number of diverging paths detected
- `path_angles: List[float]` - Divergence angles for each path (degrees)
- `confidence: float` - Junction detection confidence (0.0-1.0)

**Validation**:
- `event_type in ["JUNCTION_ENTRY", "JUNCTION_EXIT"]`
- `num_paths >= 2` for JUNCTION_ENTRY
- `len(path_angles) == num_paths`

**Usage**:
```python
# Trigger IMM activation
if junction_event.event_type == "JUNCTION_ENTRY":
    imm_state.activate(junction_event.num_paths, junction_event.path_angles)
```

---

## Entity Relationships

```
Frame (1) ──┬── (1) SegmentationResult ──┐
            │                             ├── (1) ParallelInferenceResult (Phase 3)
            ├── (0..n) DetectedObject ───┘
            │
            ├── (1) ProcessingMetrics
            │
            ├── (1) EgoTrackState (Phase 4) ──┬── (1) RailWidthProfile
            │                                  └── (1) TrackHistory
            │
            └── (0..1) IMMState (Phase 5) ──── (2..3) PathHypothesis
                                            └── (0..1) JunctionEvent

InferenceStream (2) ──── (1) CUDAStreamState (Phase 3)
```

## Data Validation Summary

**Critical Validations** (must pass or reject):
- Frame resolution 1920x1080
- CUDA stream state consistency (no double-dispatch)
- Ego-track left/right edge point counts match
- IMM path probabilities sum to 1.0

**Warning Validations** (log but continue):
- Low detection confidence (< 0.5)
- High tracking prediction duration (>10 frames)
- Rail width outside profile bounds (>20% deviation)
- IMM ambiguous state (all paths 0.2-0.5 probability)

**Performance Validations** (monitoring):
- Parallel overlap >70% for true parallelism
- Tracking overhead <2ms per frame
- IMM overhead <8ms per frame
- GPU utilization >80% in Phase 3

---

## Example Data Flow

**Phase 3 - Parallel Processing**:
```
Frame → ParallelExecutor
      → Stream 0: SegFormer (35ms)
      → Stream 1: YOLO (25ms)
      → Synchronize (wait max(35ms, 25ms))
      → ParallelInferenceResult (overlap=25ms, speedup=1.7x)
```

**Phase 4 - Temporal Tracking**:
```
Frame → Detect rails → EgoTrackState
      → Update TrackHistory (add to buffer)
      → Update RailWidthProfile (if calibrating)
      → Check width constraints
      → If occlusion: predict from Kalman filter
      → Output: stable Ego-track
```

**Phase 5 - IMM Junction**:
```
Frame → Detect junction → JunctionEvent (ENTRY, 3 paths)
      → Activate IMMState (create 3 PathHypothesis)
      → Measure rails → Update each hypothesis
      → Compute probabilities [0.5, 0.3, 0.2]
      → Output: weighted_output = 0.5*path0 + 0.3*path1 + 0.2*path2
      → After 10 frames: path0 > 0.9 → Collapse to single path
```

