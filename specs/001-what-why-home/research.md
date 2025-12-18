# Research Document: Enhanced Rail Hazard Detection

**Feature Branch**: `001-what-why-home`
**Created**: 2025-10-14
**Status**: Phase 0 Complete

## Overview

This document consolidates research findings for enhancing rail track detection with temporal continuity tracking, vanishing point analysis, and extended ROI coverage while maintaining 25-30 FPS processing speed.

## Research Areas

### 1. Testing Framework Selection (CRITICAL - Constitution Requirement)

**Decision**: pytest with pytest-cov

**Rationale**:
- **pytest**: De facto standard for Python testing, widely adopted in computer vision projects
- **pytest-cov**: Seamless integration with pytest for coverage measurement
- **pytest-benchmark**: Available for performance regression testing (critical for FPS requirements)
- **pytest-mock**: Useful for mocking TensorRT engines in unit tests

**Alternatives Considered**:
- **unittest**: Python built-in, but less feature-rich and verbose compared to pytest
- **nose2**: Declining community support, pytest is the modern standard
- **doctest**: Insufficient for complex CV testing scenarios

**Implementation Plan**:
```python
# Directory structure
tests/
├── conftest.py           # Shared fixtures (model loading, sample data)
├── pytest.ini            # Configuration (coverage thresholds, markers)
├── unit/                 # Fast, isolated tests
├── integration/          # Pipeline tests
└── fixtures/
    ├── sample_frames/    # Curated test images
    └── expected_outputs/ # Ground truth for regression tests
```

**Coverage Strategy**:
- Target: 90% overall, 95% for safety-critical modules (rail_tracker.py, vanishing_point.py, danger_zone.py)
- Use `pytest --cov=src --cov-report=html --cov-report=term-missing`
- Branch coverage for conditional logic in rail detection
- Integration tests for full pipeline regression

**Key Testing Challenges**:
1. **TensorRT Engine Mocking**: Engines require GPU, unit tests need mock predictions
2. **Performance Testing**: Must verify 25+ FPS without actual GPU in CI/CD
3. **Visual Regression**: Danger zone visualization correctness

**Solutions**:
- Use recorded segmentation masks as fixtures (bypass TensorRT in unit tests)
- pytest-benchmark with performance thresholds as smoke tests
- Structural validation of danger zones (polygon validity, continuity) instead of pixel-perfect comparison

---

### 2. Temporal Continuity Tracking (Interactive Multiple Model Filter)

**Decision**: Kalman Filter with position and velocity tracking for rail boundaries

**Rationale**:
- Rail tracks have smooth, predictable motion (vehicle moving forward)
- Kalman Filter provides optimal state estimation with low computational cost
- State vector: [left_x, left_y, right_x, right_y, velocity_x, velocity_y]
- Measurement: detected rail edges from segmentation
- Handles temporary occlusions by predicting rail position during missing detections

**Alternatives Considered**:
- **Particle Filter**: More robust to non-Gaussian noise but computationally expensive (impacts FPS target)
- **Simple Moving Average**: Insufficient for handling gaps/occlusions
- **LSTM/RNN**: Overkill for this problem, would require training data and add latency

**Implementation Approach**:
```python
class RailTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(6, 4)  # 6 state vars, 4 measurements
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],  # left_x
            [0, 1, 0, 0, 0, 0],  # left_y
            [0, 0, 1, 0, 0, 0],  # right_x
            [0, 0, 0, 1, 0, 0]   # right_y
        ], np.float32)
        # Transition matrix for constant velocity model
        self.kalman.transitionMatrix = ...

    def update(self, detected_edges, frame_time):
        """Update tracker with new detections or predict if missing"""
        if detected_edges:
            self.kalman.correct(detected_edges)
        predicted = self.kalman.predict()
        return predicted
```

**Performance Impact**:
- Kalman Filter update: ~0.1-0.5ms per frame (negligible impact on 40ms budget)
- Benefit: Reduces false negatives in complex scenarios, improving stability

**Best Practices**:
- Initialize tracker on first reliable detection (at least 3 consecutive frames)
- Reset tracker if prediction diverges significantly from detection (>20% threshold)
- Use covariance matrix to assess confidence (skip prediction if uncertainty too high)

---

### 3. Vanishing Point Analysis

**Decision**: Probabilistic Hough Transform + RANSAC for robust vanishing point estimation

**Rationale**:
- Vanishing point in perspective projection = intersection of parallel rail lines
- Probabilistic Hough Transform detects rail edge lines efficiently
- RANSAC rejects outliers (non-rail lines, noise)
- Filters multiple rail tracks by selecting those converging to primary vanishing point

**Alternatives Considered**:
- **Edge Detection + Line Fitting**: Less robust to noise and texture variations
- **Deep Learning VP Estimation**: Requires training, adds model complexity
- **Manual VP Specification**: Inflexible, requires recalibration per camera

**Implementation Approach**:
```python
def estimate_vanishing_point(segmentation_mask, rail_classes=[4, 9]):
    # Extract rail edge pixels
    rail_mask = np.isin(segmentation_mask, rail_classes)
    edges = cv2.Canny(rail_mask.astype(np.uint8) * 255, 50, 150)

    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=50, minLineLength=100, maxLineGap=10)

    # Find line intersections
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            intersection = compute_intersection(line1, line2)
            if is_valid_vp(intersection, frame_shape):
                intersections.append(intersection)

    # RANSAC to find dominant vanishing point
    vp, inliers = ransac_vp(intersections, threshold=50)
    return vp

def filter_rails_by_vp(rail_tracks, vanishing_point, angle_threshold=10):
    """Keep only tracks that converge toward vanishing point"""
    valid_tracks = []
    for track in rail_tracks:
        track_angle = compute_convergence_angle(track, vanishing_point)
        if abs(track_angle) < angle_threshold:
            valid_tracks.append(track)
    return valid_tracks
```

**Performance Impact**:
- Hough Transform: ~2-5ms per frame
- RANSAC iterations: ~1-3ms
- Total: 3-8ms (within 40ms budget)

**Best Practices**:
- Compute vanishing point on downsampled frame (512x896) for speed
- Cache vanishing point across frames (update every 5-10 frames, not every frame)
- Use geometric priors: VP typically in upper half of frame for forward-facing camera
- Fallback: If VP estimation fails, use center-based track selection

---

### 4. ROI Extension (2/5 → 1/2 Frame Height)

**Decision**: Extend ROI from lower 432 pixels (2/5 of 1080) to lower 540 pixels (1/2 of 1080)

**Rationale**:
- Current ROI misses distant rail sections crucial for early hazard warning
- 108-pixel extension = ~20-30 meters additional detection range (camera-dependent)
- Balances detection range vs. computational load

**Performance Impact Analysis**:
- ROI area increase: (1080 * 2/5) → (1080 * 1/2) = +25% pixel area
- Segmentation model: Output is full-frame regardless of ROI (no impact)
- Edge detection processing: +25% pixels → est. +1-2ms
- Danger zone computation: Negligible (same number of edges, just longer extent)

**Mitigation Strategies**:
- Apply vanishing point filtering to reduce false tracks in distant region
- Use temporal tracking to smooth noise in distant detections
- Adaptive ROI: Expand to 1/2 only when rails detected in lower 2/5 region

**Implementation**:
```python
# Before (video_frame_tester.py)
y_roi_start = int(1080 * 3/5)  # 648

# After (enhanced version)
y_roi_start = int(1080 * 1/2)  # 540

# Adaptive approach
if rails_detected_in_lower_region:
    y_roi_start = int(1080 * 1/2)  # Extend upward
else:
    y_roi_start = int(1080 * 3/5)  # Keep original
```

---

### 5. Performance Optimization Strategies

**Goal**: Achieve 25-30 FPS (33-40ms per frame) on Nvidia Titan RTX

**Current Bottleneck Analysis** (from video_frame_tester.py, 77-83ms):
- Segmentation inference: ~30-40ms (TensorRT)
- YOLO detection: ~20-25ms (TensorRT)
- Danger zone computation: ~10-15ms (NumPy/OpenCV)
- Visualization: ~5-10ms (matplotlib)

**Optimization Decisions**:

1. **Use TensorRT Engines (Already Implemented)**
   - SegFormer: segformer_b3_transfer_best_0.7961.engine
   - YOLO: yolov8s_896x512.engine
   - **Impact**: Already optimized, no additional gain

2. **Pipeline Parallelization**
   - Run segmentation and detection in parallel using threading
   - Danger zone computation overlaps with next frame preprocessing
   - **Estimated Gain**: 10-15ms (15-20% speedup)

3. **NumPy Vectorization**
   - Replace Python loops in `find_edges`, `interpolate_end_points` with vectorized NumPy
   - Use `np.where`, `np.diff`, `np.cumsum` instead of list comprehensions
   - **Estimated Gain**: 3-5ms

4. **Reduce Visualization Overhead**
   - Decouple visualization from processing pipeline
   - Option to disable visualization during real-time operation
   - Use OpenCV drawing instead of matplotlib (3x faster)
   - **Estimated Gain**: 5-8ms

5. **Caching and Reuse**
   - Cache vanishing point (update every 10 frames)
   - Reuse Kalman filter state across frames
   - Cache morphological structuring elements
   - **Estimated Gain**: 1-2ms

**Cumulative Optimization Target**:
- Current: 77-83ms → Target: 33-40ms
- Required speedup: 40-50ms reduction
- Achievable through: Parallelization (15ms) + Vectorization (5ms) + Visualization (8ms) + Algorithm efficiency (10-15ms) = ~38-43ms

**Performance Testing Strategy**:
- Benchmark each component individually (pytest-benchmark)
- Profile with cProfile and line_profiler
- Monitor GPU utilization (nvidia-smi, ensure >80% utilization)
- Test on representative video clips (straight, curved, multi-track scenarios)

---

### 6. Integration with Existing Codebase

**Challenge**: Minimize disruption to existing TheDistanceAssessor_3_engine.py and video_frame_tester.py

**Strategy**: Incremental refactoring

**Phase 1 - Extract and Modularize**:
- Extract rail detection logic → `src/rail_detection/`
- Extract danger zone creation → `src/rail_detection/danger_zone.py`
- Extract geometry utilities → `src/utils/geometry.py`
- Keep existing files functional as-is

**Phase 2 - Enhance with New Features**:
- Add `RailTracker` class → `src/rail_detection/rail_tracker.py`
- Add `VanishingPointEstimator` → `src/rail_detection/vanishing_point.py`
- Create `EnhancedFrameProcessor` → `src/processing/frame_processor.py`

**Phase 3 - Integration**:
- Update `video_frame_tester.py` to use new modules
- Add command-line flags: `--enable-tracking`, `--enable-vp-filtering`
- Maintain backward compatibility with flag defaults

**Backward Compatibility**:
```python
# video_frame_tester_enhanced.py
if args.enable_tracking:
    tracker = RailTracker()
else:
    tracker = None

if args.enable_vp_filtering:
    vp_estimator = VanishingPointEstimator()
else:
    vp_estimator = None
```

---

## Summary of Key Decisions

| Research Area | Decision | Rationale | Performance Impact |
|--------------|----------|-----------|-------------------|
| **Testing Framework** | pytest + pytest-cov | Industry standard, good tooling | N/A (development only) |
| **Temporal Tracking** | Kalman Filter | Optimal for smooth motion, low cost | +0.1-0.5ms |
| **Vanishing Point** | Hough + RANSAC | Robust, no training required | +3-8ms |
| **ROI Extension** | 2/5 → 1/2 height | Better detection range | +1-2ms |
| **Optimization** | Parallelization + vectorization | Maximize GPU utilization | -38-43ms (gain) |
| **Architecture** | Modular refactoring | Maintainability + backward compat | No impact |

**Net Performance Change**: Current 77-83ms → Target 33-40ms ✅

**Constitution Compliance**: Testing infrastructure defined ✅

---

## Next Steps (Phase 1 - Design)

1. Create `data-model.md` defining key entities (RailTrack, DangerZone, Frame, DetectionResult)
2. Design API contracts for new modules
3. Create `quickstart.md` for developers
4. Update agent context with new technology decisions
