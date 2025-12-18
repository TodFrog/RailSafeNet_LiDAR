# Developer Quickstart: Enhanced Rail Hazard Detection

**Feature Branch**: `001-what-why-home`
**Created**: 2025-10-14
**Last Updated**: 2025-10-14

## Overview

This guide helps developers quickly understand the enhanced rail hazard detection system architecture, set up their development environment, and begin contributing.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Video Input Stream                        │
│                         (1920x1080 @ 30 FPS)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FrameProcessor                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. Segmentation (SegFormer TensorRT)        ~35ms      │   │
│  │     ↓                                                     │   │
│  │  2. Rail Extent Detection                     ~2ms       │   │
│  │     ↓                                                     │   │
│  │  3. Rail Track Identification                 ~3ms       │   │
│  │     ↓                                                     │   │
│  │  4. Temporal Tracking (Kalman)    [NEW]      ~0.5ms     │   │
│  │     ↓                                                     │   │
│  │  5. Vanishing Point Filter        [NEW]      ~3ms       │   │
│  │     ↓                                                     │   │
│  │  6. Danger Zone Computation                   ~10ms      │   │
│  │     ├─────────────────┬─────────────────┐               │   │
│  │     ▼                 ▼                 ▼               │   │
│  │  Red Zone (100mm)  Orange (400mm)  Yellow (1000mm)     │   │
│  │                                                           │   │
│  │  7. Object Detection (YOLO TensorRT)          ~20ms      │   │
│  │     ↓                                                     │   │
│  │  8. Danger Zone Intersection Check            ~1ms       │   │
│  │     ↓                                                     │   │
│  │  9. Visualization (optional)                  ~5ms       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Total: ~38ms → 26 FPS ✅  (Target: 25-30 FPS)                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output: Danger Zones + Classified Objects + Performance Metrics│
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Setup

### Prerequisites

**Development Environment:**
- **Hardware**: Nvidia GPU with CUDA support (Titan RTX or equivalent)
- **OS**: Linux (tested on Ubuntu with kernel 5.15.0)
- **Python**: 3.8+
- **CUDA**: 11.x or compatible with TensorRT 8.x

**Production Deployment:**
- **Target Platform**: **Nvidia AGX Orin**
  - GPU: Ampere architecture (2048 CUDA cores, 64 Tensor cores)
  - Memory: 32GB or 64GB LPDDR5
  - AI Performance: Up to 275 TOPS
  - Expected FPS: 25-30 (target)
- **Deployment Method**: Docker Container
  - Base image: `nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime`
  - NVIDIA Container Runtime required
  - TensorRT engines must be rebuilt for AGX Orin architecture

**Note**: See `docs/DEPLOYMENT.md` for detailed deployment guide including Docker containerization and AGX Orin setup.

### Installation

```bash
# 1. Clone repository
cd /home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR

# 2. Install dependencies
pip install -r requirements.txt

# Core dependencies (if requirements.txt not ready):
pip install numpy opencv-python torch torchvision tensorrt pycuda
pip install albumentations matplotlib scikit-learn pytest pytest-cov

# 3. Verify TensorRT engines exist
ls assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine
ls assets/models_pretrained/yolo/yolov8s_896x512.engine

# 4. Verify test data exists
ls assets/crop/tram0.mp4
# Test videos: tram0.mp4 ~ tram135.mp4 (136 videos, each 30 seconds)
# Total: 68 minutes of tram front-facing camera footage
# Resolution: 1920x1080 @ 30 FPS (900 frames per video)
```

---

## Project Structure

```
RailSafeNet_LiDAR/
├── src/
│   ├── rail_detection/        # Core detection logic
│   │   ├── segmentation.py    # SegFormer wrapper
│   │   ├── detection.py       # YOLO wrapper
│   │   ├── rail_tracker.py    # [NEW] Temporal tracking
│   │   ├── vanishing_point.py # [NEW] VP estimation
│   │   └── danger_zone.py     # Danger zone computation
│   │
│   ├── processing/            # Pipeline orchestration
│   │   ├── frame_processor.py # Main processing pipeline
│   │   ├── video_processor.py # Video/stream handling
│   │   └── performance.py     # Performance monitoring
│   │
│   └── utils/                 # Utilities
│       ├── geometry.py        # Geometric operations
│       ├── visualization.py   # Visualization helpers
│       └── config.py          # Configuration management
│
├── tests/
│   ├── unit/                  # Fast, isolated tests
│   ├── integration/           # Pipeline tests
│   └── fixtures/              # Test data
│
├── specs/001-what-why-home/   # This feature's documentation
│   ├── spec.md                # Feature specification
│   ├── plan.md                # Implementation plan
│   ├── research.md            # Research findings
│   ├── data-model.md          # Entity definitions
│   ├── contracts/             # API contracts
│   └── quickstart.md          # This file
│
└── video_frame_tester.py      # Legacy script (to be refactored)
```

---

## Running the System

### Basic Usage (Existing Script)

```bash
# Process test video using existing script
python video_frame_tester.py

# Expected output:
# 📥 Loading models...
# ✅ Models loaded successfully
# 🎬 Processing video: tram0.mp4
# 📹 Extracted 30 frames from video (30 seconds at 30 FPS)
# ✅ Frame 0: Rail=True, Zone=True, Objects=3, Danger=1, Time=0.82s
# ...

# Test videos available:
# - Location: /home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/crop/
# - Files: tram0.mp4 ~ tram135.mp4 (136 test videos)
# - Each video: 30 seconds, 900 frames, 1920x1080 resolution
```

### Using Enhanced Modules (After Implementation)

```python
from src.rail_detection import SegmentationEngine, DetectionEngine
from src.processing import FrameProcessor, VideoProcessor
from src.utils.config import RailDetectionConfig

# Load configuration
config = RailDetectionConfig(
    segmentation_engine_path="assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine",
    detection_engine_path="assets/models_pretrained/yolo/yolov8s_896x512.engine",
    enable_tracking=True,
    enable_vp_filtering=True
)

# Initialize engines
seg_engine = SegmentationEngine(config.segmentation_engine_path)
det_engine = DetectionEngine(config.detection_engine_path)

# Create frame processor
frame_processor = FrameProcessor(
    segmentation_engine=seg_engine,
    detection_engine=det_engine,
    enable_tracking=config.enable_tracking,
    enable_vp_filtering=config.enable_vp_filtering
)

# Process video
video_processor = VideoProcessor(frame_processor)
metrics = video_processor.process_video(
    video_path="/path/to/test/video.mp4",
    output_dir="./output",
    visualize=True
)

print(f"Average FPS: {metrics.average_fps:.1f}")
print(f"Success Rate: {metrics.success_rate * 100:.1f}%")
```

---

## Development Workflow

### 1. Setting Up Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-benchmark pytest-mock

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html
```

### 2. Running Performance Benchmarks

```python
# tests/benchmarks/test_performance.py

import pytest
from src.processing import FrameProcessor

@pytest.mark.benchmark
def test_frame_processing_speed(frame_processor, sample_frame, benchmark):
    """Verify frame processing meets 25 FPS target (40ms)."""
    result = benchmark(frame_processor.process_frame, sample_frame)

    # Assert timing target
    assert benchmark.stats['mean'] < 0.040  # 40ms = 25 FPS
```

### 3. Adding New Features

**Example: Add new rail detection algorithm**

1. **Define interface** in `contracts/module_interfaces.md`
2. **Create data model** if needed in `data-model.md`
3. **Write tests first** (TDD approach):
   ```python
   # tests/unit/test_new_feature.py
   def test_new_feature_basic():
       feature = NewFeature()
       result = feature.process(input_data)
       assert result.is_valid()
   ```
4. **Implement feature** in `src/rail_detection/new_feature.py`
5. **Verify coverage**: `pytest --cov=src.rail_detection.new_feature`
6. **Benchmark performance**: Add performance test
7. **Update documentation**: Update relevant specs

---

## Key Concepts

### Temporal Tracking

**What**: Kalman filter tracks rail position across frames
**Why**: Handles temporary occlusions (objects blocking rails)
**When to use**: Always enabled in production (minimal overhead)

```python
from src.rail_detection.rail_tracker import RailTracker

tracker = RailTracker()

# In frame loop:
detected_track = detect_rails(frame)  # May be None if detection fails
tracked_track = tracker.update(detected_track, frame.timestamp)

# tracked_track is corrected or predicted
if tracked_track.is_predicted:
    print(f"Using prediction (confidence: {tracker.tracking_confidence:.2f})")
```

### Vanishing Point Filtering

**What**: Filters rail tracks by convergence to perspective vanishing point
**Why**: Eliminates false detections in multi-track scenarios
**When to use**: Enable for complex rail environments (junctions, parallel tracks)

```python
from src.rail_detection.vanishing_point import VanishingPointEstimator

vp_estimator = VanishingPointEstimator(cache_frames=10)

# Estimate VP (cached for 10 frames)
vp = vp_estimator.estimate(segmentation_mask, frame_id=100)

# Filter tracks
all_tracks = detect_all_tracks(segmentation_mask)  # May find 3-4 tracks
valid_tracks = vp_estimator.filter_tracks(all_tracks, vp)  # Keep 1-2 converging tracks
```

### Danger Zones

**What**: Three concentric zones around rails (Red=100mm, Orange=400mm, Yellow=1000mm)
**Why**: Prioritize hazards by proximity to rails
**How**: Computed from rail boundaries using real-world scale calibration

```python
from src.rail_detection.danger_zone import DangerZoneComputer

dz_computer = DangerZoneComputer(
    track_width_mm=1435,  # Standard gauge
    danger_distances_mm=[100, 400, 1000]
)

zones = dz_computer.compute_zones(rail_track, frame_shape=(1080, 1920))
# zones[0] = innermost (red), zones[1] = middle (orange), zones[2] = outer (yellow)

# Check object intersection
for obj in detected_objects:
    zone_id = dz_computer.check_intersection(obj, zones)
    if zone_id == 0:
        print(f"CRITICAL: {obj.class_name} in RED zone!")
```

---

## Common Tasks

### Task 1: Process a Test Video

```bash
# Using existing script with default test video
python video_frame_tester.py

# Process specific test video (30 seconds, 900 frames)
python video_frame_tester.py --video assets/crop/tram10.mp4 --output ./results

# Process multiple test videos for comparison
python video_frame_tester.py --video assets/crop/tram0.mp4 --output ./results/tram0
python video_frame_tester.py --video assets/crop/tram50.mp4 --output ./results/tram50
python video_frame_tester.py --video assets/crop/tram135.mp4 --output ./results/tram135
```

### Task 2: Benchmark Performance

```bash
# Run performance tests
pytest tests/benchmarks/ --benchmark-only

# Generate performance report
pytest tests/benchmarks/ --benchmark-autosave --benchmark-save-data
```

### Task 3: Visualize Segmentation

```python
from src.rail_detection import SegmentationEngine
from src.utils.visualization import visualize_segmentation
import cv2

# Load engine
engine = SegmentationEngine("assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine")

# Load test frame
frame = cv2.imread("tests/fixtures/sample_frames/frame_001.jpg")

# Run segmentation
mask = engine.infer(preprocess_frame(frame))

# Visualize
vis_image = visualize_segmentation(frame, mask)
cv2.imwrite("segmentation_result.jpg", vis_image)
```

### Task 4: Debug Vanishing Point

```python
from src.rail_detection.vanishing_point import VanishingPointEstimator
import matplotlib.pyplot as plt

vp_estimator = VanishingPointEstimator()
vp = vp_estimator.estimate(segmentation_mask, frame_id=0)

print(f"VP Location: ({vp.x:.1f}, {vp.y:.1f})")
print(f"Confidence: {vp.confidence:.2f}")
print(f"Inliers: {vp.num_inliers}")

# Visualize
plt.imshow(frame)
plt.scatter(vp.x, vp.y, c='red', s=100, marker='x')
plt.title(f"Vanishing Point (conf={vp.confidence:.2f})")
plt.show()
```

---

## Testing Guidelines

### Unit Tests

**Goal**: Test individual functions in isolation
**Coverage Target**: 90%+

```python
# tests/unit/test_rail_tracker.py

def test_tracker_initialization():
    tracker = RailTracker()
    assert not tracker.is_initialized
    assert tracker.tracking_confidence == 0.0

def test_tracker_update_with_detection(sample_rail_track):
    tracker = RailTracker()
    result = tracker.update(sample_rail_track, frame_time=0.0)

    assert tracker.is_initialized
    assert tracker.tracking_confidence == 1.0
    assert not result.is_predicted

def test_tracker_prediction_without_detection():
    tracker = RailTracker()
    # Initialize with one detection
    tracker.update(sample_rail_track, frame_time=0.0)

    # Predict without detection
    result = tracker.update(None, frame_time=0.033)

    assert result.is_predicted
    assert tracker.tracking_confidence < 1.0
    assert tracker.frames_since_update == 1
```

### Integration Tests

**Goal**: Test component interactions
**Coverage Target**: 85%+

```python
# tests/integration/test_frame_processor.py

def test_full_pipeline_with_rails(frame_processor, sample_frame_with_rails):
    zones, objects, metrics = frame_processor.process_frame(sample_frame_with_rails)

    assert len(zones) == 3  # Red, orange, yellow
    assert metrics.total_time_ms < 40.0  # Performance target
    assert metrics.meets_realtime_requirement

def test_pipeline_without_rails(frame_processor, sample_frame_no_rails):
    zones, objects, metrics = frame_processor.process_frame(sample_frame_no_rails)

    assert len(zones) == 0  # No danger zones
    assert len(objects) >= 0  # Objects still detected
    # Pipeline should not crash
```

### Performance Tests

**Goal**: Verify timing requirements
**Tool**: pytest-benchmark

```python
# tests/benchmarks/test_performance.py

@pytest.mark.benchmark(group="segmentation")
def test_segmentation_speed(segmentation_engine, preprocessed_frame, benchmark):
    result = benchmark(segmentation_engine.infer, preprocessed_frame)
    assert benchmark.stats['mean'] < 0.040  # 40ms target

@pytest.mark.benchmark(group="detection")
def test_detection_speed(detection_engine, sample_frame, benchmark):
    result = benchmark(detection_engine.predict, sample_frame)
    assert benchmark.stats['mean'] < 0.025  # 25ms target
```

---

## Performance Optimization Tips

### 1. Use TensorRT Engines (Already Implemented)
- SegFormer: ~35ms (vs ~80ms PyTorch)
- YOLO: ~20ms (vs ~40ms PyTorch)

### 2. Vectorize NumPy Operations
```python
# BAD: Python loop
edges = []
for y in range(y_min, y_max):
    row_mask = (segmentation[y, :] == 4) | (segmentation[y, :] == 9)
    if np.any(row_mask):
        edges.append(find_edge(row_mask))

# GOOD: Vectorized
rail_mask = np.isin(segmentation, [4, 9])
rows_with_rails = np.any(rail_mask, axis=1)
y_indices = np.where(rows_with_rails)[0]
```

### 3. Cache Heavy Computations
```python
# Cache vanishing point for 10 frames
vp_estimator = VanishingPointEstimator(cache_frames=10)

# Cache morphological operations
@functools.lru_cache(maxsize=1)
def get_structuring_element():
    return cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
```

### 4. Profile Hotspots
```bash
# Use cProfile
python -m cProfile -o profile.stats video_frame_tester.py

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats

# Or use line_profiler for function-level profiling
pip install line_profiler
kernprof -l -v video_frame_tester.py
```

---

## Troubleshooting

### Issue: TensorRT engine fails to load

```
Error: Could not deserialize engine
```

**Solution**:
- Verify CUDA version matches engine build
- Check TensorRT version compatibility
- Regenerate engine if needed:
  ```bash
  python onnx_to_engine.py --input model.onnx --output model.engine
  ```

### Issue: Low FPS (< 20 FPS)

**Checklist**:
1. Verify GPU utilization: `nvidia-smi` should show >80%
2. Check for CPU bottlenecks: Profile with cProfile
3. Disable visualization during profiling
4. Ensure TensorRT engines are being used (not PyTorch models)

### Issue: Tracking confidence drops quickly

**Possible Causes**:
- Process noise too high (tracker overshoots)
- Measurement noise too low (tracker ignores detections)

**Solution**:
```python
# Tune Kalman filter parameters
tracker = RailTracker(
    process_noise=0.05,  # Lower = smoother tracking
    measurement_noise=2.0  # Higher = trust measurements more
)
```

### Issue: Tests failing due to missing fixtures

```bash
# Generate test fixtures
python scripts/generate_test_fixtures.py --output tests/fixtures/
```

---

## Next Steps

1. **Implement Core Modules**:
   - Start with `src/rail_detection/rail_tracker.py`
   - Then `src/rail_detection/vanishing_point.py`
   - Finally integrate into `src/processing/frame_processor.py`

2. **Establish Testing**:
   - Set up pytest configuration
   - Create fixture generators
   - Write unit tests for each module

3. **Performance Tuning**:
   - Benchmark each component
   - Identify bottlenecks
   - Optimize hotspots

4. **Integration**:
   - Refactor `video_frame_tester.py` to use new modules
   - Add command-line flags for feature toggles
   - Test on full video datasets

---

## Resources

- **Feature Spec**: `specs/001-what-why-home/spec.md`
- **Implementation Plan**: `specs/001-what-why-home/plan.md`
- **Research Findings**: `specs/001-what-why-home/research.md`
- **Data Model**: `specs/001-what-why-home/data-model.md`
- **API Contracts**: `specs/001-what-why-home/contracts/`

## Contact

For questions about this feature implementation, refer to the specification documents or create an issue in the project repository.
