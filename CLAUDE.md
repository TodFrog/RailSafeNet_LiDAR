# RailSafeNet_LiDAR Development Guidelines

Production rail hazard detection system for tram/train applications.
Last updated: 2026-01-08

## Quick Start

### Video Mode (Testing)
```bash
# Native
python3 videoAssessor_final.py --mode video

# With fullscreen
python3 videoAssessor_final.py --mode video --fullscreen

# Using script
./scripts/run_video.sh
```

### Camera Mode (Production)
```bash
# Native
python3 videoAssessor_final.py --mode camera --fullscreen

# Using script
./scripts/run_camera.sh
```

### Calibration Modes
```bash
# BEV (Bird's Eye View) calibration
python3 videoAssessor_final.py --calibrate

# Vanishing Point calibration (for hazard zones)
python3 videoAssessor_final.py --calibrate-vp

# With specific video file
python3 videoAssessor_final.py --calibrate-vp --video path/to/video.mp4
```

### Docker Deployment (Jetson)
```bash
# Build image
./scripts/build_docker.sh

# Run video mode
cd docker && docker-compose up railsafenet-video

# Run camera mode
cd docker && docker-compose up railsafenet-camera

# BEV calibration
cd docker && docker-compose up railsafenet-calibrate
```

## System Architecture

### Core Pipeline
```
Frame Input
    |
    v
+-------------------+     +------------------+
| SegFormer INT8    | --> | Polynomial       | --> Hazard Zones
| (Rail Segments)   |     | Tracker (Ph4)    |     (Red/Orange/Yellow)
+-------------------+     +------------------+
    |
    v
+-------------------+     +------------------+
| YOLO INT8         | --> | Danger Zone      | --> Alert Status
| (Object Detection)|     | Detector         |     (SAFE/CAUTION/WARNING/DANGER)
+-------------------+     +------------------+
    |
    v
+-------------------+     +------------------+
| BEV Transform     | --> | Path Analyzer    | --> Direction
| (Perspective)     |     | (Ph6)            |     (STRAIGHT/LEFT/RIGHT)
+-------------------+     +------------------+
    |
    v
+-------------------+
| Render Output     | --> Display
| (Alert + MiniBEV) |     (Fullscreen)
+-------------------+
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| FinalProcessor | `videoAssessor_final.py` | Main pipeline |
| PolynomialTracker | `src/rail_detection/polynomial_tracker.py` | Rail tracking (Phase 4) |
| BEVPathAnalyzer | `src/rail_detection/bev_path_analyzer.py` | Direction detection (Phase 6) |
| DangerZoneDetector | `src/rail_detection/danger_zone_detector.py` | Curve-following hazard zones |
| VanishingPointCalibrator | `src/rail_detection/danger_zone_detector.py` | VP calibration for hazard zones |
| AlertPanel | `src/rail_detection/alert_panel.py` | Top-left status panel |
| MiniBEVRenderer | `src/rail_detection/mini_bev_renderer.py` | Bottom-right BEV PiP |

### Performance

| Metric | Value |
|--------|-------|
| FPS | 60+ (with caching) |
| Rail Jitter | 3.14px |
| Latency | <50ms |

## Project Structure

```
RailSafeNet_LiDAR/
|-- videoAssessor_final.py      # Production main script
|-- videoAssessor.py            # Base video assessor
|
|-- src/
|   |-- rail_detection/
|   |   |-- polynomial_tracker.py    # Phase 4 - rail tracking
|   |   |-- bev_transform.py         # Phase 6 - BEV transformation
|   |   |-- bev_path_analyzer.py     # Phase 6 - direction detection
|   |   |-- danger_zone_detector.py  # Hazard zone detection
|   |   |-- alert_panel.py           # Alert UI
|   |   |-- mini_bev_renderer.py     # Mini BEV PiP
|   |   |-- ego_tracker.py           # Kalman tracking
|   |   `-- width_profile.py         # Width learning
|   `-- utils/
|       |-- data_models.py           # Data structures
|       `-- geometry.py              # Geometric utilities
|
|-- config/
|   |-- rail_tracker_config.yaml     # Tracker settings
|   `-- bev_config.yaml              # BEV calibration
|
|-- assets/
|   |-- models_pretrained/
|   |   |-- segformer/optimized/     # SegFormer INT8 engine
|   |   `-- yolo/                    # YOLO INT8 engine
|   `-- crop/                        # Test videos (tram*.mp4)
|
|-- docker/
|   |-- Dockerfile                   # Jetson container
|   |-- docker-compose.yaml          # Deployment config
|   `-- .dockerignore
|
|-- scripts/
|   |-- run_video.sh                 # Video mode runner
|   |-- run_camera.sh                # Camera mode runner
|   `-- build_docker.sh              # Docker build script
|
`-- specs/
    `-- 001-what-why-home/           # Feature specifications
```

## Configuration

### rail_tracker_config.yaml

Key settings:
- `tracking_region`: Rail scanning area
- `temporal_smoothing.alpha`: EMA smoothing (0.25 = balanced)
- `straight_line_locking`: Curvature reduction
- `hazard_zones`: Zone widths (red=100px, orange=150px, yellow=200px)
- `hazard_zones.vanishing_point`: Manual VP for perspective-correct hazard zones (set via `--calibrate-vp`)
- `performance`: Cache intervals (seg=3f, det=1f)

### bev_config.yaml

Key settings:
- `bev_transform.source_points`: BEV trapezoid corners
- `bev_transform.output_size`: BEV resolution
- `path_analysis`: Direction thresholds

## Hazard Zone Generation

### Curve-Following Algorithm

The hazard zones (Red/Orange/Yellow) now follow the actual rail centerline curve rather than using straight-line boundaries. This provides more accurate hazard detection on curved tracks.

**Algorithm:**
1. For each point along the rail centerline (`center_points`), compute zone boundaries
2. Apply VP-based perspective scaling: `scale = (y - vp_y) / (bottom_y - vp_y)`
3. Calculate zone offsets from the centerline at each y-level:
   - Red: `rail_width/2 + red_margin * scale`
   - Orange: `red_offset + orange_expansion * scale`
   - Yellow: `orange_offset + yellow_expansion * scale`
4. Zone boundaries follow `center_points[i].x +/- offset` (follows curve)

**Previous behavior:** Zone boundaries were straight lines from fixed bottom positions converging to VP (`bottom_center_x +/- offset` to VP linear interpolation).

**Current behavior:** Zone boundaries follow the rail curve with perspective scaling applied at each y-level, producing curved zones that accurately track the rail path.

### Vanishing Point Calibration

The VP calibration feature allows manual setting of the vanishing point for perspective-correct hazard zone rendering. The hazard zones converge toward this point, creating realistic perspective where zones are wider at the bottom (near camera) and narrower at the top (far from camera).

### When to Use
- When automatic VP estimation produces incorrect zone geometry
- When deploying to a new camera setup with different field of view
- When hazard zones appear skewed or misaligned with rail perspective

### How to Calibrate
```bash
# 1. Run VP calibration
python3 videoAssessor_final.py --calibrate-vp

# 2. Click where rail tracks converge (vanishing point)
# 3. Press 's' to save

# 4. Run video mode to verify
python3 videoAssessor_final.py --mode video
```

### VP Config Format
The VP is saved to `config/rail_tracker_config.yaml`:
```yaml
hazard_zones:
  vanishing_point:
    x: 960   # VP x-coordinate (pixels)
    y: 200   # VP y-coordinate (pixels)
```

If no manual VP is set, the system auto-estimates VP from rail centerline regression.

## Commands

```bash
# Run tests
pytest tests/

# Lint check
ruff check .

# Format code
ruff format .

# Type check
mypy src/
```

## Keyboard Controls

### Video Mode
- `q` - Quit
- `SPACE` - Pause/Resume
- `n` - Next video
- `r` - Restart video

### Camera Mode
- `q` - Quit
- `s` - Screenshot

### VP Calibration Mode
- `Click` - Set vanishing point location
- `s` - Save VP to config and exit
- `r` - Reset VP selection
- `q` - Quit without saving

## Deployment Workflow

### 1. Development Testing
```bash
# Test with crop videos
python3 videoAssessor_final.py --mode video
```

### 2. Docker Build
```bash
# Build on target platform (Jetson)
./scripts/build_docker.sh
```

### 3. Jetson Deployment
```bash
# Run container
cd docker && docker-compose up railsafenet-camera
```

### 4. Camera Demo Setup
1. Connect USB camera to Jetson
2. Connect Jetson to external display
3. Run camera mode with fullscreen
4. Point camera at tram display video

## Technologies

- Python 3.8+
- TensorRT 8.5+ (INT8 inference)
- OpenCV 4.x
- PyTorch 2.x (tensor ops)
- PyCUDA (GPU memory)
- Albumentations (image augmentation)

## Docker Base Image

```
nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime
```

For Jetson AGX Orin with JetPack 5.1+

## Code Style

- Follow PEP 8
- Use type hints
- Document public functions
- Keep functions focused

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->

# 🤖 Research Team Protocols

## Team Roster & Routing Rules
You represent the **Project Supervisor**. Your job is to understand my request and delegate it to the appropriate sub-agent. **Do not attempt to solve complex tasks alone if a specialist exists.**

| Agent Name | Trigger Keywords | Responsibility |
|:---|:---|:---|
| **@jetson-deployer** | Docker, Deploy, Build, Jetson, FPS, Latency, TensorRT, Optimization | Handles all Docker environment issues, build scripts, and performance tuning on Jetson hardware. |
| **@rail-algorithm-specialist** | Vision, Tracking, Curve, Polynomial, BEV, Hazard Zone, Detection | Responsible for the core math and CV logic. Call this agent when modifying `src/rail_detection/`. |
| **@lidar-fusion-architect** | LiDAR, Point Cloud, PCL, Sensor Fusion, Calibration, 3D | Consults on future architecture design for integrating LiDAR data and 3D-2D projection. |
| **@safety-assurance-engineer** | Verify, Test, QA, Safety, Log Analysis, Jitter, Warning | Validates system reliability. Call this agent to review changes before finalizing any task. |

## 🔄 Standard Operating Procedures (SOPs)

### SOP 1: Deployment Check (Pre-Docker Build)
When I ask to **"Prepare for deployment"** or **"Check Docker"**, execute:
1. Call **@jetson-deployer**: Check `Dockerfile` and `requirements.txt` for compatibility.
2. Call **@jetson-deployer**: Verify TensorRT engine paths in `assets/`.
3. Call **@safety-assurance-engineer**: Ensure no debug flags are left active in production code.

### SOP 2: Algorithm Update Loop
When I ask to **"Improve tracking"** or **"Update logic"**, execute:
1. Call **@rail-algorithm-specialist**: Implement the logic changes.
2. Call **@safety-assurance-engineer**: Verify the changes do not break safety constraints (Rail Jitter < 3.14px).
3. If passed, report "Ready for Testing".

### SOP 3: Future Design Meeting
When I ask to **"Discuss LiDAR integration"**, execute:
1. Call **@lidar-fusion-architect**: Propose the architecture.
2. Call **@rail-algorithm-specialist**: Review impact on current vision pipeline.
3. Synthesize both opinions into a summary.