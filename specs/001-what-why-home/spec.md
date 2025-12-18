# Feature Specification: Enhanced Rail Hazard Detection - Phased Optimization

**Feature Branch**: `001-what-why-home`
**Created**: 2025-10-14
**Revised**: 2025-12-04 (Phase 4 Complete - Polynomial Tracking Implementation)
**Status**: Phase 4 Complete - Ready for Phase 5 (Optional IMM)
**Input**: User description: "선로 영상 기반 위험 감지 기술 고도화 및 프레임 처리 속도 개선"

## User Scenarios & Testing *(mandatory)*

###  Phase 3 - Parallel Inference Optimization (Priority: P1)

**Technical Focus**: Convert videoAssessor.py from sequential SegFormer→YOLO processing to parallel execution using independent CUDA streams and thread-based inference coordination.

**User Value**: System engineers achieve maximum GPU utilization and significantly improved throughput, enabling real-time processing at target 25+ FPS on production hardware.

**Why this priority**: Current sequential processing leaves GPU idle 40-50% of the time. Parallel inference is the foundational optimization that unlocks performance for all subsequent features. Must be implemented first.

**Independent Test**:
1. Run identical test video (tram0.mp4, 30 seconds, 900 frames) through both sequential (current) and parallel (new) implementations
2. Measure: (a) Total processing time, (b) GPU utilization % via nvidia-smi, (c) Sustained FPS over full video
3. Target: 1.5-1.8x speedup, GPU utilization >85%, sustained 20-25 FPS on Titan RTX

**Acceptance Scenarios**:

1. **Given** videoAssessor.py is modified to use parallel CUDA streams, **When** processing a single frame, **Then** SegFormer and YOLO inference begin within 2ms of each other (confirmed via CUDA event timing), demonstrating true parallelism rather than sequential execution with threading illusion

2. **Given** the system processes 100 consecutive frames in parallel mode, **When** measuring per-frame component timing, **Then** segmentation time (35-40ms) and detection time (20-25ms) overlap by at least 70%, and total frame time is reduced from 77-83ms (sequential baseline) to 45-50ms (target: <40ms ideal)

3. **Given** parallel processing is active and both models are running, **When** monitoring GPU utilization via nvidia-smi during processing, **Then** GPU compute utilization exceeds 85% (vs. 50-60% in current sequential mode), confirming efficient use of GPU resources without idle gaps

---

### Phase 4 - Ego-Track Stability Enhancement (Priority: P2) ✅ COMPLETE

**Implementation Status**: ✅ Completed 2025-12-04

**Technical Focus**: Implement temporal tracking system using polynomial curve fitting with Exponential Moving Average (EMA) for smooth, jitter-free rail tracking. Replaced Kalman-based approach with simpler, more effective polynomial fitting that includes straight-line locking for direct track sections.

**Implementation Approach**:

- **Polynomial Fitting**: 2nd-degree polynomial (x = a*y² + b*y + c) for rail center line
- **Temporal Smoothing**: EMA with configurable alpha parameter (default: 0.15)
- **Straight-Line Locking**: Detects low curvature (|a| < threshold) and forces near-zero for stable straight sections
- **Perspective-Aware Hazard Zones**: Width scaling based on y-coordinate with configurable taper factor
- **Risk-Based Object Detection**: YOLO boxes colored by zone (Red/Orange/Yellow) using cv2.pointPolygonTest
- **Performance Optimization**: INT8 TensorRT engines + intelligent caching (30-40 FPS)

**Key Files Implemented**:

- `src/rail_detection/polynomial_tracker.py` - Core polynomial tracking module (352 lines)
- `config/rail_tracker_config.yaml` - Comprehensive configuration with 50+ tunable parameters
- `videoAssessor_phase4_polynomial.py` - Real-time integration with YOLO and hazard zones

**User Value**: Tram operators experience stable, jitter-free Ego-track visualization with realistic perspective depth perception, risk-based object highlighting, and smooth tracking through lighting changes and curves.

**Why this approach**: Polynomial fitting with EMA provides superior smoothness compared to Kalman filters, is simpler to tune (1 main parameter vs 10+), faster (~0.5-1ms vs 1-2ms), and provides intuitive visual feedback (Straight/Curved mode display).

**Independent Test**:
1. Process test videos containing: (a) temporary occlusions (shadows crossing tracks for 5-10 frames), (b) parallel tracks visible simultaneously, (c) rail junctions with 2-3 path options
2. Measure: (a) Ego-track continuity (% of frames maintaining same track ID), (b) Re-lock accuracy after occlusion (correct track recovered), (c) Initial track selection accuracy (center-most track when no history exists)
3. Target: >95% continuity, >90% re-lock accuracy, 100% correct initial selection

**Acceptance Scenarios**:

1. **Given** the system has established Ego-track history over 30+ frames (1 second at 30 FPS), **When** rails are temporarily occluded for 5-10 consecutive frames (e.g., by shadow, passing object, or camera obstruction), **Then** the system maintains Ego-track position using temporal prediction (last known position + velocity estimate) and successfully re-locks to the correct track when visibility returns, without changing track ID or creating discontinuity in danger zone visualization

2. **Given** the system encounters a frame with 2-3 parallel tracks visible, **When** no prior Ego-track history exists (first 5 frames of video or after complete track loss), **Then** the system selects the track whose center line is closest to horizontal frame center (x=960 for 1920x1080 resolution) as the initial Ego-track, providing consistent starting behavior across different videos

3. **Given** the system has processed 5-10 seconds of video (150-300 frames at 30 FPS), **When** the system measures rail width at each vertical pixel level (y-coordinate) from segmentation mask, **Then** the system builds a perspective-corrected width profile (pixel width vs. y-position curve) that accounts for perspective narrowing, and constrains future rail detections to ±20% of learned width values to reject false positives (e.g., non-rail linear features, segmentation noise)

---

### Phase 5 - IMM Filter for Junction Handling (Priority: P3 - Optional)

**Technical Focus**: Implement Interacting Multiple Model (IMM) filtering that maintains multiple parallel Kalman filters at rail junctions, assigns probability weights based on measurement consistency, and outputs probability-weighted Ego-track position for smooth transitions through ambiguous regions.

**User Value**: System maintains stable Ego-track visualization even at complex junctions where multiple path options exist, reducing false alarms and providing smooth danger zone transitions without abrupt jumps between candidate paths.

**Why this priority**: This is an advanced feature for the most challenging scenarios (junctions with 3+ diverging paths where track identity is temporarily ambiguous). Basic temporal tracking (Phase 4) handles most cases. IMM provides incremental robustness but is not essential for initial production deployment. Optional "nice-to-have" feature that can be added later if junction performance proves insufficient.

**Independent Test**:
1. Process junction test videos where tracks diverge into 2-3 distinct paths (rail switches, Y-junctions)
2. Compare IMM-enabled vs. disabled modes, measuring: (a) Track selection accuracy vs. ground truth, (b) Danger zone smoothness (no abrupt jumps), (c) False alarm rate at junction entry/exit
3. Target: 10-15% improvement in junction accuracy, 30% reduction in danger zone jitter, <5% false alarm rate

**Acceptance Scenarios**:

1. **Given** the system encounters a rail junction where the track diverges into 3 possible paths, **When** IMM filtering is enabled, **Then** the system initializes 3 parallel Kalman filters (one per path hypothesis), assigns initial probability weights [0.5, 0.3, 0.2] based on geometric likelihood (path closest to current track heading gets highest weight), and updates each filter independently with incoming measurements

2. **Given** IMM filter is tracking 3 junction path candidates with current probabilities [0.6, 0.3, 0.1], **When** new frame measurements arrive (segmentation mask showing rail pixels), **Then** the system updates probability weights using Bayesian inference (likelihood of measurement given each path model), normalizes to sum=1.0, and outputs the probability-weighted average Ego-track position (not just the highest probability path) to ensure smooth visual transitions without abrupt jumps

3. **Given** the tram has passed through a junction and is now clearly on a single path, **When** IMM filter detects only one viable path with probability >0.9 for 10+ consecutive frames, **Then** the system collapses back to single-path tracking mode (disables IMM, keeps only dominant filter) to reduce computational overhead (IMM adds 5-8ms latency) and eliminate unnecessary model complexity when path is unambiguous

---

### Edge Cases

**Phase 3 (Parallel Processing) Specific**:
- What happens when one inference engine (SegFormer or YOLO) fails but the other succeeds?
- How does the system handle CUDA out-of-memory errors during parallel execution?
- What happens if parallel streams become desynchronized (one finishes much earlier than the other)?

**Phase 4 (Temporal Tracking) Specific**:
- What happens when Ego-track is lost for extended periods (>30 frames / 1 second)?
- How does the system recover when the tram switches to a different physical track (e.g., at a switch point)?
- What happens when rail width profile learning encounters anomalous frames (dirt, leaves, snow covering tracks)?
- How does tracking behave when the camera FOV changes rapidly (sharp turns, sudden braking)?

**Phase 5 (IMM Filter) Specific**:
- What happens at junctions when all IMM path probabilities are equally distributed (no clear winner)?
- How does the system handle junctions with more than 3 diverging paths?
- What happens if IMM filter gets stuck in multi-model mode and never collapses back to single-path?

**General Edge Cases** (apply to all phases):
- What happens when no rails are visible in the frame (tunnel entrance, station platform, total obstruction)?
- How does the system handle sudden lighting changes (entering/exiting tunnels, bridges creating shadows)?
- What happens when the camera lens gets partially obscured (dirt, rain, condensation)?
- How does the system behave when rails are only partially visible at frame edges during sharp curves?
- What happens when unusual objects create segmentation ambiguity (fallen leaves covering tracks, snow, reflective puddles)?
- How does the system handle rapid camera movement or vibration causing motion blur?

## Requirements *(mandatory)*

### Functional Requirements

#### Phase 3: Parallel Processing Requirements

- **FR-P3-001**: System MUST execute SegFormer segmentation inference and YOLO object detection inference in parallel using independent CUDA streams, with both inference operations beginning within 5ms of each other
- **FR-P3-002**: System MUST achieve minimum 1.4x speedup in total frame processing time compared to sequential baseline (target: 77-83ms → 55-60ms or better)
- **FR-P3-003**: System MUST maintain GPU compute utilization above 80% during parallel processing (vs. 50-60% sequential baseline)
- **FR-P3-004**: System MUST gracefully handle partial failures where one inference engine succeeds and the other fails, producing valid output from the successful engine
- **FR-P3-005**: Parallel inference overhead (thread management, synchronization) MUST NOT exceed 5ms per frame

#### Phase 4: Temporal Tracking Requirements

- **FR-P4-001**: System MUST maintain Ego-track state (position, velocity, width profile) across frames, storing minimum 30 frames of history (1 second at 30 FPS)
- **FR-P4-002**: When no prior Ego-track exists, system MUST select initial track as the one closest to horizontal frame center (x=960 for 1920x1080)
- **FR-P4-003**: System MUST maintain Ego-track identity (same track ID) through temporary occlusions lasting up to 10 frames (333ms at 30 FPS)
- **FR-P4-004**: System MUST learn perspective-corrected rail width profile over 5-10 seconds (150-300 frames) and constrain future detections to ±20% of learned width
- **FR-P4-005**: System MUST provide visual indication when operating in prediction mode (no current detection, using temporal estimate)
- **FR-P4-006**: Tracking overhead MUST NOT exceed 2ms per frame

#### Phase 5: IMM Filter Requirements (Optional)

- **FR-P5-001**: System MUST detect junction entry conditions (2+ valid path hypotheses with probability >0.15 each) and automatically enable IMM mode
- **FR-P5-002**: System MUST maintain up to 3 parallel Kalman filters at junctions, one per viable path hypothesis
- **FR-P5-003**: System MUST update path probabilities using Bayesian inference with measurement likelihood ratios
- **FR-P5-004**: System MUST output probability-weighted Ego-track position (not just maximum probability path) for smooth transitions
- **FR-P5-005**: System MUST automatically collapse IMM mode to single-path when one hypothesis exceeds 0.9 probability for 10+ frames
- **FR-P5-006**: IMM filter overhead MUST NOT exceed 8ms per frame

#### General Performance Requirements

- **FR-G-001**: System MUST process video frames at minimum sustained rate of 25 FPS for continuous 10-minute operation after Phase 3 completion
- **FR-G-002**: System MUST maintain frame processing time variance (standard deviation) less than 20% of mean processing time
- **FR-G-003**: System MUST handle edge cases (no rails, occlusions, lighting changes) gracefully without crashing or blocking pipeline

#### Visualization and Monitoring

- **FR-V-001**: System MUST display current processing mode (sequential/parallel, tracking enabled/disabled, IMM active/inactive) in frame overlay
- **FR-V-002**: System MUST provide per-frame timing breakdown (segmentation time, detection time, tracking time, total time) for performance analysis
- **FR-V-003**: System MUST log warnings when performance degrades below target (FPS <25, GPU util <80%, tracking failures)

### Key Entities

#### Existing Entities (from Phase 2, unchanged)

- **Video Frame**: Single image from video stream; 1920x1080 resolution; contains railway scene with metadata (timestamp, frame_id)
- **Rail Segmentation Mask**: Pixel-wise classification; identifies rail pixels using class values [1, 4, 9]; same resolution as frame
- **Danger Zone**: Polygonal hazard region; bounded by left/right rail edges; spans vertical extent of visible rails
- **Detected Object**: Object identified by YOLO; has bbox, class label, confidence, danger zone intersection status

#### New Entities (Phase 3: Parallel Processing)

- **InferenceStream**: Wraps CUDA stream for independent model execution; manages memory transfers and synchronization
- **ParallelExecutor**: Coordinates parallel SegFormer and YOLO execution; handles synchronization and result aggregation

#### New Entities (Phase 4: Temporal Tracking)

- **EgoTrackState**: Maintains rail track position, velocity, confidence, and history over multiple frames
- **RailWidthProfile**: Learned rail width as function of y-position; accounts for perspective; used to constrain detections
- **TrackHistory**: Circular buffer storing last N frames of Ego-track data; enables temporal prediction during occlusions

#### New Entities (Phase 5: IMM Filter)

- **PathHypothesis**: Single Kalman filter tracking one possible junction path; has state vector, covariance, probability weight
- **IMMState**: Collection of active path hypotheses at junction; manages probability updates and model selection
- **JunctionDetector**: Identifies junction entry/exit conditions based on track geometry and segmentation patterns

## Success Criteria *(mandatory)*

### Measurable Outcomes

#### Phase 3: Parallel Processing Targets

- **SC-P3-001**: Frame processing time reduced from 77-83ms (sequential baseline) to 45-55ms (parallel), representing 1.4-1.8x speedup
- **SC-P3-002**: GPU utilization during processing exceeds 80% (measured via nvidia-smi), up from 50-60% sequential baseline
- **SC-P3-003**: Sustained processing rate reaches 20-25 FPS on Nvidia Titan RTX (baseline: 12-13 FPS), demonstrating 1.6-1.9x throughput improvement
- **SC-P3-004**: Parallel inference timing shows true overlap: segmentation and detection inference times overlap by minimum 70%

#### Phase 4: Temporal Tracking Targets

- **SC-P4-001**: Ego-track continuity exceeds 95% (track ID remains stable across 95%+ of frames in test videos with occlusions/junctions)
- **SC-P4-002**: Re-lock accuracy after temporary occlusions (5-10 frames) exceeds 90% (system recovers correct track, not adjacent track)
- **SC-P4-003**: Initial track selection (no history) achieves 100% accuracy in selecting center-most track
- **SC-P4-004**: Rail width profile constrains false positives: false detection rate <5% after width profile is learned (150+ frames)
- **SC-P4-005**: Tracking overhead confirmed <2ms per frame, maintaining overall performance budget

#### Phase 5: IMM Filter Targets (Optional)

- **SC-P5-001**: Junction handling accuracy improves 10-15% compared to Phase 4 tracking-only (measured on junction test videos)
- **SC-P5-002**: Danger zone jitter at junctions reduced by 30% (measured as standard deviation of zone boundary positions)
- **SC-P5-003**: False alarm rate at junction entry/exit points below 5%
- **SC-P5-004**: IMM mode correctly activates at 90%+ of junction entry points and correctly deactivates within 10 frames of junction exit
- **SC-P5-005**: IMM overhead confirmed <8ms per frame

#### Operational Readiness (All Phases)

- **SC-OP-001**: System successfully processes 100 consecutive frames without crashes or pipeline failures in 99%+ of test runs
- **SC-OP-002**: System handles all documented edge cases gracefully without manual intervention
- **SC-OP-003**: Performance metrics (FPS, GPU util, timing breakdown) are accurately logged and accessible for analysis

### Deployment Target

**Development Platform**: Nvidia Titan RTX
- Used for development, testing, and optimization
- GPU: 24GB GDDR6, 576 Tensor Cores, 4608 CUDA cores
- Expected performance after Phase 3: 25-30 FPS
- Expected performance after Phase 4: 22-28 FPS (tracking overhead)
- Expected performance after Phase 5: 20-25 FPS (IMM overhead)

**Production Platform**: Nvidia AGX Orin
- Target deployment platform for actual tram operation
- GPU: Ampere architecture (2048 CUDA cores, 64 Tensor cores)
- CPU: 12-core ARM Cortex-A78AE
- Memory: 32GB or 64GB LPDDR5
- AI Performance: Up to 275 TOPS
- Power: 15W - 60W configurable (typical 35-40W for this application)
- Expected performance: 18-25 FPS (target minimum: 25 FPS)

**Deployment Method**: Docker Container
- Application packaged as Docker container
- Base image: `nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime`
- TensorRT engines must be rebuilt for AGX Orin architecture
- Container includes all dependencies (CUDA, TensorRT, Python, OpenCV)

**Note**: Performance targets are for Titan RTX development platform. AGX Orin targets will be validated in separate deployment phase.

### Assumptions

**General Assumptions**:
- Target hardware has CUDA-capable NVIDIA GPU
- Video input is Full HD (1920x1080) at 30 FPS
- Current TensorRT engines (SegFormer B3, YOLO v8s) are retained
- Test footage includes diverse scenarios: straight sections, curves, parallel tracks, junctions
- Performance measurements exclude video I/O time

**Phase-Specific Assumptions**:

**Phase 3**:
- GPU supports concurrent CUDA stream execution (true for all modern NVIDIA GPUs)
- Memory bandwidth is sufficient for parallel transfers (validated: 24GB Titan RTX bandwidth >600 GB/s)
- TensorRT engines do not have internal synchronization points that block parallelism

**Phase 4**:
- Rail width is approximately constant in real world (standard gauge: 1435mm)
- Perspective effect on rail width is predictable and learnable from 5-10 seconds of video
- Temporary occlusions do not exceed 333ms (10 frames at 30 FPS)
- Track switching events are rare (<1% of operational time)

**Phase 5**:
- Junctions with more than 3 diverging paths are rare and can be handled by selecting top-3 most likely paths
- Path probability calculations can use simplified measurement likelihood (no full Bayesian network needed)
- Junction regions constitute <10% of total tram route distance

### Test Dataset

- **Location**: `/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/crop/`
- **Files**: `tram0.mp4` through `tram135.mp4` (136 test videos total)
- **Format**: Each video is 30 seconds long at 30 FPS (900 frames per video)
- **Resolution**: 1920x1080 (Full HD)
- **Total footage**: 68 minutes (122,400 frames)
- **Content**: Representative tram front-facing camera footage including:
  - Straight rail sections (majority of footage)
  - Curved sections
  - Parallel tracks
  - Rail junctions and switches
  - Various lighting conditions (daylight, shadows, artificial lighting)

**Test Subset for Validation**:
- **Phase 3 testing**: tram0.mp4, tram10.mp4, tram25.mp4 (variety of scenarios, 2700 frames)
- **Phase 4 testing**: Videos with occlusions and parallel tracks (TBD: to be identified)
- **Phase 5 testing**: Videos with clear junctions (TBD: to be identified)

## Implementation Notes

**Phasing Rationale**:
1. **Phase 3 first**: Parallel processing provides foundation speedup that benefits all subsequent work
2. **Phase 4 second**: Temporal tracking requires reasonable frame rate (Phase 3) to work effectively
3. **Phase 5 last**: IMM filter is optional enhancement that builds on temporal tracking infrastructure

**Dependencies**:
- Phase 4 depends on Phase 3 (needs faster frame rate for effective tracking)
- Phase 5 depends on Phase 4 (IMM extends temporal tracking to multiple hypotheses)
- All phases depend on Phase 2 completion (data models, base inference engines)

**Validation Strategy**:
- Each phase must pass acceptance tests before proceeding to next phase
- Phase 3 must achieve minimum 1.4x speedup before Phase 4 begins
- Phase 4 must achieve 95% continuity before Phase 5 begins
- Phase 5 is optional and can be skipped if Phase 4 performance is sufficient
