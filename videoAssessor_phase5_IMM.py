#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Distance Assessor - Phase 4: Polynomial Tracking with Hazard Zones (Optimized)
======================================================================================
Phase 4 Features:
- Polynomial curve fitting with temporal smoothing (EMA)
- Straight-line locking for smooth direct sections
- Perspective-aware hazard zones (narrowing toward horizon)
- Risk-based object detection coloring (Yellow/Orange/Red)
- INT8 engines with intelligent caching for maximum FPS

Optimizations:
- Method 1+3: Intelligent caching (SegFormer 3f, YOLO 2f)
- INT8 Quantization: 30-50% additional speedup
- Expected: 30-40+ FPS

Usage:
    python3 videoAssessor_phase4_polynomial.py
"""

import cv2
import os
import sys
import time
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import deque

# Fix DISPLAY environment variable
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':1'

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Project imports
from scripts.metrics_filtered_cls import image_morpho
from videoAssessor import get_clues, find_edges, select_video_file

# ============================================================================
# CACHE CONFIGURATION (Loaded from YAML)
# ============================================================================
# These will be loaded from config/rail_tracker_config.yaml
# Default values if config not found:
DEFAULT_SEG_CACHE_INTERVAL = 3  # SegFormer: every 3 frames
DEFAULT_DET_CACHE_INTERVAL = 1  # YOLO: every frame

# -------------------------------------------------------------------
# TensorRT Engine Classes (Updated API)
# -------------------------------------------------------------------
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def allocate_buffers(engine, stream=None):
    inputs, outputs, bindings = [], [], []
    if stream is None:
        stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(tensor_name)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

class TRTEngine:
    def __init__(self, engine_path, cuda_stream=None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)

        if -1 in self.input_shape:
            self.context.set_input_shape(self.input_name, self.input_shape)

        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, cuda_stream)

class TRTSegmentationEngine(TRTEngine):
    def __init__(self, engine_path, cuda_stream=None):
        super().__init__(engine_path, cuda_stream)
        self.expected_height = self.input_shape[2]
        self.expected_width = self.input_shape[3]
        print(f"✓ SegFormer INT8 [{self.expected_height}, {self.expected_width}]")

    def infer(self, input_data):
        np.copyto(self.inputs[0].host, input_data.ravel())
        trt_outputs = do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        output_name = self.engine.get_tensor_name(1)
        output_shape = self.engine.get_tensor_shape(output_name)
        return trt_outputs[0].reshape(output_shape)

class TRTYOLOEngine(TRTEngine):
    def __init__(self, engine_path, cuda_stream=None):
        super().__init__(engine_path, cuda_stream)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        print(f"✓ YOLO INT8 [{self.input_height}, {self.input_width}]")
        self.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 7: 'truck',
                      15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                      24: 'backpack', 25: 'umbrella', 28: 'suitcase', 36: 'skateboard'}

    def predict(self, image):
        original_shape = image.shape[:2]
        image_resized = cv2.resize(image, (self.input_width, self.input_height))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_norm = (image_rgb.astype(np.float32) / 255.0).transpose((2, 0, 1))

        np.copyto(self.inputs[0].host, image_norm.ravel())
        trt_outputs = do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)

        output_name = self.engine.get_tensor_name(1)
        output_shape = self.engine.get_tensor_shape(output_name)
        output_data = trt_outputs[0].reshape(output_shape)

        results = self.post_process(output_data, original_shape)
        return [results]

    def post_process(self, output, original_shape):
        output = np.squeeze(output).T
        boxes, scores, class_ids = [], [], []
        scale_x, scale_y = original_shape[1] / self.input_width, original_shape[0] / self.input_height

        for detection in output:
            box, class_scores = detection[:4], detection[4:]
            class_id = np.argmax(class_scores)
            max_score = class_scores[class_id]

            if max_score > self.conf_threshold:
                x_center, y_center, width, height = box
                boxes.append([x_center * scale_x, y_center * scale_y, width * scale_x, height * scale_y])
                scores.append(float(max_score))
                class_ids.append(int(class_id))

        final_boxes, final_scores, final_class_ids = [], [], []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                final_boxes = [boxes[i] for i in indices]
                final_scores = [scores[i] for i in indices]
                final_class_ids = [class_ids[i] for i in indices]

        class MockResults:
            def __init__(self, boxes, scores, class_ids):
                self.boxes = self.MockBoxes(boxes, scores, class_ids)
            class MockBoxes:
                def __init__(self, boxes, scores, class_ids):
                    self.xywh = torch.tensor(boxes) if boxes else torch.empty(0, 4)
                    self.cls = torch.tensor(class_ids) if class_ids else torch.empty(0)
                def tolist(self): return self.xywh.tolist() if len(self.xywh) > 0 else []

        return MockResults(final_boxes, final_scores, final_class_ids)

# -------------------------------------------------------------------
# Phase 5: IMM-SVSF for Robust Junction Handling
# -------------------------------------------------------------------
"""
IMM-SVSF Implementation for Rail Junction Tracking

Based on "The Interacting Multiple Model Smooth Variable Structure Filter
for Trajectory Prediction" paper, this implementation provides robust
ego-track selection at rail junctions using:

1. SVSF (Smooth Variable Structure Filter):
   - Replaces Kalman Filter with sliding mode control principles
   - Robust to model uncertainty and rapid maneuvers
   - Key innovation: SVSF gain = gain_magnitude × saturation(error/boundary)
   - Uses boundary layer (psi) to prevent chattering
   - Convergence rate (gamma) controls response speed

2. Triple-Model IMM:
   - Model 1 (Straight): gamma=0.3, psi=8.0 (stable, smooth)
   - Model 2 (Left Turn): gamma=0.7, psi=3.0 (fast, aggressive)
   - Model 3 (Right Turn): gamma=0.7, psi=3.0 (fast, aggressive)

3. IMM Algorithm:
   - Mixing: Weighted combination of filter states based on transition matrix
   - Prediction: Each SVSF predicts next state independently
   - Update: All filters update with gated measurement
   - Probability Update: Bayesian fusion using likelihood × prior

4. Junction Handling:
   - Gating threshold: 50px (reject outliers)
   - Multi-candidate association: Select closest within gate
   - Mode collapse: Automatic when one model probability > 0.9
"""

class SVSF:
    """
    Smooth Variable Structure Filter (SVSF) for robust state estimation.

    Based on "The Interacting Multiple Model Smooth Variable Structure Filter
    for Trajectory Prediction" paper. SVSF uses sliding mode control principles
    to provide robust estimation under model uncertainty and rapid maneuvers.

    State vector: [x, dx, ddx] (position, velocity, acceleration)
    """

    def __init__(self, initial_x: float, initial_y: float, acc_bias: float = 0.0,
                 gamma: float = 0.5, psi: float = 5.0):
        """
        Initialize SVSF filter.

        Args:
            initial_x: Initial x position
            initial_y: Initial y position (for context, not used in state)
            acc_bias: Acceleration bias for this model (0 = straight, <0 = left, >0 = right)
            gamma: Convergence rate coefficient (0 <= gamma < 1)
            psi: Boundary layer width (chattering prevention)
        """
        # State: [x, dx, ddx]
        self.state = np.array([initial_x, 0.0, acc_bias], dtype=np.float32)

        # Covariance estimate
        self.P = np.array([
            [100.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # State transition matrix (dt = 1 pixel in y-direction)
        # x' = x + dx + 0.5*ddx
        # dx' = dx + ddx
        # ddx' = ddx (constant acceleration)
        self.F = np.array([
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Measurement matrix (observe only x position)
        self.H = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        # Process noise
        self.Q = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.1]
        ], dtype=np.float32)

        # Measurement noise
        self.R = np.array([[5.0]], dtype=np.float32)

        # SVSF parameters
        self.gamma = gamma  # Convergence rate
        self.psi = psi  # Boundary layer width
        self.acc_bias = acc_bias

        # Previous errors for SVSF gain calculation
        self.e_prev = np.array([0.0], dtype=np.float32)

    def saturation(self, x: np.ndarray, boundary: float) -> np.ndarray:
        """
        Saturation function with boundary layer.

        sat(x/boundary) = {
            x/boundary    if |x| <= boundary
            sign(x)       if |x| > boundary
        }
        """
        result = np.zeros_like(x)
        for i in range(len(x)):
            if abs(x[i]) <= boundary:
                result[i] = x[i] / boundary if boundary > 0 else 0.0
            else:
                result[i] = np.sign(x[i])
        return result

    def predict(self) -> np.ndarray:
        """
        SVSF Prediction Step.

        Returns:
            Predicted state [x, dx, ddx]
        """
        # A priori state estimate
        self.state = self.F @ self.state

        # A priori covariance estimate
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state.copy()

    def update(self, measurement: float) -> np.ndarray:
        """
        SVSF Update Step with sliding mode gain.

        Args:
            measurement: Measured x position

        Returns:
            Updated state [x, dx, ddx]
        """
        # Measurement
        z = np.array([measurement], dtype=np.float32)

        # Predicted measurement
        z_pred = self.H @ self.state

        # Innovation (measurement residual)
        e = z - z_pred

        # SVSF Gain Calculation (key difference from Kalman Filter)
        # K_k = diag(|e_{k|k-1}| + γ|e_{k-1|k-1}|) ∘ sat(ψ^{-1} * e_{k|k-1})

        # Gain magnitude term
        gain_mag = np.abs(e) + self.gamma * np.abs(self.e_prev)

        # Saturation term (direction with smoothing)
        sat_term = self.saturation(e, self.psi)

        # SVSF gain (element-wise multiplication)
        K_svsf = gain_mag * sat_term

        # State update (using SVSF gain instead of Kalman gain)
        # Note: SVSF uses the gain directly as correction, not matrix multiplication
        self.state[0] += K_svsf[0]  # Update only x position

        # Update covariance (simplified - maintains uncertainty estimate)
        # Using standard Kalman covariance update as approximation
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = 1.0 / (S[0, 0] + 1e-6)
        K_cov = self.P @ self.H.T * S_inv
        I_KH = np.eye(3) - np.outer(K_cov.flatten(), self.H[0])
        self.P = I_KH @ self.P

        # Store error for next iteration
        self.e_prev = e.copy()

        return self.state.copy()

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.state.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current covariance estimate."""
        return self.P.copy()


class TripleModelIMM:
    """
    3-Model Interacting Multiple Model (IMM) with SVSF for rail junction data association.

    Models:
    - M1 (Straight): Zero acceleration, small SVSF gain (straight track stability)
    - M2 (Left Bias): Negative acceleration, large SVSF gain (fast left junction response)
    - M3 (Right Bias): Positive acceleration, large SVSF gain (fast right junction response)

    Uses SVSF instead of Kalman Filter for robustness to model uncertainty.
    """

    def __init__(self, initial_x: float, initial_y: float):
        """
        Initialize IMM with 3 SVSF filters.

        State vector: [x, dx, ddx] (position, velocity, acceleration)
        """
        # Model accelerations
        self.acc_straight = 0.0
        self.acc_left = -0.5
        self.acc_right = 0.5

        # Model probabilities (initially favor straight)
        self.model_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Transition probability matrix (high diagonal = prefer current mode)
        self.transition_matrix = np.array([
            [0.95, 0.025, 0.025],  # From Straight
            [0.025, 0.95, 0.025],  # From Left
            [0.025, 0.025, 0.95]   # From Right
        ], dtype=np.float32)

        # Initialize 3 SVSF filters with different parameters
        self.filters = []

        # Model 1 (Straight): Small gamma (slow response), very large psi (noise-resistant)
        svsf_straight = SVSF(initial_x, initial_y, self.acc_straight, gamma=0.3, psi=15.0)
        self.filters.append(svsf_straight)

        # Model 2 (Left): Large gamma (fast response), small psi (aggressive)
        svsf_left = SVSF(initial_x, initial_y, self.acc_left, gamma=0.7, psi=3.0)
        self.filters.append(svsf_left)

        # Model 3 (Right): Large gamma (fast response), small psi (aggressive)
        svsf_right = SVSF(initial_x, initial_y, self.acc_right, gamma=0.7, psi=3.0)
        self.filters.append(svsf_right)

        # Gating threshold (pixels)
        self.gate_threshold = 50.0

        # Current state estimate (weighted average)
        self.state = np.array([initial_x, 0.0, 0.0], dtype=np.float32)
        self.covariance = np.eye(3, dtype=np.float32) * 100.0

    def predict(self) -> float:
        """
        IMM Prediction Step: Mix SVSF models and predict next state.

        Returns:
            Predicted x position (weighted average)
        """
        # Step 1: Calculate mixing probabilities
        mixing_probs = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                mixing_probs[i, j] = self.transition_matrix[i, j] * self.model_probs[i]
            normalizer = np.sum(mixing_probs[i, :])
            if normalizer > 0:
                mixing_probs[i, :] /= normalizer

        # Step 2: Mix states for each filter
        mixed_states = []
        mixed_covs = []

        for j in range(3):
            # Mix state
            mixed_state = np.zeros(3, dtype=np.float32)
            for i in range(3):
                mixed_state += mixing_probs[i, j] * self.filters[i].get_state()

            # Mix covariance
            mixed_cov = np.zeros((3, 3), dtype=np.float32)
            for i in range(3):
                diff = self.filters[i].get_state() - mixed_state
                mixed_cov += mixing_probs[i, j] * (
                    self.filters[i].get_covariance() + np.outer(diff, diff)
                )

            mixed_states.append(mixed_state)
            mixed_covs.append(mixed_cov)

        # Step 3: Set mixed states and predict each SVSF filter
        for j in range(3):
            svsf = self.filters[j]
            svsf.state = mixed_states[j]
            svsf.P = mixed_covs[j]

            # Predict using SVSF
            svsf.predict()

        # Step 4: Calculate weighted prediction
        predicted_x = sum(self.model_probs[i] * self.filters[i].get_state()[0] for i in range(3))

        return predicted_x

    def update(self, measurements: List[float]) -> Optional[float]:
        """
        IMM Update Step: Gate measurements, associate best candidate, update SVSF filters.

        Args:
            measurements: List of x-position candidates at current y-level

        Returns:
            Selected x position, or None if no valid measurement
        """
        # Handle edge cases
        if len(measurements) == 0:
            # No measurement: use prediction only
            return None

        if len(measurements) == 1:
            # Single candidate: use it directly
            z_best = measurements[0]
        else:
            # Multiple candidates: gate and associate
            predicted_x = self.predict()

            # Find closest candidate within gate
            z_best = None
            min_dist = float('inf')

            for z in measurements:
                dist = abs(z - predicted_x)
                if dist < self.gate_threshold and dist < min_dist:
                    z_best = z
                    min_dist = dist

            if z_best is None:
                # All candidates outside gate (outliers)
                return None

        # Update each SVSF filter with selected measurement
        likelihoods = np.zeros(3, dtype=np.float32)

        for i in range(3):
            svsf = self.filters[i]

            # Get prediction before update
            state_pred = svsf.get_state()
            x_pred = state_pred[0]

            # Update using SVSF
            svsf.update(z_best)

            # Calculate likelihood based on innovation
            # Using residual between measurement and prediction
            innovation = z_best - x_pred

            # Estimate innovation covariance (using SVSF covariance)
            P = svsf.get_covariance()
            S = P[0, 0] + 5.0  # Add measurement noise variance

            # Gaussian likelihood
            likelihood = np.exp(-0.5 * innovation**2 / S) / np.sqrt(2 * np.pi * S)
            likelihoods[i] = likelihood

        # Update model probabilities (Bayesian update)
        prob_bar = self.transition_matrix.T @ self.model_probs
        self.model_probs = likelihoods * prob_bar

        # Normalize
        prob_sum = np.sum(self.model_probs)
        if prob_sum > 0:
            self.model_probs /= prob_sum
        else:
            # Fallback: reset to straight
            self.model_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Update combined state estimate (weighted average of SVSF states)
        self.state = sum(self.model_probs[i] * self.filters[i].get_state() for i in range(3))

        return z_best

    def get_state(self) -> Tuple[float, np.ndarray]:
        """
        Get current state estimate.

        Returns:
            (x_position, model_probabilities)
        """
        return self.state[0], self.model_probs.copy()

    def get_dominant_mode(self) -> str:
        """Get the currently dominant mode."""
        idx = np.argmax(self.model_probs)
        modes = ['Straight', 'Left', 'Right']
        return modes[idx]

# -------------------------------------------------------------------
# Phase 4: Polynomial Rail Tracker with Perspective-Aware Hazard Zones
# -------------------------------------------------------------------
class PolynomialRailTracker:
    """Polynomial tracker with perspective-aware hazard zones."""

    def __init__(self, height: int, width: int, config_path: Optional[str] = None):
        self.height = height
        self.img_width = width
        self.config = self._load_config(config_path)

        # Extract configuration
        track_cfg = self.config['tracking_region']
        self.start_y = height - track_cfg['bottom_offset_px']
        self.end_y = int(height * track_cfg['top_percentage'])
        self.scan_step = track_cfg['scan_step_px']

        poly_cfg = self.config['polynomial_fitting']
        self.min_measurements = poly_cfg['min_measurements']
        self.weight_bias = poly_cfg['weight_bias']

        smooth_cfg = self.config['temporal_smoothing']
        self.alpha = smooth_cfg['alpha']

        straight_cfg = self.config['straight_line_locking']
        self.straight_threshold = straight_cfg['curvature_threshold']
        self.reduction_factor = straight_cfg['reduction_factor']

        width_cfg = self.config['width_constraints']
        self.min_width = width_cfg['min_width_px']
        self.max_width = width_cfg['max_width_px']
        self.clamp_min = width_cfg['clamp_min_px']
        self.clamp_max = width_cfg['clamp_max_px']

        edge_cfg = self.config['edge_selection']
        self.search_offset = edge_cfg['search_offset_range']

        perf_cfg = self.config['performance']
        self.output_step = perf_cfg['output_step_px']

        adv_cfg = self.config['advanced']
        self.fit_width_poly = adv_cfg['fit_width_polynomial']
        self.reduce_width_in_straight = adv_cfg['reduce_width_curvature_in_straight']

        # Hazard zone configuration
        hazard_cfg = self.config['hazard_zones']
        self.red_zone_width = hazard_cfg['red_zone_width_px']
        self.orange_zone_width = hazard_cfg['orange_zone_width_px']
        self.yellow_zone_width = hazard_cfg['yellow_zone_width_px']
        self.perspective_taper = hazard_cfg['perspective_taper']
        self.zone_alpha = hazard_cfg['zone_alpha']
        self.red_color = tuple(hazard_cfg['red_color'])
        self.orange_color = tuple(hazard_cfg['orange_color'])
        self.yellow_color = tuple(hazard_cfg['yellow_color'])

        # Visualization
        vis_cfg = self.config['visualization']
        self.rail_zone_color = (vis_cfg['rail_zone_color']['B'],
                                vis_cfg['rail_zone_color']['G'],
                                vis_cfg['rail_zone_color']['R'])
        self.rail_zone_alpha = vis_cfg['rail_zone_alpha']
        self.edge_color = (vis_cfg['edge_line_color']['B'],
                          vis_cfg['edge_line_color']['G'],
                          vis_cfg['edge_line_color']['R'])
        self.edge_thickness = vis_cfg['edge_line_thickness']
        self.center_color = (vis_cfg['center_line_color']['B'],
                            vis_cfg['center_line_color']['G'],
                            vis_cfg['center_line_color']['R'])
        self.center_thickness = vis_cfg['center_line_thickness']

        # Normalization
        self.y_mid = (self.start_y + self.end_y) / 2.0
        self.y_scale = (self.start_y - self.end_y) / 2.0

        # State
        self.prev_coeffs_center = None
        self.prev_coeffs_width = None
        self.mode = "Unknown"
        self.frame_count = 0
        self.straight_frames = 0
        self.curved_frames = 0
        self.hybrid_frames = 0

        # IMM for junction handling (initialized on first frame)
        self.imm = None
        self.use_imm = True  # Enable/disable IMM junction handling

        print(f"✓ Polynomial Tracker [alpha={self.alpha:.2f}, taper={self.perspective_taper:.2f}]")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "rail_tracker_config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mid) / self.y_scale

    def _apply_temporal_smoothing(self, curr_coeffs: np.ndarray, prev_coeffs: Optional[np.ndarray]) -> np.ndarray:
        if prev_coeffs is None:
            return curr_coeffs
        return self.alpha * curr_coeffs + (1.0 - self.alpha) * prev_coeffs

    def _apply_straight_line_locking(self, coeffs: np.ndarray) -> np.ndarray:
        c, b, a = coeffs
        if abs(a) < self.straight_threshold:
            a = a * self.reduction_factor
            self.mode = "Straight"
            self.straight_frames += 1
        else:
            self.mode = "Curved"
            self.curved_frames += 1
        return np.array([c, b, a], dtype=np.float32)

    def _compute_perspective_scale(self, y: np.ndarray) -> np.ndarray:
        """Compute perspective scaling factor: 1.0 at bottom, taper toward top."""
        # Normalize y to [0, 1]: 0=top, 1=bottom
        y_norm = (y - self.end_y) / (self.start_y - self.end_y)
        # Apply perspective taper: scale = 1.0 - taper * (1 - y_norm)
        scale = 1.0 - self.perspective_taper * (1.0 - y_norm)
        return scale

    def _fit_hybrid_track(self, y_samples: np.ndarray, x_samples: np.ndarray,
                          weights: np.ndarray, y_norm: np.ndarray) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
        """
        Bottom-Locked Hybrid Fitting algorithm.

        Returns:
            (coeffs, mode, linear_coeffs)
            - coeffs: Final polynomial coefficients [c, b, a]
            - mode: 'Hybrid' or 'Global'
            - linear_coeffs: Linear coefficients [c_lin, b_lin] if hybrid mode, else None
        """
        # 1. Fit global quadratic (全体データで2次曲線フィット)
        W = np.diag(weights)
        X_quad = np.column_stack((np.ones_like(y_norm), y_norm, y_norm**2))
        XTW = X_quad.T @ W
        global_coeffs = np.linalg.solve(XTW @ X_quad, XTW @ x_samples)

        # 2. Extract bottom 40% data (Normalized Y > 0.2)
        # Y正規化: -1(top) ~ +1(bottom), なので y_norm > 0.2 が下部40%
        bottom_mask = y_norm > 0.2
        if np.sum(bottom_mask) < 3:  # Need at least 3 points for linear fit
            return global_coeffs, 'Global', None

        y_bottom = y_norm[bottom_mask]
        x_bottom = x_samples[bottom_mask]
        w_bottom = weights[bottom_mask]

        # 3. Fit local linear on bottom 40% (下部40%で1次直線フィット)
        W_bottom = np.diag(w_bottom)
        X_linear = np.column_stack((np.ones_like(y_bottom), y_bottom))
        XTW_lin = X_linear.T @ W_bottom
        linear_coeffs = np.linalg.solve(XTW_lin @ X_linear, XTW_lin @ x_bottom)

        # 4. Calculate RMSE difference in bottom region
        x_pred_quad = global_coeffs[0] + global_coeffs[1] * y_bottom + global_coeffs[2] * y_bottom**2
        x_pred_linear = linear_coeffs[0] + linear_coeffs[1] * y_bottom

        rmse_diff = np.sqrt(np.mean((x_pred_quad - x_pred_linear)**2))

        # 5. Decide mode based on threshold (default: 5.0px)
        threshold = 5.0  # Can be made configurable via YAML

        if rmse_diff > threshold:
            # Hybrid mode: bottom is straight, top is curved
            self.mode = "Hybrid"
            self.hybrid_frames += 1
            # Return global coeffs but with linear_coeffs for blending
            return global_coeffs, 'Hybrid', linear_coeffs
        else:
            # Global mode: use standard straight-line locking
            self.mode = "Global"
            return global_coeffs, 'Global', None

    def process_frame(self, edges_dict: dict) -> Dict:
        """Process frame and return tracking results with perspective hazard zones."""
        self.frame_count += 1

        # Extract measurements with IMM-based junction handling
        measurements = []
        for y in range(self.start_y, self.end_y, -self.scan_step):
            found_y = -1
            for offset in range(self.search_offset):
                if (y + offset) in edges_dict:
                    found_y = y + offset
                    break
                if (y - offset) in edges_dict:
                    found_y = y - offset
                    break

            if found_y != -1:
                candidates = edges_dict[found_y]

                # Calculate center positions for all candidates
                center_candidates = [(pair, (pair[0] + pair[1]) / 2.0, abs(pair[1] - pair[0]))
                                   for pair in candidates]
                # Filter by width
                valid_candidates = [(pair, cx, w) for pair, cx, w in center_candidates
                                  if self.min_width < w < self.max_width]

                if len(valid_candidates) == 0:
                    continue

                # Use IMM for data association if enabled and multiple candidates exist
                if self.use_imm and len(valid_candidates) > 1:
                    # Extract center x positions
                    x_candidates = [cx for _, cx, _ in valid_candidates]

                    if len(measurements) == 0:
                        # First measurement: initialize IMM with center-most track
                        best_idx = np.argmin([abs(cx - self.img_width / 2) for cx in x_candidates])
                        selected_x = x_candidates[best_idx]
                        self.imm = TripleModelIMM(selected_x, y)
                    else:
                        # Use IMM to select best candidate
                        selected_x = self.imm.update(x_candidates)
                        if selected_x is None:
                            # All candidates rejected: use prediction
                            selected_x = self.imm.predict()

                    # Find the pair corresponding to selected x
                    best_idx = np.argmin([abs(cx - selected_x) for _, cx, _ in valid_candidates])
                    best_pair, center_x, width = valid_candidates[best_idx]

                else:
                    # Single candidate or IMM disabled: use simple nearest-neighbor
                    if len(measurements) == 0:
                        # First measurement: choose center-most
                        best_idx = np.argmin([abs(cx - self.img_width / 2) for _, cx, _ in valid_candidates])
                    else:
                        # Choose closest to previous
                        prev_center = measurements[-1][1]
                        best_idx = np.argmin([abs(cx - prev_center) for _, cx, _ in valid_candidates])

                    best_pair, center_x, width = valid_candidates[best_idx]

                    # Initialize IMM if enabled and first measurement
                    if self.use_imm and len(measurements) == 0:
                        self.imm = TripleModelIMM(center_x, y)

                measurements.append((y, center_x, width))

        if len(measurements) < self.min_measurements:
            return {'success': False, 'mode': self.mode}

        # Fit polynomials
        y_samples = np.array([m[0] for m in measurements], dtype=np.float32)
        x_samples = np.array([m[1] for m in measurements], dtype=np.float32)
        w_samples = np.array([m[2] for m in measurements], dtype=np.float32)

        y_norm = self._normalize_y(y_samples)
        weights = 1.0 + self.weight_bias * y_norm

        try:
            # Use hybrid fitting algorithm
            curr_coeffs_center, fit_mode, linear_coeffs = self._fit_hybrid_track(
                y_samples, x_samples, weights, y_norm
            )

            # Fit width polynomial
            W = np.diag(weights)
            X = np.column_stack((np.ones_like(y_norm), y_norm, y_norm**2))
            XTW = X.T @ W
            curr_coeffs_width = np.linalg.solve(XTW @ X, XTW @ w_samples) if self.fit_width_poly else np.array([np.mean(w_samples), 0, 0], dtype=np.float32)

            # Apply temporal smoothing
            coeffs_center = self._apply_temporal_smoothing(curr_coeffs_center, self.prev_coeffs_center)
            coeffs_width = self._apply_temporal_smoothing(curr_coeffs_width, self.prev_coeffs_width)

            # Check IMM probability for force_straight (IMM-based straight locking)
            force_straight = False
            if self.use_imm and self.imm is not None:
                straight_prob = self.imm.model_probs[0]  # Model 1 (Straight) probability
                if straight_prob >= 0.7:  # 70% threshold
                    force_straight = True

            # Apply mode-specific processing
            if force_straight:
                # IMM forces straight: set quadratic coefficient to 0
                coeffs_center[2] = 0.0
                self.mode = "Straight (IMM Locked)"
                self.straight_frames += 1
            elif fit_mode == 'Hybrid':
                # Store linear coefficients for later blending
                self.mode = "Hybrid"
                # Don't apply straight-line locking in hybrid mode
            else:
                # Global mode: apply straight-line locking
                coeffs_center = self._apply_straight_line_locking(coeffs_center)

            if self.mode in ["Straight", "Straight (IMM Locked)", "Global"] and self.reduce_width_in_straight:
                coeffs_width[2] = coeffs_width[2] * self.reduction_factor

            self.prev_coeffs_center = coeffs_center.copy()
            self.prev_coeffs_width = coeffs_width.copy()

        except np.linalg.LinAlgError:
            return {'success': False, 'mode': self.mode}

        # Generate curves
        y_dense = np.arange(self.end_y, self.start_y + 1, self.output_step, dtype=np.float32)
        y_dense_norm = self._normalize_y(y_dense)

        # Calculate center line based on mode
        if fit_mode == 'Hybrid' and linear_coeffs is not None:
            # Hybrid blending: sigmoid weight function
            # Bottom (y_norm → +1): weight → 1.0 (use linear)
            # Top (y_norm → -1): weight → 0.0 (use quadratic)
            # Sigmoid center at y_norm = 0.2, steepness k = 5.0
            sigmoid_weight = 1.0 / (1.0 + np.exp(-5.0 * (y_dense_norm - 0.2)))

            # Quadratic prediction
            x_quad = coeffs_center[0] + coeffs_center[1] * y_dense_norm + coeffs_center[2] * y_dense_norm**2

            # Linear prediction
            x_linear = linear_coeffs[0] + linear_coeffs[1] * y_dense_norm

            # Blend: bottom uses linear, top uses quadratic
            center_dense = sigmoid_weight * x_linear + (1.0 - sigmoid_weight) * x_quad
        else:
            # Global mode: standard quadratic
            center_dense = coeffs_center[0] + coeffs_center[1] * y_dense_norm + coeffs_center[2] * y_dense_norm**2

        width_dense = coeffs_width[0] + coeffs_width[1] * y_dense_norm + coeffs_width[2] * y_dense_norm**2
        width_dense = np.clip(width_dense, self.clamp_min, self.clamp_max)

        # Rail edges
        left_x = center_dense - width_dense / 2.0
        right_x = center_dense + width_dense / 2.0

        # Perspective-aware hazard zones
        perspective_scale = self._compute_perspective_scale(y_dense)

        # Red zone (with perspective)
        left_red = left_x - self.red_zone_width * perspective_scale
        right_red = right_x + self.red_zone_width * perspective_scale

        # Orange zone
        left_orange = left_red - self.orange_zone_width * perspective_scale
        right_orange = right_red + self.orange_zone_width * perspective_scale

        # Yellow zone
        left_yellow = left_orange - self.yellow_zone_width * perspective_scale
        right_yellow = right_orange + self.yellow_zone_width * perspective_scale

        hazard_zones = {
            'yellow': {
                'left': np.column_stack((left_yellow, y_dense)).astype(np.float32),
                'right': np.column_stack((right_yellow, y_dense)).astype(np.float32)
            },
            'orange': {
                'left': np.column_stack((left_orange, y_dense)).astype(np.float32),
                'right': np.column_stack((right_orange, y_dense)).astype(np.float32)
            },
            'red': {
                'left': np.column_stack((left_red, y_dense)).astype(np.float32),
                'right': np.column_stack((right_red, y_dense)).astype(np.float32)
            }
        }

        # Get IMM mode if available
        imm_mode = None
        imm_probs = None
        if self.imm is not None:
            imm_mode = self.imm.get_dominant_mode()
            _, imm_probs = self.imm.get_state()

        return {
            'success': True,
            'left_edge': np.column_stack((left_x, y_dense)).astype(np.float32),
            'right_edge': np.column_stack((right_x, y_dense)).astype(np.float32),
            'center_line': np.column_stack((center_dense, y_dense)).astype(np.float32),
            'hazard_zones': hazard_zones,
            'mode': self.mode,
            'imm_mode': imm_mode,
            'imm_probs': imm_probs
        }

    def get_statistics(self) -> Dict:
        if self.frame_count == 0:
            return {
                "total_frames": 0,
                "straight_percentage": 0.0,
                "curved_percentage": 0.0,
                "hybrid_percentage": 0.0
            }
        return {
            "total_frames": self.frame_count,
            "straight_frames": self.straight_frames,
            "curved_frames": self.curved_frames,
            "hybrid_frames": self.hybrid_frames,
            "straight_percentage": (self.straight_frames / self.frame_count) * 100,
            "curved_percentage": (self.curved_frames / self.frame_count) * 100,
            "hybrid_percentage": (self.hybrid_frames / self.frame_count) * 100
        }

# -------------------------------------------------------------------
# Cached Frame Processor
# -------------------------------------------------------------------
class CachedPolynomialProcessor:
    def __init__(self, model_seg, model_det, image_size=[512, 896], config_path=None):
        self.model_seg = model_seg
        self.model_det = model_det
        self.image_size = image_size
        self.transform_img = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.tracker = None
        self.frame_counter = 0
        self.cached_seg_mask = None
        self.cached_det_results = None

        # Load cache intervals from config
        self.config = self._load_config(config_path)
        perf_cfg = self.config.get('performance', {})
        self.seg_cache_interval = perf_cfg.get('segmentation_cache_interval', DEFAULT_SEG_CACHE_INTERVAL)
        self.det_cache_interval = perf_cfg.get('detection_cache_interval', DEFAULT_DET_CACHE_INTERVAL)

        print(f"✓ Cached Processor [Seg:{self.seg_cache_interval}f, Det:{self.det_cache_interval}f]")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "rail_tracker_config.yaml"

        config_path = Path(config_path)
        if not config_path.exists():
            print(f"⚠️ Config not found: {config_path}, using defaults")
            return {'performance': {
                'segmentation_cache_interval': DEFAULT_SEG_CACHE_INTERVAL,
                'detection_cache_interval': DEFAULT_DET_CACHE_INTERVAL
            }}

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def process_frame(self, frame):
        """Process frame with caching."""
        height, width = frame.shape[:2]
        if self.tracker is None:
            self.tracker = PolynomialRailTracker(height, width)

        self.frame_counter += 1
        seg_from_cache = False
        det_from_cache = False

        # Segmentation (with caching)
        if (self.frame_counter - 1) % self.seg_cache_interval == 0 or self.cached_seg_mask is None:
            image_tr = self.transform_img(image=frame)['image'].unsqueeze(0)
            output = self.model_seg.infer(image_tr.numpy().astype(np.float32))
            id_map = np.argmax(
                F.softmax(torch.from_numpy(output), dim=1).cpu().detach().numpy().squeeze(),
                axis=0
            ).astype(np.uint8)
            id_map = image_morpho(id_map)
            id_map = cv2.resize(id_map, (width, height), interpolation=cv2.INTER_NEAREST)
            self.cached_seg_mask = id_map
        else:
            id_map = self.cached_seg_mask
            seg_from_cache = True

        # Detection (with caching)
        if (self.frame_counter - 1) % self.det_cache_interval == 0 or self.cached_det_results is None:
            detection_results = self.model_det.predict(frame)
            self.cached_det_results = detection_results
        else:
            detection_results = self.cached_det_results
            det_from_cache = True

        # Extract edges
        clues = get_clues(id_map, 6)
        edges_dict = {}
        if clues:
            edges_dict = find_edges(id_map, clues, values=[4, 9], min_width=2)

        # Polynomial tracking
        tracking_result = self.tracker.process_frame(edges_dict)
        tracking_result['detection_results'] = detection_results
        tracking_result['seg_from_cache'] = seg_from_cache
        tracking_result['det_from_cache'] = det_from_cache

        return tracking_result

# -------------------------------------------------------------------
# Visualization with Risk-Based Object Coloring
# -------------------------------------------------------------------
def point_in_polygon(point, poly_pts):
    """Check if point is inside polygon."""
    return cv2.pointPolygonTest(poly_pts, point, False) >= 0

def classify_object_risk(center_x, center_y, hazard_zones):
    """Classify object into Red/Orange/Yellow zone."""
    # Check red zone
    left_red = hazard_zones['red']['left'].astype(np.int32)
    right_red = hazard_zones['red']['right'].astype(np.int32)
    poly_red = np.vstack([left_red, right_red[::-1]])
    if point_in_polygon((center_x, center_y), poly_red):
        return 'red'

    # Check orange zone
    left_orange = hazard_zones['orange']['left'].astype(np.int32)
    right_orange = hazard_zones['orange']['right'].astype(np.int32)
    poly_orange = np.vstack([left_orange, right_orange[::-1]])
    if point_in_polygon((center_x, center_y), poly_orange):
        return 'orange'

    # Check yellow zone
    left_yellow = hazard_zones['yellow']['left'].astype(np.int32)
    right_yellow = hazard_zones['yellow']['right'].astype(np.int32)
    poly_yellow = np.vstack([left_yellow, right_yellow[::-1]])
    if point_in_polygon((center_x, center_y), poly_yellow):
        return 'yellow'

    return None

def draw_tracking_and_zones(frame, result, tracker):
    """Draw all tracking results."""
    if not result['success']:
        return frame

    overlay = frame.copy()

    # Draw hazard zones
    hazard_zones = result['hazard_zones']

    # Yellow zone
    left_yellow = hazard_zones['yellow']['left'].astype(np.int32).reshape((-1, 1, 2))
    right_yellow = hazard_zones['yellow']['right'].astype(np.int32).reshape((-1, 1, 2))
    poly_yellow = np.vstack([left_yellow, right_yellow[::-1]])
    cv2.fillPoly(overlay, [poly_yellow], tracker.yellow_color)

    # Orange zone
    left_orange = hazard_zones['orange']['left'].astype(np.int32).reshape((-1, 1, 2))
    right_orange = hazard_zones['orange']['right'].astype(np.int32).reshape((-1, 1, 2))
    poly_orange = np.vstack([left_orange, right_orange[::-1]])
    cv2.fillPoly(overlay, [poly_orange], tracker.orange_color)

    # Red zone
    left_red = hazard_zones['red']['left'].astype(np.int32).reshape((-1, 1, 2))
    right_red = hazard_zones['red']['right'].astype(np.int32).reshape((-1, 1, 2))
    poly_red = np.vstack([left_red, right_red[::-1]])
    cv2.fillPoly(overlay, [poly_red], tracker.red_color)

    cv2.addWeighted(overlay, tracker.zone_alpha, frame, 1.0 - tracker.zone_alpha, 0, frame)

    # Rail zone (green)
    overlay = frame.copy()
    left_edge = result['left_edge'].astype(np.int32).reshape((-1, 1, 2))
    right_edge = result['right_edge'].astype(np.int32).reshape((-1, 1, 2))
    poly_rail = np.vstack([left_edge, right_edge[::-1]])
    cv2.fillPoly(overlay, [poly_rail], tracker.rail_zone_color)
    cv2.addWeighted(overlay, tracker.rail_zone_alpha, frame, 1.0 - tracker.rail_zone_alpha, 0, frame)

    # Edge lines
    cv2.polylines(frame, [left_edge], False, tracker.edge_color, tracker.edge_thickness, cv2.LINE_AA)
    cv2.polylines(frame, [right_edge], False, tracker.edge_color, tracker.edge_thickness, cv2.LINE_AA)

    # Center line
    center_line = result['center_line'].astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [center_line], False, tracker.center_color, tracker.center_thickness, cv2.LINE_AA)

    # Mode indicator
    mode = result['mode']
    if mode == "Straight":
        mode_color = (0, 255, 0)  # Green
    elif mode == "Hybrid":
        mode_color = (255, 0, 255)  # Magenta (bottom straight, top curved)
    elif mode == "Curved":
        mode_color = (0, 165, 255)  # Orange
    else:
        mode_color = (128, 128, 128)  # Gray (Global/Unknown)

    cv2.putText(frame, f"Mode: {mode}", (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Mode: {mode}", (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2, cv2.LINE_AA)

    # IMM mode indicator (if available)
    if result.get('imm_mode') is not None:
        imm_mode = result['imm_mode']
        imm_probs = result.get('imm_probs', None)

        if imm_probs is not None:
            # Show IMM mode with probabilities
            imm_text = f"IMM: {imm_mode} [S:{imm_probs[0]:.2f} L:{imm_probs[1]:.2f} R:{imm_probs[2]:.2f}]"
        else:
            imm_text = f"IMM: {imm_mode}"

        # Color based on dominant mode
        if imm_mode == 'Straight':
            imm_color = (0, 255, 0)  # Green
        elif imm_mode == 'Left':
            imm_color = (255, 255, 0)  # Cyan
        else:  # Right
            imm_color = (0, 255, 255)  # Yellow

        cv2.putText(frame, imm_text, (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, imm_text, (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, imm_color, 1, cv2.LINE_AA)

    return frame

def draw_risk_based_detections(frame, detection_results, hazard_zones, model_det):
    """Draw detections with risk-based coloring."""
    if not detection_results or not hazard_zones:
        return frame

    det_result = detection_results[0]
    if hasattr(det_result, 'boxes') and hasattr(det_result.boxes, 'xywh'):
        boxes_xywh = det_result.boxes.xywh.tolist()
        boxes_cls = det_result.boxes.cls.tolist()

        for box_xywh, cls_id in zip(boxes_xywh, boxes_cls):
            x_center, y_center, width, height = box_xywh
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Classify risk based on center point
            risk = classify_object_risk(int(x_center), int(y_center), hazard_zones)

            # Choose color based on risk
            if risk == 'red':
                color = (0, 0, 255)  # Red
            elif risk == 'orange':
                color = (0, 165, 255)  # Orange
            elif risk == 'yellow':
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 255, 255)  # White (outside zones)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cls_name = model_det.names.get(int(cls_id), f'class_{int(cls_id)}')
            label = f'{cls_name}'
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    print("="*70)
    print("🚀 Phase 4: Polynomial Tracker + INT8 Cache (Optimized)")
    print("="*70)

    # Paths
    video_dir = "assets/crop"
    seg_engine = "assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961_896x512_int8.engine"
    det_engine = "assets/models_pretrained/yolo/yolov8s_896x512_int8.engine"

    # Load INT8 models
    print("Loading INT8 engines...")
    try:
        model_seg = TRTSegmentationEngine(seg_engine)
        model_det = TRTYOLOEngine(det_engine)
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    # Initialize cached processor
    processor = CachedPolynomialProcessor(model_seg, model_det, image_size=[512, 896])

    print("✓ Ready")
    print(f"⚡ Cache: Seg 1/{processor.seg_cache_interval}f, Det 1/{processor.det_cache_interval}f")
    print("Keys: 'q'=quit, 'SPACE'=pause, 'n'=next, 'r'=restart")
    print("="*70)

    while True:
        video_path = select_video_file(video_dir)
        if video_path is None:
            break

        print(f"\n📹 {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open: {video_path}")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📊 {fps} FPS, {total_frames} frames")

        fps_deque = deque(maxlen=30)
        frame_count = 0
        paused = False
        cache_stats = {'seg_hits': 0, 'seg_miss': 0, 'det_hits': 0, 'det_miss': 0}

        # Warm-up
        processor.frame_counter = 0
        ret, warm_frame = cap.read()
        if ret:
            _ = processor.process_frame(warm_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        processor.frame_counter = 0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                start_time = time.time()

                try:
                    result = processor.process_frame(frame)

                    # Update cache stats
                    if result['success']:
                        if result['seg_from_cache']:
                            cache_stats['seg_hits'] += 1
                        else:
                            cache_stats['seg_miss'] += 1
                        if result['det_from_cache']:
                            cache_stats['det_hits'] += 1
                        else:
                            cache_stats['det_miss'] += 1

                    vis_frame = frame.copy()

                    # Draw tracking and zones
                    vis_frame = draw_tracking_and_zones(vis_frame, result, processor.tracker)

                    # Draw risk-based detections
                    if result['success']:
                        vis_frame = draw_risk_based_detections(vis_frame, result.get('detection_results', []),
                                                               result.get('hazard_zones'), model_det)

                    # FPS
                    frame_fps = 1 / (time.time() - start_time)
                    fps_deque.append(frame_fps)
                    avg_fps = sum(fps_deque) / len(fps_deque)

                    # Info
                    info = f"Frame: {frame_count}/{total_frames} | FPS: {avg_fps:.1f}"
                    cv2.putText(vis_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
                    cv2.putText(vis_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Cache stats
                    seg_rate = cache_stats['seg_hits'] / (cache_stats['seg_hits'] + cache_stats['seg_miss']) * 100 if (cache_stats['seg_hits'] + cache_stats['seg_miss']) > 0 else 0
                    det_rate = cache_stats['det_hits'] / (cache_stats['det_hits'] + cache_stats['det_miss']) * 100 if (cache_stats['det_hits'] + cache_stats['det_miss']) > 0 else 0
                    cache_text = f"INT8 CACHE | Seg:{seg_rate:.0f}% Det:{det_rate:.0f}%"
                    cv2.putText(vis_frame, cache_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Tracking stats
                    if processor.tracker:
                        stats = processor.tracker.get_statistics()
                        stats_text = f"Track: {stats['straight_percentage']:.0f}% Straight, {stats['curved_percentage']:.0f}% Curved, {stats['hybrid_percentage']:.0f}% Hybrid"
                        cv2.putText(vis_frame, stats_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow('Phase 4: Polynomial + INT8 Cache', vis_frame)

                except Exception as e:
                    print(f"Frame {frame_count} error: {e}")
                    import traceback
                    traceback.print_exc()
                    cv2.imshow('Phase 4: Polynomial + INT8 Cache', frame)

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            elif key == ord('n'):
                break
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                processor.frame_counter = 0
                fps_deque.clear()
                cache_stats = {'seg_hits': 0, 'seg_miss': 0, 'det_hits': 0, 'det_miss': 0}

        # Summary
        if frame_count > 0 and processor.tracker:
            stats = processor.tracker.get_statistics()
            print(f"\n📊 Summary:")
            print(f"   Frames: {frame_count}, Avg FPS: {sum(fps_deque)/len(fps_deque):.1f}")
            print(f"   Tracking: {stats['straight_percentage']:.1f}% Straight")
            print(f"   Cache: Seg {seg_rate:.0f}%, Det {det_rate:.0f}%")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("🎬 Closed")
