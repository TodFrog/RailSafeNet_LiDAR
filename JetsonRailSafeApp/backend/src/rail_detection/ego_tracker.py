#!/usr/bin/env python3
"""
Ego-Track Polynomial Kalman Filter for Phase 4.5

This module implements a Polynomial Kalman Filter that tracks the geometry
of the rails (curvature and slope) rather than just position. This allows
robust tracking on curves and intersections.

Model: x = a*y^2 + b*y + c
State: [c, b, a, dc, db, da]
"""

import numpy as np
from typing import Optional, Tuple, List
import logging

from src.utils.data_models import EgoTrackState, TrackHistory, RailWidthProfile

logger = logging.getLogger(__name__)


class PolynomialKalmanFilter:
    """
    Kalman filter for tracking 2nd degree polynomial coefficients.

    State vector: [c, b, a, dc, db, da]
    - c: x-intercept (position)
    - b: slope
    - a: curvature
    - dc, db, da: velocities of coefficients
    """

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt

        # State transition matrix (constant velocity for coefficients)
        # [c  b  a  dc db da]
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix (we observe c, b, a)
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1  # Measure c
        self.H[1, 1] = 1  # Measure b
        self.H[2, 2] = 1  # Measure a

        # Process noise covariance
        # Allow more flexibility in position (c) than curvature (a)
        q_c = 10.0   # Position noise (pixels/s^2)
        q_b = 1.0    # Slope noise
        q_a = 0.01   # Curvature noise
        
        self.Q = np.diag([
            q_c * dt**2, q_b * dt**2, q_a * dt**2,
            q_c * dt,    q_b * dt,    q_a * dt
        ]).astype(np.float32)

        # Measurement noise covariance
        r_c = 5.0    # Position measurement noise (pixels)
        r_b = 0.1    # Slope measurement noise
        r_a = 0.001  # Curvature measurement noise
        self.R = np.diag([r_c, r_b, r_a]).astype(np.float32)

        # Initial state covariance
        self.P = np.eye(6, dtype=np.float32) * 100.0

    def predict(self, state: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state_pred = self.F @ state
        P_pred = self.F @ covariance @ self.F.T + self.Q
        return state_pred, P_pred

    def update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        y = measurement - (self.H @ state)
        S = self.H @ covariance @ self.H.T + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        state_updated = state + K @ y
        I = np.eye(6, dtype=np.float32)
        P_updated = (I - K @ self.H) @ covariance
        return state_updated, P_updated


class EgoTracker:
    """
    Polynomial-based temporal tracker for ego-track.
    """

    def __init__(
        self,
        max_frames_lost: int = 30,
        min_confidence: float = 0.3,
        width_profile: Optional[RailWidthProfile] = None,
        image_height: int = 512
    ):
        self.max_frames_lost = max_frames_lost
        self.min_confidence = min_confidence
        self.width_profile = width_profile
        self.image_height = float(image_height)

        self.kalman = PolynomialKalmanFilter()
        self.history = TrackHistory(buffer=[], max_size=30)
        self.current_track: Optional[EgoTrackState] = None
        self.next_track_id = 0

        # Statistics
        self.total_frames = 0
        self.detected_frames = 0
        self.predicted_frames = 0

        logger.info("Polynomial EgoTracker initialized")

    def update(
        self,
        frame_id: int,
        detection: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Optional[EgoTrackState]:
        self.total_frames += 1

        if detection is not None:
            left_edge, right_edge = detection
            track_state = self._update_with_detection(frame_id, left_edge, right_edge)
            self.detected_frames += 1
        else:
            track_state = self._predict_without_detection(frame_id)
            if track_state is not None:
                self.predicted_frames += 1

        self.current_track = track_state
        if track_state is not None:
            self.history.add(track_state)

        return track_state

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize y to [-1, 1] range for numerical stability."""
        return (y - self.image_height / 2.0) / (self.image_height / 2.0)

    def _fit_polynomial(self, left_edge: np.ndarray, right_edge: np.ndarray) -> Optional[np.ndarray]:
        """
        Fit x = ay^2 + by + c to the track center line.
        Returns [c, b, a] (note order for state vector).
        """
        # Filter common y-range
        min_y = max(left_edge[:, 1].min(), right_edge[:, 1].min())
        max_y = min(left_edge[:, 1].max(), right_edge[:, 1].max())
        
        if max_y - min_y < 20: # Need at least 20px span
            return None

        # Sample points
        y_samples = np.arange(min_y, max_y, 5) # Sample every 5 pixels
        if len(y_samples) < 5:
            return None
            
        # Interpolate x for both edges
        left_x = np.interp(y_samples, left_edge[:, 1], left_edge[:, 0])
        right_x = np.interp(y_samples, right_edge[:, 1], right_edge[:, 0])
        
        center_x = (left_x + right_x) / 2.0
        
        # Normalize y for fitting
        y_norm = self._normalize_y(y_samples)
        
        # Fit polynomial x = a*y^2 + b*y + c
        # np.polyfit returns [a, b, c] (highest power first)
        try:
            coeffs = np.polyfit(y_norm, center_x, 2)
            # Convert to [c, b, a] for state vector
            return np.array([coeffs[2], coeffs[1], coeffs[0]], dtype=np.float32)
        except np.linalg.LinAlgError:
            return None

    def _update_with_detection(self, frame_id: int, left_edge: np.ndarray, right_edge: np.ndarray) -> Optional[EgoTrackState]:
        # Validate detection first
        if not self._validate_detection(left_edge, right_edge):
             logger.debug(f"Frame {frame_id}: Invalid detection")
             return self._predict_without_detection(frame_id)

        # Fit polynomial
        coeffs = self._fit_polynomial(left_edge, right_edge)
        if coeffs is None:
             logger.debug(f"Frame {frame_id}: Polynomial fit failed")
             return self._predict_without_detection(frame_id)

        # Kalman Update
        if self.current_track is None:
            # Initialize
            kalman_state = np.zeros(6, dtype=np.float32)
            kalman_state[:3] = coeffs # c, b, a
            kalman_covariance = self.kalman.P.copy()
        else:
            # Predict & Update
            kalman_state, kalman_covariance = self.kalman.predict(
                self.current_track.kalman_state,
                self.current_track.kalman_covariance
            )
            kalman_state, kalman_covariance = self.kalman.update(
                kalman_state, kalman_covariance, coeffs
            )

        # Generate smoothed edges from updated state
        smoothed_left, smoothed_right = self._generate_edges_from_state(kalman_state)
        
        # Velocity (dc/dt, db/dt) - simplified to just lateral velocity of c
        velocity = (float(kalman_state[3]), 0.0) 

        track_state = EgoTrackState(
            frame_id=frame_id,
            track_id=self.current_track.track_id if self.current_track else self.next_track_id,
            left_edge=smoothed_left,
            right_edge=smoothed_right,
            velocity=velocity,
            frames_since_detection=0,
            is_predicted=False,
            kalman_state=kalman_state,
            kalman_covariance=kalman_covariance,
            polynomial_coeffs=tuple(kalman_state[:3])
        )
        
        if self.current_track is None:
            self.next_track_id += 1
            
        return track_state

    def _predict_without_detection(self, frame_id: int) -> Optional[EgoTrackState]:
        if self.current_track is None:
            return None
            
        frames_lost = self.current_track.frames_since_detection + 1
        if frames_lost > self.max_frames_lost:
            return None

        # Predict
        kalman_state, kalman_covariance = self.kalman.predict(
            self.current_track.kalman_state,
            self.current_track.kalman_covariance
        )
        
        # Generate edges
        left_edge, right_edge = self._generate_edges_from_state(kalman_state)
        velocity = (float(kalman_state[3]), 0.0)

        return EgoTrackState(
            frame_id=frame_id,
            track_id=self.current_track.track_id,
            left_edge=left_edge,
            right_edge=right_edge,
            velocity=velocity,
            frames_since_detection=frames_lost,
            is_predicted=True,
            kalman_state=kalman_state,
            kalman_covariance=kalman_covariance,
            polynomial_coeffs=tuple(kalman_state[:3])
        )

    def _generate_edges_from_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c, b, a = state[:3]
        
        # Generate y points (full height)
        # Use 100 points for smooth rendering
        y_pixels = np.linspace(0, self.image_height - 1, 100).astype(np.float32)
        y_norm = self._normalize_y(y_pixels)
        
        # Calculate center x: x = a*y^2 + b*y + c
        center_x = a * y_norm**2 + b * y_norm + c
        
        # Apply width profile
        widths = np.zeros_like(y_pixels)
        if self.width_profile and self.width_profile.is_calibrated:
            # Vectorized width lookup would be better, but loop is fine for now
            for i, y in enumerate(y_pixels):
                widths[i] = self.width_profile.get_expected_width(y)
        else:
            # Fallback width logic (perspective)
            # Top (y=0) -> 40px, Bottom (y=H) -> 160px
            widths = 40 + (160 - 40) * (y_pixels / self.image_height)
            
        left_x = center_x - widths / 2.0
        right_x = center_x + widths / 2.0
        
        # Stack
        left_edge = np.column_stack((left_x, y_pixels)).astype(np.float32)
        right_edge = np.column_stack((right_x, y_pixels)).astype(np.float32)
        
        return left_edge, right_edge

    def _validate_detection(self, left_edge: np.ndarray, right_edge: np.ndarray) -> bool:
        """
        Validate detected edges.
        """
        # Check minimum points
        if left_edge.shape[0] < 2 or right_edge.shape[0] < 2:
            return False

        # Check shape
        if left_edge.shape[1] != 2 or right_edge.shape[1] != 2:
            return False

        # Calculate average width
        avg_width = self._calculate_average_width(left_edge, right_edge)
        if avg_width is None:
            return False

        # Validate width using profile if available
        if self.width_profile is not None and self.width_profile.is_calibrated:
            # Get average y-coordinate
            avg_y = (np.mean(left_edge[:, 1]) + np.mean(right_edge[:, 1])) / 2.0
            expected_width = self.width_profile.get_expected_width(avg_y)

            if expected_width is not None:
                lower, upper = self.width_profile.get_width_bounds(avg_y)
                if not (lower <= avg_width <= upper):
                    return False

        return True

    def _calculate_average_width(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray
    ) -> Optional[float]:
        """
        Calculate average width between edges with perspective-aware validation.
        """
        if len(left_edge) == 0 or len(right_edge) == 0:
            return None

        left_x_mean = np.mean(left_edge[:, 0])
        right_x_mean = np.mean(right_edge[:, 0])
        avg_y = (np.mean(left_edge[:, 1]) + np.mean(right_edge[:, 1])) / 2.0

        width = abs(right_x_mean - left_x_mean)

        # Perspective-aware width bounds (y-dependent)
        min_width, max_width = self._get_perspective_width_bounds(avg_y)

        if width < min_width or width > max_width:
            return None

        return float(width)

    def _get_perspective_width_bounds(self, y: float) -> Tuple[float, float]:
        """
        Get perspective-aware width bounds based on y-coordinate.
        """
        # Normalize y to [0, 1] where 0=top, 1=bottom
        y_norm = np.clip(y / self.image_height, 0.0, 1.0)

        # Linear interpolation for min/max width
        # Top (y_norm=0.5, half way): min=40, max=100
        # Bottom (y_norm=1.0): min=80, max=200
        min_width = 40 + (80 - 40) * (y_norm - 0.5) * 2  # 40 at top, 80 at bottom
        max_width = 100 + (200 - 100) * (y_norm - 0.5) * 2  # 100 at top, 200 at bottom

        # Clamp to reasonable values
        min_width = max(30, min(min_width, 80))
        max_width = max(100, min(max_width, 220))

        return float(min_width), float(max_width)

    def set_width_profile(self, width_profile: RailWidthProfile):
        self.width_profile = width_profile
        logger.info("Width profile updated in EgoTracker")

    def get_continuity_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return (self.detected_frames + self.predicted_frames) / self.total_frames

    def get_statistics(self) -> dict:
        return {
            'total_frames': self.total_frames,
            'detected_frames': self.detected_frames,
            'predicted_frames': self.predicted_frames,
            'continuity_rate': self.get_continuity_rate(),
            'current_track_id': self.current_track.track_id if self.current_track else None,
            'frames_since_detection': self.current_track.frames_since_detection if self.current_track else None,
            'is_tracking': self.current_track is not None
        }

    def reset(self):
        self.current_track = None
        self.history = TrackHistory(buffer=[], max_size=30)
        self.total_frames = 0
        self.detected_frames = 0
        self.predicted_frames = 0
        logger.info("EgoTracker reset")
