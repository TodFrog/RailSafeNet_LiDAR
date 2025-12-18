import pytest
import numpy as np
from src.rail_detection.ego_tracker import EgoTracker, PolynomialKalmanFilter
from src.utils.data_models import RailWidthProfile

class TestPolynomialTracker:
    @pytest.fixture
    def tracker(self):
        return EgoTracker(image_height=512)

    def test_kalman_initialization(self):
        kf = PolynomialKalmanFilter()
        assert kf.F.shape == (6, 6)
        assert kf.H.shape == (3, 6)
        
    def test_polynomial_fitting(self, tracker):
        # Generate synthetic curve: x = 0.001*y^2 + 0.1*y + 100
        # Image height 512, y from 0 to 511
        y = np.linspace(0, 511, 100)
        y_norm = (y - 256) / 256
        
        # True coefficients for normalized y
        # x = a*y_norm^2 + b*y_norm + c
        a_true = 50.0
        b_true = 20.0
        c_true = 256.0
        
        center_x = a_true * y_norm**2 + b_true * y_norm + c_true
        
        # Create edges (width 100)
        left_edge = np.column_stack((center_x - 50, y))
        right_edge = np.column_stack((center_x + 50, y))
        
        coeffs = tracker._fit_polynomial(left_edge, right_edge)
        
        assert coeffs is not None
        # Check c, b, a (coeffs order is c, b, a)
        # Note: Regularization might bias the result slightly towards 0 curvature
        # so we use a larger tolerance for 'a'
        assert np.isclose(coeffs[0], c_true, atol=5.0)
        assert np.isclose(coeffs[1], b_true, atol=5.0)
        assert np.isclose(coeffs[2], a_true, atol=10.0)

    def test_outlier_rejection(self, tracker):
        # 1. Establish a track first (straight line)
        y = np.linspace(0, 511, 100)
        center_x = np.full_like(y, 256.0)
        left_edge = np.column_stack((center_x - 50, y))
        right_edge = np.column_stack((center_x + 50, y))
        
        # Update a few times to stabilize
        for i in range(5):
            tracker.update(i, (left_edge, right_edge))
            
        # 2. Introduce outliers (simulating a junction branch)
        # Main track continues straight at x=256
        # Branch diverges to x=400 at y=0
        
        # Create mixed points
        # 70% points on main track, 30% on branch
        y_branch = np.linspace(0, 200, 30)
        x_branch = np.linspace(400, 256, 30) # Diverging branch
        
        branch_left = np.column_stack((x_branch - 50, y_branch))
        branch_right = np.column_stack((x_branch + 50, y_branch))
        
        # Combine with main track points
        combined_left = np.vstack((left_edge, branch_left))
        combined_right = np.vstack((right_edge, branch_right))
        
        # Shuffle to mix them up (tracker sorts by y anyway but good for simulation)
        p = np.random.permutation(len(combined_left))
        combined_left = combined_left[p]
        combined_right = combined_right[p]
        
        # Update with outliers
        state = tracker.update(5, (combined_left, combined_right))
        
        assert state is not None
        
        # The fitted polynomial should still be close to the main track (x=256)
        # c (position) should be close to 256
        # b (slope) should be close to 0
        # a (curvature) should be close to 0
        
        c, b, a = state.polynomial_coeffs
        
        # Without outlier rejection, the branch would pull the line significantly
        # With rejection, it should stay close to 256
        assert np.isclose(c, 256.0, atol=10.0)
        assert abs(b) < 10.0 # Slope should remain small
        assert abs(a) < 5.0  # Curvature should remain small

    def test_prediction_maintains_curvature(self, tracker):
        # Feed curved track for a few frames to establish state
        y = np.linspace(0, 511, 100)
        y_norm = (y - 256) / 256
        
        # Curve: x = 50*y_norm^2 + 256
        center_x = 50.0 * y_norm**2 + 256.0
        left_edge = np.column_stack((center_x - 50, y))
        right_edge = np.column_stack((center_x + 50, y))
        
        # Update 5 times
        for i in range(5):
            tracker.update(i, (left_edge, right_edge))
            
        # Now predict (no detection)
        pred_state = tracker.update(5, None)
        
        assert pred_state is not None
        assert pred_state.is_predicted is True
        
        # Check if curvature is maintained
        c, b, a = pred_state.polynomial_coeffs
        # Regularization might dampen 'a' slightly, but it should still be significant
        assert a > 30.0 
        
        # Check predicted edges
        pred_left = pred_state.left_edge
        top_point = pred_left[0] # y=0
        # Should be around x=306 (center) - 20 (half width) = 286
        assert 270 < top_point[0] < 310
