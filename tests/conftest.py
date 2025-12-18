"""
Shared pytest fixtures for RailSafeNet LiDAR tests.
"""

import pytest
import numpy as np
import os


# Engine paths
SEGMENTATION_ENGINE_PATH = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine"
DETECTION_ENGINE_PATH = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/yolo/yolov8s_896x512.engine"


@pytest.fixture
def sample_frame():
    """Generate a sample 1920x1080 RGB frame."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_with_rails():
    """Generate a sample frame with rail-like patterns."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Add rail-like vertical lines
    frame[500:1080, 900:920, :] = 128  # Left rail
    frame[500:1080, 1000:1020, :] = 128  # Right rail
    return frame


@pytest.fixture
def mock_segmentation_mask_with_rails():
    """Generate a mock segmentation mask with rail pixels (class 4 and 9)."""
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    # Rail track (class 4)
    mask[600:1080, 900:920] = 4
    mask[600:1080, 1000:1020] = 4
    # Rail track bed (class 9)
    mask[600:1080, 920:1000] = 9
    return mask


@pytest.fixture
def mock_segmentation_mask_no_rails():
    """Generate a mock segmentation mask without rails."""
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    # Just background and other classes, no class 1, 4, or 9
    mask[0:500, :] = 2  # Some other class
    return mask


@pytest.fixture
def mock_segmentation_mask_parallel_tracks():
    """Generate a mock segmentation mask with two parallel rail tracks."""
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    # First track (left)
    mask[600:1080, 600:620] = 4
    mask[600:1080, 700:720] = 4
    mask[600:1080, 620:700] = 9

    # Second track (right)
    mask[600:1080, 1200:1220] = 4
    mask[600:1080, 1300:1320] = 4
    mask[600:1080, 1220:1300] = 9
    return mask


@pytest.fixture
def mock_detection_results():
    """Generate mock YOLO detection results."""
    return [
        {
            'object_id': 0,
            'class_id': 0,  # person
            'class_name': 'person',
            'bbox_xywh': (960, 800, 100, 200),
            'bbox_xyxy': (910, 700, 1010, 900),
            'confidence': 0.92,
            'is_moving': True
        },
        {
            'object_id': 1,
            'class_id': 2,  # car
            'class_name': 'car',
            'bbox_xywh': (500, 600, 150, 100),
            'bbox_xyxy': (425, 550, 575, 650),
            'confidence': 0.88,
            'is_moving': True
        }
    ]


@pytest.fixture
def segmentation_engine_path():
    """Get path to segmentation TensorRT engine."""
    return SEGMENTATION_ENGINE_PATH


@pytest.fixture
def detection_engine_path():
    """Get path to detection TensorRT engine."""
    return DETECTION_ENGINE_PATH


@pytest.fixture
def test_video_path():
    """Get path to a test video."""
    return "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/crop/tram0.mp4"


@pytest.fixture
def engine_available():
    """Check if TensorRT engines are available."""
    return (os.path.exists(SEGMENTATION_ENGINE_PATH) and
            os.path.exists(DETECTION_ENGINE_PATH))


# ============================================================================
# Phase 3: Parallel Processing Fixtures
# ============================================================================

@pytest.fixture
def sample_normalized_frame():
    """Generate a normalized frame for SegFormer input (C, H, W)."""
    # Normalized to [0, 1] range, shape (3, 512, 896)
    return np.random.rand(3, 512, 896).astype(np.float32)


@pytest.fixture
def mock_segmentation_output():
    """Generate mock SegFormer output (num_classes, H, W)."""
    # 19 classes for CityScapes, shape (19, 512, 896)
    output = np.random.rand(19, 512, 896).astype(np.float32)
    # Make class 1 (rail) have highest probability in some regions
    output[1, 256:512, 400:500] = 2.0  # Rail region
    return output


@pytest.fixture
def mock_yolo_results():
    """Generate mock YOLO inference results."""
    class MockBox:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = xyxy
            self.conf = conf
            self.cls = cls

    class MockResult:
        def __init__(self):
            self.boxes = [
                MockBox(
                    xyxy=np.array([[910, 700, 1010, 900]]),
                    conf=np.array([0.92]),
                    cls=np.array([0])  # person
                ),
                MockBox(
                    xyxy=np.array([[425, 550, 575, 650]]),
                    conf=np.array([0.88]),
                    cls=np.array([2])  # car
                )
            ]

    return [MockResult()]


@pytest.fixture
def cuda_available():
    """Check if CUDA is available for testing."""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        return True
    except Exception:
        return False


@pytest.fixture
def timing_measurements():
    """Generate sample timing measurements for parallel processing tests."""
    return {
        'seg_time_ms': 15.5,
        'det_time_ms': 12.3,
        'total_time_ms': 18.7,  # Parallel overlap
        'sequential_time_ms': 27.8  # seg + det
    }
