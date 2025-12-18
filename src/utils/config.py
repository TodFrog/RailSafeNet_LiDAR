"""
Configuration management for RailSafeNet LiDAR system.
"""

from dataclasses import dataclass, field
from typing import List
import json
import yaml
from pathlib import Path


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
    track_width_mm: int = 1435  # Standard gauge
    danger_distances_mm: List[int] = field(default_factory=lambda: [100, 400, 1000])

    # Performance parameters
    target_fps: float = 25.0
    max_variance_percent: float = 20.0

    # Visualization
    enable_visualization: bool = True
    save_visualizations: bool = False

    @classmethod
    def from_file(cls, config_path: str) -> 'RailDetectionConfig':
        """
        Load configuration from JSON or YAML file.

        Args:
            config_path: Path to configuration file (.json or .yaml)

        Returns:
            RailDetectionConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is unsupported
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Determine file type and load
        if path.suffix.lower() in ['.json']:
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        return cls(**config_dict)

    def to_file(self, config_path: str):
        """
        Save configuration to JSON or YAML file.

        Args:
            config_path: Path to save configuration file (.json or .yaml)

        Raises:
            ValueError: If file format is unsupported
        """
        path = Path(config_path)

        # Convert to dictionary
        config_dict = {
            'segmentation_engine_path': self.segmentation_engine_path,
            'detection_engine_path': self.detection_engine_path,
            'enable_tracking': self.enable_tracking,
            'enable_vp_filtering': self.enable_vp_filtering,
            'roi_height_fraction': self.roi_height_fraction,
            'tracking_process_noise': self.tracking_process_noise,
            'tracking_measurement_noise': self.tracking_measurement_noise,
            'vp_cache_frames': self.vp_cache_frames,
            'vp_angle_threshold': self.vp_angle_threshold,
            'track_width_mm': self.track_width_mm,
            'danger_distances_mm': self.danger_distances_mm,
            'target_fps': self.target_fps,
            'max_variance_percent': self.max_variance_percent,
            'enable_visualization': self.enable_visualization,
            'save_visualizations': self.save_visualizations,
        }

        # Determine file type and save
        if path.suffix.lower() in ['.json']:
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    @classmethod
    def default(cls) -> 'RailDetectionConfig':
        """
        Create default configuration.

        Returns:
            RailDetectionConfig with default values
        """
        return cls(
            segmentation_engine_path="/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine",
            detection_engine_path="/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/yolo/yolov8s_896x512.engine"
        )

    def validate(self):
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if not Path(self.segmentation_engine_path).exists():
            raise ValueError(f"Segmentation engine not found: {self.segmentation_engine_path}")

        if not Path(self.detection_engine_path).exists():
            raise ValueError(f"Detection engine not found: {self.detection_engine_path}")

        if not (0.0 < self.roi_height_fraction <= 1.0):
            raise ValueError(f"ROI height fraction must be in (0, 1], got {self.roi_height_fraction}")

        if self.vp_cache_frames < 1:
            raise ValueError(f"VP cache frames must be >= 1, got {self.vp_cache_frames}")

        if self.vp_angle_threshold <= 0:
            raise ValueError(f"VP angle threshold must be positive, got {self.vp_angle_threshold}")

        if self.track_width_mm <= 0:
            raise ValueError(f"Track width must be positive, got {self.track_width_mm}")

        if not all(d > 0 for d in self.danger_distances_mm):
            raise ValueError(f"All danger distances must be positive: {self.danger_distances_mm}")

        if self.target_fps <= 0:
            raise ValueError(f"Target FPS must be positive, got {self.target_fps}")

        if self.max_variance_percent < 0:
            raise ValueError(f"Max variance must be non-negative, got {self.max_variance_percent}")
