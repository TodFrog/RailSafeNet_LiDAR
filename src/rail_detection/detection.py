"""
YOLO TensorRT detection engine wrapper.

This module provides a clean interface to the TensorRT-optimized YOLO model.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA context initialization
from ..utils.data_models import DetectedObject
from .segmentation import HostDeviceMem, allocate_buffers, do_inference_v2


class DetectionEngine:
    """TensorRT-optimized YOLO object detection engine."""

    # YOLO class names (COCO dataset)
    CLASS_NAMES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
        37: 'surfboard', 38: 'tennis racket', 39: 'bottle'
    }

    def __init__(self,
                 engine_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize detection engine.

        Args:
            engine_path: Path to TensorRT engine file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Raises:
            FileNotFoundError: If engine file does not exist
            RuntimeError: If TensorRT engine cannot be loaded
        """
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)

        try:
            with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}")

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

        # Get input shape
        input_binding_name = self.engine.get_binding_name(0)
        self._input_shape = self.engine.get_binding_shape(input_binding_name)
        self.input_height = self._input_shape[2]
        self.input_width = self._input_shape[3]

        print(f"✓ Detection TensorRT engine loaded")
        print(f"  Expected input shape: {self._input_shape}")
        print(f"  Confidence threshold: {conf_threshold}")
        print(f"  IoU threshold: {iou_threshold}")

    def predict(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Run object detection on image.

        Args:
            image: RGB image (shape: [H, W, 3], dtype: uint8)

        Returns:
            List of DetectedObject with bounding boxes, classes, confidences

        Raises:
            ValueError: If image format is incorrect
            RuntimeError: If inference fails
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be RGB (H, W, 3), got {image.shape}")

        original_shape = image.shape[:2]

        try:
            # Preprocess image
            image_resized = cv2.resize(image, (self.input_width, self.input_height))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_norm = (image_rgb.astype(np.float32) / 255.0).transpose((2, 0, 1))

            # Add batch dimension
            image_input = image_norm[np.newaxis, ...]  # Shape: [1, 3, H, W]

            # Copy to host buffer
            np.copyto(self.inputs[0].host, image_input.ravel())

            # Execute inference
            trt_outputs = do_inference_v2(
                self.context, self.bindings, self.inputs, self.outputs, self.stream
            )

            # Get output shape
            output_name = self.engine.get_binding_name(1)
            output_shape = self.context.get_binding_shape(self.engine.get_binding_index(output_name))

            # Reshape output
            output_data = trt_outputs[0].reshape(output_shape)

            # Post-process detections
            detections = self._post_process(output_data, original_shape)

            return detections

        except Exception as e:
            raise RuntimeError(f"Detection inference failed: {e}")

    def _post_process(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[DetectedObject]:
        """
        Post-process YOLO output with NMS.

        Args:
            output: Raw YOLO output (shape: [1, num_boxes, 4 + num_classes])
            original_shape: Original image shape (height, width)

        Returns:
            List of DetectedObject after NMS
        """
        # Remove batch dimension and transpose if needed
        output = np.squeeze(output)
        if output.ndim == 2 and output.shape[0] < output.shape[1]:
            output = output.T  # Shape: [num_boxes, 4 + num_classes]

        boxes = []
        scores = []
        class_ids = []

        # Calculate scale factors
        scale_x = original_shape[1] / self.input_width
        scale_y = original_shape[0] / self.input_height

        # Parse detections
        for detection in output:
            # Format: [x_center, y_center, width, height, class_scores...]
            box = detection[:4]
            class_scores = detection[4:]

            # Get best class
            class_id = np.argmax(class_scores)
            max_score = class_scores[class_id]

            if max_score > self.conf_threshold:
                # Convert to original image scale
                x_center = box[0] * scale_x
                y_center = box[1] * scale_y
                width = box[2] * scale_x
                height = box[3] * scale_y

                boxes.append([x_center, y_center, width, height])
                scores.append(float(max_score))
                class_ids.append(int(class_id))

        # Apply NMS
        final_detections = []
        if len(boxes) > 0:
            # Convert boxes to format for NMS [x_center, y_center, width, height]
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.conf_threshold, self.iou_threshold
            )

            if len(indices) > 0:
                indices = indices.flatten()

                for i, idx in enumerate(indices):
                    x_center, y_center, width, height = boxes[idx]

                    # Convert to xyxy format
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Create DetectedObject
                    class_id = class_ids[idx]
                    class_name = self.CLASS_NAMES.get(class_id, f"class_{class_id}")

                    # Determine if object is moving based on class
                    is_moving = class_id in DetectedObject.ACCEPTED_MOVING

                    detection = DetectedObject(
                        object_id=i,
                        class_id=class_id,
                        class_name=class_name,
                        bbox_xywh=(x_center, y_center, width, height),
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=scores[idx],
                        is_moving=is_moving
                    )

                    final_detections.append(detection)

        return final_detections

    @property
    def class_names(self) -> Dict[int, str]:
        """Get mapping of class IDs to human-readable names."""
        return self.CLASS_NAMES

    def close(self):
        """Release GPU resources."""
        # Free device memory
        for inp in self.inputs:
            inp.device.free()
        for out in self.outputs:
            out.device.free()

        # Delete context and engine
        del self.context
        del self.engine

        print("✓ Detection engine resources released")
