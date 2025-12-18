"""
SegFormer TensorRT segmentation engine wrapper.

This module provides a clean interface to the TensorRT-optimized SegFormer model.
"""

import numpy as np
from typing import Tuple
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA context initialization


class HostDeviceMem:
    """Host and device memory pair for TensorRT inference."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}"

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """
    Allocate host and device buffers for TensorRT engine.

    Args:
        engine: TensorRT engine

    Returns:
        Tuple of (inputs, outputs, bindings, stream)
    """
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        binding_name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(binding_name)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding_name))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        # Append to the appropriate list
        if engine.binding_is_input(binding_name):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    """
    Execute TensorRT inference.

    Args:
        context: TensorRT execution context
        bindings: List of binding indices
        inputs: List of input HostDeviceMem objects
        outputs: List of output HostDeviceMem objects
        stream: CUDA stream

    Returns:
        List of output arrays
    """
    # Transfer input data to device
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Execute inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back to host
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # Synchronize stream
    stream.synchronize()

    return [out.host for out in outputs]


class SegmentationEngine:
    """TensorRT-optimized SegFormer semantic segmentation engine."""

    def __init__(self, engine_path: str, image_size: Tuple[int, int] = (512, 896)):
        """
        Initialize segmentation engine.

        Args:
            engine_path: Path to TensorRT engine file
            image_size: Input size for model (height, width)

        Raises:
            FileNotFoundError: If engine file does not exist
            RuntimeError: If TensorRT engine cannot be loaded
        """
        self.engine_path = engine_path
        self.image_size = image_size

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

        # Set context for dynamic shape models (if applicable)
        if not self.engine.has_implicit_batch_dimension:
            input_shape = self.engine.get_binding_shape(0)
            self.context.set_binding_shape(0, input_shape)

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

        # Get input shape
        input_binding_name = self.engine.get_binding_name(0)
        self._input_shape = self.engine.get_binding_shape(input_binding_name)

        print(f"✓ Segmentation TensorRT engine loaded")
        print(f"  Expected input shape: {self._input_shape}")

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run segmentation inference on image.

        Args:
            image: Preprocessed image tensor (shape: [1, 3, H, W], dtype: float32)

        Returns:
            Segmentation mask (shape: [1080, 1920], dtype: uint8)
            Class IDs from 0-12

        Raises:
            ValueError: If image shape is incorrect
            RuntimeError: If inference fails
        """
        # Validate input shape
        if image.ndim != 4:
            raise ValueError(f"Input must be 4D (batch, channels, height, width), got {image.ndim}D")

        if image.dtype != np.float32:
            raise ValueError(f"Input must be float32, got {image.dtype}")

        try:
            # Copy input data to host buffer
            np.copyto(self.inputs[0].host, image.ravel())

            # Execute inference
            trt_outputs = do_inference_v2(
                self.context, self.bindings, self.inputs, self.outputs, self.stream
            )

            # Get output shape
            output_name = self.engine.get_binding_name(1)
            output_shape = self.context.get_binding_shape(self.engine.get_binding_index(output_name))

            # Reshape output
            output = trt_outputs[0].reshape(output_shape)

            # Apply softmax and argmax to get class predictions
            # Output shape is typically [1, num_classes, H, W]
            if output.ndim == 4:
                # Softmax along class dimension
                exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
                softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)

                # Argmax to get class IDs
                class_predictions = np.argmax(softmax_output, axis=1).squeeze()
            else:
                # Already processed
                class_predictions = output.squeeze()

            # Resize to target resolution (1080, 1920)
            import cv2
            if class_predictions.shape != (1080, 1920):
                class_predictions = cv2.resize(
                    class_predictions.astype(np.uint8),
                    (1920, 1080),
                    interpolation=cv2.INTER_NEAREST
                )

            return class_predictions.astype(np.uint8)

        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape (batch, channels, height, width)."""
        return self._input_shape

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

        print("✓ Segmentation engine resources released")
