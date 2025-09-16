# TensorRT Optimization Strategy for RailSafeNet

## Overview

This document outlines the TensorRT optimization strategy for deploying the transfer learning trained SegFormer B3 model on NVIDIA AGX Orin for real-time tram safety applications.

## Current Status Analysis

### Known Issues from Previous Attempts
- **SegFormer B3 Standard Optimization Failed**: TensorRT optimization failed with the existing model
- **YOLOv8n Success**: Successfully optimized with ONNX → TensorRT pipeline
- **Hybrid Solution Used**: Mixed int8/int16 quantization achieved 14-15 FPS
- **Demo Performance**: Real-time inference on 30-second tram videos

### Root Cause Analysis
The original `SegFormer_B3_1024_finetuned.pth` has a complex architecture that doesn't align with standard TensorRT optimization paths:
- **13-class architecture** vs standard 19-class implementations
- **Custom model structure** embedded in complete model object
- **Non-standard layer configurations** causing TensorRT incompatibilities

## TensorRT Optimization Strategy

### Phase 1: Model Architecture Preparation (Post Training)

#### 1.1 Model Structure Standardization
```python
# Export trained model in ONNX-compatible format
def prepare_model_for_tensorrt(model_path):
    model = load_trained_transfer_model(model_path)
    
    # Ensure standard SegFormer architecture
    # Remove any custom layers that may cause issues
    # Standardize input/output dimensions
    
    return optimized_model
```

#### 1.2 ONNX Export Strategy
```python
ONNX_EXPORT_CONFIG = {
    'input_size': (1, 3, 1024, 1024),
    'output_size': (1, 13, 1024, 1024),  # or 19 depending on training
    'opset_version': 11,
    'do_constant_folding': True,
    'export_params': True
}
```

### Phase 2: TensorRT Optimization Pipeline

#### 2.1 Multi-Stage Optimization Approach

**Stage 1: ONNX Validation**
- Export trained model to ONNX format
- Validate ONNX model inference matches PyTorch
- Optimize ONNX graph (constant folding, node fusion)

**Stage 2: TensorRT Engine Building**
```python
TENSORRT_CONFIG = {
    'max_batch_size': 1,
    'workspace_size': 2 << 30,  # 2GB
    'precision': 'fp16',        # Start with fp16
    'calibration_dataset': None # For int8 if needed
}
```

**Stage 3: Precision Strategy**
1. **FP16 First**: Target 20+ FPS with minimal accuracy loss
2. **INT8 Calibration**: If FP16 insufficient, use calibration dataset
3. **Mixed Precision**: Combine FP16/INT8 for optimal speed/accuracy

#### 2.2 Quantization Strategy

**Calibration Dataset Preparation**:
```python
# Use representative railway images for calibration
CALIBRATION_CONFIG = {
    'dataset_size': 100,  # Representative images
    'batch_size': 1,
    'input_format': 'CHW',
    'normalization': 'ImageNet'
}
```

**Layer-wise Precision Control**:
- **Backbone (Encoder)**: INT8 (less critical for accuracy)
- **Decoder Head**: FP16 (critical for segmentation quality)
- **Final Classifier**: FP16 (maintain class distinction)

### Phase 3: AGX Orin Deployment Optimization

#### 3.1 Hardware-Specific Optimizations

**AGX Orin Specifications**:
- **GPU**: 2048-core NVIDIA Ampere GPU
- **Memory**: 32GB shared memory
- **Power**: 15W-60W configurable TDP

**Optimization Targets**:
- **Performance**: ≥30 FPS for real-time processing
- **Accuracy**: <2% IoU degradation from original model
- **Power**: <40W total system power consumption
- **Latency**: <33ms per frame (30 FPS requirement)

#### 3.2 Model Deployment Pipeline

```python
DEPLOYMENT_CONFIG = {
    'engine_path': '/opt/railsafenet/models/segformer_b3.trt',
    'input_binding': 'input',
    'output_binding': 'output',
    'stream_optimization': True,
    'dla_core': 0,  # Use Deep Learning Accelerator if beneficial
    'memory_pool_size': '1GB'
}
```

### Phase 4: Performance Benchmarking

#### 4.1 Benchmark Metrics
- **Inference Time**: Per-frame processing time
- **Throughput**: Frames per second sustained
- **Memory Usage**: Peak GPU memory consumption
- **Accuracy**: IoU comparison with original model
- **Power Consumption**: Watts during inference

#### 4.2 Test Scenarios
1. **Single Image Processing**: Isolated inference timing
2. **Video Stream Processing**: Sustained performance
3. **Thermal Performance**: Long-running stability
4. **Edge Cases**: Performance under different conditions

### Phase 5: Fallback Strategies

#### 5.1 If TensorRT Optimization Fails

**Option 1: ONNX Runtime Optimization**
- Use ONNX Runtime with CUDA/TensorRT execution provider
- Often more compatible than native TensorRT
- Slightly lower performance but better compatibility

**Option 2: PyTorch JIT Compilation**
```python
# TorchScript optimization
model_scripted = torch.jit.script(model)
model_optimized = torch.jit.optimize_for_inference(model_scripted)
```

**Option 3: Hybrid Architecture**
- Keep backbone in TensorRT (if possible)
- Run decoder in optimized PyTorch
- Balance between speed and compatibility

#### 5.2 Performance Mitigation
- **Input Resolution Scaling**: Use 768x768 instead of 1024x1024
- **Temporal Optimization**: Process every 2nd frame for 60Hz input
- **ROI Processing**: Focus processing on relevant image regions

## Implementation Timeline

### Week 1: Model Preparation
- [ ] Complete transfer learning training
- [ ] Export best model to ONNX format
- [ ] Validate ONNX inference accuracy

### Week 2: TensorRT Optimization
- [ ] Build TensorRT engine with FP16
- [ ] Performance benchmarking on development hardware
- [ ] INT8 calibration if needed

### Week 3: AGX Orin Testing
- [ ] Deploy to target hardware
- [ ] End-to-end performance validation
- [ ] Integration with TheDistanceAssessor.py

### Week 4: Production Readiness
- [ ] Docker containerization
- [ ] Performance monitoring setup
- [ ] Documentation and deployment guide

## Risk Assessment

### High Risk Items
1. **Complex Architecture Compatibility**: Custom model structure may resist optimization
2. **Accuracy Degradation**: Quantization may impact rail detection accuracy
3. **Memory Constraints**: 1024x1024 input may exceed AGX Orin limits

### Mitigation Strategies
1. **Architecture Simplification**: Modify model structure if needed
2. **Graduated Quantization**: Start with FP16, move to INT8 carefully
3. **Dynamic Input Scaling**: Adjust resolution based on available memory

## Success Criteria

### Minimum Requirements
- **Performance**: ≥20 FPS sustained performance
- **Accuracy**: <5% IoU degradation
- **Memory**: <4GB GPU memory usage

### Target Performance
- **Performance**: ≥30 FPS sustained performance  
- **Accuracy**: <2% IoU degradation
- **Memory**: <3GB GPU memory usage
- **Power**: <40W system power

### Production Ready
- **Performance**: ≥30 FPS with thermal stability
- **Accuracy**: Matches PyTorch performance in critical scenarios
- **Integration**: Seamless integration with existing pipeline
- **Deployment**: One-command Docker deployment

## Monitoring and Validation

### Performance Metrics
```python
MONITORING_METRICS = {
    'inference_time_ms': [],
    'fps_sustained': [],
    'memory_usage_mb': [],
    'accuracy_iou': [],
    'power_consumption_w': [],
    'thermal_state': []
}
```

### Validation Dataset
- **Test Videos**: Same 30-second tram videos used in demo
- **Accuracy Validation**: Compare outputs with original model
- **Edge Cases**: Low light, weather conditions, complex scenes

---

*This optimization strategy balances performance requirements with practical deployment constraints, providing multiple fallback options to ensure successful production deployment.*