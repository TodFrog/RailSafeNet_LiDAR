# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RailSafeNet** is an industrial joint research project aimed at developing a real-world deployable tram safety system. The goal is to create an international standards-compliant safety braking system that combines:

- **Camera-based perception and decision making** using the RailSafeNet pipeline
- **LiDAR integration with YOLO** for accurate distance calculation of dynamic objects (people, cars, bicycles)
- **Real-time safety assessment** for autonomous braking decisions

## Hardware Environment

**Training Setup:**
- Python 3.9
- 4x NVIDIA Titan RTX (24GB each) - 96GB total GPU memory
- Multi-GPU distributed training support

**Production Deployment:**
- Target platform: NVIDIA AGX Orin (edge inference)
- Model deployment and updates via Docker containers
- Real-time inference optimized for embedded systems

## Environment Setup

Create conda environment and install dependencies:
```bash
conda create -n RailSafeNet python=3.9
conda activate RailSafeNet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r assets/requirements.txt
```

## Main Commands

### Inference and Execution
- **Primary inference script**: `python TheDistanceAssessor.py` - Main pipeline execution for tram safety object detection and distance assessment
- **Model training scripts**:
  - `python train_SegFormer.py` - Train SegFormer B3 model for semantic segmentation  
  - `python train_DeepLabv3.py` - Train DeepLabV3 model for semantic segmentation
  - `python train_yolo.py` - Train YOLO model for object detection
- **Hyperparameter sweeps**:
  - `python sweep_SegFormer.py` - Hyperparameter sweep for SegFormer
  - `python sweep_DeepLabv3.py` - Hyperparameter sweep for DeepLabV3

### Testing and Evaluation
- `python scripts/test_pilsen.py` - Test models on Pilsen Railway Dataset
- `python scripts/test_all_cls.py` - Test models on all classes
- `python scripts/test_filtered_cls.py` - Test models on filtered classes
- `python scripts/metrics_all_cls.py` - Compute metrics for all classes
- `python scripts/metrics_filtered_cls.py` - Compute metrics for filtered classes

### Model Optimization Pipeline (Updated 2025-09-18)
- **Production model creation**: `python create_production_model.py` - Convert transfer learning model to optimized production format
- **Model comparison**: `CUDA_VISIBLE_DEVICES=3 python test_model_comparison.py` - Compare original vs optimized model performance
- **Transfer learning**: `CUDA_VISIBLE_DEVICES=2 wandb agent [sweep-id]` - Run WandB hyperparameter sweeps for model optimization

#### Current Optimized Model Performance
- **Model**: `segformer_b3_production_optimized_rail_0.7500.pth`
- **Rail IoU**: 75% (significant improvement from 62.49% baseline)
- **Inference time**: ~0.189s per image
- **Location**: `/home/mmc-server4/RailSafeNet/models/`
- **Comparison results**: `/home/mmc-server4/RailSafeNet/comparison_results/`

## Architecture Overview

**RailSafeNet** is an industrial-grade tram safety system designed for real-world deployment with international safety standards compliance.

### Core Components

1. **TheDistanceAssessor.py** - Main inference pipeline that:
   - Loads pre-trained SegFormer B3 and YOLOv8s models
   - Processes input images to detect rails and objects
   - Calculates distances and safety assessments for dynamic objects
   - Outputs visualizations with safety information
   - **Future integration point for LiDAR data fusion**

2. **Multi-Modal Sensor Fusion Architecture**:
   - **Camera-based perception**: SegFormer B3 for semantic segmentation, YOLOv8s for object detection
   - **LiDAR integration** (planned): Enhanced distance calculation for dynamic objects (people, cars, bicycles)
   - **Safety decision module**: International standards-compliant braking decisions
   - Results processed by Distance Assessor for real-time safety analysis

3. **Production-Ready Training Pipeline**:
   - Multi-GPU distributed training on 4x NVIDIA Titan RTX
   - Supports both SegFormer and DeepLabV3 for segmentation tasks
   - Custom PyTorch dataloaders for RailSem19 and Pilsen Railway datasets
   - WandB integration for experiment tracking and model versioning
   - Hyperparameter sweep capabilities for model optimization

### Key Directories

- `assets/`: Contains datasets, pre-trained models, and configuration files
  - `assets/models_pretrained/`: Fine-tuned model weights (SegFormer B3, YOLOv8s)
  - `assets/pilsen_railway_dataset/`: Custom railway dataset for training/testing
  - `assets/rs19val/`: RailSem19 validation dataset
  - `assets/requirements.txt`: Python dependencies
- `scripts/`: Utility scripts for data loading, testing, and metrics computation

### Dataset Structure

**RailSem19**: Standard railway semantic segmentation dataset
- **Images**: `/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val`
- **Masks**: `/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val`

**Pilsen Railway Dataset**: Custom dataset with tram-specific annotations

Both datasets use JSON configuration files for class mappings and data splits.

**Important Path Updates Needed:**
- Code requires path modifications to point to correct dataset locations
- Update dataset paths in training scripts from old paths to new Server/Datasets_hdd locations

### Model Configuration

- **Input size**: Configurable (default 1024x1024 for SegFormer)
- **Batch size**: Typically 16 for training
- **Classes**: Railway-specific classes including rails, objects, background
- **YOLO configuration**: `scripts/pilsen.yaml` contains training parameters

### Key Features

- **Real-time inference** capabilities for production tram safety monitoring
- **Multi-GPU training** support optimized for 4x NVIDIA Titan RTX setup
- **Distance calculation** between dynamic objects and railway infrastructure
- **International standards compliance** for safety-critical applications
- **Multi-modal sensor fusion** (camera + planned LiDAR integration)
- **Production deployment ready** with industrial joint research backing
- Visualization output with safety zones and object tracking for system validation
- Support for both research (training/evaluation) and production (inference) workflows

## Development Notes

- **Target deployment**: Industrial tram systems with real-world safety requirements
- **Dynamic object focus**: People, cars, bicycles detection and distance measurement
- **Safety standards**: International compliance for autonomous braking systems
- **Scalability**: Multi-GPU training architecture supports larger models and datasets

### File Management Principles (Added 2025-09-18)
**IMPORTANT**: Always follow these principles when modifying code:

1. **Modify Existing Files First**: Never create new files unnecessarily. Always check for existing files with similar functionality and modify them instead.

2. **Explicit Documentation**: Always document the process of finding and modifying existing files. State clearly:
   - Which existing files were found and their purposes
   - What modifications were made and why
   - Any new files created and justification for creation

3. **Duplicate Prevention**: Before creating any new file, search for existing files that serve the same or similar purpose. Clean up duplicate files regularly.

4. **Path Consistency**: Always use full absolute paths instead of relative paths. Ensure consistency across all configuration files.

**Example Process**:
- ✅ "Found existing `production_segformer_pytorch.py` for model loading, modifying it to use new model path"
- ✅ "Modified existing `create_production_model.py` to add ONNX/TensorRT optimization features"
- ❌ "Creating new `model_optimizer.py` for ONNX conversion" (without checking existing files)

## Deployment & Model Management

**Production Environment:**
- **Edge Platform**: NVIDIA AGX Orin for real-time inference on trams
- **Containerization**: Docker-based model deployment and version management
- **Model Updates**: Remote model updates via Docker container management
- **Commercial Product**: Designed for actual product sales and deployment

**Repository Management:**
- **IMPORTANT**: This is the original author's repository - DO NOT push to GitHub
- Local development and testing only
- Future migration to private repository for commercial development

**AGX Orin Optimization:**
- Models optimized for edge inference performance
- TensorRT integration for accelerated inference
- Memory-efficient model loading for embedded deployment
- Real-time processing requirements for safety-critical applications

## Model Performance & Optimization

**Pre-trained Model Issues:**
- **SegFormer_B3_1024_finetuned.pth**: Works excellently in TheDistanceAssessor.py but has finetuning difficulties
  - Model structure contains only 13 class distinctions
  - Complete model structure embedded in .pth file makes modification complex
  - Standard TensorRT optimization failed due to unique model architecture

**Optimization Results:**
- **YOLOv8n**: Successfully optimized with ONNX and TensorRT
- **SegFormer B3**: TensorRT optimization failed, required hybrid quantization approach
- **Hybrid Quantization**: Using int8/int16 mixed precision achieved 14-15 FPS performance

**Demo Implementation:**
- **Evaluation Setup**: `/home/mmc-server4/RailSafeNet_mini_DT/run_hybrid_demo.py`
- **Test Data**: 30-second tram driving video evaluation
- **Performance**: Real-time inference at 14-15 FPS on target hardware

**Known Issues:**
- SegFormer model architecture incompatibility with standard optimization tools
- Need for custom quantization strategies for complex model structures
- Dataset path updates required throughout codebase

## Development Roadmap

### Phase 1: Model Optimization & Fine-tuning
**Objective**: Achieve accurate rail detection with IoU ≥ 0.6

1. **SegFormer Model Selection & Fine-tuning**:
   - Option A: Use existing `SegFormer_B3_1024_finetuned.pth` (13 classes)
   - Option B: Use Hugging Face `SegFormer_B3.pth` and fine-tune from scratch
   - Fine-tuning on 4x NVIDIA Titan RTX
   - Target: Rail detection IoU ≥ 0.6

2. **Model Optimization Success**:
   - Optimize YOLOv8n.pt for object detection
   - Optimize SegFormer_B3.pth for segmentation
   - Ensure both models work efficiently together

### Phase 2: LiDAR Integration & Sensor Fusion
**Objective**: Accurate distance calculation for dynamic objects

1. **Sensor Calibration**:
   - Camera-LiDAR calibration matrix establishment
   - Coordinate system alignment
   - **Temporal synchronization**: ±1ms precision requirement
   - **Sensor fusion algorithm**: Kalman Filter implementation

2. **Distance Calculation Enhancement**:
   - Integrate LiDAR data with YOLO detections
   - **Target objects**: Only people with YOLO-LiDAR overlap (ignore occluded objects)
   - **Danger zone**: Use TheDistanceAssessor's existing area definition (tunable parameters)
   - Accurate distance measurement for overlapping detections only
   - Modify `TheDistanceAssessor.py` to LiDAR version

3. **3D Projection Accuracy Validation**:
   - **Recommended dataset**: KITTI (available at `/home/mmc-server4/Server/Datasets_hdd/`)
   - KITTI provides synchronized camera-LiDAR data with ground truth
   - Validate 3D→2D projection accuracy using KITTI's calibration parameters
   - Benchmark coordinate transformation precision

### Phase 3: Performance Validation
**Objective**: Real-time performance with high accuracy

1. **Video Testing**:
   - Test data: `/assets/crop/tramxx.mp4` (30-second videos)
   - Target performance: ≥30 FPS with ≥90% accuracy
   - Frame-by-frame analysis and validation
   - **Note**: Test diversity limited by RailSem19 dataset constraints

2. **Performance Metrics Breakdown**:
   - Object-specific accuracy (people vs cars vs bicycles)
   - Distance-based accuracy (near-field vs far-field)
   - False positive/negative analysis for overlapping detections
   - YOLO-LiDAR fusion accuracy metrics

### Phase 4: Production Deployment
**Objective**: Deploy to NVIDIA AGX Orin

1. **Docker Containerization**:
   - Create optimized Docker image for AGX Orin
   - Include all dependencies and models
   - Version management system

2. **Embedded Testing**:
   - Real hardware testing on AGX Orin
   - Performance benchmarking
   - **Safety validation**: Tram-specific safety standards (adapted from ISO 26262 automotive standards)
   - Fail-safe mechanism implementation
   - Sensor failure response protocols

## Additional Implementation Requirements

### **Sensor Fusion Specifications**:
- **Synchronization tolerance**: ±1ms between camera and LiDAR
- **Fusion algorithm**: Kalman Filter for state estimation
- **Detection criteria**: Only YOLO-LiDAR overlapping objects classified as danger

### **Danger Zone Configuration**:
- Based on existing `TheDistanceAssessor` area definitions
- Tunable parameters for different operational scenarios
- Focus on people detection (primary safety concern)

### **Available Datasets**:
- **Validation dataset**: KITTI (recommended for 3D projection accuracy)
  - Location: `/home/mmc-server4/Server/Datasets_hdd/KITTI`
  - Provides camera-LiDAR synchronization ground truth
- **Training limitation**: RailSem19 dataset constraints limit weather/lighting diversity

### **CI/CD & Monitoring**:
- Automated model validation pipeline
- Performance regression testing
- Over-the-air update verification system
- Real-time performance monitoring and logging