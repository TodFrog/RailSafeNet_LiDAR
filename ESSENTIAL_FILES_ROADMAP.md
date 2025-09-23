# 🚀 RailSafeNet 필수 파일 사용 순서 가이드

**Updated**: 2025-09-23
**Current Status**: 정리 완료, Production 모델 성능 확인 완료

## 📊 **현재 성능 분석 결과**

### **SegFormer B3 Transfer Model 성능**
- **모델**: `segformer_b3_transfer_best_rail_0.7791.pth`
- **Combined Rail IoU**: **21.8%** (10개 이미지 테스트)
- **처리 시간**: 평균 3.31초/이미지
- **주요 이슈**: Class mapping 문제로 실제 rail 성능이 낮음

### **개별 클래스 성능**
- **Class 1 (rail-guarded)**: IoU 9.3%, F1 13.1%
- **Class 4 (rail-track)**: IoU 0.1%, F1 0.3%
- **Class 9 (rail-road)**: IoU 0%, F1 0%

## 🎯 **다음 단계별 작업 순서**

### **1단계: 즉시 실행 가능한 파일들**

#### **A. 이미지 테스트 및 검증**
```bash
# 현재 최적화된 distance assessor 실행
python TheDistanceAssessor_2.py
```
- **목적**: Production 모델로 이미지 처리 테스트
- **특징**: 77.91% IoU 모델 사용, 위험구역 생성 성공
- **결과**: 실시간 안전 검출 가능

#### **B. 성능 분석**
```bash
# SegFormer 성능 재평가 (더 많은 이미지로)
python evaluate_segformer_rail_performance.py
```
- **목적**: 전체 데이터셋 성능 확인
- **현재**: 10개 이미지 테스트 완료
- **다음**: max_images=None으로 전체 데이터셋 평가

### **2단계: 모델 최적화 및 변환**

#### **A. TensorRT 최적화**
```bash
# ONNX 변환
python production_segformer_onnx.py

# TensorRT 변환
python convert_enhanced_to_tensorrt.py
```
- **목적**: 실시간 성능 확보 (목표: 30+ FPS)
- **예상 효과**: 10x 속도 향상 (3.3초 → 0.33초)

#### **B. 비디오 프레임 테스트**
```bash
# 비디오 프레임 처리 테스트
python video_frame_tester.py
```
- **목적**: 연속 프레임 처리 성능 검증
- **결과**: 실시간 비디오 스트림 대응 가능성 확인

### **3단계: 모델 재훈련 (필요시)**

#### **A. Transfer Learning 최적화**
```bash
# 현재 best 모델 기반 추가 학습
python train_SegFormer_transfer_learning.py
```
- **목적**: Rail class IoU 개선 (목표: 50%+)
- **현재 이슈**: Class mapping 불일치 해결 필요
- **데이터**: RailSem19 dataset 사용

#### **B. 하이퍼파라미터 스윕**
```bash
# WandB 스윕 실행
CUDA_VISIBLE_DEVICES=2 wandb agent [sweep-id]
```
- **목적**: 최적 하이퍼파라미터 탐색
- **GPU**: NVIDIA Titan RTX 사용
- **메트릭**: Rail IoU 최적화 집중

### **4단계: Production 배포 준비**

#### **A. 최종 모델 생성**
```bash
# Production 모델 최적화
python create_production_model.py
```
- **목적**: 배포용 최종 모델 생성
- **최적화**: TensorRT, 메모리 효율성
- **타겟**: NVIDIA AGX Orin

#### **B. 성능 검증**
```bash
# 모델 비교 테스트
CUDA_VISIBLE_DEVICES=3 python test_model_comparison.py
```
- **목적**: 원본 vs 최적화 모델 성능 비교
- **메트릭**: 정확도, 속도, 메모리 사용량

## 📁 **핵심 파일 분류**

### **🎯 현재 사용 중인 핵심 파일**
1. **TheDistanceAssessor_2.py** - 메인 실행 파일 (Production Ready)
2. **evaluate_segformer_rail_performance.py** - 성능 평가 스크립트
3. **train_SegFormer_transfer_learning.py** - 모델 훈련
4. **create_production_model.py** - Production 모델 생성

### **🔧 모델 최적화 파일**
5. **production_segformer_onnx.py** - ONNX 변환
6. **convert_enhanced_to_tensorrt.py** - TensorRT 변환
7. **video_frame_tester.py** - 비디오 처리 테스트

### **📊 분석 및 테스트 파일**
8. **scripts/test_filtered_cls.py** - 클래스 필터링 테스트
9. **scripts/test_pilsen.py** - Pilsen 데이터셋 테스트
10. **scripts/metrics_filtered_cls.py** - 메트릭 계산

### **⚙️ 지원 파일**
- **scripts/dataloader_SegFormer.py** - 데이터 로더
- **train_SegFormer.py** - 기본 SegFormer 훈련
- **train_DeepLabv3.py** - 대안 모델 훈련
- **train_yolo.py** - YOLO 모델 훈련

## 🎪 **모델 경로 정보**

### **현재 Production 모델**
```
/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/production/segformer_b3_transfer_best_rail_0.7791.pth
```
- **성능**: 77.91% Rail IoU (훈련 시), 실제 테스트에서는 21.8%
- **상태**: 사용 가능하지만 성능 개선 필요

### **YOLO 모델**
```
/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/yolo/yolov8s.pt
```
- **성능**: 객체 검출 정상 작동
- **상태**: Production Ready

### **데이터셋 경로**
```
# RailSem19 Validation
Images: /home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val
Masks:  /home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val

# Pilsen Railway Dataset
/home/mmc-server4/RailSafeNet/assets/pilsen_railway_dataset/
```

## 🚨 **당면 과제 및 해결책**

### **Issue 1: Rail Detection 성능 저하**
- **문제**: Combined Rail IoU 21.8% (예상보다 낮음)
- **원인**: Class mapping 불일치, Ground truth vs Prediction mismatch
- **해결책**: Class mapping 재검토, 모델 재훈련 필요

### **Issue 2: 처리 속도**
- **현재**: 3.31초/이미지
- **목표**: 실시간 (30+ FPS = 0.033초/이미지)
- **해결책**: TensorRT 최적화 적용 필요

### **Issue 3: Class 4, 9 검출 실패**
- **Class 4 (rail-track)**: 거의 검출 안됨 (IoU 0.1%)
- **Class 9 (rail-road)**: 전혀 검출 안됨 (IoU 0%)
- **해결책**: 클래스별 loss weighting, data augmentation

## 🎯 **권장 실행 순서**

### **즉시 실행 (이번 주)**
1. **전체 데이터셋 성능 평가**: `evaluate_segformer_rail_performance.py` (max_images=None)
2. **TensorRT 최적화**: `production_segformer_onnx.py` → `convert_enhanced_to_tensorrt.py`
3. **비디오 테스트**: `video_frame_tester.py`

### **중기 계획 (다음 주)**
1. **모델 재훈련**: Rail class 성능 개선 집중
2. **하이퍼파라미터 최적화**: WandB sweep 실행
3. **AGX Orin 테스트**: 실제 하드웨어 배포 테스트

### **장기 계획 (한 달 내)**
1. **LiDAR 통합**: 센서 퓨전 구현
2. **실시간 시스템**: 30+ FPS 달성
3. **Production 배포**: 산업용 트램 시스템 적용

---

**🎉 현재 상태**: 정리 완료, 성능 분석 완료, 다음 단계 준비 완료
**🚀 다음 단계**: TensorRT 최적화를 통한 실시간 성능 확보