# Remote Photoplethysmography with Self-Supervised Learning

**⚠️ DEMONSTRATION REPOSITORY ONLY**
This repository contains demonstration code developed during a healthcare data preprocessing research internship. It is intended for portfolio purposes only and contains no confidential data or proprietary information.

## Overview

This project implements a self-supervised learning approach for remote photoplethysmography (rPPG) using facial video analysis. The system can estimate heart rate from facial videos without requiring ground truth physiological signals during training.

## Key Features

- **Self-Supervised Learning**: Novel frequency-domain contrastive learning approach
- **Distributed Training**: Multi-GPU training support with PyTorch DDP
- **Data Privacy**: Anonymized preprocessing workflows for healthcare datasets
- **Signal Processing**: Advanced FFT-based heart rate estimation
- **Model Architecture**: PhysNet-based 3D CNN for spatiotemporal feature extraction

## Technical Approach

### Model Architecture
- **PhysNet**: 3D CNN backbone for extracting spatiotemporal features from facial videos
- **Frequency Contrast Module**: Self-supervised learning through frequency domain augmentation
- **Multi-view Temporal Learning**: Contrastive learning across different temporal windows

### Training Strategy
- **Self-Supervised Pretraining**: Frequency domain consistency learning
- **Distributed Training**: PyTorch DistributedDataParallel for scalable training
- **Data Augmentation**: Temporal and frequency domain augmentations

## Dataset Information

This work utilizes publicly available datasets for research purposes:
- **UBFC-rPPG**: Available with academic approval
- **PURE**: Available for research use
- Custom preprocessing for healthcare datasets (anonymized)

**Note**: All sensitive healthcare data has been anonymized and is not included in this repository.

## Requirements

```bash
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.5.0
scipy>=1.7.0
numpy>=1.21.0
matplotlib>=3.3.0
facenet-pytorch>=2.5.0
wandb>=0.12.0
