# System Architecture Overview

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Face Detection  │───▶│   Face Crop     │
│  (RGB Frames)   │    │    & Tracking    │    │  & Alignment    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   HR Estimate   │◀───│  Signal Processing│◀───│   PhysNet CNN   │
│   (60-180 BPM)  │    │   & FFT Analysis  │    │ (3D Spatiotemporal)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Model Architecture Details

### PhysNet 3D CNN
- **Input**: (B, 3, T, H, W) - Batch of RGB video clips
- **Encoder**: 3D convolutions with temporal pooling
- **Decoder**: Upsampling to reconstruct temporal resolution  
- **Output**: (B, S²+1, T) - Spatial signals + averaged rPPG

### Self-Supervised Learning Pipeline
```
Input Video Clips ──┐
                    ├─── Frequency Augmentation ──┐
                    └─── Original ────────────────┘
                                                   │
                                                   ▼
                                            Contrastive Learning
                                                   │
                                                   ▼
                                          Feature Representations
```

## Technical Innovations
- **Frequency Domain Augmentation**: Novel approach for physiological consistency
- **Multi-view Temporal Learning**: Robust representation through temporal variations
- **Distributed Training**: Scalable across multiple GPUs
- **Privacy-Preserving**: Anonymization techniques for healthcare data
