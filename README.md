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

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/remote-photoplethysmography-ssl.git
cd remote-photoplethysmography-ssl
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare datasets** (requires approval for academic datasets)
```bash
python scripts/preprocess_data.py --dataset_path /path/to/dataset
```

4. **Train the model**
```bash
# Single GPU
python scripts/train.py --config configs/training_config.yaml

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train.py --config configs/training_config.yaml
```

## Research Contributions

- **Novel Self-Supervised Learning**: Developed frequency-domain contrastive learning for rPPG
- **Healthcare Data Processing**: Created privacy-preserving preprocessing workflows
- **Performance Optimization**: Achieved competitive results with minimal labeled data
- **Scalable Training**: Implemented distributed training for large-scale datasets

## Academic Background

This work was developed during a healthcare data preprocessing research internship, supporting AI-driven healthcare applications with focus on:
- Data governance and privacy compliance
- Feature engineering and model optimization
- Cross-functional collaboration in healthcare AI

## Disclaimer

- This repository is for **demonstration and portfolio purposes only**
- Contains **no confidential or proprietary information**
- All healthcare data has been **anonymized** according to data governance standards
- Not intended for commercial use or medical diagnosis
- Datasets require separate approval for academic research

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@misc{rppg_ssl_2024,
  title={Self-Supervised Learning for Remote Photoplethysmography},
  author={[Your Name]},
  year={2024},
  note={Demonstration repository for healthcare AI research}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this demonstration repository, please open an issue.

**Note**: This repository demonstrates technical capabilities and methodologies only. No confidential information or sensitive data is included.
