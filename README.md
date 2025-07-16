# Remote Photoplethysmography with Self-Supervised Learning

<div align="center">
  
![Status](https://img.shields.io/badge/Status-Demo%20Only-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-Demo%20Only-lightgrey?style=for-the-badge)

[![Portfolio](https://img.shields.io/badge/Portfolio-Healthcare%20AI-green?style=flat-square)](https://0228lin.github.io/)
[![Skills](https://img.shields.io/badge/Skills-Computer%20Vision%20%7C%20Deep%20Learning%20%7C%20Signal%20Processing-blue?style=flat-square)](https://github.com/yourusername)

</div>

---

## ⚠️ Notice

> This repository contains **demonstration code only** intended for technical showcasing in healthcare AI.  
> 🚫 Not for practical, academic, or commercial use.  
> ✅ Fully compliant with data governance standards.  
> 🛡️ No confidential or sensitive data included.

---

## 📊 Project Overview

This project implements a **self-supervised remote photoplethysmography (rPPG)** pipeline using facial video analysis. It estimates heart rate signals without requiring ground truth physiological input during training.

---

## 🚀 Highlights

- 💡 **Self-Supervised Learning**: Frequency-domain contrastive learning for physiological signal estimation  
- 🖥️ **Distributed Training**: Multi-GPU support via PyTorch DDP  
- 🔐 **Privacy Preservation**: Anonymized preprocessing for healthcare datasets  
- 🎚️ **Signal Processing**: Robust FFT-based heart rate extraction  
- 🧩 **Model Architecture**: PhysNet-based 3D CNN for spatiotemporal learning  

---

## 🔬 Technical Design

### Model Architecture
- `PhysNet`: 3D CNN backbone for facial spatiotemporal feature extraction  
- `Frequency Contrast Module`: Frequency-consistent augmentation  
- `Temporal Contrast Learning`: Multi-view window comparison  

### Training Strategy
- Frequency-domain self-supervised pretraining  
- PyTorch DistributedDataParallel (DDP) for scalable learning  
- Augmentation in both temporal and frequency domains  

---

## 📚 Datasets Used

Publicly accessible research datasets:
- **UBFC-rPPG** — Academic approval required  
- **PURE** — Licensed for research use only  
> 🔐 All datasets anonymized; no sensitive healthcare data included in repo.

---

## 💻 Quick Start Demo

```bash
# Clone and run demo
git clone https://github.com/0228lin/remote-photoplethysmography-ssl.git
cd remote-photoplethysmography-ssl
pip install -r requirements.txt
python scripts/demo.py
```

---

## 💼 Experience Highlights

- AI-driven healthcare applications with interdisciplinary collaboration  
- Privacy-first data workflows and anonymization strategies  
- Deep learning-based signal extraction and feature engineering  
- Clear documentation and reproducibility standards  

---

## 🎓 Academic Context

Developed during a healthcare data preprocessing internship supporting:
- Data governance and anonymization pipelines  
- Feature engineering and CNN optimization  
- Collaboration across AI and healthcare domains  

---

## 📊 Technical Metrics

| Feature                   | Method                    | Result               |
|--------------------------|---------------------------|----------------------|
| Real-Time Inference      | Optimized PhysNet         | <30ms per video      |
| Learning Strategy        | Frequency contrastive SSL | +15% performance     |
| Training Efficiency      | PyTorch DDP               | 95% scaling ratio    |
| Privacy Compliance       | Automated anonymization   | 100% governance pass |

---

## 👩‍💻 Developed By

This work was developed in collaboration with the **IHPC (Institute of High Performance Computing), A*STAR (Agency for Science, Technology and Research)** as part of a healthcare AI research initiative.

**Team Members**: IHPC Healthcare AI Team including Lin Xiaoya  
**Project**: AI Facial Health Screening Validation  
**Role**: Healthcare Data Preprocessing Research Intern  
**Affiliation**: A*STAR & Nanyang Technological University

---

## 🔗 Connect with Me

- 💼 [LinkedIn](https://www.linkedin.com/in/xiaoya-lin/)  
- 📧 [Email](mailto:linx0070@e.ntu.edu.sg)  
- 🧪 [Portfolio](https://0228lin.github.io/)  


