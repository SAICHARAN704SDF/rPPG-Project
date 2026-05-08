# rPPG-Project

## 📋 Overview

**rPPG-Project** is a comprehensive remote photoplethysmography (rPPG) solution for heart rate estimation and health monitoring using deep learning models and modern neural network architectures.

## ✨ Features

- **Multiple Model Architectures**
  - RhythmMamba models (best & standard versions)
  - Deep RPPG model
  - Transformer-based RPPG
  - BPM estimation model

- **Real-time Processing**
  - Live video stream processing (`live_stable.py`)
  - Testing and validation (`live_test.py`)
  - Model inference pipeline

- **Pre-trained Models**
  - `rhythmmamba_best.pth` - Best performing RhythmMamba model
  - `rythmmamba.pth` - Standard RhythmMamba model
  - `checkpoint.pth` - Checkpoint for transformer model
  - `transformer_rppg.pth` - Transformer-based RPPG model
  - `deep_rppg_model.pkl` - Deep learning RPPG model
  - `bpm_model.pkl` - BPM estimation model

## 🚀 Quick Start

### Requirements
- Python 3.8+
- PyTorch
- OpenCV
- NumPy

### Installation

```bash
# Clone the repository
git clone https://github.com/SAICHARAN704SDF/rPPG-Project.git
cd rPPG-Project

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Run Live Stable Processing:**
```bash
python live_stable.py
```

**Run Live Testing:**
```bash
python live_test.py
```

## 📁 Project Structure

```
rPPG-Project/
├── live_stable.py          # Stable production inference
├── live_test.py            # Testing and validation
├── modelrythm.py           # RhythmMamba model implementation
├── models/
│   ├── rhythmmamba_best.pth
│   ├── rythmmamba.pth
│   ├── checkpoint.pth
│   ├── transformer_rppg.pth
│   ├── deep_rppg_model.pkl
│   ├── bpm_model.pkl
│   └── Archive.zip
└── README.md
```

## 🔧 Model Details

### RhythmMamba Models
State-of-the-art sequential models for HR estimation with temporal dependencies.

### Transformer Models
Deep transformer architecture for capturing complex temporal patterns in video data.

### Deep RPPG Model
Convolutional neural network optimized for remote PPG signal extraction.

## 📊 Performance

All models are pre-trained and ready for inference. They have been validated on standard rPPG datasets.

## 📝 License & Copyright

**Copyright © 2026 - All rights reserved**

This project and all associated code, models, and documentation are the exclusive property of **Charan Kailasa** (SAICHARAN704SDF).

### License Terms:
-
- **Proprietary Rights**: All rights are reserved. Unauthorized copying, distribution, or modification of this work is strictly prohibited.
- **Permitted Use**: This software is provided for personal, educational, and authorized commercial use only with explicit permission from the copyright holder.
- **Restrictions**: 
  - Do not redistribute without permission
  - Do not modify and claim as your own
  - Do not use for commercial purposes without authorization
  - Do not reverse engineer or extract models for redistribution

For licensing inquiries or permissions, contact: **saicharankailasa@gmail.com**

## 👨‍💻 Author

** Charan Kailasa**
- GitHub: [@SAICHARAN704SDF](https://github.com/SAICHARAN704SDF)
- Email: saicharankailasa@gmail.com

## 🙏 Acknowledgments

Special thanks to the deep learning and computer vision research communities for foundational work in remote photoplethysmography.

---

**Last Updated:** May 8, 2026
