# <img src="assets/favicon.svg" width="40" align="center" alt="MALIN logo"/> MALIN — Face Analysis Toolkit

> Anti-spoofing · Deepfake detection · Emotion recognition

## Scope — three tracks

| Track | Input | Threat model | Deployment |
|-------|-------|--------------|------------|
| **[1. Access Control Kiosk](antispoof/)** | Live webcam, IR camera, or Intel RealSense D435 | Physical spoof: printed photos, screen replay, video replay | Real-time, kiosk terminal (GPU optional) |
| **[2. Deepfake Detection](deepfake/)** | Pre-recorded images/videos | Synthetic/manipulated face content in media | Offline / batch analysis |
| **[3. Emotion Recognition](emotion/)** | Live webcam | Facial expression classification | Real-time, CPU |

## Features

**Access Control Kiosk (Track 1):**
- Face registration & recognition (InsightFace embeddings)
- Multi-layer anti-spoofing: LBP texture, YCbCr skin color, HSV saturation, IR ratio, depth flatness (D435), blink detection
- Intel RealSense D435 support — RGB + IR + Depth from a single device, with depth-based closest-face selection
- PyQt5 GUI with Blur / Blink / Texture toggle buttons

**Deepfake Detection (Track 2):**
- XceptionNet binary classifier — v3.4: **97.70% ACC, 0.9967 AUC** (108k, 13 sources)
- Gradio web UI with verdict badges and P(fake) timeline

**Emotion Recognition (Track 3):**
- Real-time 7-emotion classification (DeepFace + InsightFace)
- PyQt5 GUI with emotion bar chart and smoothing

## Demo

### Track 1 — Access Control Kiosk (PyQt5)

Real-time webcam with face registration, multi-layer anti-spoofing (LBP texture + YCbCr skin + HSV saturation + IR ratio + depth flatness + blink), and runtime toggle buttons for each detection layer. Optional Intel RealSense D435 input combines RGB / IR / Depth from a single device.

| Authorized | Display Attack | Print Attack |
|:----------:|:--------------:|:------------:|
| <img src="docs/screenshots/kiosk_1776911530.png" width="260"> | <img src="docs/screenshots/kiosk_1776911609.png" width="260"> | <img src="docs/screenshots/kiosk_1776916366.png" width="260"> |

### Track 2 — Deepfake Detector (Gradio)

Terminal-themed web UI with Image / Video tabs, verdict badge (REAL / FAKE / MIXED), and `Deepfake Probability Timeline` plot for video.

Same StyleGAN-generated face — v2 misclassifies as REAL, v3 correctly detects FAKE:

| v2 (Before) | v3 (After) |
|:-----------:|:----------:|
| <img src="docs/screenshots/gradio_03_stylegan_ood.png" width="340"> | <img src="docs/screenshots/gradio_04_stylegan_v3_fixed.png" width="340"> |

<details>
<summary>More screenshots (UI, video analysis)</summary>

**Gradio UI**

<img src="docs/screenshots/gradio_01_initial.png" width="700" alt="Gradio UI">

**Video analysis — face-swap detection success.** MIXED verdict with per-frame P(fake) timeline.

<img src="docs/screenshots/gradio_02_video_mixed.png" width="700" alt="Video MIXED">

</details>

### Track 3 — Emotion Recognition (PyQt5)

Real-time 7-emotion classification (angry, disgust, fear, happy, sad, surprise, neutral) with InsightFace face detection and DeepFace emotion analysis. Background thread processing for smooth camera feed.

| Neutral | Happy | Surprise | Angry |
|:-------:|:-----:|:--------:|:-----:|
| <img src="docs/screenshots/emotion_neutral.png" width="220"> | <img src="docs/screenshots/emotion_happy.png" width="220"> | <img src="docs/screenshots/emotion_surprise.png" width="220"> | <img src="docs/screenshots/emotion_angry.png" width="220"> |

## Project Structure

```
face-defense/
├── shared/                     # Common utilities
│   ├── camera.py               # Camera abstraction (cv2 / Intel RealSense D435)
│   ├── metrics.py              # AUC, EER, ACER metrics
│   ├── visualization.py        # ROC, score distribution plots
│   ├── constants.py            # ImageNet normalization
│   └── face_utils.py           # FaceDatabase, blink EAR, crop
├── antispoof/                  # Track 1 — Kiosk anti-spoofing
│   ├── models/cdcn_model.py    # CDCN network
│   ├── data/                   # CelebA-Spoof dataset
│   └── scripts/                # Demo, training, benchmark
├── deepfake/                   # Track 2 — Deepfake detection
│   ├── data/                   # FF++, FFT, video datasets
│   └── scripts/                # Demo, training, benchmark
├── emotion/                    # Track 3 — Emotion recognition
├── notebooks/                  # Benchmark evaluation
└── assets/                     # MALIN branding
```

## Getting Started

### Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA support (recommended)
- Conda (Miniconda or Anaconda)

### Installation

```bash
conda create -n face-defense python=3.10 -y
conda activate face-defense

# 1. PyTorch (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Track 1 & 2 core dependencies
pip install insightface onnxruntime-gpu opencv-python PyQt5 timm
pip install mediapipe==0.10.14
pip install onnx==1.16.1
pip install pyrealsense2          # optional, only needed for the --d435 flag

# 3. Track 2 deepfake web UI
pip install gradio

# 4. Track 3 emotion recognition (TensorFlow-based — install last)
pip install deepface
pip install tensorflow==2.15.1 tf-keras==2.15.1

# 5. Project install
pip install -e .
```

> **Install order matters.** Install `mediapipe` *before* `deepface`/`tensorflow` to avoid `protobuf` version conflicts.

#### Version compatibility

| Package | Pinned version | Reason |
|---------|---------------|--------|
| `mediapipe` | 0.10.14 | 0.10.33+ removed the `solutions` API |
| `onnx` | 1.16.1 | Newer versions clash with `ml_dtypes` |
| `tensorflow` | 2.15.1 | 2.21+ requires protobuf 7.x → conflicts with mediapipe |
| `tf-keras` | 2.15.1 | Resolves Keras-detach issue in TF 2.15 |

### Quick Start

```bash
# Track 1 — Access Control Kiosk
#   Standard webcam:
python antispoof/scripts/demo_gui.py --camera 0
#   Webcam + separate USB IR camera:
python antispoof/scripts/demo_gui.py --camera 0 --ir_camera 1
#   Intel RealSense D435 (RGB + IR + Depth, 1 m max range):
python antispoof/scripts/demo_gui.py --d435 --max_depth 1.0

# Track 2 — Deepfake Detector (Gradio web UI)
python deepfake/scripts/demo_deepfake_gui.py

# Track 3 — Emotion Recognition
python emotion/scripts/demo_emotion_gui.py --camera 0
```

See each track's README for detailed usage, training, and benchmark instructions:
- [Anti-Spoofing (Track 1)](antispoof/README.md)
- [Deepfake Detection (Track 2)](deepfake/README.md)
- [Emotion Recognition (Track 3)](emotion/README.md)

## References

### Papers
- [CDCN: Central Difference Convolutional Network](https://arxiv.org/abs/2003.04092) — Face anti-spoofing via depth map
- [FaceForensics++](https://arxiv.org/abs/1901.08971) — Deepfake detection benchmark (XceptionNet)
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Efficient image classification

### Datasets
- [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) — 561K anti-spoofing images
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Deepfake detection dataset
- [LCC-FASD](https://csit.am/2019/proceedings/PRIP/PRIP2.pdf) — Cross-dataset evaluation
- [NUAA](https://www.kaggle.com/datasets/olgabelitskaya/photo-paper-datasets) — Webcam anti-spoofing

### Libraries
- [InsightFace](https://github.com/deepinsight/insightface) — Face detection & recognition
- [MediaPipe](https://github.com/google/mediapipe) — Face mesh & landmark detection
- [DeepFace](https://github.com/serengil/deepface) — Face analysis (emotion, age, gender)
- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch image models
