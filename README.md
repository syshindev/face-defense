# Face Defense

Real-time face anti-spoofing and deepfake detection system for access control.

## Features

- **Face Registration & Recognition** — InsightFace embedding-based identity verification
- **Anti-Spoofing** — Detects printed photos, screen replays, and 3D masks
- **Liveness Detection** — EAR-based blink detection for passive liveness verification
- **IR Camera Support** — Instant spoof detection with near-infrared camera (940nm)
- **Deepfake Detection** — XceptionNet / EfficientNet-B4 multi-class classification
- **GUI Demo** — PyQt5-based access control demo with real-time camera feed

## Architecture

```
Input Frame
    │
    v
Face Detection (InsightFace)
    │
    ├─── IR Camera ──> Instant Spoof Detection (screen/print)
    │
    ├─── Liveness ──> Blink Detection (EAR)
    │
    ├─── Anti-Spoof ──> CDCN Depth Map Analysis
    │
    └─── Deepfake ──> XceptionNet / EfficientNet-B4
    │
    v
REAL / SPOOF + Face Recognition (Authorized / Unauthorized)
```

## Benchmark Results

### Anti-Spoofing (CDCN)

| Dataset | AUC | ACER |
|---------|-----|------|
| CelebA-Spoof (same-domain, 67K) | 0.9985 | 0.0272 |
| LCC-FASD (cross-dataset, 7.5K) | 0.8097 | 0.2519 |

### Model Comparison (LCC-FASD cross-dataset)

| Metric | Silent-FAS | CDCN |
|--------|-----------|------|
| AUC | 0.7757 | **0.8097** |
| ACER | 0.3070 | **0.2519** |

## Project Structure

```
face-defense/
├── configs/                    # YAML configuration files
├── face_defense/
│   ├── core/                   # Pipeline, registry, base classes
│   ├── data/                   # Dataset loaders, preprocessing
│   │   ├── celeba_spoof_dataset.py
│   │   └── ff_dataset.py
│   ├── models/
│   │   ├── anti_spoof/         # Silent-FAS, CDCN
│   │   ├── deepfake/           # XceptionNet, EfficientNet
│   │   └── liveness/           # Blink, texture, depth
│   ├── evaluation/             # Metrics, visualization
│   └── utils/                  # Device management
├── scripts/
│   ├── demo_gui.py             # PyQt5 access control demo
│   ├── demo_access.py          # OpenCV access control demo
│   ├── demo_webcam.py          # Webcam liveness demo
│   ├── train_cdcn.py           # CDCN training script
│   ├── train_deepfake.py       # Deepfake training script
│   ├── benchmark_cdcn.py       # CDCN benchmark evaluation
│   ├── finetune_cdcn_nuaa.py   # NUAA fine-tuning
│   └── extract_frames.py       # FF++ frame extraction
├── notebooks/                  # Benchmark evaluation notebooks
└── third_party/                # External repos (Silent-FAS)
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

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install insightface onnxruntime-gpu opencv-python mediapipe PyQt5 timm
pip install -e .
```

## Usage

### GUI Demo (Access Control)
```bash
python scripts/demo_gui.py --camera 0
```

### GUI Demo with IR Camera
```bash
python scripts/demo_gui.py --camera 0 --ir_camera 1
```

### OpenCV Demo
```bash
python scripts/demo_access.py --camera 0
```

### Webcam Liveness Demo
```bash
python scripts/demo_webcam.py --camera 0
```

### Training

```bash
# Anti-spoofing (CDCN)
python scripts/train_cdcn.py --data_root data/CelebA_Spoof --epochs 50

# Deepfake detection (XceptionNet)
python scripts/train_deepfake.py --data_root data/ff-c23-frames --model legacy_xception --epochs 30
```

### Benchmark
```bash
python scripts/benchmark_cdcn.py
```

See [benchmark results](notebooks/benchmark_eval.ipynb) for detailed evaluation.

## Security Levels

| Level | Components | Detects | GPU |
|-------|-----------|---------|-----|
| Basic | Liveness (Blink) | Photo, video replay | Not required |
| Standard | Liveness + IR Camera | + Print, display attacks | Not required |
| Advanced | Liveness + IR + CDCN | + High-quality forgeries | Required |
| Maximum | All + Deepfake Detection | + AI-generated faces | Required |

## Key Scripts

| Script | Description |
|--------|-------------|
| [demo_gui.py](scripts/demo_gui.py) | PyQt5 access control demo |
| [demo_access.py](scripts/demo_access.py) | OpenCV access control demo |
| [demo_webcam.py](scripts/demo_webcam.py) | Webcam liveness demo |
| [train_cdcn.py](scripts/train_cdcn.py) | CDCN anti-spoofing training |
| [train_deepfake.py](scripts/train_deepfake.py) | Deepfake detection training |
| [benchmark_cdcn.py](scripts/benchmark_cdcn.py) | CDCN benchmark evaluation |
| [finetune_cdcn_nuaa.py](scripts/finetune_cdcn_nuaa.py) | NUAA fine-tuning |
| [extract_frames.py](scripts/extract_frames.py) | FF++ video frame extraction |

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
- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch image models
