# Face Defense

Real-time face anti-spoofing and deepfake detection system for access control.

## Features

- **Face Registration & Recognition** — InsightFace embedding-based identity verification
- **Anti-Spoofing** — Detects printed photos, screen replays, and 3D masks
- **Liveness Detection** — EAR-based blink detection for passive liveness verification
- **IR Camera Support** — Instant spoof detection with near-infrared camera (940nm)
- **Deepfake Detection** — XceptionNet binary classifier (real vs fake) on FaceForensics++ c23
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
    └─── Deepfake ──> XceptionNet (binary real/fake)
    │
    v
REAL / SPOOF + Face Recognition (Authorized / Unauthorized)
```

## Benchmark Results

### Anti-Spoofing (CDCN)

| Dataset | AUC | ACER |
|---------|-----|------|
| CelebA-Spoof (same-domain, 67K) | 0.9985 | 0.0272 |
| LCC-FASD (cross-dataset, 7.5K) | 0.8174 | 0.2463 |

### Model Comparison (LCC-FASD cross-dataset)

| Metric | Silent-FAS (historical) | CDCN |
|--------|-------------------------|------|
| AUC | 0.7757 | **0.8174** |
| ACER | 0.3070 | **0.2463** |

> Silent-FAS numbers predate the removal of its dependencies (commit `fefcd70`); kept for historical reference.

### Deepfake Detection (XceptionNet, binary)

Trained on FaceForensics++ c23 (extracted with `extract_frames.py`) with stratified 80/20 per-class split. Reached **Val Acc 92.35%** on the training distribution. Evaluated on a separately preprocessed FF++ / Celeb-DF test set (5,497 rows).

| Split | ACC | AUC | EER |
|-------|-----|-----|-----|
| Overall                 | 0.5005 | 0.6664 | 0.3943 |
| FF++ (in-domain)        | 0.4423 | 0.7085 | 0.3516 |
| Celeb-DF (cross-domain) | 0.5356 | 0.6014 | 0.4401 |

> The gap between Val Acc (92%) and Test ACC (~50%) reflects a preprocessing mismatch: training frames were cropped with our extractor, the test set uses a different Kaggle face-crop pipeline. Per-source accuracy shows the model predicts fake almost everywhere (fake recall 94–98%, real recall 11–31%), a well-known cross-pipeline brittleness of deepfake detectors. AUC 0.71 on FF++ indicates the model learned useful signal but depends on upstream alignment. Retraining on the test-matched dataset (`ff-celebdf-frames` with its own train/val CSVs) is planned as a follow-up.

## Project Structure

```
face-defense/
├── face_defense/
│   ├── data/                   # Dataset loaders
│   │   ├── celeba_spoof_dataset.py
│   │   └── ff_dataset.py       # FF++ binary real/fake
│   ├── models/
│   │   └── anti_spoof/
│   │       └── cdcn_model.py   # CDCN network
│   └── evaluation/             # Metrics, visualization
├── scripts/
│   ├── demo_gui.py             # PyQt5 access control demo
│   ├── demo_access.py          # OpenCV access control demo
│   ├── demo_webcam.py          # Webcam liveness demo
│   ├── train_cdcn.py           # CDCN training
│   ├── train_deepfake.py       # Deepfake training
│   ├── benchmark_cdcn.py       # CDCN benchmark
│   ├── benchmark_deepfake.py   # Deepfake benchmark
│   ├── finetune_cdcn_nuaa.py   # NUAA fine-tuning
│   └── extract_frames.py       # FF++ frame extraction
└── notebooks/                  # Benchmark evaluation
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

The right panel includes an **IR MODE** toggle:
- With `--ir_camera`: starts **ON** (blink AND IR material must both pass)
- Without: starts **OFF** (blink-only fallback); toggling to ON shows an `IR MODE: NO CAMERA` warning state for demo purposes

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

# Deepfake detection (XceptionNet, binary)
python scripts/train_deepfake.py --data_root data/ff-c23-frames --model legacy_xception --epochs 30
```

### Benchmark
```bash
# Anti-spoofing
python scripts/benchmark_cdcn.py

# Deepfake (FF++ + Celeb-DF test set)
python scripts/benchmark_deepfake.py \
  --data_root data/ff-celebdf-frames \
  --csv data/ff-celebdf-frames/test_labels.csv \
  --checkpoint checkpoints/legacy_xception_best.pth
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
| [benchmark_deepfake.py](scripts/benchmark_deepfake.py) | Deepfake benchmark on FF++/Celeb-DF |
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
