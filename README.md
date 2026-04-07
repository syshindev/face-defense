# Face Defense

Multi-layer face anti-spoofing and deepfake detection research framework using open-source models.

## Overview

Defends against 5 types of face spoofing attacks:
- Printed photo attacks
- Screen replay attacks
- 3D mask attacks
- Deepfake / synthetic faces
- Replay attacks

## Architecture

```
              ┌────────────────────────┐
              │  Input Frame           │
              │  Image / Video         │
              └────────────┬───────────┘
                           │
                           v
              ┌────────────────────────┐
              │  Face Detection        │
              │  InsightFace RetinaFace│
              └────────────┬───────────┘
                           │
                           v
              ┌────────────────────────┐
              │  Stage 1: Anti-Spoof   │───> SPOOF (early exit)
              │  Silent-FAS / CDCN     │
              └────────────┬───────────┘
                           │ pass
                           v
              ┌────────────────────────┐
              │  Stage 2: Liveness     │
              │  Blink / LBP / Depth   │
              └────────────┬───────────┘
                           │
                           v
              ┌────────────────────────┐
              │  Stage 3: Deepfake     │
              │  XceptionNet/EffNet    │
              └────────────┬───────────┘
                           │
                           v
              ┌────────────────────────┐
              │  Stage 4: Fusion       │
              │  Weighted Average      │
              └────────────┬───────────┘
                           │
                           v
              ┌────────────────────────┐
              │  REAL / SPOOF          │
              └────────────────────────┘
```

### Stage 1: Anti-Spoofing
| Model | Description |
|-------|-------------|
| Silent-FAS | Lightweight anti-spoofing (~20ms), multi-scale inference |
| CDCN | Depth map prediction, CVPR 2020 winner |
| deepface | Built-in anti-spoofing baseline |

### Stage 2: Liveness Detection
| Model | Description |
|-------|-------------|
| Blink Detector | Eye Aspect Ratio via MediaPipe |
| Texture Analyzer | LBP + Laplacian variance |
| Depth Estimator | MediaPipe face mesh Z-coordinates |

### Stage 3: Deepfake Detection
| Model | Description |
|-------|-------------|
| XceptionNet | Standard deepfake detector |
| EfficientNet-B4 | DFDC challenge winning architecture |

### Stage 4: Score Fusion
- Weighted average across all stages
- Cascade early-exit for real-time performance

## Project Structure

```
face-defense/
├── configs/              # YAML configuration files
├── face_defense/
│   ├── core/             # Pipeline, registry, base classes
│   ├── data/             # Dataset loaders, preprocessing
│   ├── models/
│   │   ├── anti_spoof/   # Silent-FAS, CDCN, deepface
│   │   ├── deepfake/     # XceptionNet, EfficientNet
│   │   ├── liveness/     # Blink, texture, depth
│   │   └── ensemble/     # Score fusion, cascade policy
│   ├── training/         # Training loop
│   ├── evaluation/       # Metrics, evaluator, visualization
│   └── utils/            # Device management
├── scripts/              # Entry-point scripts
├── notebooks/            # Research notebooks
└── third_party/          # External repos (Silent-FAS, etc.)
```

## Getting Started

### Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA support (recommended)
- Conda (Miniconda or Anaconda)

### Installation

```bash
# Create conda environment
conda create -n face-defense python=3.10 -y
conda activate face-defense

# Install dlib via conda (pip build fails on Windows due to cp949 encoding issue)
conda install -c conda-forge dlib -y

# Install remaining dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Clone Silent-Face-Anti-Spoofing for anti-spoof inference
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git third_party/Silent-Face-Anti-Spoofing
```

## Usage

### Single Image Analysis
```bash
python scripts/demo_image.py --image path/to/image.jpg
```

### Webcam Demo
```bash
python scripts/demo_webcam.py --config configs/pipeline/lightweight.yaml
```

### Training
```bash
python scripts/train.py --config configs/anti_spoof/cdcn.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --config configs/pipeline/full_defense.yaml
```
