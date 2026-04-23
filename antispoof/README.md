# Anti-Spoofing — Track 1

Real-time face authentication kiosk with multi-layer spoof detection.

## Detection Layers

| Layer | Method | Detects | Speed |
|-------|--------|---------|-------|
| Texture | LBP entropy analysis | Screen replay (phone/monitor) | Instant |
| Skin | YCbCr skin color ratio | Printed photos (B&W, color) | Instant |
| IR | RGB/IR brightness ratio | Print + screen attacks | Instant |
| Blink | EAR eye aspect ratio | All static images | 3-5 sec |
| Smoothing | 5/10 majority vote | Reduces false positives | — |

## Architecture

```
Webcam frame ─┐                     ┌─ IR frame
              v                     v
       Face detection          IR ratio check
       (InsightFace)           ├─ ratio < 1.0  → print attack
              │                └─ ratio > 2.0  → screen attack
              v
        Texture check (LBP entropy + YCbCr skin ratio)
        ├─ lbp < 6.10     → screen attack
        └─ skin < 0.20    → print attack
              │
              v
        Smoothing (5/10 majority vote)
              │
              v
        Blink liveness (EAR, optional backup)
              │
              v
        Face embedding match (cosine vs registered users)
              │
              v
        AUTHORIZED  |  UNAUTHORIZED  |  DENIED
```

## Usage

```bash
# GUI demo
python antispoof/scripts/demo_gui.py --camera 0

# GUI demo with IR camera
python antispoof/scripts/demo_gui.py --camera 0 --ir_camera 1

# Debug mode (show detection values)
DEBUG=1 python antispoof/scripts/demo_gui.py --camera 0 --ir_camera 1

# OpenCV demo
python antispoof/scripts/demo_access.py --camera 0

# Webcam liveness demo
python antispoof/scripts/demo_webcam.py --camera 0
```

Toggle buttons on the right panel:
- **IR MODE** — IR camera material check, shows real-time ACTIVE/INACTIVE status
- **BLINK** — EAR-based blink liveness detection
- **TEXTURE** — LBP entropy + YCbCr skin color analysis

## Training

```bash
# CDCN anti-spoofing
python antispoof/scripts/train_cdcn.py --data_root data/CelebA_Spoof --epochs 50

# Fine-tune on NUAA
python antispoof/scripts/finetune_cdcn_nuaa.py --data_root data/NUAA --epochs 20
```

## Benchmark (CDCN)

| Dataset | AUC | ACER |
|---------|-----|------|
| CelebA-Spoof (same-domain, 67K) | 0.9985 | 0.0272 |
| LCC-FASD (cross-dataset, 7.5K) | 0.8174 | 0.2463 |

### Model Comparison (LCC-FASD cross-dataset)

| Metric | Silent-FAS (historical) | CDCN |
|--------|-------------------------|------|
| AUC | 0.7757 | **0.8174** |
| ACER | 0.3070 | **0.2463** |

## Scripts

| Script | Description |
|--------|-------------|
| [demo_gui.py](scripts/demo_gui.py) | PyQt5 access control demo |
| [demo_access.py](scripts/demo_access.py) | OpenCV access control demo |
| [demo_webcam.py](scripts/demo_webcam.py) | Webcam liveness demo |
| [train_cdcn.py](scripts/train_cdcn.py) | CDCN anti-spoofing training |
| [finetune_cdcn_nuaa.py](scripts/finetune_cdcn_nuaa.py) | CDCN fine-tuning on NUAA |
| [benchmark_cdcn.py](scripts/benchmark_cdcn.py) | CDCN benchmark evaluation |
