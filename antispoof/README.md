# Anti-Spoofing — Track 1

Real-time face authentication kiosk with multi-layer spoof detection.

## Detection Layers

| Layer | Method | Detects | Speed |
|-------|--------|---------|-------|
| Texture | LBP entropy analysis | Screen replay (phone/monitor) | Instant |
| Skin | YCbCr skin color ratio | Printed photos (B&W, color) | Instant |
| Saturation | HSV saturation analysis | Distance-agnostic display vs print classification | Instant |
| IR | RGB/IR brightness ratio | Print + screen attacks | Instant |
| Depth | RealSense D435 depth std (flatness) | Planar 2D spoofs — D435 only | Instant |
| Blink | EAR eye aspect ratio (dynamic threshold by face size) | All static images | 3-5 sec |
| Smoothing | 5/10 majority vote | Reduces false positives | — |

## Architecture

```
Camera (cv2 webcam OR Intel RealSense D435)
  │   produces RGB + IR (optional) + Depth (D435 only)
  v
Face detection — InsightFace, multi-face
  │   D435 → closest face within --max_depth
  │   cv2  → largest face (>= --min_face px)
  v
Texture check (HSV saturation → LBP → YCbCr skin)
  │   sat > 120     → DISPLAY
  │   sat < 55      → PRINT
  │   skin < 0.20   → PRINT
  │   lbp < 5.50    → DISPLAY or PRINT (sat tiebreak at 80)
  v
IR check
  │   D435 → depth std < 8.0  AND  ratio > 0.80  → DISPLAY/PRINT
  │   cv2  → ratio < 1.0 → PRINT,  ratio > 2.0 → DISPLAY
  v
Smoothing — 5/10 majority vote over recent frames
  v
Blink liveness — multi-face nose match, dynamic EAR 0.31~0.35 by face size
  v
Face embedding match — cosine vs registered users (threshold >= 0.4)
  v
AUTHORIZED  |  UNAUTHORIZED  |  DENIED
```

## Usage

```bash
# Standard webcam (texture + blink only)
python antispoof/scripts/demo_gui.py --camera 0

# Webcam + separate USB IR camera (legacy 2-camera setup)
python antispoof/scripts/demo_gui.py --camera 0 --ir_camera 1

# Intel RealSense D435 (RGB + IR + Depth combined, 1 m max range)
python antispoof/scripts/demo_gui.py --d435 --max_depth 1.0

# Debug mode (print per-frame detection values)
DEBUG=1 python antispoof/scripts/demo_gui.py --d435 --max_depth 1.0

# OpenCV demo
python antispoof/scripts/demo_access.py --camera 0

# Webcam liveness demo
python antispoof/scripts/demo_webcam.py --camera 0
```

CLI flags:
- `--camera N` — RGB webcam index (default 0)
- `--ir_camera N` — separate USB IR camera index, for legacy 2-camera setups (default -1, off)
- `--d435` — use Intel RealSense D435 (RGB + IR + Depth integrated, requires `pyrealsense2`)
- `--max_depth M` — D435 only; ignore faces farther than M meters (0 = unlimited)
- `--min_face PX` — non-D435 only; ignore faces narrower than PX pixels (default 120; auto-disabled with --d435)

Toggle buttons on the right panel:
- **BLUR** — bbox-based background blur, keeps the target face crisp while obscuring bystanders
- **BLINK** — EAR-based blink liveness detection
- **TEXTURE** — HSV saturation + LBP entropy + YCbCr skin color analysis

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
