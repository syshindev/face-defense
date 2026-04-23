# Deepfake Detection — Track 2

XceptionNet binary classifier for image-level deepfake detection.

## Architecture

```
Image / video file ──> Face crop ──> XceptionNet (binary) ──> P(fake)
                                                          │
                                      Report per-frame / per-source metrics
```

## Model Versions

| Version | Training data | Test ACC | Test AUC | Notes |
|---------|---------------|----------|----------|-------|
| v1 | FF++ c23 (custom extraction) | 50.05% | 0.6664 | Preprocessing mismatch |
| v2 | FF++ + Celeb-DF (34k) | 95.63% | 0.9929 | Matched preprocessing |
| v3 | v2 + StyleGAN3 + SDXL + FFHQ (64k) | 97.60% | 0.9972 | OOD expansion |
| **v3.4** | **v3 + WildDeepfake + SD1.5 + CelebA-HQ (108k)** | **97.70%** | **0.9967** | **Maximum data, 13 sources** |

### Per-category results (v2 → v3 → v3.4)

| Category | Source | v2 ACC | v3 ACC | v3.4 ACC |
|----------|--------|--------|--------|----------|
| Face-swap | FF++ + Celeb-DF | 0.9563 | 0.9638 | **0.9609** |
| Whole-image GAN | StyleGAN3 | 0.0594 | 0.9990 | **0.9951** |
| Diffusion (SDXL) | SDXL | 0.0099 | 1.0000 | **1.0000** |
| Diffusion (SD 1.5) | SD 1.5 | — | — | **1.0000** |
| Real (FFHQ) | FFHQ | 0.8999 | 0.9960 | **0.9979** |
| Real (CelebA-HQ) | CelebA-HQ | — | — | **0.9990** |
| WildDeepfake | Internet video crops | — | — | **0.9980** |
| FF++ video crops | InsightFace-cropped | — | — | **0.9607** |

## Usage

```bash
# Gradio web UI
python deepfake/scripts/demo_deepfake_gui.py

# CLI — image
python deepfake/scripts/demo_deepfake.py --image path/to/image.jpg

# CLI — video
python deepfake/scripts/demo_deepfake.py --video path/to/video.mp4
```

## Training

```bash
# v1 — folder mode
python deepfake/scripts/train_deepfake.py --data_root data/ff-c23-frames --model legacy_xception --epochs 30

# v2 — CSV mode
python deepfake/scripts/train_deepfake.py \
  --data_root data/ff-celebdf-frames \
  --train_csv data/ff-celebdf-frames/train_labels.csv \
  --val_csv data/ff-celebdf-frames/val_labels.csv \
  --model legacy_xception --epochs 30 --ckpt_suffix _v2
```

## Benchmark

```bash
python deepfake/scripts/benchmark_deepfake.py \
  --data_root data/ff-celebdf-frames \
  --csv data/ff-celebdf-frames/test_labels.csv \
  --checkpoint checkpoints/legacy_xception_v2_best.pth \
  --plot_dir plots --plot_tag _v2
```

See [benchmark_eval.ipynb](../notebooks/benchmark_eval.ipynb) for full analysis.

## Limitation

YouTube compressed video detection remains challenging — h264 re-encoding destroys the subtle artifacts the model relies on. Positioned as an **image-level media authenticator**.

## Scripts

| Script | Description |
|--------|-------------|
| [demo_deepfake.py](scripts/demo_deepfake.py) | CLI deepfake detector (image/video) |
| [demo_deepfake_gui.py](scripts/demo_deepfake_gui.py) | Gradio web UI |
| [train_deepfake.py](scripts/train_deepfake.py) | Training (CSV + augmentation) |
| [benchmark_deepfake.py](scripts/benchmark_deepfake.py) | Benchmark (per-category) |
| [extract_video_frames.py](scripts/extract_video_frames.py) | FF++ video face crop |
| [generate_stylegan_faces.py](scripts/generate_stylegan_faces.py) | StyleGAN3 face generation |
| [generate_diffusion_faces.py](scripts/generate_diffusion_faces.py) | SDXL/SD1.5 face generation |
| [merge_train_csv.py](scripts/merge_train_csv.py) | Multi-source CSV merger |
