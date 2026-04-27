# Emotion Recognition — Track 3

Real-time facial emotion recognition using DeepFace + InsightFace.

## Features

- 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
- InsightFace face detection + DeepFace emotion classification
- PyQt5 GUI with real-time emotion bar chart
- Background thread analysis for smooth camera feed
- Rolling-average smoothing for stable results

## Usage

```bash
python emotion/scripts/demo_emotion_gui.py --camera 0

# Adjust smoothing window (default: 5)
python emotion/scripts/demo_emotion_gui.py --camera 0 --smooth 10
```

## Scripts

| Script | Description |
|--------|-------------|
| [demo_emotion_gui.py](scripts/demo_emotion_gui.py) | PyQt5 emotion recognition demo |

## Dependencies

```bash
pip install deepface tf-keras
```

## Note

DeepFace emotion model is based on FER2013 (~65% benchmark accuracy), but performs well in practice for common emotions (happy, sad, neutral, surprise). Some subtle expressions (disgust, contempt) may be less reliable.
