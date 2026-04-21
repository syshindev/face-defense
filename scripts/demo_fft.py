import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from insightface.app import FaceAnalysis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_defense.data.fft_dataset import compute_fft_spectrum


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="FFT frequency-domain deepfake detection on video")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fft_xception_best.pth")
    parser.add_argument("--model", type=str, default="legacy_xception")
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--frame_step", type=int, default=15)
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    face_app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_thresh=args.det_thresh, det_size=(640, 640))

    model = timm.create_model(args.model, pretrained=False, num_classes=2).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Device: {device} | FFT model loaded")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames @ {fps:.1f} fps, step={args.frame_step}")

    scores = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.frame_step == 0:
            faces = face_app.get(frame)
            if faces:
                face = max(faces, key=lambda f: f.det_score)
                x1, y1, x2, y2 = face.bbox.astype(int)
                w, h = x2 - x1, y2 - y1
                mx, my = int(w * 0.2), int(h * 0.2)
                H, W = frame.shape[:2]
                x1, y1 = max(0, x1 - mx), max(0, y1 - my)
                x2, y2 = min(W, x2 + mx), min(H, y2 + my)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    spectrum = compute_fft_spectrum(crop, args.image_size)
                    spectrum = (spectrum - IMAGENET_MEAN) / IMAGENET_STD
                    tensor = torch.from_numpy(spectrum).permute(2, 0, 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(tensor)
                        prob = F.softmax(logits, dim=1)[0, 1].item()
                    scores.append(prob)
        frame_idx += 1
    cap.release()

    if not scores:
        print("No faces detected.")
        return

    arr = np.array(scores)
    n_fake = int((arr >= args.threshold).sum())
    fake_ratio = n_fake / len(arr)
    mean_p = float(arr.mean())

    if fake_ratio >= 0.7:
        verdict = "FAKE"
    elif fake_ratio >= 0.3:
        verdict = "MIXED"
    else:
        verdict = "REAL"

    print(f"\n=== FFT Analysis ===")
    print(f"Predictions: {len(arr)}")
    print(f"P(fake) mean={mean_p:.4f} median={np.median(arr):.4f} "
          f"min={arr.min():.4f} max={arr.max():.4f}")
    print(f"Flagged fake: {n_fake}/{len(arr)} ({100*fake_ratio:.1f}%)")
    print(f"Overall verdict: {verdict}")


if __name__ == "__main__":
    main()
