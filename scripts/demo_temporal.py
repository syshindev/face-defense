import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_temporal import TemporalDeepfakeDetector


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal deepfake detection on video")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/temporal_lstm_best.pth")
    parser.add_argument("--backbone", type=str, default="legacy_xception")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def crop_face(frame, bbox, margin=0.2):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    H, W = frame.shape[:2]
    x1 = max(0, int(x1) - mx)
    y1 = max(0, int(y1) - my)
    x2 = min(W, int(x2) + mx)
    y2 = min(H, int(y2) + my)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return face


def preprocess(face_bgr, image_size):
    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1).float()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Face detector
    face_app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_thresh=args.det_thresh, det_size=(640, 640))

    # Temporal model
    model = TemporalDeepfakeDetector(
        backbone_name=args.backbone,
        backbone_ckpt=None,
        freeze_backbone=True,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded: {args.checkpoint}")

    # Extract face crops from video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {fps:.1f} fps")

    face_crops = []
    frame_indices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.frame_step == 0:
            faces = face_app.get(frame)
            if faces:
                face = max(faces, key=lambda f: f.det_score)
                crop = crop_face(frame, face.bbox)
                if crop is not None:
                    tensor = preprocess(crop, args.image_size)
                    face_crops.append(tensor)
                    frame_indices.append(frame_idx)
        frame_idx += 1
    cap.release()

    print(f"Extracted {len(face_crops)} face crops")

    if len(face_crops) < args.seq_len:
        print("Not enough face crops for a sequence. Exiting.")
        return

    # Build sequences and predict
    scores = []
    seq_times = []

    for i in range(0, len(face_crops) - args.seq_len + 1, args.seq_len):
        seq = torch.stack(face_crops[i:i + args.seq_len], dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(seq)
            prob_fake = F.softmax(logits, dim=1)[0, 1].item()
        scores.append(prob_fake)
        start_time = frame_indices[i] / fps
        end_time = frame_indices[i + args.seq_len - 1] / fps
        seq_times.append((start_time, end_time))

        verdict = "FAKE" if prob_fake >= args.threshold else "REAL"
        print(f"  Seq {len(scores):3d} ({start_time:6.1f}s ~ {end_time:6.1f}s): "
              f"P(fake)={prob_fake:.4f}  {verdict}")

    if not scores:
        print("No sequences analyzed.")
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

    print(f"\n=== Summary ===")
    print(f"Sequences: {len(arr)}")
    print(f"P(fake) mean={mean_p:.4f} median={np.median(arr):.4f} "
          f"min={arr.min():.4f} max={arr.max():.4f}")
    print(f"Flagged fake: {n_fake}/{len(arr)} ({100*fake_ratio:.1f}%)")
    print(f"Overall verdict: {verdict}")


if __name__ == "__main__":
    main()
