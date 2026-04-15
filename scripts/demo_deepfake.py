import os
import sys
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from insightface.app import FaceAnalysis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Run deepfake detector on an image or video")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Path to a single image file")
    src.add_argument("--video", type=str, help="Path to a video file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/legacy_xception_v2_best.pth")
    parser.add_argument("--model", type=str, default="legacy_xception")
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--frame_step", type=int, default=30, help="Video: analyze every N frames")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.2, help="Face crop margin (fraction of bbox)")
    parser.add_argument("--det_thresh", type=float, default=0.3,
                        help="InsightFace detection threshold (default 0.3 — lower than stock 0.5 to survive occlusions)")
    parser.add_argument("--save_annotated", type=str, default=None,
                        help="Video mode: save annotated video to this path")
    return parser.parse_args()


def build_face_app(det_thresh):
    app = FaceAnalysis(name="buffalo_l",
                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(640, 640))
    return app


def load_model(model_name, checkpoint, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=2).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def crop_face(frame, bbox, margin):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    H, W = frame.shape[:2]
    x1 = max(0, int(x1) - mx)
    y1 = max(0, int(y1) - my)
    x2 = min(W, int(x2) + mx)
    y2 = min(H, int(y2) + my)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess(face_bgr, image_size):
    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def predict(model, face_bgr, image_size, device):
    tensor = preprocess(face_bgr, image_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob_fake = F.softmax(logits, dim=1)[0, 1].item()
    return prob_fake


def verdict(score, threshold):
    return "FAKE" if score >= threshold else "REAL"


def analyze_image(args, face_app, model, device):
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: cannot read image {args.image}")
        return

    faces = face_app.get(img)
    if not faces:
        print("No face detected. Running on the full image as fallback.")
        score = predict(model, img, args.image_size, device)
        print(f"P(fake)={score:.4f}  verdict={verdict(score, args.threshold)}")
        return

    face = max(faces, key=lambda f: f.det_score)
    crop, _ = crop_face(img, face.bbox, args.margin)
    score = predict(model, crop, args.image_size, device)
    print(f"P(fake)={score:.4f}  verdict={verdict(score, args.threshold)}  det_score={face.det_score:.3f}")


def analyze_video(args, face_app, model, device):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames @ {fps:.1f} fps, step={args.frame_step}")

    writer = None
    if args.save_annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_annotated, fourcc, fps, (w, h))

    scores = []
    analyzed = 0
    frame_idx = 0
    last_annotations = []  # list of (bbox, score) persisted between sampled frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_step == 0:
            faces = face_app.get(frame)
            last_annotations = []
            if faces:
                for face in faces:
                    crop, (x1, y1, x2, y2) = crop_face(frame, face.bbox, args.margin)
                    if crop.size == 0:
                        continue
                    score = predict(model, crop, args.image_size, device)
                    scores.append(score)
                    last_annotations.append(((x1, y1, x2, y2), score))
                analyzed += 1

        for (x1, y1, x2, y2), score in last_annotations:
            color = (0, 0, 255) if score >= args.threshold else (0, 255, 0)
            label = f"{verdict(score, args.threshold)} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if writer:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    if not scores:
        print("No face detected in any sampled frame.")
        return

    arr = np.array(scores)
    n_fake = int((arr >= args.threshold).sum())
    mean_p = float(arr.mean())
    summary_verdict = "FAKE" if mean_p >= args.threshold else "REAL"
    print(f"\nAnalyzed frames: {analyzed}")
    print(f"P(fake) mean={mean_p:.4f} median={np.median(arr):.4f} "
          f"min={arr.min():.4f} max={arr.max():.4f}")
    print(f"Frames flagged fake (>= {args.threshold}): {n_fake}/{analyzed} "
          f"({100*n_fake/analyzed:.1f}%)")
    print(f"Overall verdict: {summary_verdict}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}  |  Ckpt: {args.checkpoint}")

    face_app = build_face_app(args.det_thresh)
    model = load_model(args.model, args.checkpoint, device)

    if args.image:
        analyze_image(args, face_app, model, device)
    else:
        analyze_video(args, face_app, model, device)


if __name__ == "__main__":
    main()
