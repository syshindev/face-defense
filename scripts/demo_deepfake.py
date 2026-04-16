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
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Video: rolling-average window (in analyzed samples) for per-face score smoothing")
    parser.add_argument("--iou_match", type=float, default=0.3,
                        help="IoU threshold for matching the same face across sampled frames")
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


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_track(tracks, bbox, iou_thresh):
    best_i, best_iou = -1, 0.0
    for i, t in enumerate(tracks):
        s = iou(bbox, t["bbox"])
        if s > best_iou:
            best_iou, best_i = s, i
    return best_i if best_iou >= iou_thresh else -1


def update_tracks_from_frame(frame, tracks, face_app, model, image_size,
                             margin, iou_thresh, smooth_window, device):
    faces = face_app.get(frame)
    new_tracks = []
    used = set()
    scores = []
    for face in faces or []:
        crop, bbox = crop_face(frame, face.bbox, margin)
        if crop.size == 0:
            continue
        score = predict(model, crop, image_size, device)
        scores.append(score)
        idx = match_track(tracks, bbox, iou_thresh)
        if idx >= 0 and idx not in used:
            history = tracks[idx]["history"] + [score]
            used.add(idx)
        else:
            history = [score]
        history = history[-smooth_window:]
        new_tracks.append({"bbox": bbox, "history": history})
    return new_tracks, scores


def draw_tracks(frame, tracks, threshold):
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        smoothed = float(np.mean(t["history"]))
        color = (0, 0, 255) if smoothed >= threshold else (0, 255, 0)
        label = f"{verdict(smoothed, threshold)} {smoothed:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


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

    raw_scores = []
    analyzed = 0
    frame_idx = 0
    tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_step == 0:
            tracks, scores = update_tracks_from_frame(
                frame, tracks, face_app, model, args.image_size, args.margin,
                args.iou_match, args.smooth_window, device,
            )
            if scores:
                raw_scores.extend(scores)
                analyzed += 1

        draw_tracks(frame, tracks, args.threshold)

        if writer:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    if not raw_scores:
        print("No face detected in any sampled frame.")
        return

    arr = np.array(raw_scores)
    n_fake = int((arr >= args.threshold).sum())
    mean_p = float(arr.mean())
    summary_verdict = "FAKE" if mean_p >= args.threshold else "REAL"
    print(f"\nAnalyzed sampled-frame face predictions: {len(arr)} (frames={analyzed})")
    print(f"P(fake) raw: mean={mean_p:.4f} median={np.median(arr):.4f} "
          f"min={arr.min():.4f} max={arr.max():.4f}")
    print(f"Raw predictions flagged fake (>= {args.threshold}): {n_fake}/{len(arr)} "
          f"({100*n_fake/len(arr):.1f}%)")
    print(f"Overall verdict (raw mean): {summary_verdict}")


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
