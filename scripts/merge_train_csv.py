import argparse
import os
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop faces from extra datasets using InsightFace (same pipeline as ff-celebdf)"
    )
    parser.add_argument("--in_dir", type=str, required=True, help="Directory of raw images")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for cropped faces")
    parser.add_argument("--image_size", type=int, default=299, help="Output crop size")
    parser.add_argument("--margin", type=float, default=0.3, help="Crop margin around detected face")
    parser.add_argument("--det_thresh", type=float, default=0.3)
    return parser.parse_args()


def crop_face(img, bbox, margin, out_size):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    H, W = img.shape[:2]
    x1 = max(0, int(x1) - mx)
    y1 = max(0, int(y1) - my)
    x2 = min(W, int(x2) + mx)
    y2 = min(H, int(y2) + my)
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (out_size, out_size))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l",
                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_thresh=args.det_thresh, det_size=(640, 640))

    files = sorted([
        f for f in os.listdir(args.in_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Processing {len(files)} images from {args.in_dir}")

    saved = 0
    no_face = 0
    for i, fname in enumerate(files):
        out_path = os.path.join(args.out_dir, fname)
        if os.path.exists(out_path):
            saved += 1
            continue
        img = cv2.imread(os.path.join(args.in_dir, fname))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            no_face += 1
            continue
        face = max(faces, key=lambda f: f.det_score)
        cropped = crop_face(img, face.bbox, args.margin, args.image_size)
        if cropped is None:
            no_face += 1
            continue
        cv2.imwrite(out_path, cropped)
        saved += 1
        if (saved + no_face) % 500 == 0:
            print(f"  [{i+1}/{len(files)}] saved={saved} no_face={no_face}", flush=True)

    print(f"\nDone. saved={saved} no_face={no_face} total={len(files)}")


if __name__ == "__main__":
    main()
