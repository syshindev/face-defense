import argparse
import os
import time

import cv2
from insightface.app import FaceAnalysis


REAL_DIRS = {"original"}
FAKE_DIRS = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures",
             "FaceShifter", "DeepFakeDetection"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract face crops from FF++ videos using InsightFace "
                    "(same pipeline as inference, solving the preprocessing mismatch)"
    )
    parser.add_argument("--video_root", type=str,
                        default="data/ff-c23/FaceForensics++_C23")
    parser.add_argument("--out_dir", type=str, default="data/extra/ff_video_crops")
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--frame_step", type=int, default=30,
                        help="Extract one frame every N frames")
    parser.add_argument("--max_per_video", type=int, default=20,
                        help="Max frames to extract per video")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Max videos per category (0 = all)")
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


def process_video(video_path, out_dir, face_app, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    saved = 0
    frame_idx = 0

    while saved < args.max_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        faces = face_app.get(frame)
        if faces:
            face = max(faces, key=lambda f: f.det_score)
            cropped = crop_face(frame, face.bbox, args.margin, args.image_size)
            if cropped is not None:
                fname = f"{video_name}_f{frame_idx:04d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), cropped)
                saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    args = parse_args()

    face_app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_thresh=args.det_thresh, det_size=(640, 640))

    real_out = os.path.join(args.out_dir, "real")
    fake_out = os.path.join(args.out_dir, "fake")
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    total_saved = 0
    start = time.time()

    for category in sorted(os.listdir(args.video_root)):
        cat_path = os.path.join(args.video_root, category)
        if not os.path.isdir(cat_path) or category == "csv":
            continue

        if category in REAL_DIRS:
            out = real_out
            label = "real"
        elif category in FAKE_DIRS:
            out = fake_out
            label = "fake"
        else:
            continue

        videos = sorted([f for f in os.listdir(cat_path) if f.endswith(".mp4")])
        if args.max_videos > 0:
            videos = videos[:args.max_videos]

        print(f"[{category}] ({label}) {len(videos)} videos", flush=True)
        cat_saved = 0

        for i, vname in enumerate(videos):
            vpath = os.path.join(cat_path, vname)
            n = process_video(vpath, out, face_app, args)
            cat_saved += n
            total_saved += n

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"  [{i+1}/{len(videos)}] cat_saved={cat_saved} "
                      f"total={total_saved} elapsed={elapsed:.0f}s", flush=True)

        print(f"  -> {category}: {cat_saved} frames saved", flush=True)

    elapsed = time.time() - start
    real_count = len([f for f in os.listdir(real_out) if f.endswith(".jpg")])
    fake_count = len([f for f in os.listdir(fake_out) if f.endswith(".jpg")])
    print(f"\nDone. real={real_count} fake={fake_count} total={total_saved} "
          f"elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
