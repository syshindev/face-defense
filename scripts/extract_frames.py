import os
import argparse
import cv2
from tqdm import tqdm


# Class mapping for FF++ dataset
CLASSES = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
    "FaceShifter": 5,
    "DeepFakeDetection": 6,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from FF++ videos")
    parser.add_argument("--data_root", type=str, required=True, help="Path to FaceForensics++_C23/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for extracted frames")
    parser.add_argument("--interval", type=int, default=30, help="Extract 1 frame every N frames")
    parser.add_argument("--image_size", type=int, default=299, help="Output image size")
    return parser.parse_args()


def extract_video_frames(video_path, output_dir, interval, image_size):
    # Extract frames from a single video at given interval
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame = cv2.resize(frame, (image_size, image_size))
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            fname = f"{video_name}_{frame_count:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


def main():
    args = parse_args()

    total_extracted = 0

    for class_name, label in CLASSES.items():
        class_dir = os.path.join(args.data_root, class_name)
        if not os.path.exists(class_dir):
            print(f"Skipping {class_name} (not found)")
            continue

        # Create output folder
        output_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(output_dir, exist_ok=True)

        # Get all mp4 files
        videos = sorted([f for f in os.listdir(class_dir) if f.endswith(".mp4")])
        print(f"\n{class_name}: {len(videos)} videos")

        class_count = 0
        for video_file in tqdm(videos, desc=class_name):
            video_path = os.path.join(class_dir, video_file)
            count = extract_video_frames(video_path, output_dir, args.interval, args.image_size)
            class_count += count

        print(f"  Extracted: {class_count} frames")
        total_extracted += class_count

    print(f"\nTotal extracted: {total_extracted} frames")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
