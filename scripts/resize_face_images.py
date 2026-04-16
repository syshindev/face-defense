import argparse
import os
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize face-only images (StyleGAN/FFHQ/Diffusion) to a fixed size. "
                    "Use this instead of crop_extra_faces.py when images are already face-centered."
    )
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--size", type=int, default=299)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(args.in_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Resizing {len(files)} images from {args.in_dir} -> {args.out_dir} ({args.size}x{args.size})")

    saved = 0
    failed = 0
    start = time.time()

    for i, fname in enumerate(files):
        out_path = os.path.join(args.out_dir, fname)
        if os.path.exists(out_path):
            saved += 1
            continue
        img = cv2.imread(os.path.join(args.in_dir, fname))
        if img is None:
            failed += 1
            continue
        resized = cv2.resize(img, (args.size, args.size))
        cv2.imwrite(out_path, resized)
        saved += 1

        if saved % 1000 == 0:
            elapsed = time.time() - start
            print(f"  [{saved}/{len(files)}] {elapsed:.0f}s", flush=True)

    total = time.time() - start
    print(f"\nDone. saved={saved} failed={failed} total={total:.0f}s")


if __name__ == "__main__":
    main()
