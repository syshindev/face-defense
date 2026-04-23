import argparse
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a grid image showing sample faces from each source"
    )
    parser.add_argument("--stylegan_dir", type=str, default="data/extra/stylegan_cropped")
    parser.add_argument("--diffusion_dir", type=str, default="data/extra/diffusion_cropped")
    parser.add_argument("--ffhq_dir", type=str, default="data/extra/ffhq_cropped")
    parser.add_argument("--per_row", type=int, default=5)
    parser.add_argument("--tile_size", type=int, default=160)
    parser.add_argument("--out_path", type=str, default="docs/screenshots/v3_data_samples.png")
    return parser.parse_args()


def pick_samples(directory, n):
    files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:n]
    return [os.path.join(directory, f) for f in files]


def build_row(paths, tile_size, bg_color):
    tiles = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            img = np.full((tile_size, tile_size, 3), 40, dtype=np.uint8)
        img = cv2.resize(img, (tile_size, tile_size))
        tiles.append(img)
    if not tiles:
        return None
    return np.hstack(tiles)


def add_label(canvas, text, y, bg, fg):
    h = 30
    strip = np.full((h, canvas.shape[1], 3), bg, dtype=np.uint8)
    cv2.putText(strip, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv2.LINE_AA)
    return strip


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    sources = [
        ("[ STYLEGAN - WHOLE-IMAGE GAN ]", args.stylegan_dir, (0, 170, 255)),
        ("[ DIFFUSION - SDXL TEXT-TO-IMAGE ]", args.diffusion_dir, (255, 212, 0)),
        ("[ FFHQ - REAL FACES (FLICKR) ]", args.ffhq_dir, (65, 255, 0)),
    ]

    rows = []
    for label, directory, color in sources:
        if not os.path.isdir(directory):
            print(f"SKIP {directory} (not found)")
            continue
        paths = pick_samples(directory, args.per_row)
        if not paths:
            continue
        row = build_row(paths, args.tile_size, (12, 12, 12))
        label_strip = add_label(row, label, 0, bg=(26, 26, 26), fg=color)
        rows.append(np.vstack([label_strip, row]))

    if not rows:
        print("No sources found, nothing to grid.")
        return

    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 12, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)
        padded.append(np.full((8, max_w, 3), 12, dtype=np.uint8))

    grid = np.vstack(padded)
    cv2.imwrite(args.out_path, grid)
    print(f"Saved grid: {args.out_path} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
