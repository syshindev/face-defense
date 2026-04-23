import argparse
import os
import random
import subprocess
import tempfile
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-generate h264-compressed versions of training images "
                    "to simulate YouTube compression pipeline"
    )
    parser.add_argument("--in_dir", type=str, required=True,
                        help="Directory of original images")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for h264-augmented images")
    parser.add_argument("--mode", type=str, default="mix",
                        choices=["single", "double", "resolution", "mix"],
                        help="single: one h264 pass, double: two passes, "
                             "resolution: resize+h264, mix: random choice per image")
    parser.add_argument("--crf_range", type=str, default="20,38",
                        help="CRF range (lower=better quality)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _make_even(img_path):
    """Resize to even dimensions for h264, return temp path and original size."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    new_w = w if w % 2 == 0 else w + 1
    new_h = h if h % 2 == 0 else h + 1
    if new_w != w or new_h != h:
        img = cv2.resize(img, (new_w, new_h))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img)
    return tmp.name, (w, h)


def _restore_size(out_path, orig_size):
    """Resize back to original dimensions."""
    if orig_size is None:
        return
    img = cv2.imread(out_path)
    if img is not None:
        img = cv2.resize(img, orig_size)
        cv2.imwrite(out_path, img)


def h264_single(img_path, out_path, crf):
    """Single h264 encode/decode."""
    even_path, orig_size = _make_even(img_path)
    if even_path is None:
        return False
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_mp4 = tmp.name
    try:
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", even_path,
            "-c:v", "libx264", "-crf", str(crf),
            "-frames:v", "1", "-pix_fmt", "yuv420p",
            tmp_mp4
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp_mp4,
            "-frames:v", "1",
            out_path
        ], check=True)
        _restore_size(out_path, orig_size)
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        for f in [tmp_mp4, even_path]:
            if os.path.exists(f):
                os.unlink(f)


def h264_double(img_path, out_path, crf1, crf2):
    """Double h264 encode/decode (simulates upload + YouTube re-encode)."""
    even_path, orig_size = _make_even(img_path)
    if even_path is None:
        return False
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t1, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t2, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as mid:
        tmp1, tmp2, mid_jpg = t1.name, t2.name, mid.name
    try:
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", even_path,
            "-c:v", "libx264", "-crf", str(crf1),
            "-frames:v", "1", "-pix_fmt", "yuv420p",
            tmp1
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp1, "-frames:v", "1", mid_jpg
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", mid_jpg,
            "-c:v", "libx264", "-crf", str(crf2),
            "-frames:v", "1", "-pix_fmt", "yuv420p",
            tmp2
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp2, "-frames:v", "1", out_path
        ], check=True)
        _restore_size(out_path, orig_size)
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        for f in [tmp1, tmp2, mid_jpg, even_path]:
            if os.path.exists(f):
                os.unlink(f)


def h264_resolution(img_path, out_path, crf, scale):
    """Resize + h264 + resize back (simulates YouTube resolution change)."""
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    # Make dimensions even (h264 requirement)
    new_w = new_w if new_w % 2 == 0 else new_w + 1
    new_h = new_h if new_h % 2 == 0 else new_h + 1

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t1, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t2:
        scaled_jpg, tmp_mp4 = t1.name, t2.name
    try:
        resized = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(scaled_jpg, resized)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", scaled_jpg,
            "-c:v", "libx264", "-crf", str(crf),
            "-frames:v", "1", "-pix_fmt", "yuv420p",
            tmp_mp4
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp_mp4, "-frames:v", "1", out_path
        ], check=True)
        # Resize back to original
        decoded = cv2.imread(out_path)
        if decoded is not None:
            back = cv2.resize(decoded, (w, h))
            cv2.imwrite(out_path, back)
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        for f in [scaled_jpg, tmp_mp4]:
            if os.path.exists(f):
                os.unlink(f)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    crf_lo, crf_hi = [int(x) for x in args.crf_range.split(",")]

    files = sorted([f for f in os.listdir(args.in_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"Processing {len(files)} images, mode={args.mode}")

    saved = 0
    failed = 0
    start = time.time()

    for i, fname in enumerate(files):
        in_path = os.path.join(args.in_dir, fname)
        out_path = os.path.join(args.out_dir, fname)
        if os.path.exists(out_path):
            saved += 1
            continue

        crf = rng.randint(crf_lo, crf_hi)

        if args.mode == "single":
            ok = h264_single(in_path, out_path, crf)
        elif args.mode == "double":
            crf2 = rng.randint(crf_lo, crf_hi)
            ok = h264_double(in_path, out_path, crf, crf2)
        elif args.mode == "resolution":
            scale = rng.uniform(0.5, 0.9)
            ok = h264_resolution(in_path, out_path, crf, scale)
        else:  # mix
            choice = rng.choice(["single", "double", "resolution"])
            if choice == "single":
                ok = h264_single(in_path, out_path, crf)
            elif choice == "double":
                crf2 = rng.randint(crf_lo, crf_hi)
                ok = h264_double(in_path, out_path, crf, crf2)
            else:
                scale = rng.uniform(0.5, 0.9)
                ok = h264_resolution(in_path, out_path, crf, scale)

        if ok:
            saved += 1
        else:
            failed += 1

        if (saved + failed) % 500 == 0:
            elapsed = time.time() - start
            rate = (saved + failed) / max(elapsed, 1)
            print(f"  [{saved + failed}/{len(files)}] saved={saved} failed={failed} "
                  f"rate={rate:.1f}/s", flush=True)

    total = time.time() - start
    print(f"\nDone. saved={saved} failed={failed} total={total:.0f}s")


if __name__ == "__main__":
    main()
