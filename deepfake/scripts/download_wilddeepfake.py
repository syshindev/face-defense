import argparse
import os
import random
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download WildDeepfake from HuggingFace and sample N images"
    )
    parser.add_argument("--out_dir", type=str, default="data/extra/wilddeepfake")
    parser.add_argument("--count", type=int, default=30000,
                        help="Total images to sample (real + fake combined)")
    parser.add_argument("--dataset_id", type=str, default="xingjunm/WildDeepfake")
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    real_dir = os.path.join(args.out_dir, "real")
    fake_dir = os.path.join(args.out_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    from datasets import load_dataset

    print(f"Loading dataset '{args.dataset_id}' (streaming)...")
    ds_train = load_dataset(args.dataset_id, split="train", streaming=True)

    rng = random.Random(args.seed)
    saved_real = 0
    saved_fake = 0
    target_per_class = args.count // 2
    skipped = 0
    start = time.time()

    for i, item in enumerate(ds_train):
        if saved_real >= target_per_class and saved_fake >= target_per_class:
            break

        try:
            key = item["__key__"]
            img = item["png"]
        except (KeyError, TypeError):
            skipped += 1
            continue

        is_fake = "/fake/" in key
        is_real = "/real/" in key
        if not is_fake and not is_real:
            skipped += 1
            continue

        if is_real and saved_real >= target_per_class:
            continue
        if is_fake and saved_fake >= target_per_class:
            continue

        if img.mode != "RGB":
            img = img.convert("RGB")

        img_resized = img.resize((args.image_size, args.image_size))

        if is_real:
            path = os.path.join(real_dir, f"wild_real_{saved_real:06d}.jpg")
            saved_real += 1
        else:
            path = os.path.join(fake_dir, f"wild_fake_{saved_fake:06d}.jpg")
            saved_fake += 1

        img_resized.save(path, quality=92)

        total_saved = saved_real + saved_fake
        if total_saved % 500 == 0:
            elapsed = time.time() - start
            rate = total_saved / max(elapsed, 1)
            print(f"  real={saved_real} fake={saved_fake} skipped={skipped} "
                  f"rate={rate:.1f}/s", flush=True)

    total = time.time() - start
    print(f"\nDone. real={saved_real} fake={saved_fake} skipped={skipped} total={total:.0f}s")


if __name__ == "__main__":
    main()
