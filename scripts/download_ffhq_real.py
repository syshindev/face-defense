import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Download FFHQ real face images from HuggingFace")
    parser.add_argument("--out_dir", type=str, default="data/extra/ffhq_real")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--dataset_id", type=str, default="merkol/ffhq-256",
                        help="HuggingFace dataset id (must have an 'image' column)")
    parser.add_argument("--image_key", type=str, default="image",
                        help="Column name for the PIL image in the dataset")
    parser.add_argument("--start_idx", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from datasets import load_dataset

    print(f"Loading dataset '{args.dataset_id}' (streaming)...")
    ds = load_dataset(args.dataset_id, split="train", streaming=True)

    saved = 0
    skipped = 0
    start = time.time()

    for i, item in enumerate(ds):
        if saved + skipped >= args.count:
            break
        idx = args.start_idx + i
        path = os.path.join(args.out_dir, f"ffhq_{idx:06d}.jpg")
        if os.path.exists(path):
            skipped += 1
            continue
        try:
            img = item[args.image_key]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path, quality=92)
            saved += 1
        except Exception as e:
            print(f"  [{idx}] FAIL {e}", flush=True)
            continue

        if saved % 200 == 0:
            elapsed = time.time() - start
            rate = saved / max(elapsed, 1)
            eta = (args.count - saved - skipped) / max(rate, 1e-6)
            print(f"  saved={saved} skipped={skipped} "
                  f"rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    total = time.time() - start
    print(f"\nDone. saved={saved} skipped={skipped} total={total:.0f}s")


if __name__ == "__main__":
    main()
