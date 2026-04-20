import argparse
import os
import random

import pandas as pd


KAGGLE_PREFIX = "/kaggle/input/ff-andcelebdf-frame-dataset-by-wish/"
FF_CELEBDF_PROJECT_ROOT = "data/ff-celebdf-frames/"


def rewrite_kaggle_path(p):
    # Convert Kaggle-absolute paths to project-relative so training can use
    # --data_root=. for all sources (ff-celebdf + extra).
    if isinstance(p, str) and p.startswith(KAGGLE_PREFIX):
        return FF_CELEBDF_PROJECT_ROOT + p[len(KAGGLE_PREFIX):]
    return p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge ff-celebdf-frames CSV with extra datasets into a unified train/val/test split"
    )
    parser.add_argument("--base_csv", type=str, default="data/ff-celebdf-frames/train_labels.csv",
                        help="Original ff-celebdf training CSV")
    parser.add_argument("--base_val_csv", type=str, default="data/ff-celebdf-frames/val_labels.csv")
    parser.add_argument("--base_test_csv", type=str, default="data/ff-celebdf-frames/test_labels.csv")
    parser.add_argument("--stylegan_dir", type=str, default="data/extra/stylegan_cropped")
    parser.add_argument("--diffusion_dir", type=str, default="data/extra/diffusion_cropped")
    parser.add_argument("--ffhq_dir", type=str, default="data/extra/ffhq_cropped")
    parser.add_argument("--celebahq_dir", type=str, default="data/extra/celebahq_cropped")
    parser.add_argument("--diffusion_sd15_dir", type=str, default="data/extra/diffusion_sd15_cropped")
    parser.add_argument("--wild_real_dir", type=str, default="data/extra/wilddeepfake/real")
    parser.add_argument("--wild_fake_dir", type=str, default="data/extra/wilddeepfake/fake")
    parser.add_argument("--video_real_dir", type=str, default="data/extra/ff_video_crops/real")
    parser.add_argument("--video_fake_dir", type=str, default="data/extra/ff_video_crops/fake")
    parser.add_argument("--out_dir", type=str, default="data/v3_splits")
    parser.add_argument("--wild_count", type=int, default=0,
                        help="Limit WildDeepfake per class (0 = use all)")
    parser.add_argument("--video_count", type=int, default=0,
                        help="Limit ff_video_crops per class (0 = use all)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def scan_dir(directory, label, source):
    rows = []
    if not os.path.isdir(directory):
        print(f"  SKIP {directory} (not found)")
        return rows
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append({
                "filepath": os.path.join(directory, fname),
                "source": source,
                "label": label,
            })
    print(f"  {source}: {len(rows)} images (label={label})")
    return rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    base_train = pd.read_csv(args.base_csv)
    base_val = pd.read_csv(args.base_val_csv)
    base_test = pd.read_csv(args.base_test_csv)
    for df in (base_train, base_val, base_test):
        df["filepath"] = df["filepath"].map(rewrite_kaggle_path)
    print(f"Base: train={len(base_train)} val={len(base_val)} test={len(base_test)}")

    print("\nScanning extra datasets:")
    extra_rows = []
    extra_rows.extend(scan_dir(args.stylegan_dir, label=1, source="stylegan"))
    extra_rows.extend(scan_dir(args.diffusion_dir, label=1, source="diffusion"))
    extra_rows.extend(scan_dir(args.ffhq_dir, label=0, source="ffhq_real"))
    extra_rows.extend(scan_dir(args.celebahq_dir, label=0, source="celebahq_real"))
    extra_rows.extend(scan_dir(args.diffusion_sd15_dir, label=1, source="diffusion_sd15"))
    wild_real = scan_dir(args.wild_real_dir, label=0, source="wild_real")
    wild_fake = scan_dir(args.wild_fake_dir, label=1, source="wild_fake")
    if args.wild_count > 0:
        rng.shuffle(wild_real)
        rng.shuffle(wild_fake)
        wild_real = wild_real[:args.wild_count]
        wild_fake = wild_fake[:args.wild_count]
        print(f"  -> WildDeepfake limited to {args.wild_count} per class")
    extra_rows.extend(wild_real)
    extra_rows.extend(wild_fake)
    video_real = scan_dir(args.video_real_dir, label=0, source="ff_video_real")
    video_fake = scan_dir(args.video_fake_dir, label=1, source="ff_video_fake")
    if args.video_count > 0:
        rng.shuffle(video_real)
        rng.shuffle(video_fake)
        video_real = video_real[:args.video_count]
        video_fake = video_fake[:args.video_count]
        print(f"  -> ff_video_crops limited to {args.video_count} per class")
    extra_rows.extend(video_real)
    extra_rows.extend(video_fake)

    if not extra_rows:
        print("No extra data found. Exiting.")
        return

    rng.shuffle(extra_rows)
    n = len(extra_rows)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    n_train = n - n_test - n_val

    extra_train = extra_rows[:n_train]
    extra_val = extra_rows[n_train:n_train + n_val]
    extra_test = extra_rows[n_train + n_val:]

    extra_train_df = pd.DataFrame(extra_train)
    extra_val_df = pd.DataFrame(extra_val)
    extra_test_df = pd.DataFrame(extra_test)

    base_cols = ["filepath", "source", "label"]
    train_df = pd.concat([base_train[base_cols], extra_train_df], ignore_index=True)
    val_df = pd.concat([base_val[base_cols], extra_val_df], ignore_index=True)
    test_df = pd.concat([base_test[base_cols], extra_test_df], ignore_index=True)

    train_path = os.path.join(args.out_dir, "train_v3.csv")
    val_path = os.path.join(args.out_dir, "val_v3.csv")
    test_path = os.path.join(args.out_dir, "test_v3.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n=== v3 Splits ===")
    print(f"Train: {len(train_df)} ({train_path})")
    print(f"  sources: {train_df['source'].value_counts().to_dict()}")
    print(f"  labels:  {train_df['label'].value_counts().to_dict()}")
    print(f"Val:   {len(val_df)} ({val_path})")
    print(f"Test:  {len(test_df)} ({test_path})")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name} breakdown:")
        for src in sorted(df["source"].unique()):
            mask = df["source"] == src
            n_real = int((df.loc[mask, "label"] == 0).sum())
            n_fake = int((df.loc[mask, "label"] == 1).sum())
            print(f"  {src}: {mask.sum()} (real={n_real}, fake={n_fake})")


if __name__ == "__main__":
    main()
