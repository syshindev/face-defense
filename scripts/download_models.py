import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Download pretrained model weights")
    parser.add_argument("--model", type=str, default="all", help="Model to download (silent_fas, cdcn, xception, efficientnet, all)")
    parser.add_argument("--output", type=str, default="weights/", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Downloading model: {args.model}")
    print(f"Output directory: {args.output}")

    # TODO: Implement download logic for each model
    # silent_fas: clone repo + download weights
    # cdcn: download checkpoint
    # xception: download from DeepfakeBench
    # efficientnet: download pretrained weights
    print("Download not yet implemented. Coming in Phase 2")


if __name__ == "__main__":
    main()
