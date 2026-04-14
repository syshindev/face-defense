import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Download pretrained model weights")
    parser.add_argument("--model", type=str, default="all", help="Model to download (cdcn, xception, efficientnet, all)")
    parser.add_argument("--output", type=str, default="weights/", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Downloading model: {args.model}")
    print(f"Output directory: {args.output}")

    # TODO: Implement download logic for each model
    # cdcn: download checkpoint
    # xception: download pretrained weights
    # efficientnet: download pretrained weights
    print("Download not yet implemented.")


if __name__ == "__main__":
    main()
