import argparse
import os
import sys

import cv2

from face_defense.core.config import load_config
from face_defense.core.pipeline import build_pipeline
from face_defense.data.preprocessing import FacePreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Face Defense Single Image Analysis")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, default="configs/pipeline/lightweight.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    # Add Silent-FAS to path if available
    fas_root = os.path.join(os.path.dirname(__file__), "..", "third_party", "Silent-Face-Anti-Spoofing")
    if os.path.exists(fas_root):
        sys.path.insert(0, os.path.abspath(fas_root))
        os.chdir(os.path.abspath(fas_root))

    config = load_config(args.config)
    pipeline = build_pipeline(config)
    preprocessor = FacePreprocessor()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot read image {args.image}")
        return

    faces = preprocessor.process(image)
    print(f"Detected {len(faces)} face(s)\n")

    for i, (face_img, det) in enumerate(faces):
        result = pipeline.run(face_img)
        verdict = "REAL" if result.is_real else "SPOOF"
        print(f"Face {i+1}:")
        print(f"  Verdict: {verdict}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Attack type: {result.attack_type}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        if result.exited_early:
            print(f"  Early exit at: {result.exit_stage}")
        print()


if __name__ == "__main__":
    main()
