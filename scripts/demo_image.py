import argparse

import cv2

from face_defense.core.config import load_config
from face_defense.data.preprocessing import FacePreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Face Defense Single Image Analysis")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, default="configs/pipeline/full_defense.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    preprocessor = FacePreprocessor()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot read image {args.image}")
        return

    faces = preprocessor.process(image)
    print(f"Detected {len(faces)} face(s)")

    for i, (face_img, det) in enumerate(faces):
        # TODO: Replace with pipeline.run(face_img)
        print(f"  Face {i+1}: bbox={det['bbox'].tolist()}, score={det['score']:.2f}")


if __name__ == "__main__":
    main()
