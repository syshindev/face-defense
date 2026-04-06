import argparse

import cv2

from face_defense.core.config import load_config
from face_defense.data.preprocessing import FacePreprocessor
from face_defense.data.webcam_source import WebcamSource


def parse_args():
    parser = argparse.ArgumentParser(description="Face Defense Webcam Demo")
    parser.add_argument("--config", type=str, default="configs/pipeline/lightweight.yaml")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def draw_result(frame, bbox, result):
    # Draw bounding box and verdict on frame
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
    label = "REAL" if result["is_real"] else "SPOOF"
    confidence = result["confidence"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, f"{label} ({confidence:.1%})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
    )
    return frame


def main():
    args = parse_args()
    config = load_config(args.config)
    preprocessor = FacePreprocessor()

    # TODO: Build pipeline from config and run inference
    # pipeline = build_pipeline(config)

    with WebcamSource(camera_id=args.camera) as cam:
        print("Press 'q' to quit")
        while True:
            frame = cam.read()
            if frame is None:
                continue

            faces = preprocessor.process(frame)
            for face_img, det in faces:
                # TODO: Replace with pipeline.run(face_img)
                result = {"is_real": True, "confidence": 0.0}
                frame = draw_result(frame, det["bbox"], result)

            cv2.imshow("Face Defense Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

if __name__ == "__main__":
    main()
    