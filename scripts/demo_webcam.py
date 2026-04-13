import argparse

import cv2
import numpy as np
import torch

from face_defense.data.preprocessing import FacePreprocessor
from face_defense.models.anti_spoof.cdcn_model import CDCN


def parse_args():
    parser = argparse.ArgumentParser(description="Face Defense Webcam Demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cdcn_best.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load CDCN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CDCN(in_channels=3, theta=0.7)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Device: {device}, Model loaded")

    # Face detector
    preprocessor = FacePreprocessor(face_size=256, margin=0.2)

    cap = cv2.VideoCapture(args.camera)
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = preprocessor.process(frame)

        for face_img, det in faces:
            # CDCN inference
            img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                depth_map = model(tensor)
            score = depth_map.mean().item()
            is_real = score >= args.threshold

            # Draw result
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 0) if is_real else (0, 0, 255)
            label = "REAL" if is_real else "SPOOF"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Defense - Anti Spoofing Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
