import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np
import mediapipe as mp

from shared.face_utils import compute_ear, LEFT_EYE, RIGHT_EYE


def parse_args():
    parser = argparse.ArgumentParser(description="Face Defense Webcam Demo")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


# MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    print("Press 'q' to quit")

    BLINK_TIMEOUT = 5.0
    EAR_CLOSE = 0.33
    EAR_OPEN = 0.37
    MAX_BLINK_DURATION = 0.5

    was_closed = False
    close_time = 0.0
    last_blink_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cv2.imshow("Face Defense - Anti Spoofing Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Face bbox
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        # EAR computation
        left_ear = compute_ear(landmarks, LEFT_EYE)
        right_ear = compute_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        # Blink detection
        now = cv2.getTickCount() / cv2.getTickFrequency()

        if avg_ear < EAR_CLOSE:
            if not was_closed:
                was_closed = True
                close_time = now
        elif was_closed and avg_ear >= EAR_OPEN:
            blink_duration = now - close_time
            if blink_duration < MAX_BLINK_DURATION:
                last_blink_time = now
            was_closed = False

        # Verdict
        time_since_blink = now - last_blink_time if last_blink_time > 0 else 999

        if time_since_blink < BLINK_TIMEOUT:
            verdict = "REAL"
            color = (0, 255, 0)
        else:
            verdict = "SPOOF"
            color = (0, 0, 255)

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, verdict, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Defense - Anti Spoofing Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
