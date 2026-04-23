import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np
import mediapipe as mp

from shared.face_utils import compute_ear, LEFT_EYE, RIGHT_EYE

# MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
)


def analyze_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    # 1. Z-depth variance (real face = high, flat = low)
    z_values = np.array([lm.z for lm in landmarks])
    z_std = float(z_values.std())
    z_range = float(z_values.max() - z_values.min())

    # 2. EAR (eye aspect ratio)
    left_ear = compute_ear(landmarks, LEFT_EYE)
    right_ear = compute_ear(landmarks, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2.0

    # 3. Texture - Laplacian variance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 4. Face bbox from landmarks
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))

    return {
        "z_std": z_std,
        "z_range": z_range,
        "ear": avg_ear,
        "lap_var": lap_var,
        "bbox": (x1, y1, x2, y2),
    }


def main():
    cap = cv2.VideoCapture(0)
    print("Press 'r' to start recording REAL scores")
    print("Press 's' to start recording SPOOF scores")
    print("Press 'p' to print summary")
    print("Press 'q' to quit")

    real_scores = []
    spoof_scores = []
    mode = None  # None, "real", "spoof"

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result = analyze_frame(frame)

        if result:
            x1, y1, x2, y2 = result["bbox"]

            # Record scores based on mode
            if mode == "real":
                real_scores.append(result["lap_var"])
                color = (0, 255, 0)
                cv2.putText(frame, f"RECORDING REAL ({len(real_scores)})", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif mode == "spoof":
                spoof_scores.append(result["lap_var"])
                color = (0, 0, 255)
                cv2.putText(frame, f"RECORDING SPOOF ({len(spoof_scores)})", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                color = (255, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Display scores
            y = 30
            for key, val in result.items():
                if key == "bbox":
                    continue
                text = f"{key}: {val:.4f}"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 30

        cv2.imshow("Debug - Anti Spoofing Scores", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            mode = "real" if mode != "real" else None
            print(f"Mode: {'REAL recording' if mode == 'real' else 'stopped'}")
        elif key == ord("s"):
            mode = "spoof" if mode != "spoof" else None
            print(f"Mode: {'SPOOF recording' if mode == 'spoof' else 'stopped'}")
        elif key == ord("p"):
            if real_scores:
                r = np.array(real_scores)
                print(f"\nREAL  lap_var: mean={r.mean():.2f}, min={r.min():.2f}, max={r.max():.2f}, std={r.std():.2f}")
            if spoof_scores:
                s = np.array(spoof_scores)
                print(f"SPOOF lap_var: mean={s.mean():.2f}, min={s.min():.2f}, max={s.max():.2f}, std={s.std():.2f}")
        elif key == ord("q"):
            # Print final summary
            if real_scores:
                r = np.array(real_scores)
                print(f"\nREAL  lap_var: mean={r.mean():.2f}, min={r.min():.2f}, max={r.max():.2f}, std={r.std():.2f}")
            if spoof_scores:
                s = np.array(spoof_scores)
                print(f"SPOOF lap_var: mean={s.mean():.2f}, min={s.min():.2f}, max={s.max():.2f}, std={s.std():.2f}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
