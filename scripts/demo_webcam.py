import argparse
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


def parse_args():
    parser = argparse.ArgumentParser(description="Face Defense Webcam Demo")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


# MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def compute_ear(landmarks, eye_indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_lap_var(frame):
    # Laplacian variance: real skin has richer texture than screen/print
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_moire(gray):
    # Moire pattern detection using FFT high-frequency energy
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    # High frequency region (outer ring)
    mask = np.zeros((h, w), dtype=bool)
    y, x = np.ogrid[:h, :w]
    outer = (x - cx) ** 2 + (y - cy) ** 2 >= (min(h, w) // 4) ** 2
    mask[outer] = True

    high_freq = magnitude[mask].mean() if mask.sum() > 0 else 0
    return float(high_freq)


def compute_color_score(frame):
    # Color analysis: screen/print has different color distribution
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Screens tend to have lower saturation variance than real skin
    sat_std = hsv[:, :, 1].std()
    return float(sat_std)


def compute_reflection(gray):
    # Reflection detection: screens have bright specular highlights
    _, bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    bright_ratio = bright.sum() / (gray.shape[0] * gray.shape[1] * 255)
    return float(bright_ratio)


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    print("Press 'q' to quit")

    # Multi-frame movement tracking
    prev_landmarks = None
    movement_history = deque(maxlen=30)

    # Scoring thresholds
    LAP_THRESHOLD = 90.0
    MOVEMENT_THRESHOLD = 0.001
    MOIRE_THRESHOLD = 3.0
    COLOR_THRESHOLD = 30.0
    REFLECTION_THRESHOLD = 0.02

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

        # Crop face region for analysis
        fx1 = max(0, x1)
        fy1 = max(0, y1)
        fx2 = min(w, x2)
        fy2 = min(h, y2)
        face_crop = frame[fy1:fy2, fx1:fx2]

        if face_crop.size == 0:
            continue

        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # 1. Laplacian variance (texture sharpness)
        lap_var = compute_lap_var(face_crop)
        lap_pass = lap_var >= LAP_THRESHOLD

        # 2. Micro-movement detection
        curr_pts = np.array([[lm.x, lm.y] for lm in landmarks])
        if prev_landmarks is not None:
            movement = np.mean(np.abs(curr_pts - prev_landmarks))
            movement_history.append(movement)
        prev_landmarks = curr_pts.copy()

        avg_movement = np.mean(movement_history) if len(movement_history) > 10 else 0.01
        move_pass = avg_movement >= MOVEMENT_THRESHOLD

        # 3. Moire pattern detection
        moire_score = compute_moire(gray_face)
        moire_pass = moire_score < MOIRE_THRESHOLD

        # 4. Color analysis
        color_score = compute_color_score(face_crop)
        color_pass = color_score >= COLOR_THRESHOLD

        # 5. Reflection detection
        reflection = compute_reflection(gray_face)
        reflect_pass = reflection < REFLECTION_THRESHOLD

        # Combined verdict (majority voting)
        checks = [lap_pass, move_pass, moire_pass, color_pass, reflect_pass]
        pass_count = sum(checks)
        is_real = pass_count >= 3

        # Draw result
        color = (0, 255, 0) if is_real else (0, 0, 255)
        label = "REAL" if is_real else "SPOOF"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({pass_count}/5)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display individual scores
        scores_text = [
            f"Texture:    {'PASS' if lap_pass else 'FAIL'} ({lap_var:.1f})",
            f"Movement:   {'PASS' if move_pass else 'FAIL'} ({avg_movement:.4f})",
            f"Moire:      {'PASS' if moire_pass else 'FAIL'} ({moire_score:.2f})",
            f"Color:      {'PASS' if color_pass else 'FAIL'} ({color_score:.1f})",
            f"Reflection: {'PASS' if reflect_pass else 'FAIL'} ({reflection:.4f})",
        ]
        for i, text in enumerate(scores_text):
            c = (0, 255, 0) if checks[i] else (0, 0, 255)
            cv2.putText(frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        cv2.imshow("Face Defense - Anti Spoofing Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
