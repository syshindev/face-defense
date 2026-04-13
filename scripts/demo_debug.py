import cv2
import numpy as np
import mediapipe as mp

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
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result = analyze_frame(frame)

        if result:
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display scores
            y = 30
            for key, val in result.items():
                if key == "bbox":
                    continue
                text = f"{key}: {val:.4f}"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 30

        cv2.imshow("Debug - Anti Spoofing Scores", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
