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

# Cheek region landmarks for rPPG
LEFT_CHEEK = [50, 101, 36, 205, 187, 123, 116, 117, 118, 119]
RIGHT_CHEEK = [280, 330, 266, 425, 411, 352, 345, 346, 347, 348]


def compute_lap_var(frame):
    # Laplacian variance: real skin has richer texture than screen/print
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_roi_mean_green(frame, landmarks, indices):
    # Extract mean green channel value from landmark ROI
    h, w = frame.shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    green = frame[:, :, 1]
    roi_pixels = green[mask == 255]
    if len(roi_pixels) == 0:
        return 0.0
    return float(roi_pixels.mean())


def compute_rppg_score(signal_buffer):
    # Analyze green channel signal for pulse presence
    if len(signal_buffer) < 60:
        return 0.0, False

    signal = np.array(signal_buffer)
    # Detrend: remove linear trend
    signal = signal - np.linspace(signal[0], signal[-1], len(signal))
    # Normalize
    if signal.std() < 1e-6:
        return 0.0, False
    signal = (signal - signal.mean()) / signal.std()

    # FFT to find pulse frequency
    fps = 30.0
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(signal))

    # Heart rate range: 0.8Hz ~ 3.0Hz (48~180 BPM)
    valid = (freqs >= 0.8) & (freqs <= 3.0)
    if not valid.any():
        return 0.0, False

    valid_mag = fft_mag[valid]
    valid_freqs = freqs[valid]

    # Peak frequency and SNR
    peak_idx = valid_mag.argmax()
    peak_mag = valid_mag[peak_idx]
    peak_freq = valid_freqs[peak_idx]
    bpm = peak_freq * 60

    # SNR: peak energy vs rest
    total_energy = valid_mag.sum()
    if total_energy < 1e-6:
        return 0.0, False
    snr = peak_mag / (total_energy - peak_mag + 1e-6)

    # Score: higher SNR = more likely real pulse
    score = min(snr / 0.5, 1.0)
    has_pulse = snr >= 0.15

    return score, has_pulse


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    print("Press 'q' to quit")
    print("rPPG needs ~3 seconds to stabilize")

    # rPPG signal buffer (green channel mean over time)
    green_buffer = deque(maxlen=150)  # 5 seconds at 30fps

    # Texture history for smoothing
    lap_history = deque(maxlen=15)

    # rPPG result smoothing
    rppg_history = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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

        # Crop face region
        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(w, x2), min(h, y2)
        face_crop = frame[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            continue

        # 1. Texture analysis (Laplacian variance)
        lap_var = compute_lap_var(face_crop)
        lap_history.append(lap_var)
        avg_lap = np.mean(lap_history)

        # 2. rPPG: extract green channel from cheek/forehead ROI
        green_val = get_roi_mean_green(frame, landmarks, LEFT_CHEEK + RIGHT_CHEEK)
        green_buffer.append(green_val)
        rppg_score, has_pulse = compute_rppg_score(green_buffer)
        rppg_history.append(1.0 if has_pulse else 0.0)
        avg_rppg = np.mean(rppg_history)

        # Combined verdict
        texture_pass = avg_lap >= 80.0
        pulse_pass = avg_rppg >= 0.5

        # Both must pass, or pulse alone if texture is borderline
        if pulse_pass and texture_pass:
            is_real = True
        elif pulse_pass and avg_lap >= 50.0:
            is_real = True
        else:
            is_real = False

        # Status text
        buffer_pct = min(len(green_buffer) / 60 * 100, 100)

        # Draw result
        color = (0, 255, 0) if is_real else (0, 0, 255)
        label = "REAL" if is_real else "SPOOF"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display scores
        info = [
            f"Texture:  {'PASS' if texture_pass else 'FAIL'} ({avg_lap:.1f})",
            f"Pulse:    {'PASS' if pulse_pass else 'FAIL'} ({rppg_score:.2f})",
            f"Buffer:   {buffer_pct:.0f}%",
        ]
        for i, text in enumerate(info):
            c = (0, 255, 0) if i < 2 and [texture_pass, pulse_pass][i] else (200, 200, 200)
            if i == 2:
                c = (200, 200, 200)
            cv2.putText(frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        cv2.imshow("Face Defense - Anti Spoofing Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
