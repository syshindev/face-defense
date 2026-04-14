import argparse
import os
import json
import time

import cv2
import numpy as np
import mediapipe as mp
from insightface.app import FaceAnalysis

# Eye landmark indices for blink detection
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def parse_args():
    parser = argparse.ArgumentParser(description="Face Defense Access Control Demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ir_camera", type=int, default=-1, help="IR camera index (-1 = disabled)")
    parser.add_argument("--db_path", type=str, default="face_db")
    parser.add_argument("--similarity_threshold", type=float, default=0.4)
    return parser.parse_args()


class FaceDatabase:
    # Simple face embedding database using JSON + npy files

    def __init__(self, db_path):
        self.db_path = db_path
        self.meta_path = os.path.join(db_path, "meta.json")
        self.users = {}
        os.makedirs(db_path, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.users = json.load(f)

    def _save(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)

    def register(self, name, embedding):
        emb_file = f"{name}.npy"
        np.save(os.path.join(self.db_path, emb_file), embedding)
        self.users[name] = emb_file
        self._save()
        print(f"Registered: {name}")

    def recognize(self, embedding, threshold):
        best_name = None
        best_sim = -1

        for name, emb_file in self.users.items():
            stored = np.load(os.path.join(self.db_path, emb_file))
            sim = np.dot(embedding, stored) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored) + 1e-8
            )
            if sim > best_sim:
                best_sim = sim
                best_name = name

        return best_name, float(best_sim)

    def delete(self, name):
        if name not in self.users:
            print(f"User not found: {name}")
            return
        emb_path = os.path.join(self.db_path, self.users[name])
        if os.path.exists(emb_path):
            os.remove(emb_path)
        del self.users[name]
        self._save()
        print(f"Deleted: {name}")

    def list_users(self):
        return list(self.users.keys())

    def count(self):
        return len(self.users)


def compute_ear(landmarks, eye_indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def check_ir_spoof(ir_frame, face_bbox):
    # IR-based spoof detection
    x1, y1, x2, y2 = face_bbox
    h, w = ir_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_roi = ir_frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return False, "no_face"

    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    mean_val = gray.mean()
    std_val = gray.std()

    # Screen: very dark in IR
    if mean_val < 30:
        return False, "display"

    # Print: low variance, uniform reflection
    if std_val < 15:
        return False, "print"

    return True, "real"


def draw_result_panel(frame, result_info, panel_w=350):
    h = frame.shape[0]
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    # Title
    cv2.putText(panel, "ACCESS CONTROL", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(panel, (20, 55), (panel_w - 20, 55), (100, 100, 100), 1)

    status = result_info.get("status", "")
    name = result_info.get("name", "")
    similarity = result_info.get("similarity", 0)
    liveness = result_info.get("liveness", "")

    # Status mapping
    status_map = {
        "authorized": ((0, 255, 0), "AUTHORIZED", "Authentication Success"),
        "unauthorized_real": ((0, 165, 255), "UNAUTHORIZED", "Unregistered User"),
        "spoof_print": ((0, 0, 255), "UNAUTHORIZED", "Print Attack Detected"),
        "spoof_display": ((0, 0, 255), "UNAUTHORIZED", "Display Attack Detected"),
        "spoof_blink": ((0, 0, 255), "UNAUTHORIZED", "Liveness Check Failed"),
        "registering": ((0, 200, 255), "REGISTERING", "Look at the camera..."),
        "scanning": ((0, 200, 255), "SCANNING", "Verifying..."),
    }
    status_color, status_text, result_text = status_map.get(
        status, ((150, 150, 150), "WAITING", "No Face Detected")
    )

    y = 90

    # Status badge
    cv2.rectangle(panel, (20, y - 5), (panel_w - 20, y + 35), status_color, -1)
    cv2.putText(panel, status_text, (30, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 60

    # Result
    cv2.putText(panel, "Result:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 25
    cv2.putText(panel, result_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 40

    # User name
    if name:
        cv2.putText(panel, "User:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 25
        cv2.putText(panel, name, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 40

    # Similarity
    if similarity > 0:
        cv2.putText(panel, "Similarity:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 25
        sim_color = (0, 255, 0) if similarity >= 0.4 else (0, 0, 255)
        cv2.putText(panel, f"{similarity:.2f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sim_color, 1)
        y += 40

    # Liveness
    cv2.putText(panel, "Liveness:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 25
    if liveness == "PASS":
        cv2.putText(panel, "PASS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif liveness == "FAIL":
        cv2.putText(panel, "FAIL", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(panel, "...", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 50

    # Controls
    cv2.line(panel, (20, h - 90), (panel_w - 20, h - 90), (100, 100, 100), 1)
    reg_count = result_info.get("registered", 0)
    cv2.putText(panel, f"Registered: {reg_count}", (20, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.putText(panel, "[n] Register  [d] Delete  [q] Quit", (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return panel


def main():
    args = parse_args()

    # Face recognition model
    face_app = FaceAnalysis(name="buffalo_l",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Blink detection
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Face database
    db = FaceDatabase(args.db_path)
    print(f"Registered users: {db.count()}")

    # Cameras
    cap = cv2.VideoCapture(args.camera)
    ir_cap = None
    use_ir = args.ir_camera >= 0
    if use_ir:
        ir_cap = cv2.VideoCapture(args.ir_camera)
        if not ir_cap.isOpened():
            print("IR camera not found, falling back to blink detection")
            use_ir = False

    # Prompt registration if database is empty
    if db.count() == 0:
        print("No registered users. Press 'n' to register or start without registration.")

    # Blink state
    EAR_CLOSE = 0.33
    EAR_OPEN = 0.37
    BLINK_TIMEOUT = 5.0
    was_closed = False
    close_time = 0.0
    last_blink_time = 0.0

    print("Press 'n' to register, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        ir_frame = None
        if use_ir and ir_cap:
            ir_ret, ir_frame = ir_cap.read()
            if not ir_ret:
                ir_frame = None

        result_info = {"status": "waiting", "name": "", "similarity": 0, "liveness": "", "registered": db.count()}
        now = time.time()

        # Face detection + recognition
        faces = face_app.get(frame)

        if len(faces) > 0:
            face = max(faces, key=lambda f: f.det_score)
            x1, y1, x2, y2 = face.bbox.astype(int)
            embedding = face.normed_embedding

            # Liveness check
            is_live = False
            spoof_type = None

            if use_ir and ir_frame is not None:
                is_live, spoof_type = check_ir_spoof(ir_frame, (x1, y1, x2, y2))
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(rgb)

                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0].landmark
                    left_ear = compute_ear(landmarks, LEFT_EYE)
                    right_ear = compute_ear(landmarks, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0

                    if avg_ear < EAR_CLOSE:
                        if not was_closed:
                            was_closed = True
                            close_time = now
                    elif was_closed and avg_ear >= EAR_OPEN:
                        if (now - close_time) < 0.5:
                            last_blink_time = now
                        was_closed = False

                time_since_blink = now - last_blink_time if last_blink_time > 0 else 999
                is_live = time_since_blink < BLINK_TIMEOUT
                if not is_live:
                    spoof_type = "blink"

            if not is_live:
                # Spoof detected
                status = f"spoof_{spoof_type}" if spoof_type in ("display", "print") else "spoof_blink"
                result_info = {"status": status, "liveness": "FAIL", "registered": db.count()}
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "SPOOF", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Live face - check identity
                name, sim = db.recognize(embedding, args.similarity_threshold)
                if db.count() > 0 and sim >= args.similarity_threshold:
                    result_info = {
                        "status": "authorized",
                        "name": name,
                        "similarity": sim,
                        "liveness": "PASS",
                        "registered": db.count(),
                    }
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "AUTHORIZED", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    result_info = {
                        "status": "unauthorized_real",
                        "name": name if name else "",
                        "similarity": sim if sim > 0 else 0,
                        "liveness": "PASS",
                        "registered": db.count(),
                    }
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "UNKNOWN", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Draw result panel
        panel = draw_result_panel(frame, result_info)
        display = np.hstack([frame, panel])

        cv2.imshow("Face Defense - Access Control Demo", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            if len(faces) > 0:
                face = max(faces, key=lambda f: f.det_score)
                idx = db.count() + 1
                name = f"User_{idx:03d}"
                db.register(name, face.normed_embedding)
                print(f"Registered: {name} | Total: {db.count()}")
            else:
                print("No face detected")
        elif key == ord("d"):
            users = db.list_users()
            if not users:
                print("No registered users")
            else:
                last = users[-1]
                db.delete(last)
                print(f"Deleted: {last} | Remaining: {db.count()}")

    cap.release()
    if ir_cap:
        ir_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
