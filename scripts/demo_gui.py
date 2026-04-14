import sys
import os
import json
import time

import cv2
import numpy as np
import mediapipe as mp
from insightface.app import FaceAnalysis
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


class FaceDatabase:
    def __init__(self, db_path="face_db"):
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

    def recognize(self, embedding, threshold=0.4):
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

    def delete_last(self):
        users = list(self.users.keys())
        if not users:
            return None
        last = users[-1]
        emb_path = os.path.join(self.db_path, self.users[last])
        if os.path.exists(emb_path):
            os.remove(emb_path)
        del self.users[last]
        self._save()
        return last

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


class MainWindow(QMainWindow):
    def __init__(self, camera_id=0, ir_camera_id=-1):
        super().__init__()
        self.setWindowTitle("Face Defense - Access Control System")
        self.setMinimumSize(1000, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # Models
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.db = FaceDatabase()

        # Camera
        self.cap = cv2.VideoCapture(camera_id)
        self.ir_cap = None
        self.use_ir = ir_camera_id >= 0
        if self.use_ir:
            self.ir_cap = cv2.VideoCapture(ir_camera_id)
            if not self.ir_cap.isOpened():
                self.use_ir = False

        # Blink state
        self.EAR_CLOSE = 0.33
        self.EAR_OPEN = 0.37
        self.BLINK_TIMEOUT = 5.0
        self.was_closed = False
        self.close_time = 0.0
        self.last_blink_time = 0.0

        # Current face for registration
        self.current_embedding = None

        self._init_ui()
        self._start_timer()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left: Camera view
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #000; border: 1px solid #333;")
        main_layout.addWidget(self.camera_label, stretch=3)

        # Right: Result panel
        right_panel = QFrame()
        right_panel.setStyleSheet(
            "background-color: #2a2a2a; border: 1px solid #333; border-radius: 8px;"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(8)

        # Title
        title = QLabel("ACCESS CONTROL")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("color: #444;")
        right_layout.addWidget(sep1)

        # Status badge
        self.status_label = QLabel("WAITING")
        self.status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "background-color: #555; color: white; padding: 10px; border-radius: 5px;"
        )
        right_layout.addWidget(self.status_label)

        # Result
        self.result_label = self._create_info_label("Result", "No Face Detected")
        right_layout.addWidget(self.result_label)

        # User
        self.user_label = self._create_info_label("User", "-")
        right_layout.addWidget(self.user_label)

        # Similarity
        self.sim_label = self._create_info_label("Similarity", "-")
        right_layout.addWidget(self.sim_label)

        # Liveness
        self.live_label = self._create_info_label("Liveness", "-")
        right_layout.addWidget(self.live_label)

        right_layout.addStretch()

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #444;")
        right_layout.addWidget(sep2)

        # Registered count
        self.reg_label = QLabel(f"Registered: {self.db.count()}")
        self.reg_label.setFont(QFont("Arial", 10))
        self.reg_label.setStyleSheet("color: #888;")
        right_layout.addWidget(self.reg_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_register = QPushButton("Register")
        self.btn_register.setStyleSheet(
            "background-color: #2196F3; color: white; padding: 8px 16px; "
            "border: none; border-radius: 4px; font-size: 12px;"
        )
        self.btn_register.clicked.connect(self.register_face)
        btn_layout.addWidget(self.btn_register)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet(
            "background-color: #f44336; color: white; padding: 8px 16px; "
            "border: none; border-radius: 4px; font-size: 12px;"
        )
        self.btn_delete.clicked.connect(self.delete_face)
        btn_layout.addWidget(self.btn_delete)

        right_layout.addLayout(btn_layout)

        main_layout.addWidget(right_panel, stretch=1)

    def _create_info_label(self, title, value):
        widget = QLabel(f"<span style='color:#888; font-size:11px;'>{title}</span><br>"
                        f"<span style='font-size:13px;'>{value}</span>")
        widget.setTextFormat(Qt.RichText)
        return widget

    def _update_info_label(self, label, title, value, color="white"):
        label.setText(
            f"<span style='color:#888; font-size:11px;'>{title}</span><br>"
            f"<span style='color:{color}; font-size:13px;'>{value}</span>"
        )

    def _start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        now = time.time()
        faces = self.face_app.get(frame)

        if len(faces) == 0:
            self._set_status("WAITING", "#555", "No Face Detected")
            self._update_info_label(self.user_label, "User", "-")
            self._update_info_label(self.sim_label, "Similarity", "-")
            self._update_info_label(self.live_label, "Liveness", "-")
            self.current_embedding = None
        else:
            face = max(faces, key=lambda f: f.det_score)
            x1, y1, x2, y2 = face.bbox.astype(int)
            self.current_embedding = face.normed_embedding

            # Liveness check
            is_live = False
            spoof_type = None

            if self.use_ir and self.ir_cap:
                ir_ret, ir_frame = self.ir_cap.read()
                if ir_ret:
                    is_live, spoof_type = self._check_ir(ir_frame, (x1, y1, x2, y2))
            else:
                is_live, spoof_type = self._check_blink(frame, now)

            if not is_live:
                # Spoof
                if spoof_type == "display":
                    self._set_status("UNAUTHORIZED", "#f44336", "Display Attack Detected")
                elif spoof_type == "print":
                    self._set_status("UNAUTHORIZED", "#f44336", "Print Attack Detected")
                else:
                    self._set_status("UNAUTHORIZED", "#f44336", "Liveness Check Failed")
                self._update_info_label(self.user_label, "User", "-")
                self._update_info_label(self.sim_label, "Similarity", "-")
                self._update_info_label(self.live_label, "Liveness", "FAIL", "#f44336")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "SPOOF", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self._update_info_label(self.live_label, "Liveness", "PASS", "#4CAF50")

                if self.db.count() == 0:
                    self._set_status("UNAUTHORIZED", "#FF9800", "No Registered Users")
                    self._update_info_label(self.user_label, "User", "-")
                    self._update_info_label(self.sim_label, "Similarity", "-")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "UNKNOWN", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    name, sim = self.db.recognize(self.current_embedding)
                    self._update_info_label(self.sim_label, "Similarity", f"{sim:.2f}",
                                            "#4CAF50" if sim >= 0.4 else "#f44336")

                    if sim >= 0.4:
                        self._set_status("AUTHORIZED", "#4CAF50", "Authentication Success")
                        self._update_info_label(self.user_label, "User", name)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "AUTHORIZED", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        self._set_status("UNAUTHORIZED", "#FF9800", "Unregistered User")
                        self._update_info_label(self.user_label, "User", "-")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(frame, "UNKNOWN", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Display frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(img).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled)

    def _set_status(self, text, color, result):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"background-color: {color}; color: white; padding: 10px; "
            f"border-radius: 5px; font-size: 14px; font-weight: bold;"
        )
        self._update_info_label(self.result_label, "Result", result)

    def _check_blink(self, frame, now):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb)

        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            left_ear = compute_ear(landmarks, LEFT_EYE)
            right_ear = compute_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.EAR_CLOSE:
                if not self.was_closed:
                    self.was_closed = True
                    self.close_time = now
            elif self.was_closed and avg_ear >= self.EAR_OPEN:
                if (now - self.close_time) < 0.5:
                    self.last_blink_time = now
                self.was_closed = False

        time_since = now - self.last_blink_time if self.last_blink_time > 0 else 999
        if time_since < self.BLINK_TIMEOUT:
            return True, None
        return False, "blink"

    def _check_ir(self, ir_frame, bbox):
        x1, y1, x2, y2 = bbox
        h, w = ir_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = ir_frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, "no_face"
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        if gray.mean() < 30:
            return False, "display"
        if gray.std() < 15:
            return False, "print"
        return True, None

    def register_face(self):
        if self.current_embedding is not None:
            idx = self.db.count() + 1
            name = f"User_{idx:03d}"
            self.db.register(name, self.current_embedding)
            self.reg_label.setText(f"Registered: {self.db.count()}")
        else:
            self._set_status("WAITING", "#555", "No face to register")

    def delete_face(self):
        deleted = self.db.delete_last()
        if deleted:
            self.reg_label.setText(f"Registered: {self.db.count()}")
        else:
            self._set_status("WAITING", "#555", "No users to delete")

    def closeEvent(self, event):
        self.cap.release()
        if self.ir_cap:
            self.ir_cap.release()
        event.accept()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ir_camera", type=int, default=-1)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(camera_id=args.camera, ir_camera_id=args.ir_camera)
    window.show()
    sys.exit(app.exec_())
