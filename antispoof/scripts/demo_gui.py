import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np
import mediapipe as mp
from insightface.app import FaceAnalysis
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QListWidget,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from shared.face_utils import FaceDatabase, compute_ear, LEFT_EYE, RIGHT_EYE


class MainWindow(QMainWindow):
    def __init__(self, camera_id=0, ir_camera_id=-1):
        super().__init__()
        self.setWindowTitle("FACE DEFENSE // ACCESS CONTROL")
        self.setMinimumSize(1000, 600)
        self.setStyleSheet(
            "background-color: #0c0c0c; color: #c0c0c0; "
            "font-family: 'Consolas', 'Courier New', monospace;"
        )

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
        self.has_ir = ir_camera_id >= 0
        if self.has_ir:
            self.ir_cap = cv2.VideoCapture(ir_camera_id)
            if not self.ir_cap.isOpened():
                self.has_ir = False
        self.ir_mode_enabled = self.has_ir
        self.blink_mode_enabled = True
        self.texture_mode_enabled = True
        self._texture_frame_count = 0
        self._texture_last_result = (True, None)
        self._spoof_history = []
        self._SPOOF_WINDOW = 10

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
        self.camera_label.setStyleSheet("background-color: #000; border: 2px solid #2a2a2a;")
        main_layout.addWidget(self.camera_label, stretch=3)

        # Right: Result panel
        right_panel = QFrame()
        right_panel.setStyleSheet(
            "background-color: #111111; border: 2px solid #2a2a2a;"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(8)

        # Title
        title = QLabel("ACCESS CONTROL")
        title.setFont(QFont("Consolas", 12, QFont.Bold))
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("color: #ffaa00; border: none;")
        right_layout.addWidget(title)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("color: #333; border: none;")
        right_layout.addWidget(sep1)

        # Status badge
        self.status_label = QLabel("STANDBY")
        self.status_label.setFont(QFont("Consolas", 13, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "background-color: #1a1a1a; color: #888; padding: 10px; "
            "border: 1px solid #333;"
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
        sep2.setStyleSheet("color: #333; border: none;")
        right_layout.addWidget(sep2)

        # Registered users
        self.reg_label = QLabel(f"Registered: {self.db.count()}")
        self.reg_label.setFont(QFont("Consolas", 10))
        self.reg_label.setStyleSheet("color: #ffaa00; border: none;")
        right_layout.addWidget(self.reg_label)

        self.user_list = QListWidget()
        self.user_list.setMaximumHeight(100)
        self.user_list.setStyleSheet(
            "background-color: #0c0c0c; color: #ffaa00; border: 1px solid #333; "
            "font-size: 11px; padding: 2px; font-family: 'Consolas';"
        )
        self._update_user_list()
        right_layout.addWidget(self.user_list)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_register = QPushButton("[ REGISTER ]")
        self.btn_register.setStyleSheet(self._button_qss("#ffaa00"))
        self.btn_register.clicked.connect(self.register_face)
        btn_layout.addWidget(self.btn_register)

        self.btn_delete = QPushButton("[ DELETE ]")
        self.btn_delete.setStyleSheet(self._button_qss("#ff3333"))
        self.btn_delete.clicked.connect(self.delete_face)
        btn_layout.addWidget(self.btn_delete)

        right_layout.addLayout(btn_layout)

        self.btn_ir_toggle = QPushButton()
        self.btn_ir_toggle.clicked.connect(self.toggle_ir_mode)
        right_layout.addWidget(self.btn_ir_toggle)
        self._refresh_ir_button()

        self.btn_blink_toggle = QPushButton()
        self.btn_blink_toggle.clicked.connect(self.toggle_blink_mode)
        right_layout.addWidget(self.btn_blink_toggle)
        self._refresh_blink_button()

        self.btn_texture_toggle = QPushButton()
        self.btn_texture_toggle.clicked.connect(self.toggle_texture_mode)
        right_layout.addWidget(self.btn_texture_toggle)
        self._refresh_texture_button()

        main_layout.addWidget(right_panel, stretch=1)

    def _create_info_label(self, title, value):
        widget = QLabel(f"<span style='color:#666; font-size:10px;'>[ {title} ]</span><br>"
                        f"<span style='font-size:13px; color:#aaa;'>{value}</span>")
        widget.setTextFormat(Qt.RichText)
        widget.setStyleSheet("border: none;")
        return widget

    def _update_info_label(self, label, title, value, color="#aaa"):
        label.setText(
            f"<span style='color:#666; font-size:10px;'>[ {title} ]</span><br>"
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
            self._set_status("STANDBY", "#555", "No Face Detected")
            self._update_info_label(self.user_label, "User", "-")
            self._update_info_label(self.sim_label, "Similarity", "-")
            self._update_info_label(self.live_label, "Liveness", "-")
            self.current_embedding = None
        else:
            face = max(faces, key=lambda f: f.det_score)
            x1, y1, x2, y2 = face.bbox.astype(int)
            self.current_embedding = face.normed_embedding

            # Liveness check
            is_live, spoof_type = self._check_liveness(frame, now, (x1, y1, x2, y2))

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
                self._update_info_label(self.live_label, "Liveness", "FAIL", "#ff3333")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "SPOOF", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                self._update_info_label(self.live_label, "Liveness", "PASS", "#00ff41")

                if self.db.count() == 0:
                    self._set_status("DENIED", "#ffaa00", "No Registered Users")
                    self._update_info_label(self.user_label, "User", "-")
                    self._update_info_label(self.sim_label, "Similarity", "-")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "UNKNOWN", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    name, sim = self.db.recognize(self.current_embedding)
                    self._update_info_label(self.sim_label, "Similarity", f"{sim:.2f}",
                                            "#00ff41" if sim >= 0.4 else "#ff3333")

                    if sim >= 0.4:
                        self._set_status("AUTHORIZED", "#00ff41", "Authentication Success")
                        self._update_info_label(self.user_label, "User", name)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "AUTHORIZED", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        self._set_status("DENIED", "#ff3333", "Unregistered User")
                        self._update_info_label(self.user_label, "User", "-")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(frame, "UNKNOWN", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

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
            f"background-color: #1a1a1a; color: {color}; padding: 10px; "
            f"border: 1px solid {color}; font-size: 13px; font-weight: bold;"
        )
        self._update_info_label(self.result_label, "Result", result)

    def _check_liveness(self, frame, now, bbox):
        is_spoof = False
        spoof_reason = None

        # Texture check (instant, frame-level)
        if self.texture_mode_enabled:
            tex_live, tex_reason = self._check_texture(frame, bbox)
            if not tex_live:
                is_spoof = True
                spoof_reason = tex_reason

        # IR check
        if not is_spoof and self.ir_mode_enabled and self.has_ir and self.ir_cap:
            ir_ret, ir_frame = self.ir_cap.read()
            if ir_ret:
                ir_live, ir_reason = self._check_ir(ir_frame, frame, bbox)
                if not ir_live:
                    is_spoof = True
                    spoof_reason = ir_reason

        # Smoothing — majority vote over last N frames
        self._spoof_history.append(is_spoof)
        if len(self._spoof_history) > self._SPOOF_WINDOW:
            self._spoof_history.pop(0)
        spoof_count = sum(self._spoof_history)
        smoothed_spoof = spoof_count >= self._SPOOF_WINDOW // 2

        if smoothed_spoof:
            return False, spoof_reason or "display"

        # Blink check (slow, requires user action)
        if self.blink_mode_enabled:
            blink_live, blink_reason = self._check_blink(frame, now)
            if not blink_live:
                return False, blink_reason

        return True, None

    def toggle_ir_mode(self):
        self.ir_mode_enabled = not self.ir_mode_enabled
        self._refresh_ir_button()

    def _refresh_ir_button(self):
        if self.ir_mode_enabled:
            if self.has_ir:
                text, color = "[ IR MODE: ON ]", "#00d4ff"
            else:
                text, color = "[ IR MODE: NO CAMERA ]", "#ffaa00"
        else:
            text, color = "[ IR MODE: OFF ]", "#666"
        self.btn_ir_toggle.setText(text)
        self.btn_ir_toggle.setStyleSheet(self._button_qss(color))

    def toggle_blink_mode(self):
        self.blink_mode_enabled = not self.blink_mode_enabled
        self._refresh_blink_button()

    def _refresh_blink_button(self):
        if self.blink_mode_enabled:
            text, color = "[ BLINK: ON ]", "#00ff41"
        else:
            text, color = "[ BLINK: OFF ]", "#666"
        self.btn_blink_toggle.setText(text)
        self.btn_blink_toggle.setStyleSheet(self._button_qss(color))

    def toggle_texture_mode(self):
        self.texture_mode_enabled = not self.texture_mode_enabled
        self._refresh_texture_button()

    def _refresh_texture_button(self):
        if self.texture_mode_enabled:
            text, color = "[ TEXTURE: ON ]", "#e040fb"
        else:
            text, color = "[ TEXTURE: OFF ]", "#666"
        self.btn_texture_toggle.setText(text)
        self.btn_texture_toggle.setStyleSheet(self._button_qss(color))

    @staticmethod
    def _compute_lbp(gray):
        """Compute LBP (Local Binary Pattern) using vectorized numpy."""
        padded = gray.astype(np.int16)
        center = padded[1:-1, 1:-1]
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]
        lbp = np.zeros_like(center, dtype=np.uint8)
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[1 + dy:padded.shape[0] - 1 + dy,
                              1 + dx:padded.shape[1] - 1 + dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)
        return lbp

    def _check_texture(self, frame, bbox):
        """LBP texture + Laplacian analysis for spoof detection."""
        # Run every 3 frames, reuse last result otherwise
        self._texture_frame_count += 1
        if self._texture_frame_count % 3 != 0:
            return self._texture_last_result

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, "no_face"

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        # 1. LBP entropy — prints have higher entropy (paper grain/print dots)
        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-8
        lbp_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

        # 2. Laplacian variance — screens have much higher sharpness
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 3. Skin color ratio in YCbCr
        roi_resized = cv2.resize(roi, (64, 64))
        ycrcb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2YCrCb)
        cb = ycrcb[:, :, 2]
        cr = ycrcb[:, :, 1]
        skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        skin_ratio = skin_mask.sum() / skin_mask.size

        print(f"TEX lbp={lbp_entropy:.2f} skin={skin_ratio:.3f}")

        if lbp_entropy < 5.90:
            result = (False, "display")
        elif lbp_entropy > 6.40 and skin_ratio < 0.40:
            result = (False, "print")
        elif skin_ratio < 0.15:
            result = (False, "display")
        else:
            result = (True, None)

        self._texture_last_result = result
        return result

    @staticmethod
    def _button_qss(color):
        return (
            f"background-color: #1a1a1a; color: {color}; padding: 8px 16px; "
            f"border: 1px solid {color}; font-size: 11px; font-family: 'Consolas';"
        )

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

    def _check_ir(self, ir_frame, rgb_frame, bbox):
        x1, y1, x2, y2 = bbox

        # IR ROI
        ih, iw = ir_frame.shape[:2]
        ix1, iy1 = max(0, x1), max(0, y1)
        ix2, iy2 = min(iw, x2), min(ih, y2)
        ir_roi = ir_frame[iy1:iy2, ix1:ix2]
        if ir_roi.size == 0:
            return False, "no_face"
        ir_gray = cv2.cvtColor(ir_roi, cv2.COLOR_BGR2GRAY) if len(ir_roi.shape) == 3 else ir_roi

        # RGB ROI
        rh, rw = rgb_frame.shape[:2]
        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(rw, x2), min(rh, y2)
        rgb_roi = rgb_frame[ry1:ry2, rx1:rx2]
        if rgb_roi.size == 0:
            return False, "no_face"
        rgb_gray = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2GRAY) if len(rgb_roi.shape) == 3 else rgb_roi

        ir_mean = ir_gray.mean()
        ir_std = ir_gray.std()
        rgb_mean = rgb_gray.mean()
        ratio = rgb_mean / max(ir_mean, 1.0)
        print(f"IR={ir_mean:.1f} std={ir_std:.1f} RGB={rgb_mean:.1f} ratio={ratio:.2f}")

        if ratio > 2.5:
            return False, "display"
        if ir_std < 20:
            return False, "print"
        return True, None

    def _update_user_list(self):
        self.user_list.clear()
        for name in self.db.users.keys():
            self.user_list.addItem(name)

    def register_face(self):
        if self.current_embedding is not None:
            idx = self.db.count() + 1
            name = f"User_{idx:03d}"
            self.db.register(name, self.current_embedding)
            self.reg_label.setText(f"Registered: {self.db.count()}")
            self._update_user_list()
        else:
            self._set_status("WAITING", "#555", "No face to register")

    def delete_face(self):
        selected = self.user_list.currentItem()
        if selected:
            name = selected.text()
            self.db.delete(name)
            self.reg_label.setText(f"Registered: {self.db.count()}")
            self._update_user_list()
        else:
            self._set_status("WAITING", "#555", "Select a user to delete")

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
