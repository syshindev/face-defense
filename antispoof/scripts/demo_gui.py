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
    QVBoxLayout, QHBoxLayout, QFrame, QListWidget, QDialog, QLineEdit,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

from shared.face_utils import FaceDatabase, compute_ear, LEFT_EYE, RIGHT_EYE
from shared.camera import create_camera


class MainWindow(QMainWindow):
    def __init__(self, camera_id=0, ir_camera_id=-1, min_face=0,
                 use_d435=False, max_depth=0.0):
        super().__init__()
        self.min_face = min_face
        self.max_depth = max_depth
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
            max_num_faces=3,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.db = FaceDatabase()

        # Camera
        self.camera = create_camera(use_d435=use_d435,
                                    color_id=camera_id, ir_id=ir_camera_id)
        self.has_ir = self.camera.has_ir
        self.blink_mode_enabled = True
        self.texture_mode_enabled = True
        self.blur_mode_enabled = False
        self._current_bbox = None
        self._texture_frame_count = 0
        self._texture_last_result = (True, None)
        self._spoof_history = []
        self._SPOOF_WINDOW = 10
        self._no_face_since = 0.0

        # Blink state
        self.EAR_CLOSE = 0.33
        self.EAR_OPEN = 0.34
        self.BLINK_TIMEOUT = 5.0
        self.was_closed = False
        self.close_time = 0.0
        self.last_blink_time = 0.0

        # Current face for registration
        self.current_embedding = None

        self._scale = 1.0
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
        self._right_layout = right_layout
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(8)

        # Title
        self.title_label = QLabel("ACCESS CONTROL")
        self.title_label.setFont(QFont("Consolas", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignLeft)
        self.title_label.setStyleSheet("color: #ffaa00; border: none;")
        right_layout.addWidget(self.title_label)

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

        # Distance (D435 only)
        if self.camera.is_d435:
            self.depth_label = self._create_info_label("Distance", "-")
            right_layout.addWidget(self.depth_label)

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

        self.btn_blur_toggle = QPushButton()
        self.btn_blur_toggle.clicked.connect(self.toggle_blur_mode)
        right_layout.addWidget(self.btn_blur_toggle)
        self._refresh_blur_button()

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
        s = self._scale
        t = int(10 * s)
        v = int(13 * s)
        widget = QLabel(f"<span style='color:#666; font-size:{t}px;'>[ {title} ]</span><br>"
                        f"<span style='font-size:{v}px; color:#aaa;'>{value}</span>")
        widget.setTextFormat(Qt.RichText)
        widget.setStyleSheet("border: none;")
        return widget

    def _update_info_label(self, label, title, value, color="#aaa"):
        s = self._scale
        t = int(10 * s)
        v = int(13 * s)
        label.setText(
            f"<span style='color:#666; font-size:{t}px;'>[ {title} ]</span><br>"
            f"<span style='color:{color}; font-size:{v}px;'>{value}</span>"
        )

    def _start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.camera.read_color()
        if not ret:
            return

        now = time.time()
        faces = self.face_app.get(frame)

        if len(faces) == 0:
            self._current_bbox = None
            self._reset_panel("No Face Detected")
            if self._no_face_since == 0.0:
                self._no_face_since = now
            elif now - self._no_face_since > 0.5:
                self._spoof_history.clear()
            self.current_embedding = None
        else:
            self._no_face_since = 0.0

            # Distance limit: ignore faces smaller than min_face (far away)
            if self.min_face > 0:
                faces = [f for f in faces
                         if (f.bbox[2] - f.bbox[0]) >= self.min_face]
                if not faces:
                    self._reset_panel("Out of Range")
                    self._show_frame(frame)
                    return

            # D435: depth-based filter + selection (strip far faces before picking)
            if self.camera.is_d435:
                face_dists = []
                for f in faces:
                    fx1, fy1, fx2, fy2 = f.bbox.astype(int)
                    fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    d = self.camera.get_depth_at(fcx, fcy)
                    face_dists.append((f, d))

                # Keep only faces within max_depth
                if self.max_depth > 0:
                    nearby = [(f, d) for f, d in face_dists if 0 < d <= self.max_depth]
                else:
                    nearby = [(f, d) for f, d in face_dists if d > 0]

                if not nearby:
                    self._reset_panel("Out of Range")
                    self._show_frame(frame)
                    return

                # Pick the closest face
                face, dist = min(nearby, key=lambda x: x[1])
                if hasattr(self, 'depth_label'):
                    self._update_info_label(self.depth_label, "Distance",
                                            f"{dist:.2f} m", "#ffaa00")
            else:
                # Non-D435: pick the largest face by bbox width
                face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])

            x1, y1, x2, y2 = face.bbox.astype(int)
            self._current_bbox = (x1, y1, x2, y2)
            self.current_embedding = face.normed_embedding

            # Liveness check
            is_live, spoof_type = self._check_liveness(frame, now, (x1, y1, x2, y2))

            if not is_live:
                # Spoof
                if spoof_type == "display":
                    self._set_status("UNAUTHORIZED", "#ff3333", "Display Attack Detected")
                elif spoof_type == "print":
                    self._set_status("UNAUTHORIZED", "#ff3333", "Print Attack Detected")
                else:
                    self._set_status("UNAUTHORIZED", "#ff3333", "Liveness Check Failed")
                self._update_info_label(self.user_label, "User", "-")
                self._update_info_label(self.sim_label, "Similarity", "-")
                self._update_info_label(self.live_label, "Liveness", "FAIL", "#ff3333")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                self._update_info_label(self.live_label, "Liveness", "PASS", "#00ff41")

                if self.db.count() == 0:
                    self._set_status("DENIED", "#ffaa00", "No Registered Users")
                    self._update_info_label(self.user_label, "User", "-")
                    self._update_info_label(self.sim_label, "Similarity", "-")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                else:
                    name, sim = self.db.recognize(self.current_embedding)
                    self._update_info_label(self.sim_label, "Similarity", f"{sim:.2f}",
                                            "#00ff41" if sim >= 0.4 else "#ff3333")

                    if sim >= 0.4:
                        self._set_status("AUTHORIZED", "#00ff41", "Authentication Success")
                        self._update_info_label(self.user_label, "User", name)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        self._set_status("DENIED", "#ff3333", "Unregistered User")
                        self._update_info_label(self.user_label, "User", "-")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        self._show_frame(frame)

    def _show_frame(self, frame):
        if self.blur_mode_enabled and self._current_bbox is not None:
            blurred = cv2.GaussianBlur(frame, (51, 51), 0)
            x1, y1, x2, y2 = self._current_bbox
            h, w = frame.shape[:2]
            margin = int((x2 - x1) * 0.3)
            bx1 = max(0, x1 - margin)
            by1 = max(0, y1 - margin)
            bx2 = min(w, x2 + margin)
            by2 = min(h, y2 + margin)
            blurred[by1:by2, bx1:bx2] = frame[by1:by2, bx1:bx2]
            frame = blurred
        elif self.blur_mode_enabled:
            frame = cv2.GaussianBlur(frame, (51, 51), 0)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(img).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled)

    def _reset_panel(self, reason="No Face Detected"):
        """Reset the panel to standby."""
        self._set_status("STANDBY", "#555", reason)
        self._update_info_label(self.user_label, "User", "-")
        self._update_info_label(self.sim_label, "Similarity", "-")
        self._update_info_label(self.live_label, "Liveness", "-")
        if hasattr(self, 'depth_label'):
            self._update_info_label(self.depth_label, "Distance", "-")

    def _set_status(self, text, color, result):
        s = self._scale
        pad = int(10 * s)
        fs = int(13 * s)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"background-color: #1a1a1a; color: {color}; padding: {pad}px; "
            f"border: 1px solid {color}; font-size: {fs}px; font-weight: bold;"
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

        # IR check (always on when available)
        if not is_spoof and self.has_ir:
            ir_ret, ir_frame = self.camera.read_ir()
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
        if (len(self._spoof_history) >= self._SPOOF_WINDOW
                and spoof_count >= 5):
            return False, spoof_reason or "display"

        # Blink check
        if self.blink_mode_enabled:
            blink_live, blink_reason = self._check_blink(frame, now, bbox)
            if not blink_live:
                return False, blink_reason

        return True, None

    def toggle_blur_mode(self):
        self.blur_mode_enabled = not self.blur_mode_enabled
        self._refresh_blur_button()

    def _refresh_blur_button(self):
        if self.blur_mode_enabled:
            text, color = "[ BLUR: ON ]", "#ffaa00"
        else:
            text, color = "[ BLUR: OFF ]", "#666"
        self.btn_blur_toggle.setText(text)
        self.btn_blur_toggle.setStyleSheet(self._button_qss(color))

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
        """LBP texture + skin color + HSV saturation analysis for spoof detection."""
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

        # 1. LBP entropy — real skin has higher entropy than print/display
        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-8
        lbp_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

        # 2. Skin color ratio in YCbCr
        roi_resized = cv2.resize(roi, (64, 64))
        ycrcb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2YCrCb)
        cb = ycrcb[:, :, 2]
        cr = ycrcb[:, :, 1]
        skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        skin_ratio = skin_mask.sum() / skin_mask.size

        # 3. HSV saturation — displays (backlit) are high-sat, prints (ink) are low-sat
        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        sat_mean = hsv[:, :, 1].mean()

        if os.environ.get("DEBUG"):
            print(f"TEX lbp={lbp_entropy:.2f} skin={skin_ratio:.3f} sat={sat_mean:.1f}")

        if sat_mean > 120:
            # Very high saturation → display (backlit, phone close-up)
            result = (False, "display")
        elif sat_mean < 55:
            # Very low saturation → print (ink/paper, distance-agnostic)
            result = (False, "print")
        elif skin_ratio < 0.20:
            result = (False, "print")
        elif lbp_entropy < 5.50:
            # Mid saturation + low entropy → split by saturation
            if sat_mean > 80:
                result = (False, "display")
            else:
                result = (False, "print")
        else:
            result = (True, None)

        self._texture_last_result = result
        return result

    def _button_qss(self, color):
        s = self._scale
        pv, ph = int(8 * s), int(16 * s)
        fs = int(11 * s)
        return (
            f"background-color: #1a1a1a; color: {color}; padding: {pv}px {ph}px; "
            f"border: 1px solid {color}; font-size: {fs}px; font-family: 'Consolas';"
        )

    def _check_blink(self, frame, now, bbox=None):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb)

        if mesh_results.multi_face_landmarks:
            # Among multiple faces, pick the one matching bbox
            landmarks = None
            if bbox is not None:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = bbox
                margin = int((x2 - x1) * 0.5)
                best_dist = float('inf')
                bcx, bcy = (x1 + x2) // 2, (y1 + y2) // 2
                for face_lm in mesh_results.multi_face_landmarks:
                    nose = face_lm.landmark[1]
                    nx, ny = int(nose.x * w), int(nose.y * h)
                    if (x1 - margin <= nx <= x2 + margin and
                            y1 - margin <= ny <= y2 + margin):
                        dist = abs(nx - bcx) + abs(ny - bcy)
                        if dist < best_dist:
                            best_dist = dist
                            landmarks = face_lm.landmark
            else:
                landmarks = mesh_results.multi_face_landmarks[0].landmark

            if landmarks is None:
                # No face matched bbox → fall back to last blink time
                time_since = now - self.last_blink_time if self.last_blink_time > 0 else 999
                if time_since < self.BLINK_TIMEOUT:
                    return True, None
                return False, "blink"

            left_ear = compute_ear(landmarks, LEFT_EYE)
            right_ear = compute_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            # Adapt threshold to face size:
            # close (200px+): 0.31, far (100px): 0.35
            if bbox is not None:
                face_w = bbox[2] - bbox[0]
                scale = max(0.0, min(1.0, (200 - face_w) / 100.0))
                ear_close = 0.31 + scale * 0.04
                ear_open = ear_close + 0.02
            else:
                ear_close = self.EAR_CLOSE
                ear_open = self.EAR_OPEN

            if os.environ.get("DEBUG"):
                print(f"EAR={avg_ear:.3f} thr={ear_close:.2f}")

            if avg_ear < ear_close:
                if not self.was_closed:
                    self.was_closed = True
                    self.close_time = now
            elif self.was_closed and avg_ear >= ear_open:
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
        if os.environ.get("DEBUG"):
            print(f"IR={ir_mean:.1f} std={ir_std:.1f} RGB={rgb_mean:.1f} ratio={ratio:.2f}")

        if self.camera.is_d435:
            # D435: depth flatness + ratio joint decision
            # Real faces are 3D (nose/cheek/ear relief); prints/screens are flat
            dep_ret, depth_map = self.camera.read_depth()
            depth_std_val = 0.0
            if dep_ret:
                dep_roi = depth_map[iy1:iy2, ix1:ix2]
                valid = dep_roi[dep_roi > 0].astype(np.float32)
                if valid.size > 100:
                    depth_std_val = valid.std()
            if os.environ.get("DEBUG"):
                print(f"DEPTH std={depth_std_val:.1f}")

            is_flat = depth_std_val < 8.0
            is_ratio_suspect = ratio > 0.80

            if is_flat and is_ratio_suspect:
                # Saturation-based subtype (more stable than ratio)
                roi_color = rgb_frame[ry1:ry2, rx1:rx2]
                if roi_color.size > 0:
                    hsv_roi = cv2.cvtColor(cv2.resize(roi_color, (64, 64)),
                                           cv2.COLOR_BGR2HSV)
                    ir_sat = hsv_roi[:, :, 1].mean()
                else:
                    ir_sat = 0
                if ir_sat > 80:
                    return False, "display"
                return False, "print"
            else:
                return True, None
        else:
            # Legacy USB IR: RGB/IR ratio decision
            if ratio < 1.00:
                return False, "print"
            if ratio > 2.00:
                return False, "display"
            return True, None

    def _update_user_list(self):
        self.user_list.clear()
        for name in self.db.users.keys():
            self.user_list.addItem(name)

    def _show_register_dialog(self):
        s = self._scale
        fs = int(14 * s)
        pad = int(12 * s)
        dlg = QDialog(self)
        dlg.setWindowTitle("Register User")
        dlg.setStyleSheet(
            f"background-color: #111111; color: #c0c0c0; "
            f"font-family: 'Consolas'; font-size: {fs}px;"
        )
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(pad * 2, pad * 2, pad * 2, pad * 2)
        layout.setSpacing(pad)

        label = QLabel("Enter name (blank = auto)")
        label.setStyleSheet(f"color: #ffaa00; font-size: {fs}px;")
        layout.addWidget(label)

        line = QLineEdit()
        line.setStyleSheet(
            f"background-color: #0c0c0c; color: #fff; border: 1px solid #ffaa00; "
            f"padding: {pad}px; font-size: {fs}px;"
        )
        line.setPlaceholderText("e.g. John Doe")
        layout.addWidget(line)

        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.setStyleSheet(self._button_qss("#00ff41"))
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(self._button_qss("#666"))
        btn_cancel.clicked.connect(dlg.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        dlg.setMinimumWidth(int(300 * s))
        if dlg.exec_() == QDialog.Accepted:
            return True, line.text()
        return False, ""

    def register_face(self):
        if self.current_embedding is not None:
            ok, name = self._show_register_dialog()
            if ok:
                name = name.strip() if name.strip() else f"User_{self.db.count() + 1:03d}"
                self.db.register(name, self.current_embedding)
                self.reg_label.setText(f"Registered: {self.db.count()}")
                self._update_user_list()
                self._set_status("AUTHORIZED", "#00ff41", f"{name} registered")
        else:
            self._set_status("STANDBY", "#555", "No face to register")

    def delete_face(self):
        selected = self.user_list.currentItem()
        if selected:
            name = selected.text()
            self.db.delete(name)
            self.reg_label.setText(f"Registered: {self.db.count()}")
            self._update_user_list()
        else:
            self._set_status("STANDBY", "#555", "Select a user to delete")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        h = event.size().height()
        new_scale = max(1.0, h / 600.0)
        if abs(new_scale - self._scale) > 0.05:
            self._scale = new_scale
            self._apply_scale()

    def _apply_scale(self):
        s = self._scale
        margin = int(20 * s)
        self._right_layout.setContentsMargins(margin, margin, margin, margin)
        self._right_layout.setSpacing(int(8 * s))

        self.title_label.setFont(QFont("Consolas", int(12 * s), QFont.Bold))
        self.status_label.setFont(QFont("Consolas", int(13 * s), QFont.Bold))
        self.reg_label.setFont(QFont("Consolas", int(10 * s)))

        self.user_list.setStyleSheet(
            f"background-color: #0c0c0c; color: #ffaa00; border: 1px solid #333; "
            f"font-size: {int(11 * s)}px; padding: {int(2 * s)}px; font-family: 'Consolas';"
        )

        # Refresh buttons
        self._refresh_blur_button()
        self._refresh_blink_button()
        self._refresh_texture_button()
        self.btn_register.setStyleSheet(self._button_qss("#ffaa00"))
        self.btn_delete.setStyleSheet(self._button_qss("#ff3333"))

    def closeEvent(self, event):
        self.camera.release()
        event.accept()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ir_camera", type=int, default=-1)
    parser.add_argument("--min_face", type=int, default=120,
                        help="Minimum face width in px; smaller faces ignored (distance limit)")
    parser.add_argument("--d435", action="store_true",
                        help="Use Intel RealSense D435 (RGB + IR + Depth integrated)")
    parser.add_argument("--max_depth", type=float, default=0.0,
                        help="D435 max allowed distance in meters; 0 = unlimited")
    args = parser.parse_args()
    # D435 enforces distance via depth, so min_face is unnecessary
    if args.d435 and args.min_face == 120:
        args.min_face = 0

    app = QApplication(sys.argv)
    window = MainWindow(camera_id=args.camera, ir_camera_id=args.ir_camera,
                        min_face=args.min_face, use_d435=args.d435,
                        max_depth=args.max_depth)
    window.show()
    sys.exit(app.exec_())
