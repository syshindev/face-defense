import sys
import argparse
import time
import threading
from collections import deque

import cv2
import numpy as np
from deepface import DeepFace
from insightface.app import FaceAnalysis
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QFrame,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_COLORS = {
    "angry": "#ff3333",
    "disgust": "#00bcd4",
    "fear": "#e040fb",
    "happy": "#00ff41",
    "sad": "#4a9eff",
    "surprise": "#ffdd00",
    "neutral": "#888888",
}

EMOTION_CV_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (212, 188, 0),
    "fear": (255, 0, 255),
    "happy": (0, 255, 0),
    "sad": (255, 154, 74),
    "surprise": (0, 221, 255),
    "neutral": (200, 200, 200),
}


class EmotionWindow(QMainWindow):
    def __init__(self, camera_id=0, smooth=5):
        super().__init__()
        self.setWindowTitle("FACE DEFENSE // EMOTION RECOGNITION")
        self.setMinimumSize(1000, 600)
        self.setStyleSheet(
            "background-color: #0c0c0c; color: #c0c0c0; "
            "font-family: 'Consolas', 'Courier New', monospace;"
        )

        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        self.cap = cv2.VideoCapture(camera_id)
        self.emotion_history = deque(maxlen=smooth)

        self._lock = threading.Lock()
        self._bbox = None
        self._smoothed_emotions = None
        self._last_frame_for_analysis = None
        self._running = True

        self._init_ui()
        self._start_timer()
        self._start_analysis_thread()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #000; border: 2px solid #2a2a2a;")
        main_layout.addWidget(self.camera_label, stretch=3)

        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #111111; border: 2px solid #2a2a2a;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(8)

        title = QLabel("EMOTION RECOGNITION")
        title.setFont(QFont("Consolas", 12, QFont.Bold))
        title.setStyleSheet("color: #ffaa00; border: none;")
        right_layout.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333; border: none;")
        right_layout.addWidget(sep)

        self.emotion_label = QLabel("NO FACE")
        self.emotion_label.setFont(QFont("Consolas", 16, QFont.Bold))
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.emotion_label.setStyleSheet(
            "background-color: #1a1a1a; color: #888; padding: 12px; border: 1px solid #333;"
        )
        right_layout.addWidget(self.emotion_label)

        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Consolas", 11))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: #666; border: none;")
        right_layout.addWidget(self.confidence_label)

        right_layout.addSpacing(10)

        self.bar_labels = {}
        self.bar_fills = {}
        for emo in EMOTIONS:
            row = QHBoxLayout()
            display = emo.upper()
            name = QLabel(f"{display[:8]:>8}")
            name.setFont(QFont("Consolas", 10))
            name.setFixedWidth(70)
            name.setStyleSheet("color: #888; border: none;")
            row.addWidget(name)

            bar_bg = QFrame()
            bar_bg.setFixedHeight(16)
            bar_bg.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
            bar_layout = QHBoxLayout(bar_bg)
            bar_layout.setContentsMargins(0, 0, 0, 0)
            bar_layout.setSpacing(0)

            bar_fill = QFrame()
            bar_fill.setFixedHeight(14)
            bar_fill.setFixedWidth(0)
            bar_fill.setStyleSheet(f"background-color: {EMOTION_COLORS[emo]}; border: none;")
            bar_layout.addWidget(bar_fill, alignment=Qt.AlignLeft)

            row.addWidget(bar_bg, stretch=1)

            pct = QLabel("0%")
            pct.setFont(QFont("Consolas", 10))
            pct.setFixedWidth(40)
            pct.setAlignment(Qt.AlignRight)
            pct.setStyleSheet("color: #666; border: none;")
            row.addWidget(pct)

            right_layout.addLayout(row)
            self.bar_labels[emo] = pct
            self.bar_fills[emo] = (bar_fill, bar_bg)

        right_layout.addStretch()
        main_layout.addWidget(right_panel, stretch=1)

    def _start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

    def _start_analysis_thread(self):
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()

    def _analysis_loop(self):
        while self._running:
            with self._lock:
                frame = self._last_frame_for_analysis
                self._last_frame_for_analysis = None
            if frame is None:
                time.sleep(0.03)
                continue

            try:
                faces = self.face_app.get(frame)
            except Exception:
                continue
            if not faces:
                with self._lock:
                    self._bbox = None
                    self._smoothed_emotions = None
                    self.emotion_history.clear()
                continue

            face = max(faces, key=lambda f: f.det_score)
            bbox = face.bbox.astype(int)
            H, W = frame.shape[:2]
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(W, bbox[2]), min(H, bbox[3])

            with self._lock:
                self._bbox = (x1, y1, x2, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                results = DeepFace.analyze(
                    crop, actions=["emotion"],
                    enforce_detection=False, silent=True,
                    detector_backend="skip",
                )
                raw = results[0]["emotion"] if isinstance(results, list) else results["emotion"]
                vals = [raw.get(e, 0) / 100.0 for e in EMOTIONS]
                with self._lock:
                    self.emotion_history.append(vals)
                    avg = np.mean(self.emotion_history, axis=0)
                    self._smoothed_emotions = dict(zip(EMOTIONS, avg))
            except Exception:
                pass

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        with self._lock:
            self._last_frame_for_analysis = frame.copy()
            bbox = self._bbox
            emotions = self._smoothed_emotions

        if bbox is None:
            self.emotion_label.setText("NO FACE")
            self.emotion_label.setStyleSheet(
                "background-color: #1a1a1a; color: #888; padding: 12px; border: 1px solid #333;"
            )
            self.confidence_label.setText("")
            for emo in EMOTIONS:
                self.bar_labels[emo].setText("0%")
                self.bar_fills[emo][0].setFixedWidth(0)
        else:
            x1, y1, x2, y2 = bbox

            if emotions:
                top_emotion = max(emotions, key=emotions.get)
                score = emotions[top_emotion]
                color = EMOTION_COLORS.get(top_emotion, "#888")
                cv_color = EMOTION_CV_COLORS.get(top_emotion, (200, 200, 200))
                display_name = top_emotion.upper()

                self.emotion_label.setText(display_name)
                self.emotion_label.setStyleSheet(
                    f"background-color: #1a1a1a; color: {color}; padding: 12px; "
                    f"border: 1px solid {color};"
                )
                self.confidence_label.setText(f"{score:.0%}")

                for emo in EMOTIONS:
                    val = emotions[emo]
                    self.bar_labels[emo].setText(f"{val:.0%}")
                    bar_fill, bar_bg = self.bar_fills[emo]
                    max_w = bar_bg.width() - 2
                    bar_fill.setFixedWidth(max(0, int(val * max_w)))

                cv2.rectangle(frame, (x1, y1), (x2, y2), cv_color, 2)
                cv2.putText(frame, f"{display_name} {score:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cv_color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(img).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled)

    def closeEvent(self, event):
        self._running = False
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = EmotionWindow(camera_id=args.camera, smooth=args.smooth)
    window.show()
    sys.exit(app.exec_())
