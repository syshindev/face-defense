"""Camera abstraction layer for Face Defense.

Provides a unified interface for:
- DualCVCamera: legacy 2-camera setup (RGB + IR via cv2.VideoCapture)
- RealSenseCamera: Intel RealSense D435 (RGB + IR + Depth in one device)
- create_camera(): factory that picks the right backend
"""

import cv2
import numpy as np


class DualCVCamera:
    """Legacy two-camera setup using cv2.VideoCapture."""

    def __init__(self, color_id=0, ir_id=-1):
        self.is_d435 = False
        self._cap = cv2.VideoCapture(color_id)
        self._ir_cap = None
        self.has_ir = ir_id >= 0
        if self.has_ir:
            self._ir_cap = cv2.VideoCapture(ir_id)
            if not self._ir_cap.isOpened():
                self.has_ir = False
                self._ir_cap = None

    def read_color(self):
        """Return (success, bgr_frame)."""
        return self._cap.read()

    def read_ir(self):
        """Return (success, ir_frame). Falls back to (False, None) if no IR camera."""
        if self._ir_cap is not None:
            return self._ir_cap.read()
        return False, None

    def read_depth(self):
        """Not available for cv2 cameras."""
        return False, None

    def get_depth_at(self, x, y):
        """Not available for cv2 cameras. Returns 0.0."""
        return 0.0

    def release(self):
        self._cap.release()
        if self._ir_cap is not None:
            self._ir_cap.release()


class RealSenseCamera:
    """Intel RealSense D435 — RGB + IR + Depth from a single device."""

    def __init__(self):
        import pyrealsense2 as rs

        self.is_d435 = True
        self.has_ir = True
        self._rs = rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._pipeline.start(config)

        self._align = rs.align(rs.stream.color)

        # Per-frame cache (populated by read_color, consumed by read_ir/read_depth)
        self._color_frame = None
        self._ir_frame = None
        self._depth_frame = None

    def read_color(self):
        """Grab a frameset, cache IR/Depth, return (success, bgr_frame)."""
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except Exception:
            self._color_frame = self._ir_frame = self._depth_frame = None
            return False, None

        aligned = self._align.process(frames)

        cf = aligned.get_color_frame()
        self._ir_frame = aligned.get_infrared_frame(1)
        self._depth_frame = aligned.get_depth_frame()

        if not cf:
            self._color_frame = None
            return False, None

        self._color_frame = cf
        return True, np.asanyarray(cf.get_data())

    def read_ir(self):
        """Return (success, ir_gray) from the cached frameset."""
        if self._ir_frame is None:
            return False, None
        ir = np.asanyarray(self._ir_frame.get_data())
        return True, ir

    def read_depth(self):
        """Return (success, depth_array_u16) from the cached frameset."""
        if self._depth_frame is None:
            return False, None
        depth = np.asanyarray(self._depth_frame.get_data())
        return True, depth

    def get_depth_at(self, x, y):
        """Return depth in meters at pixel (x, y). 0.0 if unavailable."""
        if self._depth_frame is None:
            return 0.0
        try:
            return self._depth_frame.get_distance(int(x), int(y))
        except Exception:
            return 0.0

    def release(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass


def create_camera(use_d435=False, color_id=0, ir_id=-1):
    """Factory: return the appropriate camera backend.

    If use_d435 is True, try RealSenseCamera first; fall back to DualCVCamera
    on import/init failure.
    """
    if use_d435:
        try:
            cam = RealSenseCamera()
            return cam
        except Exception as e:
            print(f"[camera] D435 init failed ({e}), falling back to cv2")
    return DualCVCamera(color_id, ir_id)
