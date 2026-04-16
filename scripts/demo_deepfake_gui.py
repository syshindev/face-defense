import os
import sys
import argparse
import tempfile

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))

from demo_deepfake import (
    build_face_app, load_model, crop_face, predict, verdict,
    update_tracks_from_frame, draw_tracks,
)

MIXED_LOW = 0.3
MIXED_HIGH = 0.7
FACE_MARGIN = 0.2
IOU_MATCH = 0.3
TINT_ALPHA = 0.18
TINT_BORDER_PX = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio GUI for deepfake detection")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/legacy_xception_v2_best.pth")
    parser.add_argument("--model", type=str, default="legacy_xception")
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--share", action="store_true", help="Expose a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


ARGS = parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  Model: {ARGS.model}  |  Ckpt: {ARGS.checkpoint}")
FACE_APP = build_face_app(ARGS.det_thresh)
MODEL = load_model(ARGS.model, ARGS.checkpoint, DEVICE)


def status_badge(verdict_text, color):
    return (
        f'<div style="display:block;width:100%;box-sizing:border-box;'
        f'background:#1a1a1a;border:1px solid {color};color:{color};'
        f'padding:12px 16px;font-weight:700;letter-spacing:0.08em;'
        f'text-transform:uppercase;text-align:center;'
        f'font-family:Consolas,monospace;font-size:14px;line-height:1.5;'
        f'margin-bottom:12px;">{verdict_text}</div>'
    )


def kv(label, value, color="#c0c0c0"):
    return (
        f'<div style="margin:6px 0;">'
        f'<span style="color:#666;font-size:10px;letter-spacing:0.08em;'
        f'text-transform:uppercase;">[ {label} ]</span><br>'
        f'<span style="color:{color};font-size:14px;'
        f'font-family:Consolas,monospace;">{value}</span>'
        f'</div>'
    )


VERDICT_COLORS_RGB = {
    "REAL": (0, 255, 65),     # #00ff41
    "FAKE": (255, 51, 51),    # #ff3333
    "MIXED": (0, 212, 255),   # #00d4ff (cyan — uncertain / partial)
}


def apply_verdict_tint(img_rgb, v, thickness=TINT_BORDER_PX, alpha=TINT_ALPHA):
    if v not in VERDICT_COLORS_RGB:
        return img_rgb
    color = VERDICT_COLORS_RGB[v]
    overlay = np.full_like(img_rgb, color)
    tinted = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
    cv2.rectangle(tinted, (0, 0), (tinted.shape[1] - 1, tinted.shape[0] - 1),
                  color, thickness)
    return tinted


def analyze_image(image_rgb, threshold):
    if image_rgb is None:
        html = status_badge("AWAITING INPUT", "#666") + kv("Status", "Upload an image to run analysis.")
        return None, html

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = FACE_APP.get(bgr)
    if not faces:
        html = (
            status_badge("NO FACE DETECTED", "#888")
            + kv("Status", "No face found in the input. Try a clearer frontal face.")
        )
        return image_rgb, html

    face = max(faces, key=lambda f: f.det_score)
    crop, (x1, y1, x2, y2) = crop_face(bgr, face.bbox, FACE_MARGIN)
    if crop.size == 0:
        html = status_badge("FACE CROP FAILED", "#ffaa00") + kv("Status", "Bounding box produced an empty region.")
        return image_rgb, html

    score = predict(MODEL, crop, ARGS.image_size, DEVICE)
    vis_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    v = verdict(score, threshold)
    badge_color = "#ff3333" if v == "FAKE" else "#00ff41"
    tinted = apply_verdict_tint(vis_rgb, v)
    html = (
        status_badge(v, badge_color)
        + kv("P(fake)", f"{score:.4f}", badge_color)
        + kv("Detection confidence", f"{face.det_score:.3f}")
    )
    return tinted, html


def analyze_video(video_path, threshold, frame_step, smooth_window, save_annotated):
    if not video_path:
        html = status_badge("AWAITING INPUT", "#666") + kv("Status", "Upload a video to run analysis.")
        return None, None, html

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        html = status_badge("VIDEO OPEN FAILED", "#ff3333") + kv("Status", "Could not open the provided file.")
        return None, None, html

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    out_path = None
    if save_annotated:
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    raw_scores = []
    times = []
    tracks = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            tracks, scores = update_tracks_from_frame(
                frame, tracks, FACE_APP, MODEL, ARGS.image_size, FACE_MARGIN,
                IOU_MATCH, smooth_window, DEVICE,
            )
            raw_scores.extend(scores)
            times.extend([frame_idx / fps] * len(scores))

        draw_tracks(frame, tracks, threshold)

        if writer:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    if not raw_scores:
        html = (
            status_badge("NO FACE DETECTED", "#888")
            + kv("Status", "No face found in any sampled frame.")
            + kv("Try", "A video with clearer, larger frontal faces, or lower --frame_step")
        )
        return None, None, html

    arr = np.array(raw_scores)
    mean_p = float(arr.mean())
    median_p = float(np.median(arr))
    n_fake = int((arr >= threshold).sum())
    fake_ratio = n_fake / len(arr)
    if fake_ratio >= MIXED_HIGH:
        summary_verdict = "FAKE"
    elif fake_ratio >= MIXED_LOW:
        summary_verdict = "MIXED"
    else:
        summary_verdict = "REAL"

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, arr, "b-", linewidth=1, alpha=0.8)
    ax.axhline(threshold, color="red", linestyle="--", alpha=0.5, label=f"threshold={threshold}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(fake)")
    ax.set_title("Deepfake Probability Timeline")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    badge_color = {"FAKE": "#ff3333", "MIXED": "#00d4ff", "REAL": "#00ff41"}[summary_verdict]
    badge_text = {
        "FAKE": "FAKE",
        "MIXED": "MIXED  —  CONTAINS DEEPFAKE SEGMENTS",
        "REAL": "REAL",
    }[summary_verdict]

    if out_path and summary_verdict in VERDICT_COLORS_RGB:
        color_rgb = VERDICT_COLORS_RGB[summary_verdict]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        tint_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        cap2 = cv2.VideoCapture(out_path)
        writer2 = cv2.VideoWriter(tint_path, fourcc, fps, (w, h))
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            overlay = np.full_like(frame, color_bgr)
            tinted = cv2.addWeighted(frame, 1 - TINT_ALPHA, overlay, TINT_ALPHA, 0)
            cv2.rectangle(tinted, (0, 0), (w - 1, h - 1), color_bgr, TINT_BORDER_PX)
            writer2.write(tinted)
        cap2.release()
        writer2.release()
        try:
            os.unlink(out_path)
        except OSError:
            pass
        out_path = tint_path
    html = (
        status_badge(badge_text, badge_color)
        + kv("Video", f"{total} frames @ {fps:.1f} fps, step={frame_step}")
        + kv("Face predictions", str(len(arr)))
        + kv(f"Flagged fake (>= {threshold})",
             f"{n_fake}/{len(arr)} ({100*fake_ratio:.1f}%)", badge_color)
        + kv("P(fake) mean", f"{mean_p:.4f}")
        + kv("P(fake) median", f"{median_p:.4f}")
        + kv("Min / Max", f"{arr.min():.4f} / {arr.max():.4f}")
    )
    return out_path, fig, html


TERMINAL_THEME = gr.themes.Base(
    primary_hue="amber",
    neutral_hue="gray",
    font=(gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"),
).set(
    body_background_fill="#0c0c0c",
    body_text_color="#c0c0c0",
    background_fill_primary="#111111",
    background_fill_secondary="#0c0c0c",
    block_background_fill="#111111",
    block_border_color="#2a2a2a",
    block_border_width="1px",
    block_label_background_fill="#1a1a1a",
    block_label_text_color="#888",
    block_title_text_color="#888",
    input_background_fill="#0c0c0c",
    input_border_color="#2a2a2a",
    button_primary_background_fill="#1a1a1a",
    button_primary_background_fill_hover="#252525",
    button_primary_text_color="#ffaa00",
    button_primary_border_color="#ffaa00",
    button_secondary_background_fill="#1a1a1a",
    button_secondary_text_color="#c0c0c0",
    button_secondary_border_color="#333333",
    slider_color="#ffaa00",
    border_color_accent="#ffaa00",
    border_color_primary="#2a2a2a",
    color_accent="#ffaa00",
    color_accent_soft="#3a2a00",
    link_text_color="#ffaa00",
)

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Consolas', 'JetBrains Mono', 'Courier New', monospace !important;
}
footer { display: none !important; }
h1 {
    color: #e8e8e8 !important;
    font-weight: 700;
    font-size: 28px !important;
    letter-spacing: 0.02em;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 12px;
    margin-bottom: 0 !important;
    position: relative;
    padding-left: 18px;
}
h1::before {
    content: "";
    position: absolute;
    left: 0;
    top: 4px;
    bottom: 16px;
    width: 5px;
    background: #5a7a9a;
}
h2, h3 { color: #ffaa00 !important; letter-spacing: 0.02em; }

/* kill roundness — terminal look */
.gradio-container *, button, input, .block, .form {
    border-radius: 2px !important;
}

/* buttons */
button.primary {
    border: 1px solid #ffaa00 !important;
    background: #1a1a1a !important;
    color: #ffaa00 !important;
    font-family: inherit !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
button.primary:hover { background: #252525 !important; }
button:not(.primary) {
    border: 1px solid #333 !important;
    background: #1a1a1a !important;
    color: #c0c0c0 !important;
}

/* tabs */
.tabs > .tab-nav {
    border-bottom: 1px solid #2a2a2a !important;
}
.tabs > .tab-nav > button {
    color: #666 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 12px;
}
.tabs > .tab-nav > button.selected {
    color: #ffaa00 !important;
    border-bottom: 2px solid #ffaa00 !important;
}

/* labels above components — uppercase tracking (color already set by theme) */
.block .block-label, label span, .gr-input-label {
    text-transform: uppercase;
    font-size: 10px !important;
    letter-spacing: 0.08em;
}

/* blockquote = note box (shares the title accent color) */
blockquote {
    border-left: 3px solid #5a7a9a !important;
    background: #141414 !important;
    color: #9a9a9a !important;
    padding: 8px 12px !important;
    margin: 10px 0;
    font-size: 0.92em;
}

/* slider track */
input[type='range'] { accent-color: #ffaa00; }

/* hide number-input spinner arrows */
input[type='number']::-webkit-inner-spin-button,
input[type='number']::-webkit-outer-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}
input[type='number'] {
    -moz-appearance: textfield !important;
}


/* inline code spans — steel blue (shares the title accent color) */
code, .prose code, p code, li code {
    background: #1a1a1a !important;
    color: #5a7a9a !important;
    font-family: inherit !important;
    padding: 1px 6px !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 2px !important;
    font-size: 0.95em !important;
}

/* code block (triple backtick) */
pre, pre code {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
    color: #c0c0c0 !important;
    padding: 8px 12px !important;
}

/* result panel — brute-force every wrapper to full row width */
.result-panel.block {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    min-height: 0 !important;
}
.result-panel, .result-panel * {
    max-width: none !important;
    box-sizing: border-box !important;
}
.result-panel > *,
.result-panel > * > *,
.result-panel > * > * > * {
    width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}

/* plot panel — hidden until a figure is rendered */
.result-plot:not(:has(img, svg, canvas)) { display: none !important; }

/* keep loading overlay in image/video drop zones, remove everywhere else */
.result-panel .progress, .result-panel .progress-level,
.result-panel .progress-text, .result-panel .progress-bar,
.result-plot .progress, .result-plot .progress-level,
.result-plot .progress-text, .result-plot .progress-bar,
.meta-text, .eta-bar, .status-bar, .loading-wrap, .wait-indicator {
    display: none !important;
}

/* override Gradio's pending/wait/loading visual on the primary button so
   no spinner/animation appears inside it while analysis runs */
button.primary[disabled], button.primary.pending,
button.primary.wait, button.primary.generating,
button.primary.loading, button.primary.processing {
    background: #1a1a1a !important;
    color: #ffaa00 !important;
    border: 1px solid #ffaa00 !important;
    opacity: 0.6;
    animation: none !important;
}
/* hide only injected SVG/DIV children (spinner, loader) — keep the text label */
button.primary[disabled] > svg, button.primary[disabled] > div,
button.primary.pending > svg, button.primary.pending > div,
button.primary.wait > svg, button.primary.wait > div,
button.primary.generating > svg, button.primary.generating > div,
button.primary.loading > svg, button.primary.loading > div,
button.primary.processing > svg, button.primary.processing > div {
    display: none !important;
}
button.primary[disabled]::before, button.primary[disabled]::after,
button.primary.pending::before, button.primary.pending::after,
button.primary.wait::before, button.primary.wait::after,
button.primary.generating::before, button.primary.generating::after,
button.primary.loading::before, button.primary.loading::after {
    content: none !important;
    display: none !important;
    animation: none !important;
}

"""

DOT = lambda c: f'<span style="display:inline-block;width:10px;height:10px;background:{c};margin:0 4px 0 0;vertical-align:middle;"></span>'
LEGEND_HTML = (
    f'<div style="display:flex;gap:20px;margin:6px 0 14px 0;font-size:11px;'
    f'letter-spacing:0.08em;text-transform:uppercase;color:#888;font-family:Consolas,monospace;">'
    f'<span>{DOT("#00ff41")}REAL</span>'
    f'<span>{DOT("#ff3333")}FAKE</span>'
    f'<span>{DOT("#00d4ff")}MIXED</span>'
    f'<span>{DOT("#666")}NO INPUT / NO FACE</span>'
    f'</div>'
)

with gr.Blocks(title="FACE DEFENSE // DEEPFAKE", theme=TERMINAL_THEME, css=CUSTOM_CSS) as app:
    gr.Markdown("# Deepfake Detector")
    gr.HTML(LEGEND_HTML)
    gr.Markdown(
        "Upload an image or a video. The model (XceptionNet v2, trained on "
        "`ff-celebdf-frames`) predicts **P(fake)** per face.  \n"
        "> Works best on FaceForensics++ / Celeb-DF style face-swap deepfakes. "
        "Out-of-distribution inputs (StyleGAN whole-image generation, heavily "
        "compressed YouTube clips) will show lower accuracy."
    )

    with gr.Tab("Image"):
        with gr.Row():
            img_in = gr.Image(label="Input image", type="numpy")
            img_out = gr.Image(label="Annotated output", type="numpy")
        img_thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Threshold")
        img_result = gr.HTML(elem_classes=["result-panel"])
        img_btn = gr.Button("[ ANALYZE IMAGE ]", variant="primary")
        img_btn.click(analyze_image, inputs=[img_in, img_thresh],
                      outputs=[img_out, img_result], show_progress="full")

    with gr.Tab("Video"):
        with gr.Row():
            vid_in = gr.Video(label="Input video")
            vid_out = gr.Video(label="Annotated output")
        vid_thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Threshold")
        vid_step = gr.Slider(5, 60, value=15, step=5, label="Frame step (sample every N frames)")
        vid_smooth = gr.Slider(1, 10, value=5, step=1, label="Smoothing window (samples)")
        vid_save = gr.Checkbox(value=True, label="Save annotated video")
        vid_plot = gr.Plot(label="Deepfake Probability Timeline", elem_classes=["result-plot"])
        vid_result = gr.HTML(elem_classes=["result-panel"])
        vid_btn = gr.Button("[ ANALYZE VIDEO ]", variant="primary")
        vid_btn.click(
            analyze_video,
            inputs=[vid_in, vid_thresh, vid_step, vid_smooth, vid_save],
            outputs=[vid_out, vid_plot, vid_result],
            show_progress="full",
        )


if __name__ == "__main__":
    app.launch(server_port=ARGS.port, share=ARGS.share)
