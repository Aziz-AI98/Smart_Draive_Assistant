import os
import spaces
import gc
import cv2
import numpy as np
import pandas as pd
import gradio as gr
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# COCO Labels (only what matters for driving)
# ============================================================
RELEVANT_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign",
}

# ============================================================
# Model Loading (Cached)
# ============================================================
model_cache = {}

def get_model(model_name: str):
    """Load YOLOv8 model (cached to avoid reloading)."""
    if model_name not in model_cache:
        model_cache[model_name] = YOLO(model_name)
    return model_cache[model_name]

# ============================================================
# Detection (YOLOv8)
# ============================================================
def detect_road_objects(frame, model, score_threshold, imgsz=512):
    """Run YOLOv8 on a frame and return filtered detections."""
    results = model.predict(
        source=frame,
        conf=score_threshold,
        imgsz=imgsz,
        verbose=False,
        device="cpu",
    )
    detections = []
    if not results:
        return detections

    r = results[0]
    if r.boxes is None:
        return detections

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names

    for box, conf, cid in zip(boxes, confs, cls_ids):
        label = names.get(cid, str(cid))
        if label not in RELEVANT_CLASSES:
            continue
        x1, y1, x2, y2 = box.astype(int).tolist()
        detections.append({
            "box": [x1, y1, x2, y2],
            "label": label,
            "confidence": float(conf),
        })
    return detections

# ============================================================
# IoU + Tracker
# ============================================================
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, threshold=0.30, max_missing=12):
        self.threshold = threshold
        self.max_missing = max_missing
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        used = set()
        for track_id, track in list(self.tracks.items()):
            choices = [
                (iou(track["box"], d["box"]), i)
                for i, d in enumerate(detections)
                if i not in used and d["label"] == track["label"]
            ]
            best, idx = max(choices, default=(0, -1))
            if best >= self.threshold:
                self.tracks[track_id].update(box=detections[idx]["box"], missing=0)
                detections[idx]["track_id"] = track_id
                used.add(idx)
            else:
                self.tracks[track_id]["missing"] += 1

        self.tracks = {k: v for k, v in self.tracks.items() if v["missing"] <= self.max_missing}

        for i, d in enumerate(detections):
            if i not in used:
                d["track_id"] = self.next_id
                self.tracks[self.next_id] = {"box": d["box"], "label": d["label"], "missing": 0}
                self.next_id += 1
        return detections

# ============================================================
# Risk Assessment
# ============================================================
def risk_for_object(box, label, frame_w, frame_h):
    x1, y1, x2, y2 = box
    area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)
    center_x = (x1 + x2) / 2
    in_lane = 0.30 * frame_w < center_x < 0.70 * frame_w
    important = label in {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

    if important and in_lane and area_ratio > 0.075:
        return "DANGER", area_ratio
    if important and (in_lane or area_ratio > 0.025):
        return "CAUTION", area_ratio
    return "SAFE", area_ratio

# ============================================================
# Video Processing Pipeline
# ============================================================
@spaces.GPU
def process_driving_video(
    input_video_path,
    model_choice,
    score_threshold,
    iou_threshold,
    max_missing,
    skip_frames,
    imgsz
):
    if not input_video_path:
        return None, None, "⚠️ الرجاء رفع فيديو أولاً."

    model = get_model(model_choice)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return None, None, "❌ تعذر فتح الفيديو المرفق."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = "driving_assistant_output.mp4"
    csv_path = "driving_events.csv"

    import imageio.v2 as imageio
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
    )

    tracker = IoUTracker(threshold=iou_threshold, max_missing=max_missing)
    events = []
    frame_no = 0
    analyzed = 0

    rank = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
    colors = {"SAFE": (0, 200, 0), "CAUTION": (0, 190, 255), "DANGER": (0, 0, 255)}
    last_annotated = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_no += 1

        if frame_no % skip_frames != 0 and last_annotated is not None:
            writer.append_data(cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB))
            continue

        analyzed += 1
        frame_risk = "SAFE"
        detections = detect_road_objects(frame, model, score_threshold, imgsz)
        tracked = tracker.update(detections)

        for d in tracked:
            risk, area = risk_for_object(d["box"], d["label"], w, h)
            if rank[risk] > rank[frame_risk]:
                frame_risk = risk

            x1, y1, x2, y2 = d["box"]
            color = colors[risk]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"#{d['track_id']} {d['label']} {d['confidence']:.2f} {risk}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
            if risk != "SAFE":
                events.append({
                    "frame": frame_no,
                    "time_seconds": round(frame_no / fps, 2),
                    "track_id": d["track_id"],
                    "object": d["label"],
                    "risk": risk,
                    "box_area_ratio": round(area, 4),
                })

        cv2.rectangle(frame, (0, 0), (w, 60), colors[frame_risk], -1)
        cv2.putText(
            frame, f"RISK: {frame_risk}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )

        last_annotated = frame.copy()
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.close()

    events_df = pd.DataFrame(events).drop_duplicates() if events else pd.DataFrame(columns=["frame", "time_seconds", "track_id", "object", "risk", "box_area_ratio"])
    events_df.to_csv(csv_path, index=False)

    summary_msg = f"✅ تم الانتهاء بنجاح! تم تحليل {analyzed} إطار من أصل {total_frames}."
    return output_path, csv_path, summary_msg

# ============================================================
# Gradio UI Design
# ============================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Smart Front-View Driving Assistant")
    gr.Markdown(" نظام مساعد القيادة الذكية باستخدام **YOLOv8** وتتبع الأجسام وتقييم المخاطر المرورية.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📹 ارفع فيديو كاميرا القيادة (Dashcam)")
            
            model_choice = gr.Dropdown(
                choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                value="yolov8s.pt",
                label="اختر نموذج YOLOv8"
            )
            
            with gr.Accordion("⚙️ إعدادات متقدمة", open=False):
                skip_frames = gr.Slider(1, 5, value=1, step=1, label="تخطي الإطارات (لزيادة السرعة)")
                imgsz = gr.Slider(320, 640, value=512, step=96, label="حجم صورة المعالجة (Inference Size)")
                score_threshold = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="عتبة الثقة (Confidence Threshold)")
                iou_threshold = gr.Slider(0.1, 0.8, value=0.30, step=0.05, label="عتبة التتبع (IoU Threshold)")
                max_missing = gr.Slider(1, 30, value=12, step=1, label="أقصى إطارات مفقودة للتتبع")

            submit_btn = gr.Button("🚀 بدء التحليل", variant="primary")

        with gr.Column(scale=1):
            video_output = gr.Video(label="🎬 فيديو النتائج النهائي")
            csv_output = gr.File(label="📥 تحميل تقرير الأحداث (CSV)")
            status_output = gr.Textbox(label="حالة العملية التنفيذية")

    submit_btn.click(
        fn=process_driving_video,
        inputs=[
            video_input, model_choice, score_threshold,
            iou_threshold, max_missing, skip_frames, imgsz
        ],
        outputs=[video_output, csv_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())