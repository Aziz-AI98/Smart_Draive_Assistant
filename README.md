---
title: Smart Front View Driving Assistant
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
---

# 🚗 Smart Front-View Driving Assistant
### AI-Powered Dashcam Analysis using YOLOv8 + IoU Tracking + Risk Assessment
**Developed by:** Abdulaziz Alraddai

---

## 📌 Overview | نبذة عن المشروع

**Smart Front-View Driving Assistant** is an end-to-end Computer Vision & Deep Learning solution designed to assist drivers and autonomous systems by analyzing dashcam footage in real time.

The system performs object detection, unique object tracking across frames, and dynamic risk assessment to identify potential hazards on the road.

نظام **مساعد القيادة الذكية ذو الرؤية الأمامية** هو حل متكامل يعتمد على الذكاء الاصطناعي ورؤية الكمبيوتر لتحليل فيديوهات كاميرا القيادة (Dashcam). يقوم النظام بكشف العناصر المرورية، تتبعها بفرادة عبر الإطارات، وتقييم مستويات الخطر تلقائياً للتنبيه بالمخاطر المحتملة.

---

## ✨ Key Features | الميزات الرئيسية

- **🚀 Real-Time Object Detection:** Powered by **YOLOv8** (`yolov8n`, `yolov8s`, `yolov8m`) to detect key driving objects (Cars, Pedestrians, Trucks, Buses, Traffic Lights, etc.).
- **🎯 Robust Multi-Object Tracking:** Uses an **IoU (Intersection over Union) Tracker** to assign unique ID tracks to detected objects across frames.
- **⚠️ Dynamic Risk Assessment:** Evaluates collision risk into 3 levels:
  - 🟢 **SAFE:** Normal object distance or out-of-lane.
  - 🟡 **CAUTION:** Object in driving corridor or approaching closer.
  - 🔴 **DANGER:** Critical object directly in-lane with a high frame area ratio (> 7.5%).
- **📹 Video Export & Analytics:** Exports annotated H.264 MP4 videos and detailed CSV event logs with precise timestamps.
- **💻 Interactive Gradio UI:** User-friendly web interface hosted on Hugging Face Spaces.

---

## 🛠️ Tech Stack | التقنيات المستخدمة

- **Framework:** [Gradio](https://gradio.app/)
- **Object Detection:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Computer Vision:** OpenCV (`cv2`), ImageIO
- **Data Processing:** Pandas, NumPy
- **Language:** Python 3.10+

---

## ⚙️ How It Works | آلية العمل

1. **Video Upload:** The user uploads a dashcam video (MP4/AVI/MOV).
2. **Detection & Filtering:** YOLOv8 extracts bounding boxes for driving-relevant COCO classes.
3. **IoU Association:** Objects are tracked over time using spatial overlap metrics.
4. **Risk Logic:** Calculates position relative to the central driving corridor and object bounding box area.
5. **Output Generation:** Rendered annotated video and CSV report are generated for immediate download.

---

## 🚀 Local Installation & Setup | التشغيل المحلي

To run this project locally, follow these steps:

```bash
# 1. Clone the repository
git clone [https://huggingface.co/spaces/Aziz-AI98/driving-assistant-yolo](https://huggingface.co/spaces/Aziz-AI98/driving-assistant-yolo)
cd driving-assistant-yolo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Gradio app
python app.py