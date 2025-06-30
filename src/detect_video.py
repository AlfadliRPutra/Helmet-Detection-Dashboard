import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
from settings import MODEL_PATH
import time
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

# ================== Load Model Sekali ==================
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

CLASS_NAMES = {
    0: 'Helmet', 1: 'Motorcycle', 2: 'No Helmet', 3: 'Rider'
}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 165, 255),
    2: (0, 0, 255),
    3: (255, 0, 0),
}

# ================== Fungsi Bantu ==================
def overlaps(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return max(0, min(ax2, bx2) - max(ax1, bx1)) > 0 and max(0, min(ay2, by2) - max(ay1, by1)) > 0

# ================== Fungsi Utama UI Streamlit ==================
def show():
    st.markdown("<h2 style='text-align: center;'>📹 Deteksi Helm dengan Tracking</h2><hr>", unsafe_allow_html=True)
    st.info("Unggah video dan deteksi akan dijalankan. Pelanggaran (tanpa helm saat berkendara) akan ditampilkan.")

    uploaded_video = st.file_uploader("📁 Unggah Video...", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        input_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(input_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.subheader("🎞️ Video Asli")
        st.video(input_video_path)

        if st.button("🚀 Mulai Deteksi dan Tracking"):
            cap = cv2.VideoCapture(input_video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            tracker = DeepSort(max_age=100, n_init=3, max_cosine_distance=0.85, nn_budget=100)
            captured_motor_ids = {}
            violation_images = []

            CONFIDENCE_THRESHOLD = 0.5
            VIOLATION_INTERVAL = 20

            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_idx = 0
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                results = model(frame, verbose=False)[0]
                boxes_by_cls = defaultdict(list)
                detections = []

                for box in results.boxes:
                    conf = float(box.conf)
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    boxes_by_cls[cls].append((x1, y1, x2, y2))
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

                tracks = tracker.update_tracks(detections, frame=frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    cls = track.get_det_class()
                    track_id = track.track_id
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    color = CLASS_COLORS.get(cls, (255, 255, 255))
                    label = f'ID {track_id} - {CLASS_NAMES.get(cls)}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # === Deteksi pelanggaran: No Helmet + Rider + Motorcycle ===
                for rider_box in boxes_by_cls[3]:
                    for no_helmet_box in boxes_by_cls[2]:
                        if overlaps(rider_box, no_helmet_box):
                            motor_track_id = None
                            motor_box = None
                            for track in tracks:
                                if track.is_confirmed() and track.get_det_class() == 1:
                                    tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
                                    tracked_box = (tx1, ty1, tx2, ty2)
                                    if overlaps(rider_box, tracked_box):
                                        motor_track_id = track.track_id
                                        motor_box = tracked_box
                                        break
                            if motor_track_id is not None and motor_box is not None:
                                last = captured_motor_ids.get(motor_track_id, -999)
                                if frame_idx - last >= VIOLATION_INTERVAL:
                                    x1s = [motor_box[0], rider_box[0], no_helmet_box[0]]
                                    y1s = [motor_box[1], rider_box[1], no_helmet_box[1]]
                                    x2s = [motor_box[2], rider_box[2], no_helmet_box[2]]
                                    y2s = [motor_box[3], rider_box[3], no_helmet_box[3]]
                                    vx1, vy1 = max(min(x1s) - 5, 0), max(min(y1s) - 5, 0)
                                    vx2, vy2 = min(max(x2s) + 5, frame.shape[1]), min(max(y2s) + 5, frame.shape[0])
                                    crop = frame[vy1:vy2, vx1:vx2]
                                    violation_images.append(crop)
                                    captured_motor_ids[motor_track_id] = frame_idx

                out.write(frame)

                if frame_idx % 5 == 0:
                    progress = frame_idx / total_frames if total_frames > 0 else 0
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    eta = ((elapsed / frame_idx) * (total_frames - frame_idx)) if frame_idx > 0 else 0
                    status_text.text(f"Frame {frame_idx}/{total_frames} | Estimasi selesai: {int(eta)} detik")

            cap.release()
            out.release()

            st.success("✅ Proses selesai!")
            st.subheader("🎬 Video Hasil Deteksi dan Tracking")
            st.video(output_video_path)

            with open(output_video_path, "rb") as file:
                st.download_button("📥 Unduh Video", data=file, file_name="hasil_tracking.mp4", mime="video/mp4")

            if violation_images:
                st.subheader("📸 Gambar Pelanggaran yang Terdeteksi")
                cols = st.columns(3)
                for idx, img in enumerate(violation_images):
                    with cols[idx % 3]:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Pelanggaran #{idx+1}", use_column_width=True)
            else:
                st.info("👍 Tidak ada pelanggaran terdeteksi.")

            # Hapus file sementara
            if os.path.exists(input_video_path):
                os.remove(input_video_path)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)

