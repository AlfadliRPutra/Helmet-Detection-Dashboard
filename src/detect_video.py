import streamlit as st
import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import tempfile
import numpy as np

# ==== Konfigurasi ====
CONFIDENCE_THRESHOLD = 0.5
VIOLATION_UPDATE_INTERVAL = 10 # frame
PADDING = 5
color_map = {
    0: (0, 255, 0),       # helmet
    1: (0, 165, 255),     # motorcycle
    2: (0, 0, 255),       # no helmet
    3: (255, 0, 0),       # rider
}

# Fungsi bantu
def overlaps(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
    return overlap_x > 0 and overlap_y > 0

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def main():
    st.title("🚨 Deteksi Pelanggaran Helm + Tracking")

    uploaded_file = st.file_uploader("📁 Upload Video", type=["mp4", "mov", "avi"])
    model_path = st.text_input("🧠 Path Model YOLO", value="model.pt")
    if uploaded_file and model_path and st.button("🚀 Mulai Proses"):
        model = load_model(model_path)
        tracker = DeepSort(max_age=100, n_init=3, max_cosine_distance=0.85, nn_budget=100)
        violation_dir = tempfile.mkdtemp(prefix="pelanggaran_")
        captured_motor_ids = {}

        temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        st.info(f"Video dimensi: {width}x{height} | FPS: {fps:.2f} | Total Frame: {total_frames}")
        progress = st.progress(0)

        violation_images = []
        frame_idx = 0
        for result in model.predict(source=temp_input_path, stream=True, conf=CONFIDENCE_THRESHOLD):
            frame = result.orig_img
            frame_idx += 1
            if len(result.boxes) > 10:
                continue

            clean_frame = frame.copy()
            detections = []
            boxes_by_cls = defaultdict(list)

            for box in result.boxes:
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
                color = color_map.get(cls, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID {track_id} - {cls}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Pelanggaran
            for rider_box in boxes_by_cls[3]:
                for no_helmet_box in boxes_by_cls[2]:
                    if overlaps(rider_box, no_helmet_box):
                        motor_track_id, motor_box = None, None
                        for track in tracks:
                            if track.is_confirmed() and track.get_det_class() == 1:
                                tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
                                tracked_box = (tx1, ty1, tx2, ty2)
                                if overlaps(rider_box, tracked_box):
                                    motor_track_id = track.track_id
                                    motor_box = tracked_box
                                    break

                        if motor_track_id and motor_box:
                            last_saved = captured_motor_ids.get(motor_track_id, -999)
                            if frame_idx - last_saved >= VIOLATION_UPDATE_INTERVAL:
                                x1s = [motor_box[0], rider_box[0], no_helmet_box[0]]
                                y1s = [motor_box[1], rider_box[1], no_helmet_box[1]]
                                x2s = [motor_box[2], rider_box[2], no_helmet_box[2]]
                                y2s = [motor_box[3], rider_box[3], no_helmet_box[3]]
                                vx1 = max(min(x1s) - PADDING, 0)
                                vy1 = max(min(y1s) - PADDING, 0)
                                vx2 = min(max(x2s) + PADDING, frame.shape[1])
                                vy2 = min(max(y2s) + PADDING, frame.shape[0])
                                crop = clean_frame[vy1:vy2, vx1:vx2]

                                file_path = os.path.join(violation_dir, f"motorID_{motor_track_id}.jpg")
                                cv2.imwrite(file_path, crop)
                                captured_motor_ids[motor_track_id] = frame_idx
                                violation_images.append(file_path)

            out.write(frame)
            progress.progress(min(frame_idx / total_frames, 1.0))

        out.release()
        st.success("✅ Selesai! Video & pelanggaran berhasil diproses.")

        st.subheader("🎬 Hasil Video")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("📥 Unduh Video Hasil", f.read(), "hasil_tracking.mp4", "video/mp4")

        if violation_images:
            st.subheader("📸 Pelanggaran yang Terdeteksi")
            cols = st.columns(3)
            for i, img_path in enumerate(violation_images):
                with cols[i % 3]:
                    st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
        else:
            st.info("👍 Tidak ada pelanggaran terdeteksi.")
