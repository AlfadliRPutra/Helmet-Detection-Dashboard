import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO
from settings import MODEL_PATH
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# ================== Load Model Sekali ==================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()


def overlaps(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return max(0, min(ax2, bx2) - max(ax1, bx1)) > 0 and max(0, min(ay2, by2) - max(ay1, by1)) > 0


def crop_violation_area(frame, boxes, padding=5):
    x1s = [b[0] for b in boxes]
    y1s = [b[1] for b in boxes]
    x2s = [b[2] for b in boxes]
    y2s = [b[3] for b in boxes]
    vx1 = max(0, min(x1s) - padding)
    vy1 = max(0, min(y1s) - padding)
    vx2 = min(frame.shape[1], max(x2s) + padding)
    vy2 = min(frame.shape[0], max(y2s) + padding)
    return frame[vy1:vy2, vx1:vx2]


def show():
    st.title("🚦 Deteksi Pelanggaran Helm dengan Tracking")

    uploaded_video = st.file_uploader("📤 Upload video untuk analisis", type=["mp4", "mov", "avi"])
    conf_thresh = st.slider("🎯 Confidence Threshold", 0.2, 1.0, 0.5, 0.05)
    interval = st.slider("⏱️ Interval Simpan Pelanggaran (frame)", 1, 60, 10)
    padding = st.slider("📐 Padding Crop (px)", 0, 30, 5)

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🚀 Jalankan Deteksi"):
            input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            input_temp.write(uploaded_video.read())
            input_path = input_temp.name

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            violation_dir = tempfile.mkdtemp()

            tracker = DeepSort(max_age=100, n_init=3, max_cosine_distance=0.85, nn_budget=100)

            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            st.write(f"📏 Ukuran video: {width}x{height}, FPS: {fps:.2f}, Total Frame: {total_frames}")

            frame_idx = 0
            captured_motor_ids = {}
            progress = st.progress(0)
            violation_images = []

            for result in model.predict(source=input_path, stream=True, conf=conf_thresh):
                frame = result.orig_img
                frame_idx += 1

                detections = []
                boxes_by_cls = defaultdict(list)

                for box in result.boxes:
                    conf = float(box.conf)
                    if conf < conf_thresh:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    boxes_by_cls[cls].append((x1, y1, x2, y2))
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

                tracks = tracker.update_tracks(detections, frame=frame)
                clean_frame = frame.copy()

                # Visualisasi tracking
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    cls = track.get_det_class()
                    track_id = track.track_id
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    color_map = {
                        0: (0, 255, 0),       # helmet
                        1: (0, 165, 255),     # motorcycle
                        2: (0, 0, 255),       # no helmet
                        3: (255, 0, 0),       # rider
                    }
                    color = color_map.get(cls, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID {track_id} - {cls}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Logika pelanggaran
                for rider_box in boxes_by_cls[3]:
                    for no_helmet_box in boxes_by_cls[2]:
                        if overlaps(rider_box, no_helmet_box):
                            motor_track_id, motor_box = None, None
                            for track in tracks:
                                if track.is_confirmed() and track.get_det_class() == 1:
                                    tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
                                    if overlaps(rider_box, (tx1, ty1, tx2, ty2)):
                                        motor_track_id = track.track_id
                                        motor_box = (tx1, ty1, tx2, ty2)
                                        break

                            if motor_track_id is not None:
                                boxes = [motor_box, rider_box, no_helmet_box]
                                last_saved = captured_motor_ids.get(motor_track_id, -999)
                                if frame_idx - last_saved >= interval:
                                    crop = crop_violation_area(clean_frame, boxes, padding)
                                    save_path = os.path.join(violation_dir, f"motorID_{motor_track_id}_f{frame_idx}.jpg")
                                    cv2.imwrite(save_path, crop)
                                    captured_motor_ids[motor_track_id] = frame_idx
                                    violation_images.append(save_path)

                out.write(frame)
                progress.progress(min(1.0, frame_idx / total_frames))

            out.release()

            st.success("✅ Proses selesai!")
            st.subheader("🎬 Video Hasil:")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button("⬇️ Unduh Video", f, file_name="hasil_deteksi.mp4")

            if violation_images:
                st.subheader("📸 Gambar Pelanggaran")
                cols = st.columns(3)
                for idx, img_path in enumerate(violation_images):
                    with cols[idx % 3]:
                        st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
            else:
                st.info("👍 Tidak ada pelanggaran yang terdeteksi.")
