import streamlit as st
import cv2
import os
import tempfile
import subprocess
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from settings import MODEL_PATH

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def overlaps(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return max(0, min(ax2, bx2) - max(ax1, bx1)) > 0 and max(0, min(ay2, by2) - max(ay1, by1)) > 0

def convert_video_for_streamlit(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def show():
    st.markdown("""
        <h2 style='text-align: center;'>🎥 Deteksi Video</h2>
        <hr style="margin-top: 5px; margin-bottom: 30px;">
    """, unsafe_allow_html=True)

    st.markdown("Unggah video untuk mulai mendeteksi...")

    uploaded_video = st.file_uploader("📤 Upload video untuk analisis", type=["mp4"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        input_path = tfile.name
        raw_output_path = input_path.replace(".mp4", "_raw_output.mp4")
        streamlit_output_path = input_path.replace(".mp4", "_output_streamlit.mp4")
        violation_dir = tempfile.mkdtemp()

        model = load_model()
        tracker = DeepSort(
            max_age=100,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=50
        )

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        out = cv2.VideoWriter(raw_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        st.info(f"⏳ Memproses video ({total_frames} frame)...")
        captured_motor_ids = {}
        violation_images = []

        frame_idx = 0
        for result in model.predict(source=input_path, stream=True, conf=0.6):
            frame = result.orig_img
            frame_idx += 1

            if len(result.boxes) > 10:
                continue

            detections = []
            boxes_by_cls = defaultdict(list)

            for box in result.boxes:
                conf = float(box.conf)
                if conf < 0.6:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                boxes_by_cls[cls].append((x1, y1, x2, y2))
                
                # Hanya masukkan motorcycle ke tracker
                if cls == 1:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

            tracks = tracker.update_tracks(detections, frame=frame)
            clean_frame = frame.copy()

            # Gambar hasil dari tracker (motorcycle dengan ID)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                cls = track.get_det_class()
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                color = (0, 165, 255)  # motorcycle: oranye
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID {track_id} - {cls}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Gambar hasil prediksi langsung untuk class lain
            for cls_id, boxes in boxes_by_cls.items():
                if cls_id == 1:
                    continue  # motorcycle sudah digambar via tracker
                for (x1, y1, x2, y2) in boxes:
                    color_map = {
                        0: (0, 255, 0),     # helmet: hijau
                        2: (0, 0, 255),     # no helmet: merah
                        3: (255, 0, 0),     # rider: biru
                    }
                    color = color_map.get(cls_id, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'Cls {cls_id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Cek pelanggaran
            for rider_box in boxes_by_cls[3]:
                for no_helmet_box in boxes_by_cls[2]:
                    if overlaps(rider_box, no_helmet_box):
                        motor_track_id = None
                        motor_box = None
                        for track in tracks:
                            if track.is_confirmed() and track.get_det_class() == 1:
                                tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
                                if overlaps(rider_box, (tx1, ty1, tx2, ty2)):
                                    motor_track_id = track.track_id
                                    motor_box = (tx1, ty1, tx2, ty2)
                                    break

                        if motor_track_id and motor_box:
                            x1s = [motor_box[0], rider_box[0], no_helmet_box[0]]
                            y1s = [motor_box[1], rider_box[1], no_helmet_box[1]]
                            x2s = [motor_box[2], rider_box[2], no_helmet_box[2]]
                            y2s = [motor_box[3], rider_box[3], no_helmet_box[3]]

                            vx1, vy1 = max(0, min(x1s)), max(0, min(y1s))
                            vx2, vy2 = min(width, max(x2s)), min(height, max(y2s))

                            if frame_idx - captured_motor_ids.get(motor_track_id, -999) >= 10:
                                if vx2 > vx1 and vy2 > vy1:
                                    crop = clean_frame[vy1:vy2, vx1:vx2]
                                    save_path = os.path.join(violation_dir, f"motorID_{motor_track_id}.jpg")
                                    cv2.imwrite(save_path, crop)
                                    captured_motor_ids[motor_track_id] = frame_idx

                                    if save_path not in violation_images:
                                        violation_images.append(save_path)

            out.write(frame)

        out.release()

        convert_video_for_streamlit(raw_output_path, streamlit_output_path)
        st.success("✅ Proses selesai!")
        st.video(streamlit_output_path)

        with open(streamlit_output_path, "rb") as f:
            st.download_button("⬇️ Unduh Video Hasil", f, file_name="hasil_deteksi.mp4")

        if violation_images:
            st.subheader("📸 Gambar Pelanggaran")
            cols = st.columns(3)
            for idx, img_path in enumerate(violation_images):
                with cols[idx % 3]:
                    st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
        else:
            st.info("👍 Tidak ada pelanggaran yang terdeteksi.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        **Keterangan:**
        - <span style='color:green;'>■</span> **Hijau**: Helmet  
        - <span style='color:red;'>■</span> **Merah**: No Helmet  
        - <span style='color:blue;'>■</span> **Biru**: Rider  
        - <span style='color:orange;'>■</span> **Oranye**: Motorcycle  
    """, unsafe_allow_html=True)
