import streamlit as st
import cv2
import tempfile
import numpy as np
import os
from ultralytics import YOLO
from settings import MODEL_PATH

# Load model YOLO
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Fungsi deteksi objek untuk 1 frame
def detect_objects_yolo_ultralytics(frame, confidence_threshold=0.3):
    results = model(frame)
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    class_colors = {
        0: (0, 255, 0),     # Helmet - Hijau
        1: (0, 0, 255),     # No Helmet - Merah
        2: (255, 0, 0),     # Rider - Biru
        3: (0, 165, 255)    # Motorcycle - Oranye (BGR)
    }

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            x1, y1, x2, y2 = map(int, boxes[i])
            conf = scores[i].item()
            cls_id = int(class_ids[i].item())
            color = class_colors.get(cls_id, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# Fungsi utama untuk halaman Streamlit
def show():
    st.markdown("""
        <h2 style='text-align: center;'>🎥 Deteksi Helm pada Video</h2>
        <hr style="margin-top: 5px; margin-bottom: 30px;">
    """, unsafe_allow_html=True)

    st.markdown("""
        Unggah video dan klik **Deteksi** untuk memproses helm di tiap frame video.
    """)

    st.subheader("📤 Unggah Video")
    uploaded_video = st.file_uploader("Pilih video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Deteksi Helm"):
            with st.spinner("Memproses video, mohon tunggu..."):
                # Simpan video input sementara
                input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                input_temp.write(uploaded_video.read())
                input_path = input_temp.name
                input_temp.close()

                st.info(f"📥 Input video disimpan sementara di: `{input_path}`")

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    st.error("❌ Gagal membuka video input. Format mungkin tidak didukung.")
                    return

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                st.write(f"📏 Ukuran: {width}x{height}, 🎞️ FPS: {fps}")

                # Output video
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                if not out.isOpened():
                    st.error("❌ VideoWriter gagal dibuka. Cek codec ('mp4v') dan dimensi.")
                    return

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning(f"🚫 Gagal membaca frame ke-{frame_count}. Stop.")
                        break

                    processed_frame = detect_objects_yolo_ultralytics(frame)
                    out.write(processed_frame)
                    frame_count += 1

                cap.release()
                out.release()

                st.info(f"✅ Total frame yang diproses: {frame_count}")

                # Debug: cek file output
                if not os.path.exists(output_path):
                    st.error("❌ Gagal menyimpan video hasil. File tidak ditemukan.")
                else:
                    size = os.path.getsize(output_path)
                    if size == 0:
                        st.error("❌ File video hasil kosong (size = 0 bytes).")
                    else:
                        st.success(f"✅ Video berhasil disimpan. Ukuran: {round(size / 1024 / 1024, 2)} MB")
                        st.markdown("🔍 **Hasil Deteksi Video**:")
                        with open(output_path, 'rb') as result_video:
                            video_bytes = result_video.read()
                            st.video(video_bytes)

    else:
        st.warning("Silakan unggah video terlebih dahulu.")

    st.markdown("""
        **Catatan**:
        - 🟢 Hijau: Helmet  
        - 🔴 Merah: No Helmet  
        - 🔵 Biru: Rider  
        - 🟠 Oranye: Motorcycle  
    """)
