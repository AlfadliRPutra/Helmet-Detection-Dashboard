import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from settings import MODEL_PATH
import time # Untuk mengukur waktu

# --- Bagian ini tetap sama ---
@st.cache_resource
def load_model():
    """Memuat model YOLO dari path yang ditentukan."""
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

CLASS_NAMES = {
    0: 'Motorcycle', 1: 'Rider', 2: 'Helmet', 3: 'No Helmet'
}
CLASS_COLORS = {
    0: (0, 165, 255),    # Oranye
    1: (255, 0, 0),      # Biru
    2: (0, 255, 0),      # Hijau
    3: (0, 0, 255),      # Merah
}

# Fungsi deteksi ini sedikit diubah agar menerima dan mengembalikan frame BGR
def obj_detect_video(frame_bgr, confidence_threshold=0.4):
    """
    Fungsi deteksi yang dioptimalkan untuk frame video (input BGR, output BGR).
    """
    results = model(frame_bgr)
    
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    # Tidak perlu membuat copy karena kita akan menggambar langsung di frame
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i].tolist()
            score = scores[i].item()
            class_id = int(class_ids[i].item())

            x_min, y_min, x_max, y_max = map(int, box)
            
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            class_name = CLASS_NAMES.get(class_id, "Unknown")
            
            cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), color, 2)
            label = f'{class_name}: {score:.2f}'
            cv2.putText(frame_bgr, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame_bgr

# --- FUNGSI UTAMA UNTUK UI VIDEO ---
def show():
    st.markdown(
        "<h2 style='text-align: center;'>📹 Deteksi Helm pada Video</h2><hr>", 
        unsafe_allow_html=True
    )
    st.info("""
    Unggah file video (`.mp4`, `.mov`, `.avi`). Proses ini akan memakan waktu tergantung pada durasi video dan kekuatan CPU/GPU Anda.
    """)

    uploaded_video = st.file_uploader("Pilih file video...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Gunakan tempfile untuk menyimpan video yang diunggah agar bisa dibaca OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        
        st.video(tfile.name) # Tampilkan video asli

        if st.button("🚀 Mulai Deteksi pada Video"):
            # Placeholder untuk video hasil
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            try:
                # Buka video sumber
                cap = cv2.VideoCapture(tfile.name)
                
                # Dapatkan properti video untuk VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Buat objek VideoWriter untuk menyimpan hasil
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                # UI untuk progress
                st.subheader("⚙️ Sedang Memproses...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Proses frame dengan fungsi deteksi
                    processed_frame = obj_detect_video(frame)
                    
                    # Tulis frame yang sudah diproses ke file output
                    out.write(processed_frame)
                    
                    frame_count += 1
                    progress = frame_count / total_frames
                    
                    # Update progress bar dan status
                    progress_bar.progress(progress)
                    elapsed_time = time.time() - start_time
                    eta = (elapsed_time / frame_count) * (total_frames - frame_count)
                    status_text.text(f"Frame {frame_count}/{total_frames} | ETA: {int(eta)} detik")

                # Tutup semuanya
                cap.release()
                out.release()
                
                progress_bar.empty()
                status_text.empty()
                st.success("🎉 Video berhasil diproses!")
                
                # Tampilkan video hasil
                st.video(output_video_path)
                
                # Berikan tombol download
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="📥 Unduh Video Hasil",
                        data=file,
                        file_name=f"hasil_{uploaded_video.name}",
                        mime="video/mp4"
                    )

            finally:
                # Pastikan file sementara dihapus
                import os
                os.remove(tfile.name)
                # Jangan hapus output_video_path sampai setelah diunduh/ditampilkan

