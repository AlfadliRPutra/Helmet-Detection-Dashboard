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
    0: 'Helmet', 1: 'Motorcycle', 2: 'No Helmet', 3: 'Rider'
}
CLASS_COLORS = {
    0: (0, 255, 0),    # Oranye
    1: (0, 165, 255) ,    # Biru
    2: (0, 0, 255),     # Hijau
    3: (255, 0, 0),     # Merah
}

# --- Fungsi ini tidak diubah ---
def obj_detect_video(frame_bgr, confidence_threshold=0.4):
    """
    Fungsi deteksi yang dioptimalkan untuk frame video (input BGR, output BGR).
    """
    results = model(frame_bgr, verbose=False)
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls
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
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_bgr, (x_min, y_min - h - 10), (x_min + w, y_min - 5), color, -1)
            cv2.putText(frame_bgr, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return frame_bgr

# --- FUNGSI UTAMA UNTUK UI VIDEO (LOGIKA DIPERBARUI) ---
def show_video_page():
    st.markdown(
        "<h2 style='text-align: center;'>📹 Deteksi Helm pada Video</h2><hr>", 
        unsafe_allow_html=True
    )
    st.info("""
    Unggah file video. Deteksi akan dilakukan **setiap 1 detik** untuk mempercepat proses.
    """)

    uploaded_video = st.file_uploader("Pilih file video...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        input_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(input_video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        st.subheader("Video Asli")
        st.video(input_video_path)

        if st.button("🚀 Mulai Deteksi (Setiap 1 Detik)"):
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            
            try:
                cap = cv2.VideoCapture(input_video_path)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                # Menghindari pembagian dengan nol jika FPS tidak terbaca
                if fps == 0:
                    st.warning("Tidak dapat membaca FPS video, menggunakan default 30 FPS.")
                    fps = 30
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                st.subheader("⚙️ Sedang Memproses...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()
                
                frame_count = 0
                last_processed_frame = None # Untuk menyimpan frame terakhir yang diproses

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # <<< PERUBAHAN LOGIKA UTAMA ADA DI SINI >>>
                    # Hanya proses frame pertama setiap detiknya (frame ke-0, ke-fps, ke-2*fps, dst.)
                    if frame_count % fps == 0:
                        # Jalankan deteksi dan simpan hasilnya
                        last_processed_frame = obj_detect_video(frame.copy())
                        out.write(last_processed_frame)
                    else:
                        # Untuk frame lain, tulis hasil deteksi terakhir
                        if last_processed_frame is not None:
                            out.write(last_processed_frame)
                        else:
                            # Jika belum ada frame yang diproses, tulis frame asli
                            out.write(frame)
                    
                    frame_count += 1
                    
                    # Update UI progress
                    progress = frame_count / total_frames if total_frames > 0 else 0
                    if frame_count % 5 == 0:
                        elapsed_time = time.time() - start_time
                        eta = ((elapsed_time / frame_count) * (total_frames - frame_count)) if frame_count > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"Frame: {frame_count}/{total_frames} | Estimasi Waktu: {int(eta)}s")

                progress_bar.progress(1.0)
                status_text.success("🎉 Video berhasil diproses!")
                
                cap.release()
                out.release()
                
                st.subheader("Video Hasil Deteksi")
                st.video(output_video_path)
                
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="📥 Unduh Video Hasil",
                        data=file,
                        file_name=f"hasil_deteksi_{uploaded_video.name}",
                        mime="video/mp4"
                    )

            finally:
                if os.path.exists(input_video_path):
                    os.remove(input_video_path)
                if os.path.exists(output_video_path):
                    os.remove(output_video_path)
