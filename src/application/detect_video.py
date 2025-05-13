import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from settings import MODEL_PATH

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Fungsi deteksi objek per frame
def detect_objects_yolo_ultralytics(frame, confidence_threshold=0.3):
    results = model(frame)
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    class_colors = {
        0: (0, 255, 0),     # Helmet - Green
        1: (0, 0, 255),     # No Helmet - Red
        2: (255, 0, 0),     # Rider - Blue
        3: (0, 165, 255)    # Motorcycle - Orange (BGR)
    }

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i].tolist()
            score = scores[i].item()
            class_id = int(class_ids[i].item())

            x_min, y_min, x_max, y_max = map(int, box)
            color = class_colors.get(class_id, (255, 255, 255))  # default: white
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            label = f'{score:.2f}'
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def show():
    st.markdown(
        """
        <h2 style='text-align: center;'>🎥 Deteksi Helm pada Video</h2>
        <hr style="margin-top: 5px; margin-bottom: 30px;">
        """, unsafe_allow_html=True
    )

    st.markdown("""
    Unggah video dan klik **Deteksi** untuk memproses helm di tiap frame.
    """)

    st.subheader("📤 Unggah Video")
    uploaded_video = st.file_uploader("Pilih video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Deteksi Helm"):
            with st.spinner("Memproses video, mohon tunggu..."):
                # Simpan video input
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_input.write(uploaded_video.read())
                temp_input.close()

                cap = cv2.VideoCapture(temp_input.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Simpan video output
                temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = detect_objects_yolo_ultralytics(frame)
                    out.write(frame)

                cap.release()
                out.release()

                # Tampilkan hasil
                with open(temp_output_path, 'rb') as video_file:
                    st.markdown("🔍 **Hasil Deteksi Video**:")
                    st.video(video_file.read())
                st.success("Video selesai diproses.")

    else:
        st.warning("Silakan unggah video terlebih dahulu.")

    st.markdown("""
    **Catatan**: hijau: helmet. merah: no helmet. biru: rider. oranye: motorcycle
    """)
