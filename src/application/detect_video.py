import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLO Model (gunakan model yang sudah dilatih)
@st.cache_resource
def load_model():
    model = YOLO("models/helmet_model.pt")  # Ganti dengan path model YOLO kamu
    return model

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

# Fungsi untuk input video (unggah atau sample video)
def video_input():
    vid_file = None
    vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    if vid_bytes:
        vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
        with open(vid_file, 'wb') as out:
            out.write(vid_bytes.read())
    
    return vid_file

# Fungsi utama untuk halaman Streamlit
def show():
    st.markdown("""
        <h2 style='text-align: center;'>🎥 Deteksi Helm pada Video</h2>
        <hr style="margin-top: 5px; margin-bottom: 30px;">
    """, unsafe_allow_html=True)

    st.markdown("""
        Unggah video dan klik **Deteksi** untuk memproses helm di tiap frame video.
    """)

    # Pilih video
    uploaded_video = video_input()

    if uploaded_video is not None:
        st.video(uploaded_video)

        # Menampilkan video
        cap = cv2.VideoCapture(uploaded_video)

        # Mendapatkan FPS dan ukuran video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.sidebar.markdown(f"**FPS**: {fps}")
        st.sidebar.markdown(f"**Width**: {width}")
        st.sidebar.markdown(f"**Height**: {height}")

        output = st.empty()
        prev_time = 0
        curr_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break

            # Convert frame to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Deteksi objek (helm) pada frame
            output_img = detect_objects_yolo_ultralytics(frame)

            # Tampilkan hasil frame
            output.image(output_img)

            # Update FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

        cap.release()

    else:
        st.warning("Silakan unggah video terlebih dahulu.")

    st.markdown("""
        **Catatan**:
        - 🟢 Hijau: Helmet  
        - 🔴 Merah: No Helmet  
        - 🔵 Biru: Rider  
        - 🟠 Oranye: Motorcycle  
    """)

