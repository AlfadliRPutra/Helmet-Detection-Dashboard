import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from settings import MODEL_PATH 

# Muat model sekali menggunakan cache Streamlit
@st.cache_resource
def load_model():
    """Memuat model YOLO dari path yang ditentukan."""
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# Definisikan nama kelas dan warna untuk konsistensi
CLASS_NAMES = {
    0: 'Helmet',
    1: 'Motorcycle',
    2: 'No Helmet',
    3: 'Rider'
}

CLASS_COLORS = {
    0: (0, 255, 0),      # Hijau  - Helmet
    1: (0, 165, 255),    # Oranye - Motorcycle
    2: (0, 0, 255),      # Merah  - No Helmet
    3: (255, 0, 0)       # Biru   - Rider
}


def obj_detect(image_pil, confidence_threshold=0.4):
    """
    Fungsi untuk mendeteksi objek dan hanya menampilkan bounding box (tanpa label atau confidence).
    """
    image_np = np.array(image_pil)
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = model.predict(
        img_bgr,
        conf=0.5,
        iou=0.5
    )

    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    detect_img = img_bgr.copy()

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i].tolist()
            class_id = int(class_ids[i].item())

            x_min, y_min, x_max, y_max = map(int, box)

            color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Putih jika tidak dikenali
            cv2.rectangle(detect_img, (x_min, y_min), (x_max, y_max), color, 2)

    detect_img_rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    return detect_img_rgb

# Streamlit UI
def show():
    st.markdown(
        """
        <h2 style='text-align: center;'>🖼️ Deteksi Gambar</h2>
        <hr style="margin-top: 5px; margin-bottom: 30px;">
        """, unsafe_allow_html=True
    )
    st.markdown(""" 
    Unggah gambar untuk mulai mendeteksi...
    """)

    st.subheader("📤 Unggah Gambar untuk Deteksi")
    uploaded_image = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Buka gambar menggunakan PIL
        image = Image.open(uploaded_image)
        
        st.image(image, caption="Gambar Asli", use_container_width=True)

        if st.button("✨ Deteksi Helm"):
            with st.spinner("🧠 Menganalisis gambar..."):
                # Panggil fungsi deteksi dengan objek gambar PIL
                result_image = obj_detect(image)
                st.image(result_image, caption="Hasil Deteksi", use_container_width=True)

    else:
        st.info("ℹ️ Silakan unggah gambar untuk memulai deteksi.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(""" 
    **Keterangan:**
    - <span style='color:green;'>■</span> **Hijau**: Helmet
    - <span style='color:red;'>■</span> **Merah**: No Helmet
    - <span style='color:blue;'>■</span> **Biru**: Rider
    - <span style='color:orange;'>■</span> **Oranye**: Motorcycle
    """, unsafe_allow_html=True)
