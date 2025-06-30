import streamlit as st
from PIL import Image
import cv2
import numpy as np
# Hapus tempfile karena tidak lagi digunakan
from ultralytics import YOLO
# Pastikan file settings.py dan path model sudah benar
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
    Fungsi untuk melakukan deteksi objek pada gambar.
    Inputnya adalah objek gambar PIL, bukan path file.
    """
    # 1. Konversi gambar PIL ke array NumPy (format RGB)
    image_np = np.array(image_pil)
    
    # 2. Konversi dari RGB ke BGR karena OpenCV menggunakan BGR
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Lakukan deteksi dengan model YOLO
    results = model.predict(
        img_bgr,
        conf=0.5,
        iou=0.5
    )

    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    # Salin gambar untuk digambari bounding box
    detect_img = img_bgr.copy()

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i].tolist()
            score = scores[i].item()
            class_id = int(class_ids[i].item())

            x_min, y_min, x_max, y_max = map(int, box)
            
            # Ambil warna dan nama kelas, default jika tidak ada
            color = CLASS_COLORS.get(class_id, (255, 255, 255)) # Default Putih
            class_name = CLASS_NAMES.get(class_id, "Unknown")
            
            # Gambar kotak
            cv2.rectangle(detect_img, (x_min, y_min), (x_max, y_max), color, 2)

            # Buat label dengan nama kelas dan skor kepercayaan
            label = f'{class_name}: {score:.2f}'
            
            # Atur posisi dan gambar teks label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(detect_img, (x_min + label_size[0], y_min - 10), color, cv2.FILLED)
            # cv2.putText(detect_img, label, (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Konversi kembali dari BGR ke RGB untuk ditampilkan di Streamlit
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
