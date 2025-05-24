import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from settings import MODEL_PATH 

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)  # Pastikan path-nya relatif ke root proyek
    return model
model=load_model()


# Fungsi deteksi objek yang telah dimodifikasi
def obj_detect(img_path, confidence_threshold=0.3):
    img = cv2.imread(img_path)
    results = model(img)  # Menggunakan model YOLO yang telah dimuat

    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    detect_img = img.copy()

    # Mapping warna berdasarkan class ID
    class_colors = {
        0: (0, 165, 255)    # Oranye - Motorcycle (warna BGR: Orange)
        1: (255, 0, 0),     # Biru - Rider
        2: (0, 255, 0),     # Hijau - Helmet
        3: (0, 0, 255),     # Merah - No Helmet
    }

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i].tolist()
            score = scores[i].item()
            class_id = int(class_ids[i].item())

            x_min, y_min, x_max, y_max = map(int, box)
            color = class_colors.get(class_id, (255, 255, 255))  # Default putih jika class_id tidak dikenali
            cv2.rectangle(detect_img, (x_min, y_min), (x_max, y_max), color, 1)

            label = f'{score:.2f}'
            cv2.putText(detect_img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    return detect_img



# Streamlit UI
def show():
    st.markdown(
        """
        <h2 style='text-align: center;'>🖼️ Deteksi Helm pada Gambar</h2>
        <hr style="margin-top: 5px; margin-bottom: 30px;">
        """, unsafe_allow_html=True
    )

    st.markdown(""" 
    Di sini, kamu dapat mengunggah gambar dan melihat hasil deteksi helm pada gambar tersebut. 
    Cukup pilih gambar dan klik **Deteksi** untuk menampilkan hasilnya.
    """)

    st.subheader("📤 Unggah Gambar untuk Deteksi")
    uploaded_image = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)

        if st.button("Deteksi Helm"):
            with st.spinner("Memproses gambar..."):
                # Simpan gambar sementara untuk dibaca OpenCV
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    image.save(temp_file.name)
                    result_image = obj_detect(temp_file.name)

                st.markdown("🔍 **Hasil Deteksi**:")
                st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
    else:
        st.warning("Silakan unggah gambar untuk memulai deteksi.")

    st.markdown(""" 
    **Catatan**:
        - 🟢 Hijau: Helmet  
        - 🔴 Merah: No Helmet  
        - 🔵 Biru: Rider  
        - 🟠 Oranye: Motorcycle  
    """)

