import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from app import model  as model_obj_det


# Fungsi deteksi objek
def obj_detect(img_path, confidence_threshold=0.3):
    img = cv2.imread(img_path)
    results = model_obj_det(img)  # Using the loaded YOLO model to get predictions

    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    detect_img = img.copy()

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i].tolist()
            score = scores[i].item()
            class_id = class_ids[i].item()

            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(detect_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f'Conf: {score:.2f} | Class: {int(class_id)}'
            cv2.putText(detect_img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
    **Catatan**: Deteksi ini menggunakan model YOLO yang telah dilatih. Pastikan model sudah diload dengan benar.
    """)
