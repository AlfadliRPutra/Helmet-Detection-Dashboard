import streamlit as st

def show():
    st.markdown(
        """
        <h3 style='text-align: center; margin-top: -30px;'>📊 Model Overview</h3>
        <hr style='margin-top: 5px; margin-bottom: 30px;'>
        """, unsafe_allow_html=True
    )

    # Info model
    st.subheader("🧠 Model Info")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model:** YOLOv5s")
        st.write("**Ukuran:** 14.5 MB")
        st.write("**Parameter:** 7.2M")
    with col2:
        st.write("**Dataset:** Custom Helmet Dataset (COCO-style)")
        st.write("**Epoch:** 100")
        st.write("**Optimizer:** Adam")

    # Metrics
    st.subheader("📈 Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("mAP@0.5", "0.86")
    col2.metric("Precision", "0.89")
    col3.metric("Recall", "0.84")

    # Training curve (online image)
    st.subheader("📉 Training Loss Curve")
    st.image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        caption="Training loss vs Epoch",
        use_container_width=True
    )

    # Confusion matrix (online image)
    st.subheader("🧮 Confusion Matrix")
    st.image(
       "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        caption="Confusion Matrix",
        use_container_width=True
    )

    # Prediksi contoh (online image)
    st.subheader("🖼️ Contoh Prediksi")
    st.image(
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        caption="Contoh deteksi helm (simulasi)",
        use_container_width=True
    )

    st.info("Catatan: Model dilatih menggunakan 5000 gambar pelatihan dan 1000 gambar validasi.")
