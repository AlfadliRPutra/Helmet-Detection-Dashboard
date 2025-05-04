import streamlit as st

def show():
    st.markdown(
        """
        <h3 style='text-align: center; margin-top: -30px;'>📊 Model Overview</h3>
        <hr style='margin-top: 5px; margin-bottom: 30px;'>
        """, unsafe_allow_html=True
    )

    # Informasi dasar model
    st.subheader("🧠 Model Info")
    st.write("""
    - **Model**: YOLOv5s  
    - **Ukuran**: 14.5 MB  
    - **Jumlah Parameter**: 7.2M  
    - **Dataset**: Custom Helmet Dataset (COCO-style)  
    - **Epoch**: 100  
    - **Optimizer**: Adam  
    """)

    # Evaluasi performa
    st.subheader("📈 Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("mAP@0.5", "0.86")
    col2.metric("Precision", "0.89")
    col3.metric("Recall", "0.84")

    # Contoh visualisasi prediksi
    st.subheader("🖼️ Contoh Prediksi")
    st.image("images/sample_prediction.jpg", caption="Contoh deteksi helm", use_column_width=True)

    st.info("Catatan: Model dilatih menggunakan 5000 gambar pelatihan dan 1000 gambar validasi.")
