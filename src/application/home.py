import streamlit as st

def show():
    st.markdown(
        """
        <h2 style='text-align: center; margin-top: -40px; margin-bottom: 10px;'>
            🚀 Welcome to the Helmet Detection Dashboard
        </h2>
        <hr style='margin-top: 5px; margin-bottom: 30px;'>
        """,
        unsafe_allow_html=True
    )

    st.write("Selamat datang di aplikasi deteksi helm!")
    st.markdown("""
    Di dalam dashboard ini kamu bisa melakukan:
    - 📊 **Model Overview**: Lihat ringkasan performa model.
    - 🖼️ **Detect Image**: Unggah gambar untuk deteksi helm.
    - 🎥 **Detect Video**: Unggah video untuk proses deteksi.
    - 🎦 **Realtime Camera Detection**: Gunakan webcam secara langsung.
    """)

    st.success("Pilih menu dari sidebar untuk memulai!")

