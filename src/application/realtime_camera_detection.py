import streamlit as st

def show():
    st.markdown(
        """
        <h2 style='text-align: center;'>🎦 Realtime Camera Detection</h2>
        <hr style="margin-top: 5px; margin-bottom: 30px;">
        """, unsafe_allow_html=True
    )

    # Deskripsi Halaman
    st.markdown("""
    Di sini, kamu bisa menggunakan webcam untuk mengambil gambar secara langsung. 
    Arahkan kamera ke objek dan gambar akan otomatis tampil di bawah.
    """)

    # Input Kamera Asli
    st.subheader("📹 Gunakan Webcam untuk Pengambilan Gambar")
    camera_input = st.camera_input("Ambil gambar dengan kamera")

    if camera_input is not None:
        # Tampilkan gambar yang diambil dari kamera
        st.image(camera_input, caption="Gambar yang Diambil", use_column_width=True)
        st.info("Klik **Ambil Gambar** untuk menangkap gambar dari kamera.")
    else:
        st.warning("Silakan gunakan kamera untuk mengambil gambar.")

    st.markdown("""
    **Catatan**: Deteksi helm akan segera tersedia setelah model deteksi siap.
    """)
