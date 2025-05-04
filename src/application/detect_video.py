import streamlit as st

def show():
    st.markdown(
        """
        <h2 style='text-align: center;'>🎥 Detect Helm on Video</h2>
        <hr style="margin-top: 5px; margin-bottom: 30px;">
        """, unsafe_allow_html=True
    )

    # Deskripsi Halaman
    st.markdown("""
    Di sini, kamu dapat mengunggah video untuk mendeteksi helm pada setiap frame video tersebut.
    Cukup pilih video dan klik **Deteksi** untuk memulai proses deteksi.
    """)

    # Upload Video
    st.subheader("📤 Unggah Video untuk Deteksi")
    uploaded_video = st.file_uploader("Pilih video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Tampilkan video yang diunggah
        st.video(uploaded_video)

        # Tombol Deteksi
        with st.spinner("Memproses video..."):
            if st.button("Deteksi Helm"):
                # Simulasi: Menampilkan hasil deteksi video (ganti nanti dengan model deteksi yang sebenarnya)
                st.markdown("🔍 **Hasil Deteksi Video**:")
                st.video(uploaded_video)  # Tampilkan video asli (seharusnya hasil deteksi nanti)
                st.success("Helm berhasil terdeteksi pada video.")
    else:
        st.warning("Silakan unggah video untuk memulai deteksi.")
    
    st.markdown("""
    **Catatan**: Deteksi ini menggunakan model simulasi. Ganti dengan model nyata untuk hasil yang lebih akurat.
    """)
