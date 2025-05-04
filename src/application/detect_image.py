import streamlit as st
from PIL import Image

def show():
    st.markdown(
        """
        <h2 style='text-align: center;'>🖼️ Detect Helm on Image</h2>
        <hr style="margin-top: 5px; margin-bottom: 30px;">
        """, unsafe_allow_html=True
    )

    # Deskripsi Halaman
    st.markdown("""
    Di sini, kamu dapat mengunggah gambar dan melihat hasil deteksi helm pada gambar tersebut. 
    Cukup pilih gambar dan klik **Deteksi** untuk menampilkan hasilnya.
    """)

    # Upload Gambar
    st.subheader("📤 Unggah Gambar untuk Deteksi")
    uploaded_image = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)

        # Tombol Deteksi
        with st.spinner("Memproses gambar..."):
            if st.button("Deteksi Helm"):
                # Simulasi: Menampilkan hasil deteksi (ganti nanti dengan model deteksi yang sebenarnya)
                st.markdown("🔍 **Hasil Deteksi**:")
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Construction_workers.jpg/640px-Construction_workers.jpg", caption="Contoh Hasil Deteksi", use_container_width=True)
                st.success("Helm berhasil terdeteksi di gambar.")
    else:
        st.warning("Silakan unggah gambar untuk memulai deteksi.")
    
    st.markdown("""
    **Catatan**: Deteksi ini menggunakan model simulasi. Ganti dengan model nyata untuk hasil yang lebih akurat.
    """)
