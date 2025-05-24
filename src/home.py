import streamlit as st

def show():
    st.markdown(
        """
        <style>
        .welcome-text {
            text-align: center;
            font-size: 32px;
            color: #4CAF50;
            margin-top: -30px;
        }
        .description-text {
            font-size: 18px;
            text-align: justify;
            margin-bottom: 30px;
        }
        .card {
            background-color: #f0f8ff;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .card-header {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
        <div class="welcome-text">
            🚀 Welcome to the Helmet Detection Dashboard
        </div>
        <hr style="margin-top: 5px; margin-bottom: 30px;">

        """, unsafe_allow_html=True
    )

    # Deskripsi singkat
    st.markdown("""
    <div class="description-text">
    Selamat datang di aplikasi **Helmet Detection**!  
    Dashboard ini memungkinkan kamu untuk mengakses beberapa fitur canggih, termasuk:
    </div>
    """, unsafe_allow_html=True)

    # Kartu Menu dengan emoji dan ikon
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">📊 Model Overview</div>
            <div>Ringkasan performa model deteksi helm, termasuk metrik dan visualisasi.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">🖼️ Detect Image</div>
            <div>Unggah gambar untuk mendeteksi helm pada foto.</div>
        </div>
        """, unsafe_allow_html=True)

    # Kartu Menu tambahan
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">🎥 Detect Video</div>
            <div>Unggah video untuk deteksi helm pada setiap frame.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        
        </div>
        """, unsafe_allow_html=True)

    # Footer dengan tombol navigasi atau info lebih lanjut
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <a href="#home" style="font-size: 18px; color: #007bff; text-decoration: none; font-weight: bold;">→ Explore More Features</a>
    </div>
    """, unsafe_allow_html=True)

    st.success("Pilih menu dari sidebar untuk memulai!")
