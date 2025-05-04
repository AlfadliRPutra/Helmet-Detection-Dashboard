import streamlit as st

# Import setiap menu
from application.model_overview import show as show_model_overview
from application.detect_image import show as show_detect_image
from application.detect_video import show as show_detect_video
from application.realtime_camera import show as show_realtime_camera

st.set_page_config(page_title="Object Detection App", layout="wide")

# Sidebar
st.sidebar.title("📚 Menu")
menu = st.sidebar.radio(
    "Pilih halaman:",
    ("Model Overview", "Detect Image", "Detect Video", "Realtime Camera Detection")
)

# Header
st.markdown("<h1 style='text-align: center;'>🚀 Object Detection App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Routing ke setiap menu
if menu == "Model Overview":
    show_model_overview()
elif menu == "Detect Image":
    show_detect_image()
elif menu == "Detect Video":
    show_detect_video()
elif menu == "Realtime Camera Detection":
    show_realtime_camera()
