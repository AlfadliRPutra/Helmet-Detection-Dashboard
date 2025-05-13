import streamlit as st

# Import setiap menu
from home import show as show_home
from model_overview import show as show_model_overview
from detect_image import show as show_detect_image
from detect_video import show as show_detect_video
from realtime_camera_detection import show as show_realtime_camera_detection


st.set_page_config(page_title="Object Detection App", layout="wide")

# Sidebar
st.sidebar.title("📚 Menu")
menu = st.sidebar.radio(
    "Pilih halaman:",
    ("Home", "Model Overview", "Detect Image", "Detect Video", "Realtime Camera Detection")
)





# Routing ke setiap menu

if menu == "Model Overview":
    show_model_overview()
elif menu == "Home":
    show_home()
elif menu == "Detect Image":
    show_detect_image()
elif menu == "Detect Video":
    show_detect_video()
elif menu == "Realtime Camera Detection":
    show_realtime_camera_detection()
