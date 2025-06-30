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
        st.write("**Model:** YOLO11")
        st.write("**Ukuran:** 5.4 MB")
        st.write("**Parameter:** 2.5 M")
    with col2:
        st.write("**Dataset:** CCTV ATCS Samarinda & Phone Camera")
        st.write("**Epoch:** 50")
        st.write("**Optimizer:** Stochastic Gradient Descent")

    # Metrics
    st.subheader("📈 Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("mAP@0.5", "0.97")
    col2.metric("Precision", "0.95")
    col3.metric("Recall", "0.97")

    # Training curve (online image)
    st.subheader("📉 Training Loss Curve")
    st.image(
        "../assets/images/loss_plot.png",
        caption="Loss Model",
        use_container_width=True
    )

    # Confusion matrix (online image)
    st.subheader("🧮 Confusion Matrix")
    st.image(
       "../assets/images/loss_plot.png",
        caption="Confusion Matrix",
        use_container_width=True
    )

    
