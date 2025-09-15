# app_streamlit.py
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import glob
import pandas as pd
from pathlib import Path

# Import with error handling
try:
    import cv2
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    st.stop()

try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Ultralytics import failed: {e}")
    st.stop()

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Object Detection",
                   page_icon="🤖", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown(
    """
    <style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* CSS Variables for Theme */
    :root {
        --primary-bg: #0f0f23;
        --secondary-bg: #1a1a2e;
        --accent-color: #00d4ff;
        --accent-secondary: #ff6b6b;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --card-bg: rgba(26, 26, 46, 0.8);
        --card-border: rgba(0, 212, 255, 0.2);
        --shadow-primary: 0 8px 32px rgba(0, 212, 255, 0.15);
        --shadow-secondary: 0 4px 16px rgba(255, 107, 107, 0.1);
    }

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', system-ui, sans-serif;
        background: var(--primary-bg);
        color: var(--text-primary);
        scroll-behavior: smooth;
    }

    /* Animated Background */
    .stApp {
        background: 
            radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 107, 107, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(138, 43, 226, 0.05) 0%, transparent 50%),
            linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        background-attachment: fixed;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }

    /* Animated particles background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(0, 212, 255, 0.05) 1px, transparent 1px),
            radial-gradient(circle at 75% 75%, rgba(255, 107, 107, 0.05) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: particles 20s linear infinite;
        pointer-events: none;
        z-index: -1;
    }

    @keyframes particles {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-50px, -50px); }
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.1) 0%, 
            rgba(255, 107, 107, 0.05) 50%, 
            rgba(138, 43, 226, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin: 2rem 0 3rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-primary);
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        animation: rotate 8s linear infinite;
        z-index: -1;
    }

    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }

    .main-header h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #ff6b6b 50%, #8a2be2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }

    .main-header p {
        color: var(--text-secondary) !important;
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        font-weight: 400;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(15, 15, 35, 0.95) 0%, 
            rgba(26, 26, 46, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--card-border);
    }

    .sidebar-header {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-secondary) 100%);
        margin: -1rem -1rem 2rem -1rem;
        padding: 2rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .sidebar-header h3 {
        color: white !important;
        margin: 0;
        font-weight: 600;
        font-size: 1.3rem;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    /* Content Cards */
    .content-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: var(--shadow-primary);
    }

    .content-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-color), var(--accent-secondary));
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }

    .content-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: var(--accent-color);
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.2);
    }

    .content-card:hover::before {
        transform: scaleX(1);
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-secondary) 100%);
        color: white;
        font-weight: 600;
        border-radius: 16px;
        border: none;
        padding: 1rem 2.5rem;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 212, 255, 0.4);
    }

    .stButton button:hover::before {
        left: 100%;
    }

    /* File Uploader */
    .stFileUploader {
        background: rgba(0, 212, 255, 0.05);
        border: 2px dashed var(--accent-color);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }

    .stFileUploader:hover {
        background: rgba(0, 212, 255, 0.1);
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
    }

    /* Sliders */
    .stSlider {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .stSlider:hover {
        border-color: var(--accent-color);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.1);
    }

    /* DataFrames */
    .stDataFrame {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: var(--shadow-primary);
        transition: all 0.3s ease;
    }

    .stDataFrame:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(0, 212, 255, 0.15);
    }

    /* Text Elements */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .sidebar h1, .sidebar h2, .sidebar h3, .sidebar h4, .sidebar h5, .sidebar h6 {
        color: white !important;
    }

    label, p, span {
        color: var(--text-secondary) !important;
        font-weight: 400;
    }

    .sidebar label, .sidebar p, .sidebar span {
        color: #e2e8f0 !important;
    }

    /* Status Messages */
    .stSuccess {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: white;
        border-radius: 16px;
        border: none;
        padding: 1.5rem;
        font-weight: 500;
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.3);
    }

    .stWarning {
        background: linear-gradient(135deg, #ffb347 0%, #ff8c00 100%);
        color: white;
        border-radius: 16px;
        border: none;
        padding: 1.5rem;
        font-weight: 500;
        box-shadow: 0 8px 25px rgba(255, 140, 0, 0.3);
    }

    .stError {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border-radius: 16px;
        border: none;
        padding: 1.5rem;
        font-weight: 500;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
    }

    .stInfo {
        background: linear-gradient(135deg, var(--accent-color) 0%, #0099cc 100%);
        color: white;
        border-radius: 16px;
        border: none;
        padding: 1.5rem;
        font-weight: 500;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-color), var(--accent-secondary));
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
    }

    /* Radio Buttons */
    .stRadio > div {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    /* Metrics */
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-color), var(--accent-secondary));
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 212, 255, 0.2);
    }

    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }

    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 3px solid rgba(0, 212, 255, 0.3);
        border-radius: 50%;
        border-top-color: var(--accent-color);
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Image Containers */
    .image-container {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 35px rgba(0, 212, 255, 0.15);
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--secondary-bg);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-secondary) 100%);
        border-radius: 10px;
        border: 2px solid var(--secondary-bg);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--accent-secondary) 0%, var(--accent-color) 100%);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .content-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .stRadio > div {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- APP TITLE --------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🚀 AI Vision Studio</h1>
        <p>Next-generation object detection powered by advanced neural networks</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- SIDEBAR --------------------
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <h3>⚡ Control Panel</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 🎛️ Model Configuration")
uploaded_weights = st.sidebar.file_uploader("Upload custom weights (.pt)", type=[
                                            "pt"], help="Upload your trained YOLOv8 model weights")

st.sidebar.markdown("### 🎯 Detection Settings")
conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0,
                         0.4, 0.01, help="Minimum confidence for detections")
img_size = st.sidebar.selectbox("Image size (px)", [
                                320, 416, 640, 1280], index=2, help="Input image size for inference")

st.sidebar.markdown("### � Performance Metrics")
st.sidebar.info(
    "**🧠 Model:** YOLOv8 Nano  \n**📊 Classes:** 80 COCO objects  \n**⚡ Speed:** Real-time  \n**🎯 Accuracy:** High precision")

# -------------------- LOAD MODEL --------------------


@st.cache_resource
def load_model(weights_path="yolov8n.pt"):
    return YOLO(weights_path)


def save_uploaded_file(uploaded_file, suffix=""):
    suffix = suffix if suffix else Path(uploaded_file.name).suffix
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    return tf.name


def annotate_and_table(results, model):
    res = results[0]
    try:
        plotted = res.plot()
        annotated = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    except Exception:
        annotated = res.orig_img if hasattr(res, "orig_img") else None

    detections = []
    try:
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            for c, cf, box in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()):
                name = model.names[int(c)]
                detections.append({"Class": name, "Confidence": round(
                    float(cf), 2), "BBox": [round(float(x), 2) for x in box]})
    except:
        detections = []

    return annotated, pd.DataFrame(detections)


# -------------------- MODEL --------------------
weights_to_load = "yolov8n.pt"
if uploaded_weights:
    weights_to_load = save_uploaded_file(uploaded_weights, suffix=".pt")
    st.sidebar.success("✅ Custom weights loaded successfully")
else:
    st.sidebar.info("Using pre-trained YOLOv8n weights")

model = load_model(weights_to_load)

# -------------------- MODE SELECTION --------------------
st.markdown(
    """
    <div class="content-card">
        <h3>🎯 Choose Detection Mode</h3>
        <p>Select your preferred input method for AI-powered object detection</p>
    </div>
    """,
    unsafe_allow_html=True
)

mode = st.radio(
    "Detection Mode",
    ["� Image Analysis", "� Video Processing", "� Live Camera"],
    horizontal=True,
    help="Select the type of input you want to process"
)

# IMAGE MODE
if mode == "� Image Analysis":
    st.markdown(
        """
        <div class="content-card">
            <h3>� Image Analysis Engine</h3>
            <p>Upload high-resolution images for precise object detection and analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("📎 Drop your image here", type=[
                                "jpg", "jpeg", "png"], help="Supported: JPG, JPEG, PNG (Max: 200MB)")

    if uploaded:
        with st.spinner("� AI is analyzing your image..."):
            img = Image.open(uploaded).convert("RGB")
            results = model.predict(np.array(img), conf=conf, imgsz=img_size)
            annotated, df = annotate_and_table(results, model)

        st.markdown(
            """
            <div class="content-card">
                <h3>📊 Analysis Results</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### 🖼️ Original Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(img, caption="Source Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if annotated is not None:
                st.markdown("#### 🎯 Detection Results")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(annotated, caption="AI Detection Results", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
        with col2:
            if not df.empty:
                st.markdown("#### 🎯 Detection Summary")
                st.dataframe(df, use_container_width=True)
                
                # Enhanced metrics display
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4>🔢 Total Objects</h4>
                            <h2 style="color: var(--accent-color);">{len(df)}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_b:
                    if len(df) > 0:
                        avg_conf = df['Confidence'].mean()
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h4>📈 Avg Confidence</h4>
                                <h2 style="color: var(--accent-secondary);">{avg_conf:.2f}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                # Object class distribution
                if not df.empty:
                    st.markdown("#### 📊 Object Distribution")
                    class_counts = df['Class'].value_counts()
                    st.bar_chart(class_counts)
                    
            else:
                st.markdown(
                    """
                    <div class="content-card" style="text-align: center;">
                        <h4>🔍 No Objects Detected</h4>
                        <p>Try adjusting the confidence threshold or upload a different image</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# VIDEO MODE
elif mode == "� Video Processing":
    st.markdown(
        """
        <div class="content-card">
            <h3>� Video Processing Engine</h3>
            <p>Upload video files for advanced frame-by-frame object detection and analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_vid = st.file_uploader("🎥 Drop your video here", type=[
                                    "mp4", "mov", "avi", "mkv"], help="Supported: MP4, MOV, AVI, MKV (Max: 500MB)")

    if uploaded_vid:
        tmp = save_uploaded_file(uploaded_vid)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📹 Source Video")
            st.video(tmp)
        
        with col2:
            st.markdown("#### 🎛️ Processing Controls")
            process_button = st.button("🚀 Start AI Processing", use_container_width=True)
            
            if process_button:
                with st.spinner("🎬 AI is processing your video... This may take several minutes"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    project_dir = tempfile.mkdtemp()
                    
                    # Update progress simulation
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        status_text.text(f'Processing frame {i + 1}/100...')
                        
                    results = model.predict(
                        source=tmp, conf=conf, imgsz=img_size, project=project_dir, name="run", save=True)

                    try:
                        out_dir = str(results[0].save_dir)
                        vids = glob.glob(os.path.join(out_dir, "*"))
                        vids = [v for v in vids if Path(v).suffix.lower() in [
                            ".mp4", ".avi", ".mov", ".mkv"]]
                        if vids:
                            st.markdown(
                                """
                                <div class="content-card">
                                    <h3>✅ Processing Complete!</h3>
                                    <p>Your video has been enhanced with AI object detection</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            st.markdown("#### 🎯 Enhanced Video with AI Detection")
                            st.video(vids[0])

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.download_button(
                                    label="� Download Enhanced Video",
                                    data=open(vids[0], 'rb').read(),
                                    file_name=f"ai_enhanced_{uploaded_vid.name}",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                            with col_b:
                                st.markdown(
                                    """
                                    <div class="metric-card">
                                        <h4>🎉 Success!</h4>
                                        <p>Ready to download</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    except Exception as e:
                        st.error(f"⚠️ Processing failed: {str(e)}")
                        st.info("💡 Try using a smaller video file or different format")

# WEBCAM MODE
elif mode == "� Live Camera":
    st.markdown(
        """
        <div class="content-card">
            <h3>� Real-time AI Camera</h3>
            <p>Experience live object detection with your camera feed in real-time</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("#### 🎮 Camera Controls")
        start_webcam = st.button("� Start Live Detection", use_container_width=True)
        stop_webcam = st.button("⏹ Stop Camera", use_container_width=True)
        
        st.markdown("#### 📊 Live Analytics")
        fps_placeholder = st.empty()
        detection_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        st.markdown("#### ⚙️ Quick Settings")
        live_conf = st.slider("Live Confidence", 0.1, 1.0, conf, 0.05)

    with col1:
        if start_webcam:
            st.markdown(
                """
                <div class="content-card" style="border: 2px solid var(--accent-color); text-align: center;">
                    <h4>🔴 LIVE DETECTION ACTIVE</h4>
                    <p>AI is analyzing your camera feed in real-time</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Camera access denied. Please check your camera permissions and try again.")
                st.info("💡 Make sure no other application is using your camera")
            else:
                stframe = st.empty()
                frame_count = 0
                total_detections = 0
                confidence_sum = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Camera connection lost")
                        break

                    frame_count += 1
                    results = model.predict(frame, conf=live_conf, imgsz=img_size)
                    annotated, df = annotate_and_table(results, model)

                    if annotated is not None:
                        # Add frame overlay with stats
                        overlay_frame = annotated.copy()
                        stframe.image(overlay_frame, channels="RGB", use_column_width=True)

                    # Update live statistics
                    current_detections = len(df) if not df.empty else 0
                    total_detections += current_detections
                    
                    if not df.empty:
                        confidence_sum += df['Confidence'].mean()

                    with col2:
                        fps_placeholder.markdown(
                            f"""
                            <div class="metric-card">
                                <h4>📹 Frames</h4>
                                <h2 style="color: var(--accent-color);">{frame_count}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        detection_placeholder.markdown(
                            f"""
                            <div class="metric-card">
                                <h4>🎯 Current Objects</h4>
                                <h2 style="color: var(--accent-secondary);">{current_detections}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        if total_detections > 0:
                            avg_conf = confidence_sum / max(1, frame_count)
                            confidence_placeholder.markdown(
                                f"""
                                <div class="metric-card">
                                    <h4>📈 Avg Confidence</h4>
                                    <h2 style="color: var(--accent-color);">{avg_conf:.2f}</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    if stop_webcam:
                        break

                cap.release()
                st.markdown(
                    """
                    <div class="content-card" style="text-align: center;">
                        <h4>✅ Live Session Ended</h4>
                        <p>Camera disconnected successfully</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    """
    <div class="content-card" style="text-align: center; margin-top: 3rem;">
        <h4>🚀 AI Vision Studio</h4>
        <p>Powered by YOLOv8 • Built with Streamlit • Next-gen Computer Vision Technology</p>
        <p style="font-size: 0.9rem; color: var(--text-secondary);">
            � Experience the future of AI • 🌐 <a href="#" style="color: var(--accent-color);">Documentation</a> • 📧 <a href="#" style="color: var(--accent-secondary);">Support</a>
        </p>
        <div style="margin-top: 1rem;">
            <span style="color: var(--accent-color);">●</span>
            <span style="color: var(--accent-secondary);">●</span>
            <span style="color: var(--accent-color);">●</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
