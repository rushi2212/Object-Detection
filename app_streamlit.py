# app_streamlit.py
import subprocess
import sys

# Ensure ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "ultralytics==8.3.25", "opencv-python-headless"
    ])
    from ultralytics import YOLO

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import pandas as pd
from pathlib import Path

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Vision Studio", page_icon="🤖", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown(
    """
    <style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* CSS Variables for Theme */
    :root {
        --primary-bg: #0a0a0f;
        --secondary-bg: #1a1a2e;
        --accent-color: #00d4ff;
        --accent-secondary: #ff6b6b;
        --accent-tertiary: #4ecdc4;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --card-bg: rgba(26, 26, 46, 0.9);
        --card-border: rgba(0, 212, 255, 0.3);
        --shadow-primary: 0 12px 40px rgba(0, 212, 255, 0.2);
        --shadow-secondary: 0 8px 25px rgba(255, 107, 107, 0.15);
    }

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', system-ui, sans-serif;
        background: var(--primary-bg);
        color: var(--text-primary);
        scroll-behavior: smooth;
    }

    /* Enhanced Animated Background */
    .stApp {
        background: 
            radial-gradient(circle at 15% 85%, rgba(0, 212, 255, 0.15) 0%, transparent 60%),
            radial-gradient(circle at 85% 15%, rgba(255, 107, 107, 0.12) 0%, transparent 60%),
            radial-gradient(circle at 50% 50%, rgba(78, 205, 196, 0.08) 0%, transparent 70%),
            linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%);
        background-attachment: fixed;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }

    /* Floating particles animation */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(0, 212, 255, 0.08) 2px, transparent 2px),
            radial-gradient(circle at 80% 70%, rgba(255, 107, 107, 0.06) 1px, transparent 1px),
            radial-gradient(circle at 40% 80%, rgba(78, 205, 196, 0.05) 1.5px, transparent 1.5px);
        background-size: 80px 80px, 120px 120px, 60px 60px;
        animation: float 25s linear infinite;
        pointer-events: none;
        z-index: -1;
    }

    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(-30px, -30px) rotate(120deg); }
        66% { transform: translate(30px, -60px) rotate(240deg); }
        100% { transform: translate(0, 0) rotate(360deg); }
    }

    /* Enhanced Header Styling */
    .main-header {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.15) 0%, 
            rgba(78, 205, 196, 0.1) 30%,
            rgba(255, 107, 107, 0.08) 70%, 
            rgba(138, 43, 226, 0.12) 100%);
        backdrop-filter: blur(25px);
        border: 2px solid rgba(0, 212, 255, 0.4);
        border-radius: 28px;
        padding: 4rem 3rem;
        margin: 3rem 0 4rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-primary);
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -60%;
        left: -60%;
        width: 220%;
        height: 220%;
        background: conic-gradient(from 0deg, transparent, rgba(0, 212, 255, 0.15), transparent, rgba(78, 205, 196, 0.1), transparent);
        animation: spin 12s linear infinite;
        z-index: -1;
    }

    @keyframes spin {
        100% { transform: rotate(360deg); }
    }

    .main-header h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #4ecdc4 30%, #ff6b6b 70%, #8a2be2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.03em;
        text-shadow: 0 0 40px rgba(0, 212, 255, 0.4);
        animation: glow 3s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.3)); }
        to { filter: drop-shadow(0 0 35px rgba(0, 212, 255, 0.6)); }
    }

    .main-header p {
        color: var(--text-secondary) !important;
        font-size: 1.4rem;
        margin: 1.5rem 0 0 0;
        font-weight: 400;
        opacity: 0.9;
    }

    /* Enhanced Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(10, 10, 15, 0.98) 0%, 
            rgba(26, 26, 46, 0.95) 100%);
        backdrop-filter: blur(25px);
        border-right: 2px solid var(--card-border);
    }

    .sidebar-header {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-tertiary) 50%, var(--accent-secondary) 100%);
        margin: -1rem -1rem 2rem -1rem;
        padding: 2.5rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        border-radius: 0 0 20px 20px;
    }

    .sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.25), transparent);
        animation: shimmer 4s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .sidebar-header h3 {
        color: white !important;
        margin: 0;
        font-weight: 700;
        font-size: 1.4rem;
        text-shadow: 0 3px 12px rgba(0, 0, 0, 0.4);
    }

    /* Enhanced Content Cards */
    .content-card {
        background: var(--card-bg);
        backdrop-filter: blur(25px);
        border: 2px solid var(--card-border);
        border-radius: 24px;
        padding: 3rem;
        margin: 3rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: var(--shadow-primary);
    }

    .content-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-color), var(--accent-tertiary), var(--accent-secondary));
        transform: scaleX(0);
        transition: transform 0.5s ease;
    }

    .content-card:hover {
        transform: translateY(-12px) scale(1.02);
        border-color: var(--accent-color);
        box-shadow: 0 25px 50px rgba(0, 212, 255, 0.25);
    }

    .content-card:hover::before {
        transform: scaleX(1);
    }

    /* Enhanced Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-tertiary) 50%, var(--accent-secondary) 100%);
        color: white;
        font-weight: 700;
        border-radius: 20px;
        border: none;
        padding: 1.2rem 3rem;
        font-size: 1.2rem;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.4);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.6s;
    }

    .stButton button:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.5);
    }

    .stButton button:hover::before {
        left: 100%;
    }

    /* Enhanced File Uploader */
    .stFileUploader {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(78, 205, 196, 0.06) 100%);
        border: 3px dashed var(--accent-color);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }

    .stFileUploader::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        transition: left 0.8s;
    }

    .stFileUploader:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(78, 205, 196, 0.12) 100%);
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 212, 255, 0.3);
        border-color: var(--accent-tertiary);
    }

    .stFileUploader:hover::before {
        left: 100%;
    }

    /* Enhanced Metrics */
    .metric-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, rgba(0, 212, 255, 0.05) 100%);
        backdrop-filter: blur(25px);
        border: 2px solid var(--card-border);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
        margin: 1rem 0;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-color), var(--accent-tertiary), var(--accent-secondary));
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.25);
    }

    /* Enhanced Image Containers */
    .image-container {
        background: var(--card-bg);
        border: 2px solid var(--card-border);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        transition: all 0.4s ease;
        overflow: hidden;
        position: relative;
    }

    .image-container::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, var(--accent-color), var(--accent-tertiary), var(--accent-secondary), var(--accent-color));
        border-radius: 20px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.4s ease;
    }

    .image-container:hover {
        transform: scale(1.03);
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.2);
    }

    .image-container:hover::before {
        opacity: 1;
    }

    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(0, 212, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-tertiary) 100%);
        color: white !important;
    }

    /* Chart Container */
    .chart-container {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Enhanced Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }

    ::-webkit-scrollbar-track {
        background: var(--secondary-bg);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-tertiary) 50%, var(--accent-secondary) 100%);
        border-radius: 10px;
        border: 2px solid var(--secondary-bg);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--accent-secondary) 0%, var(--accent-color) 100%);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.8rem;
        }
        
        .content-card {
            padding: 2rem;
            margin: 2rem 0;
        }
        
        .stFileUploader {
            padding: 3rem 1rem;
        }
    }

    /* Loading Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading-text {
        animation: pulse 2s infinite;
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
        <p>Advanced object detection with state-of-the-art neural networks</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- SIDEBAR --------------------
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <h3>⚡ Control Center</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 🎛️ Model Configuration")
uploaded_weights = st.sidebar.file_uploader(
    "Upload custom weights (.pt)",
    type=["pt"],
    help="Upload your trained YOLOv8 model weights for custom object detection"
)

st.sidebar.markdown("### 🎯 Detection Settings")
conf = st.sidebar.slider(
    "Confidence threshold",
    0.0, 1.0, 0.35, 0.01,
    help="Minimum confidence score for object detection (higher = more selective)"
)

img_size = st.sidebar.selectbox(
    "Image resolution (px)",
    [320, 416, 640, 1280],
    index=2,
    help="Input image size for inference (higher = more accurate but slower)"
)

st.sidebar.markdown("### 📊 Model Information")
st.sidebar.info(
    """
    **🧠 Architecture:** YOLOv8 Nano  
    **📊 Object Classes:** 80 COCO categories  
    **⚡ Performance:** Real-time detection  
    **🎯 Accuracy:** State-of-the-art precision  
    **💾 Model Size:** ~6MB (optimized)
    """
)

# Advanced settings in expander
with st.sidebar.expander("🔧 Advanced Settings"):
    iou_threshold = st.slider("IoU Threshold", 0.1, 0.9, 0.45, 0.05,
                              help="Intersection over Union threshold for Non-Maximum Suppression")
    max_detections = st.slider("Max Detections", 10, 1000, 300, 10,
                               help="Maximum number of detections per image")

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
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                detections.append({
                    "Object": name,
                    "Confidence": f"{float(cf)*100:.1f}%",
                    "Box Area": f"{int(area)} px²",
                    "Position": f"({int(x1)}, {int(y1)})",
                    "Size": f"{int(width)}×{int(height)}"
                })
    except:
        detections = []

    return annotated, pd.DataFrame(detections)


# -------------------- MODEL LOADING --------------------
weights_to_load = "yolov8n.pt"
if uploaded_weights:
    weights_to_load = save_uploaded_file(uploaded_weights, suffix=".pt")
    st.sidebar.success("✅ Custom weights loaded successfully!")
else:
    st.sidebar.info("📦 Using pre-trained YOLOv8n model")

with st.spinner("🔄 Loading AI model..."):
    model = load_model(weights_to_load)

# -------------------- IMAGE DETECTION --------------------
st.markdown(
    """
    <div class="content-card">
        <h3>🎯 AI-Powered Object Detection</h3>
        <p>Upload high-quality images for precise object detection and comprehensive analysis with advanced neural networks</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "📸 Select or drag your image here",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
    help="Supported formats: JPG, JPEG, PNG, WebP, BMP, TIFF (Max: 200MB)"
)

if uploaded:
    # Show image info
    file_details = {
        "Filename": uploaded.name,
        "File size": f"{uploaded.size / 1024:.1f} KB",
        "File type": uploaded.type
    }

    with st.expander("📋 File Information"):
        st.json(file_details)

    with st.spinner("🤖 AI is analyzing your image... Please wait"):
        img = Image.open(uploaded).convert("RGB")

        # Run prediction with advanced settings
        results = model.predict(
            np.array(img),
            conf=conf,
            imgsz=img_size,
            iou=iou_threshold,
            max_det=max_detections
        )
        annotated, df = annotate_and_table(results, model)

    st.markdown(
        """
        <div class="content-card">
            <h3>📊 Detection Results & Analysis</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Enhanced layout with better proportions
    col1, col2 = st.columns([3, 2])

    with col1:
        # Image display with enhanced tabs
        tab1, tab2 = st.tabs(["🖼️ Original Image", "🎯 AI Detection Results"])

        with tab1:
            st.markdown('<div class="image-container">',
                        unsafe_allow_html=True)
            st.image(
                img, caption=f"Source: {uploaded.name}", use_column_width=True)

            # Image statistics
            img_array = np.array(img)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Width", f"{img_array.shape[1]} px")
            with col_b:
                st.metric("Height", f"{img_array.shape[0]} px")
            with col_c:
                st.metric("Channels", img_array.shape[2])

            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            if annotated is not None:
                st.markdown('<div class="image-container">',
                            unsafe_allow_html=True)
                st.image(
                    annotated, caption="AI-Enhanced Detection Results", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Detection confidence info
                if not df.empty:
                    avg_conf = df['Confidence'].str.rstrip(
                        '%').astype(float).mean()
                    st.info(f"🎯 Average detection confidence: {avg_conf:.1f}%")
            else:
                st.warning("⚠️ Could not generate annotated image")

    with col2:
        if not df.empty:
            st.markdown("#### 📊 Detection Summary")

            # Enhanced metrics in a grid
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>🔢 Objects Found</h4>
                        <h2 style="color: var(--accent-color);">{len(df)}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_b:
                if len(df) > 0:
                    avg_conf = df['Confidence'].str.rstrip(
                        '%').astype(float).mean()
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4>📈 Avg Confidence</h4>
                            <h2 style="color: var(--accent-secondary);">{avg_conf:.1f}%</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Detailed detection table
            st.markdown("#### 📋 Detailed Results")
            st.dataframe(df, use_container_width=True, height=300)

            # Object class distribution
            st.markdown("#### 📊 Object Distribution")
            object_counts = df['Object'].value_counts()

            st.markdown('<div class="chart-container">',
                        unsafe_allow_html=True)
            st.bar_chart(object_counts)
            st.markdown('</div>', unsafe_allow_html=True)

            # Top detected objects summary
            st.markdown("#### 🏆 Detection Summary")
            for idx, (obj_name, count) in enumerate(object_counts.head(5).items()):
                obj_df = df[df['Object'] == obj_name]
                avg_conf = obj_df['Confidence'].str.rstrip(
                    '%').astype(float).mean()

                confidence_color = "🟢" if avg_conf >= 80 else "🟡" if avg_conf >= 60 else "🟠"
                st.markdown(
                    f"**{idx+1}.** {confidence_color} **{obj_name}**: {count} object(s) • {avg_conf:.1f}% confidence"
                )

        else:
            st.markdown(
                f"""
                <div class="content-card" style="text-align: center; background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%); border-color: rgba(255, 193, 7, 0.4);">
                    <h4>🔍 No Objects Detected</h4>
                    <p>The AI model couldn't detect any objects with the current confidence threshold.</p>
                    <div style="margin: 1rem 0;">
                        <strong>💡 Try these suggestions:</strong><br>
                        • Lower the confidence threshold (currently {conf:.0%})<br>
                        • Use a higher resolution image<br>
                        • Ensure objects are clearly visible<br>
                        • Try a different image with common objects
                    </div>
                    <small style="opacity: 0.8;">Model: YOLOv8n • Threshold: {conf:.1%} • Resolution: {img_size}px</small>
                </div>
                """,
                unsafe_allow_html=True
            )

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    """
    <div class="content-card" style="text-align: center; margin-top: 4rem;">
        <h4>🚀 AI Vision Studio</h4>
        <p style="font-size: 1.1rem;">Powered by <strong>YOLOv8</strong> • Built with <strong>Streamlit</strong> • Next-generation Computer Vision</p>
        <p style="font-size: 0.95rem; color: var(--text-secondary); margin-top: 1.5rem;">
            ✨ Experience the future of AI-powered object detection<br>
            🌐 <a href="#" style="color: var(--accent-color); text-decoration: none;">Documentation</a> • 
            📧 <a href="#" style="color: var(--accent-secondary); text-decoration: none;">Support</a> • 
            🔗 <a href="#" style="color: var(--accent-tertiary); text-decoration: none;">GitHub</a>
        </p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 15px;">
            <span style="color: var(--accent-color); font-size: 1.5rem;">●</span>
            <span style="color: var(--accent-tertiary); font-size: 1.5rem;">●</span>
            <span style="color: var(--accent-secondary); font-size: 1.5rem;">●</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
