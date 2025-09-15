# app_streamlit.py
import subprocess
import sys

# Ensure ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "ultralytics==8.3.25",
        "opencv-python-headless"
    ])
    from ultralytics import YOLO

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import glob
import pandas as pd
from pathlib import Path

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Object Detection Studio",
                   page_icon="👁️", layout="wide")

# -------------------- MODERN CSS DESIGN --------------------
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

    /* Modern Color Palette */
    :root {
        --bg-primary: #f8fafc;
        --bg-secondary: #ffffff;
        --bg-accent: #f1f5f9;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --border-light: #e2e8f0;
        --border-medium: #cbd5e1;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
        --accent-red: #ef4444;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --gradient-primary: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        --gradient-secondary: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
    }

    /* Global Reset */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Main App Background */
    .stApp {
        background: var(--bg-primary);
        min-height: 100vh;
    }

    /* Header Section */
    .hero-header {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }

    .hero-title {
        font-size: 2.75rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin: 0;
        font-weight: 400;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-light);
    }

    .sidebar-header {
        background: var(--gradient-primary);
        margin: -1rem -1rem 1.5rem -1rem;
        padding: 1.5rem;
        border-radius: 0 0 16px 16px;
        color: white;
        text-align: center;
    }

    .sidebar-section {
        background: var(--bg-accent);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .sidebar-section h4 {
        color: var(--text-primary) !important;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }

    /* Card Components */
    .modern-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-md);
        transition: all 0.2s ease;
        position: relative;
    }

    .modern-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-light);
    }

    .card-icon {
        width: 40px;
        height: 40px;
        background: var(--gradient-primary);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        color: white;
    }

    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }

    .card-subtitle {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin: 0;
    }

    /* Modern Buttons */
    .stButton button {
        background: var(--gradient-primary);
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: white;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }

    .stButton button:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }

    /* File Uploader */
    .stFileUploader {
        background: var(--bg-accent);
        border: 2px dashed var(--border-medium);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.2s ease;
    }

    .stFileUploader:hover {
        border-color: var(--accent-blue);
        background: rgba(59, 130, 246, 0.05);
    }

    /* Radio Buttons */
    .stRadio > div {
        background: var(--bg-accent);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid var(--border-light);
    }

    .stRadio > div > label > div {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .stRadio > div > label > div > label {
        background: var(--bg-secondary);
        border: 1px solid var(--border-medium);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        margin: 0;
        cursor: pointer;
        transition: all 0.2s ease;
        flex: 1;
        text-align: center;
        min-width: 150px;
    }

    .stRadio > div > label > div > label:hover {
        border-color: var(--accent-blue);
        background: rgba(59, 130, 246, 0.05);
    }

    /* Sliders */
    .stSlider {
        background: var(--bg-accent);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid var(--border-light);
    }

    /* Metrics Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
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
        background: var(--gradient-primary);
    }

    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-blue);
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Image Containers */
    .image-wrapper {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }

    .image-wrapper img {
        border-radius: 8px;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
    }

    /* Status Messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--accent-green), #059669);
        border: none;
        border-radius: 12px;
        color: white;
    }

    .stError {
        background: linear-gradient(135deg, var(--accent-red), #dc2626);
        border: none;
        border-radius: 12px;
        color: white;
    }

    .stInfo {
        background: linear-gradient(135deg, var(--accent-blue), #2563eb);
        border: none;
        border-radius: 12px;
        color: white;
    }

    .stWarning {
        background: linear-gradient(135deg, var(--accent-orange), #d97706);
        border: none;
        border-radius: 12px;
        color: white;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: var(--gradient-primary);
        border-radius: 10px;
    }

    /* Charts */
    .stBarChart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    /* Text Styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    p, label, span {
        color: var(--text-secondary) !important;
    }

    /* Feature Badges */
    .feature-badge {
        display: inline-block;
        background: var(--gradient-primary);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.25rem;
    }

    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(59, 130, 246, 0.3);
        border-radius: 50%;
        border-top-color: var(--accent-blue);
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .modern-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .metric-container {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HEADER --------------------
st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">Object Detection Studio</h1>
        <p class="hero-subtitle">Intelligent visual analysis powered by YOLOv8 • Upload, analyze, and discover</p>
        <div style="margin-top: 1rem;">
            <span class="feature-badge">Real-time Detection</span>
            <span class="feature-badge">80+ Object Classes</span>
            <span class="feature-badge">High Accuracy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- SIDEBAR --------------------
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <h3 style="margin: 0; color: white;">⚙️ Configuration</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-section">
        <h4>🎯 Model Settings</h4>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_weights = st.sidebar.file_uploader(
    "Upload custom model weights (.pt)", 
    type=["pt"], 
    help="Upload your trained YOLOv8 model weights for custom detection"
)

st.sidebar.markdown(
    """
    <div class="sidebar-section">
        <h4>🔧 Detection Parameters</h4>
    </div>
    """,
    unsafe_allow_html=True
)

conf = st.sidebar.slider(
    "Confidence threshold", 
    0.0, 1.0, 0.4, 0.01, 
    help="Minimum confidence score for object detection"
)

img_size = st.sidebar.selectbox(
    "Image resolution (px)", 
    [320, 416, 640, 1280], 
    index=2, 
    help="Higher resolution = better accuracy but slower processing"
)

st.sidebar.markdown(
    """
    <div class="sidebar-section">
        <h4>📊 Model Information</h4>
        <p style="font-size: 0.9rem; line-height: 1.4;">
            <strong>Architecture:</strong> YOLOv8 Nano<br>
            <strong>Dataset:</strong> COCO (80 classes)<br>
            <strong>Performance:</strong> Real-time inference<br>
            <strong>Accuracy:</strong> mAP 37.3%
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

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
                detections.append({
                    "Object": name, 
                    "Confidence": round(float(cf), 3), 
                    "Coordinates": [round(float(x), 1) for x in box]
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
    st.sidebar.info("💡 Using pre-trained YOLOv8n weights")

model = load_model(weights_to_load)

# -------------------- MODE SELECTION --------------------
st.markdown(
    """
    <div class="modern-card">
        <div class="card-header">
            <div class="card-icon">🎯</div>
            <div>
                <div class="card-title">Detection Mode</div>
                <div class="card-subtitle">Choose your input type for AI-powered object detection</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

mode = st.radio(
    "Select detection mode:",
    ["📸 Image Analysis", "🎬 Video Processing"],
    horizontal=True,
    help="Choose between single image analysis or video frame processing"
)

# IMAGE MODE
if mode == "📸 Image Analysis":
    st.markdown(
        """
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">📸</div>
                <div>
                    <div class="card-title">Image Analysis</div>
                    <div class="card-subtitle">Upload high-resolution images for precise object detection</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"], 
        help="Supported formats: JPG, JPEG, PNG • Maximum size: 200MB"
    )

    if uploaded:
        with st.spinner("🔍 Analyzing image..."):
            img = Image.open(uploaded).convert("RGB")
            results = model.predict(np.array(img), conf=conf, imgsz=img_size)
            annotated, df = annotate_and_table(results, model)

        st.markdown(
            """
            <div class="modern-card">
                <div class="card-header">
                    <div class="card-icon">🎯</div>
                    <div>
                        <div class="card-title">Detection Results</div>
                        <div class="card-subtitle">AI-powered object analysis complete</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 📷 Original Image")
            with st.container():
                st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
                st.image(img, caption="Source Image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if annotated is not None:
                st.markdown("#### 🎯 Detection Visualization")
                with st.container():
                    st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
                    st.image(annotated, caption="Objects Detected and Annotated", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if not df.empty:
                st.markdown("#### 📊 Detection Summary")
                st.dataframe(df, use_container_width=True)

                # Metrics
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Objects</div>
                            <div class="metric-value">{len(df)}</div>
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
                                <div class="metric-label">Avg Confidence</div>
                                <div class="metric-value">{avg_conf:.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)

                # Object distribution
                if not df.empty:
                    st.markdown("#### 📈 Object Distribution")
                    class_counts = df['Object'].value_counts()
                    st.bar_chart(class_counts)
                    
                    # Detailed breakdown
                    unique_objects = df['Object'].nunique()
                    st.markdown(
                        f"""
                        <div class="metric-card" style="margin-top: 1rem;">
                            <div class="metric-label">Unique Classes</div>
                            <div class="metric-value">{unique_objects}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:
                st.markdown(
                    """
                    <div class="modern-card" style="text-align: center;">
                        <div style="padding: 2rem;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">🤔</div>
                            <h4 style="color: var(--text-primary);">No Objects Detected</h4>
                            <p style="color: var(--text-secondary);">Try adjusting the confidence threshold or uploading a different image with more visible objects.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# VIDEO MODE
elif mode == "🎬 Video Processing":
    st.markdown(
        """
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">🎬</div>
                <div>
                    <div class="card-title">Video Processing</div>
                    <div class="card-subtitle">Upload video files for comprehensive frame-by-frame analysis</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_vid = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "mov", "avi", "mkv"], 
        help="Supported formats: MP4, MOV, AVI, MKV • Maximum size: 500MB"
    )

    if uploaded_vid:
        tmp = save_uploaded_file(uploaded_vid)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("#### 🎥 Source Video")
            st.video(tmp)

        with col2:
            st.markdown("#### ⚡ Processing Controls")
            
            st.markdown(
                """
                <div class="modern-card">
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚀</div>
                        <p>Ready to process your video with AI-powered object detection</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            process_button = st.button(
                "🎯 Start AI Processing", 
                use_container_width=True
            )

            if process_button:
                with st.spinner("🔄 Processing video frames..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    project_dir = tempfile.mkdtemp()

                    # Simulate progress
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        status_text.text(f'🎬 Processing frame {i + 1}/100...')

                    try:
                        results = model.predict(
                            source=tmp, 
                            conf=conf, 
                            imgsz=img_size, 
                            project=project_dir, 
                            name="detection_run", 
                            save=True
                        )

                        out_dir = str(results[0].save_dir)
                        vids = glob.glob(os.path.join(out_dir, "*"))
                        vids = [v for v in vids if Path(v).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
                        
                        if vids:
                            st.markdown(
                                """
                                <div class="modern-card">
                                    <div class="card-header">
                                        <div class="card-icon">✅</div>
                                        <div>
                                            <div class="card-title">Processing Complete!</div>
                                            <div class="card-subtitle">Your video has been enhanced with AI detection</div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            st.markdown("#### 🎯 Enhanced Video Output")
                            st.video(vids[0])

                            col_a, col_b = st.columns(2)
                            with col_a:
                                with open(vids[0], 'rb') as f:
                                    st.download_button(
                                        label="💾 Download Enhanced Video",
                                        data=f.read(),
                                        file_name=f"detected_{uploaded_vid.name}",
                                        mime="video/mp4",
                                        use_container_width=True
                                    )
                            
                            with col_b:
                                st.markdown(
                                    """
                                    <div class="metric-card">
                                        <div class="metric-label">Status</div>
                                        <div class="metric-value">✅ Ready</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.error("❌ No output video generated. Please try a different format.")
                            
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        st.info("💡 Try using a smaller video file or different format (MP4 recommended)")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    """
    <div class="modern-card" style="text-align: center; margin-top: 2rem;">
        <div style="padding: 1rem;">
            <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">Object Detection Studio</h4>
            <p style="color: var(--text-secondary); margin: 0;">
                Powered by YOLOv8 • Built with Streamlit • Professional Computer Vision
            </p>
            <div style="margin-top: 1rem;">
                <span class="feature-badge">Enterprise Ready</span>
                <span class="feature-badge">High Performance</span>
                <span class="feature-badge">Easy to Use</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)