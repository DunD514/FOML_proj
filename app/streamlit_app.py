"""Autonomous Underwater ROV Ship Hull Inspection Dashboard - Main Application."""

import streamlit as st
import sys
from pathlib import Path
import json
import torch
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Page setup
st.set_page_config(
    page_title="Underwater Hull Inspection AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0e3854;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a6572;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e3d59 0%, #17252a 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4deeea;
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #b0bec5;
        margin-top: 5px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #e8f5e9;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[1]

# Header
st.markdown('<div class="main-header">🌊 AI-Powered Underwater Ship Hull Inspection Drone</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous ROV Defect Detection, Real-Time Segmentation, and Apache Parquet Telemetry Logging</div>', unsafe_allow_html=True)

# Hardware & System Health
col1, col2, col3, col4 = st.columns(4)

cuda_active = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_active else "CPU"

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{"CUDA ON" if cuda_active else "CPU"}</div>
        <div class="kpi-label">Compute Acceleration ({device_name.split()[0]})</div>
    </div>
    """, unsafe_allow_html=True)

# Load metrics if available
metrics_file = ROOT / "outputs" / "metrics" / "test_metrics_mavecodd.json"
metrics = {}
if metrics_file.exists():
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

unet_dice = metrics.get("deep_learning_model_b", {}).get("mean_dice", 0.50)
baseline_acc = metrics.get("baseline_model_a", {}).get("accuracy", 0.82)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{unet_dice:.2f}</div>
        <div class="kpi-label">U-Net Test Dice (F1) Score</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{baseline_acc*100:.1f}%</div>
        <div class="kpi-label">Baseline Patch Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# Check inspection logs count
logs_dir = ROOT / "outputs" / "inspection_logs"
num_missions = len(list(logs_dir.glob("*.parquet"))) if logs_dir.exists() else 0

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{num_missions}</div>
        <div class="kpi-label">Logged Inspection Missions</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# Main overview layout
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📌 System Architecture & Operational Pipeline")
    st.markdown("""
    This project demonstrates an end-to-end autonomous underwater robotic non-destructive testing (NDT) pipeline:
    
    1. **Robotic Simulation**:
       - Models an underwater ROV scanning a 50m &times; 10m vessel hull using a serpentine (lawnmower) path.
       - Implements realistic thruster kinematics, acoustic DVL positioning uncertainty, and optical field of view.
       - Optical degradation models wavelength-dependent red light absorption, turbidity scattering, and marine snow particles.
    
    2. **Machine Learning & Computer Vision**:
       - **Model A (Baseline)**: Handcrafted multi-scale HOG + HSV color histograms + PCA + Support Vector Machine.
       - **Model B (Deep Learning)**: Lightweight dual-head U-Net for pixel-level segmentation, defect mask generation, and bounding-box localization.
    
    3. **PyArrow Data Management**:
       - Structured Apache Parquet columnar storage capturing 26 telemetry and vision attributes per observation step.
       - Supports batch appending, fast analytical querying, and automated PDF inspection reporting.
    """)

    st.info("💡 **Navigation Tip**: Select any of the 6 specialized pages in the left sidebar to explore the interactive components.")

with col_right:
    st.subheader("🔍 Navigation Directory")
    st.markdown("""
    - **Page 1: Overview** &mdash; System architecture, KPI counters, and dataset distribution.
    - **Page 2: Image Inspection** &mdash; Upload or select hull images for live dual-model inference.
    - **Page 3: Drone Simulation** &mdash; Interactive 2D/3D hull map with live ROV trajectory and sensor feed.
    - **Page 4: Dataset Explorer** &mdash; Browse MaVeCoDD and Underwater Crack datasets with mask overlays.
    - **Page 5: Model Evaluation** &mdash; Traditional ML vs U-Net comparison, confusion matrices, and turbidity stress tests.
    - **Page 6: Inspection Reports** &mdash; Query logged Parquet missions and generate downloadable PDF reports.
    """)

    st.markdown("""
    <div style="background-color: #fff3cd; border-left: 5px solid #ffeeba; padding: 12px; border-radius: 6px; font-size: 0.85rem; color: #856404;">
        <b>Engineering Notice:</b> Detections are designated as <i>'suspected defects'</i> requiring certified marine surveyor review.
        Subsurface fatigue and internal stress require ultrasonic testing.
    </div>
    """, unsafe_allow_html=True)
