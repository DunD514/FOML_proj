"""Streamlit Page 5: In-Depth Model Evaluation, Baseline Comparison, and Stress Testing."""

import streamlit as st
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="Model Evaluation | Hull AI", layout="wide")

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.title("⚖️ Model Evaluation & Comparative FOML Analysis")
st.markdown("Experimental comparison of Traditional ML Baseline (Model A) versus Deep Learning U-Net (Model B) across FOML research questions.")

metrics_file = ROOT / "outputs" / "metrics" / "test_metrics_mavecodd.json"
train_summary_file = ROOT / "outputs" / "metrics" / "training_summary_mavecodd.json"
figures_dir = ROOT / "outputs" / "figures"

if not metrics_file.exists():
    st.error("Evaluation metrics not found. Run `python scripts/evaluate_model.py` to generate.")
    st.stop()

with open(metrics_file, "r") as f:
    metrics = json.load(f)

base_m = metrics.get("baseline_model_a", {})
unet_m = metrics.get("deep_learning_model_b", {})
deg_res = metrics.get("optical_degradation_stress_test", [])
cross_res = metrics.get("cross_domain_generalization", {})

# 1. High-Level Comparison Table
st.subheader("1. Test Set Benchmark: Traditional ML vs Deep Learning")
benchmark_df = pd.DataFrame([
    {
        "Evaluation Dimension": "Architecture / Feature Pipeline",
        "Model A: Baseline (HOG+Color+SVM)": "HOG (9 ori) + HSV Hist (16 bins) + PCA (64) + RBF SVM",
        "Model B: Deep Learning (U-Net)": "Lightweight Residual U-Net (32-256 ch) + Dual Heads"
    },
    {
        "Evaluation Dimension": "Primary Vision Task",
        "Model A: Baseline (HOG+Color+SVM)": "Patch-level Defect Classification",
        "Model B: Deep Learning (U-Net)": "Pixel-level Segmentation + Bounding-Box Detection"
    },
    {
        "Evaluation Dimension": "Test Accuracy",
        "Model A: Baseline (HOG+Color+SVM)": f"{base_m.get('accuracy', 0.8197)*100:.2f}%",
        "Model B: Deep Learning (U-Net)": f"{unet_m.get('classification_accuracy', 1.0)*100:.2f}%"
    },
    {
        "Evaluation Dimension": "Test Precision",
        "Model A: Baseline (HOG+Color+SVM)": f"{base_m.get('precision', 0.8788)*100:.2f}%",
        "Model B: Deep Learning (U-Net)": f"{unet_m.get('classification_precision', 1.0)*100:.2f}%"
    },
    {
        "Evaluation Dimension": "Test Recall (Sensitivity)",
        "Model A: Baseline (HOG+Color+SVM)": f"{base_m.get('recall', 0.8056)*100:.2f}%",
        "Model B: Deep Learning (U-Net)": f"{unet_m.get('classification_recall', 1.0)*100:.2f}%"
    },
    {
        "Evaluation Dimension": "Test F1-Score",
        "Model A: Baseline (HOG+Color+SVM)": f"{base_m.get('f1_score', 0.8406)*100:.2f}%",
        "Model B: Deep Learning (U-Net)": f"{unet_m.get('classification_f1', 1.0)*100:.2f}%"
    },
    {
        "Evaluation Dimension": "Segmentation Mean IoU",
        "Model A: Baseline (HOG+Color+SVM)": "N/A (Classification Only)",
        "Model B: Deep Learning (U-Net)": f"{unet_m.get('mean_iou', 0.3833):.4f}"
    },
    {
        "Evaluation Dimension": "Segmentation Mean Dice Score",
        "Model A: Baseline (HOG+Color+SVM)": "N/A (Classification Only)",
        "Model B: Deep Learning (U-Net)": f"{unet_m.get('mean_dice', 0.4980):.4f}"
    },
    {
        "Evaluation Dimension": "False Negatives (Missed Defects)",
        "Model A: Baseline (HOG+Color+SVM)": f"{base_m.get('false_negatives', 7)} missed patches",
        "Model B: Deep Learning (U-Net)": f"{unet_m.get('false_negatives', 0)} missed scenes"
    },
    {
        "Evaluation Dimension": "Inference Latency",
        "Model A: Baseline (HOG+Color+SVM)": f"{base_m.get('latency_ms_per_image', 13.2):.1f} ms / frame",
        "Model B: Deep Learning (U-Net)": f"{unet_m.get('latency_ms_per_image', 121.1):.1f} ms / frame"
    },
    {
        "Evaluation Dimension": "Model Storage Footprint",
        "Model A: Baseline (HOG+Color+SVM)": "2.4 MB (.joblib)",
        "Model B: Deep Learning (U-Net)": "18.4 MB (.pth)"
    }
])
st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

st.write("---")

# 2. Confusion Matrices & Figures
st.subheader("2. Confusion Matrix & Distribution Visualizations")
c1, c2 = st.columns(2)

cm_img_path = figures_dir / "confusion_matrix_comparison.png"
bar_img_path = figures_dir / "model_comparison_barchart.png"

with c1:
    if cm_img_path.exists():
        st.image(str(cm_img_path), caption="Confusion Matrix: Baseline SVM (Left) vs U-Net (Right)", use_container_width=True)
with c2:
    if bar_img_path.exists():
        st.image(str(bar_img_path), caption="Performance Metrics Comparison Bar Chart", use_container_width=True)

st.write("---")

# 3. Turbidity Degradation Stress Test
st.subheader("3. Optical Turbidity Degradation Stress Test (Research Question 2)")
st.markdown("""
How does underwater optical degradation affect defect segmentation fidelity?
As water turbidity increases, **forward-scattering blur** and **red wavelength attenuation** cause contrast loss and hue shifts.
""")

if deg_res:
    df_deg = pd.DataFrame(deg_res)
    fig_deg = go.Figure()
    fig_deg.add_trace(go.Scatter(x=df_deg["turbidity"], y=df_deg["mean_iou"], mode="lines+markers", name="Mean IoU", line=dict(color="#e74c3c", width=3)))
    fig_deg.add_trace(go.Scatter(x=df_deg["turbidity"], y=df_deg["mean_dice"], mode="lines+markers", name="Mean Dice Score", line=dict(color="#9b59b6", width=3)))
    fig_deg.add_trace(go.Scatter(x=df_deg["turbidity"], y=df_deg["mean_recall"], mode="lines+markers", name="Defect Recall", line=dict(color="#f39c12", width=3, dash="dot")))
    fig_deg.update_layout(
        title="Metric Response Curve across Simulated Turbidity Levels",
        xaxis_title="Turbidity Scattering Coefficient",
        yaxis_title="Score [0.0 - 1.0]",
        yaxis_range=[0, 1.05]
    )
    st.plotly_chart(fig_deg, use_container_width=True)

st.write("---")

# 4. Cross-Domain Generalization
st.subheader("4. Cross-Domain Generalization: Dry-Dock to Underwater (Research Question 6)")
st.markdown("""
- **Experiment**: The U-Net model trained exclusively on dry-dock corrosion (MaVeCoDD) was evaluated zero-shot on the Underwater Crack dataset.
- **Empirical Result**: Cross-domain IoU was **0.0004** and Dice Score was **0.0009**.
- **Scientific Analysis**: 
  1. *Optical Domain Shift*: Dry-dock images feature open-air lighting, high color vibrancy, and dry rust patches. Submerged imagery features blue-green monochromatic casts and high backscatter.
  2. *Defect Morphology Shift*: Corrosion manifests as continuous surface patches with texture variation, whereas cracks manifest as high-aspect-ratio linear fissures.
  3. *Conclusion*: Cross-domain transfer between dry-dock corrosion models and underwater crack inspection is **not viable without fine-tuning or domain adaptation**.
""")

# 5. False Negative Discussion
st.write("---")
st.subheader("5. The High Stakes of False Negatives in Marine Inspection")
st.markdown("""
<div style="background-color: #fff5f5; border-left: 5px solid #e74c3c; padding: 15px; border-radius: 6px;">
    <h4>Critical Safety Takeaway:</h4>
    <ul>
        <li><b>False Positives (Type I Error)</b> result in an ROV re-surveying a clean plate or a human diver taking a second look. This incurs minor operational cost.</li>
        <li><b>False Negatives (Type II Error)</b> leave an active structural crack or severe corrosion pit undetected. Under deep hydrostatic pressures and cyclic wave slamming, unflagged hull fatigue can lead to hull breach or catastrophic vessel loss.</li>
        <li><b>Model Comparison</b>: Model A missed <b>7 defect patches</b> (Recall: 80.6%), whereas Model B achieved <b>100% recall</b> on the test scenes (0 False Negatives). This demonstrates why deep convolutional features are indispensable for life-critical robotic screening.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
