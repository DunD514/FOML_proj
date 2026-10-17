import streamlit as st
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Image Inspection | Hull AI", layout="wide")

from src.utils.reproducibility import get_device
from src.data.dataset_loader import scan_mavecodd_samples, scan_crack_samples, load_image_and_mask
from src.data.preprocessing import apply_underwater_degradation
from src.models.predict import HullInspectionPredictor

st.title("🔬 Interactive Hull Image Inspection")
st.markdown("Run live deep-learning defect segmentation, bounding-box localization, and baseline comparison on ship hull imagery.")

# Check model availability
checkpoints_dir = ROOT / "models" / "checkpoints"
baseline_path = checkpoints_dir / "baseline_mavecodd.joblib"
unet_path = checkpoints_dir / "unet_mavecodd.pth"

if not baseline_path.exists() and not unet_path.exists():
    st.error("⚠️ Model checkpoints not found! Please run `python scripts/train_model.py` first to generate trained models.")
    st.stop()

@st.cache_resource
def load_predictor():
    dev = get_device("auto")
    return HullInspectionPredictor(
        baseline_path=baseline_path if baseline_path.exists() else None,
        unet_path=unet_path if unet_path.exists() else None,
        device=dev,
        confidence_threshold=0.35,
        mask_threshold=0.40
    )

predictor = load_predictor()

# Input Selection Tabs
tab_select, tab_upload = st.tabs(["📁 Select Sample from Dataset", "📤 Upload Custom Hull Image"])

selected_image = None
defect_hint = "corrosion"
sample_source = "mavecodd"

with tab_select:
    col_ds, col_item = st.columns([1, 2])
    with col_ds:
        dataset_choice = st.selectbox("Choose Dataset Source", ["MaVeCoDD (Corrosion)", "Underwater Crack (Cracks)", "Synthetic Sample Hull"])
    
    with col_item:
        if dataset_choice == "MaVeCoDD (Corrosion)":
            samples = scan_mavecodd_samples(ROOT / "data/raw/mavecodd", zip_fallback=ROOT / "ry392rp8cj-1.zip")
            defect_hint = "corrosion"
            sample_source = "mavecodd"
        elif dataset_choice == "Underwater Crack (Cracks)":
            samples = scan_crack_samples(ROOT / "data/raw/underwater_crack", zip_fallback=ROOT / "under-crack-images.zip", limit=50)
            defect_hint = "crack"
            sample_source = "underwater_crack"
        else:
            sample_dir = ROOT / "data/sample/images"
            if sample_dir.exists():
                samples = [{"image_path": str(f), "mask_path": str(ROOT / "data/sample/masks" / f"{f.stem}_mask.png"), "image_id": f.stem, "dataset_source": "synthetic"} for f in sorted(sample_dir.glob("*.jpg"))]
            else:
                samples = []
            defect_hint = "corrosion"
            sample_source = "synthetic"

        if samples:
            sample_names = [f"{s.get('image_id', f'sample_{i}')} ({s.get('resolution_tier', '')})" for i, s in enumerate(samples[:60])]
            chosen_idx = st.selectbox("Select Inspection Image", range(len(sample_names)), format_func=lambda i: sample_names[i])
            chosen_sample = samples[chosen_idx]
            raw_img, gt_mask = load_image_and_mask(chosen_sample)
            selected_image = raw_img
        else:
            st.warning("No samples found in chosen dataset directory.")

with tab_upload:
    uploaded = st.file_uploader("Upload Hull Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        selected_image = np.array(pil_img)
        defect_hint = st.selectbox("Expected Defect Type", ["corrosion", "crack"])
        sample_source = "user_upload"

# Optical Degradation Controls
st.sidebar.header("🌊 Simulated Optical Degradation")
st.sidebar.markdown("Simulate underwater camera conditions using the Jaffe-McGlamery optical model:")
enable_degradation = st.sidebar.checkbox("Apply Underwater Degradation", value=False)
turbidity = st.sidebar.slider("Turbidity / Scattering", 0.0, 0.8, 0.25, 0.05, disabled=not enable_degradation)
red_att = st.sidebar.slider("Red Channel Absorption", 0.0, 0.9, 0.65, 0.05, disabled=not enable_degradation)
contrast = st.sidebar.slider("Contrast Attenuation", 0.2, 1.0, 0.80, 0.05, disabled=not enable_degradation)

if selected_image is not None:
    # Resize display copy if very large for smooth browser rendering
    h, w = selected_image.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        proc_image = cv2.resize(selected_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        proc_image = selected_image.copy()

    if enable_degradation:
        proc_image = apply_underwater_degradation(
            proc_image, turbidity=turbidity, red_attenuation=red_att, contrast_factor=contrast
        )

    st.write("---")
    btn = st.button("🚀 Run Defect Inspection Model", type="primary", use_container_width=True)

    if btn or st.session_state.get("ran_inference", False):
        st.session_state["ran_inference"] = True

        with st.spinner("Analyzing image with U-Net and Baseline models..."):
            res = predictor.predict(proc_image, defect_hint=defect_hint)

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Input Observation")
            st.image(proc_image, caption=f"Input Image ({w}x{h}) | Source: {sample_source}", use_container_width=True)

        with col2:
            st.subheader("🎯 Model B Detection & Mask Overlay")
            overlay = predictor.draw_overlay(proc_image, res, show_mask=True, show_boxes=True)
            st.image(overlay, caption="Segmented Defect Mask & Localized Bounding Boxes", use_container_width=True)

        # Verdict Cards
        st.write("---")
        k1, k2, k3, k4, k5 = st.columns(5)

        is_defect = res["defect_detected"]
        k1.metric("Predicted Condition", res["predicted_defect_class"].replace("_", " ").title())
        k2.metric("Detection Confidence", f"{res['confidence_score']*100:.1f}%")
        k3.metric("Defects Localized", len(res["bounding_boxes"]))
        k4.metric("Surface Defect Area", f"{res['defect_area_fraction']*100:.2f}%")
        k5.metric("Inference Latency", f"{res['inference_time_ms']:.1f} ms")

        # Probability Map & Detailed Inspection
        st.subheader("🗺️ Defect Probability Distribution")
        fig_prob = px.imshow(
            res["probability_map"],
            color_continuous_scale="Jet",
            title="Pixel-Level Defect Probability Heatmap (U-Net Output)"
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        if res["bounding_boxes"]:
            st.subheader("📍 Extracted Defect Bounding Boxes")
            box_data = [
                {
                    "Defect ID": b["defect_id"],
                    "Category": b["defect_type"],
                    "BBox [Xmin, Ymin, Xmax, Ymax]": str(b["bbox_pixel"]),
                    "Normalized BBox": str(b["bbox_normalized"]),
                    "Centroid (Norm)": f"({b['centroid_normalized'][0]:.2f}, {b['centroid_normalized'][1]:.2f})",
                    "Pixel Area": b["pixel_area"],
                    "Area Fraction": f"{b['area_fraction']*100:.2f}%"
                }
                for b in res["bounding_boxes"]
            ]
            st.dataframe(box_data, use_container_width=True)

        # Baseline Comparison
        if res.get("baseline_result"):
            base_r = res["baseline_result"]
            st.subheader("⚖️ Model A (Baseline SVM) Comparison")
            st.markdown(f"""
            - **Baseline Prediction**: `{base_r['defect_class']}`
            - **Baseline Confidence**: `{base_r['confidence']*100:.1f}%`
            - **Baseline Latency**: `{base_r['latency_ms']:.1f} ms`
            - **Agreement between Models**: `{"✅ Models Agree" if (base_r['prediction'] == (1 if is_defect else 0)) else "⚠️ Disagreement (Review Required)"}`
            """)

        st.warning(
            "⚠️ **Formal Notice:** Image-based predictions represent suspected surface defects requiring on-site confirmation. "
            "They do not provide ultrasonic thickness or structural stress measurements."
        )
