import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Dataset Explorer | Hull AI", layout="wide")

from src.data.dataset_loader import scan_mavecodd_samples, scan_crack_samples, load_image_and_mask
from src.data.split_dataset import load_split_manifest

st.title("📂 Dataset Explorer & Annotation Inspector")
st.markdown("Detailed breakdown of primary (MaVeCoDD), secondary (Underwater Crack), and synthetic datasets.")

# Dataset selection
ds_tab = st.radio("Select Dataset to Explore", ["MaVeCoDD (Marine Vessel Corrosion)", "Underwater Crack Dataset", "Train/Val/Test Splits Manifest"], horizontal=True)

if ds_tab == "MaVeCoDD (Marine Vessel Corrosion)":
    st.subheader("🚢 MaVeCoDD: Marine Vessel Corrosion in Dry-Dock")
    st.markdown("""
    - **Origin**: Mendeley Data (`dx.doi.org/10.17632/ry392rp8cj.1`), Kamzelis et al.
    - **Subject**: Real vessel hulls undergoing dry-dock repair and blasting.
    - **Annotation Type**: Pixel-level segmentation masks (`*_label.jpg`). Defect pixels retain color; background is near-zero black.
    - **Resolution Tiers**: HiRes (up to 4032 &times; 3024) and LoRes (1920 &times; 1080).
    """)

    mav_samples = scan_mavecodd_samples(ROOT / "data/raw/mavecodd", zip_fallback=ROOT / "ry392rp8cj-1.zip")
    st.metric("Total Valid Paired Samples", len(mav_samples))

    if mav_samples:
        df_mav = pd.DataFrame(mav_samples)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("##### Resolution Distribution")
            fig_pie = px.pie(df_mav, names="resolution_tier", title="MaVeCoDD Resolution Split", color_discrete_sequence=["#0984e3", "#00cec9"])
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.write("##### Sample Gallery & Mask Inspector")
            sample_idx = st.slider("Select Sample Index", 0, len(mav_samples)-1, 0)
            s = mav_samples[sample_idx]
            img, mask = load_image_and_mask(s)

            alpha = st.slider("Mask Overlay Opacity", 0.0, 1.0, 0.4, 0.05)
            color_mask = np.zeros_like(img)
            color_mask[mask > 0] = [231, 76, 60]  # Red overlay
            blended = cv2.addWeighted(img, 1.0 - alpha, color_mask, alpha, 0)

            c_img1, c_img2 = st.columns(2)
            c_img1.image(img, caption=f"Raw Image: {s['image_id']} ({img.shape[1]}x{img.shape[0]})", use_container_width=True)
            c_img2.image(blended, caption=f"Ground-Truth Mask (Defect: {(np.sum(mask>0)/mask.size)*100:.2f}% of pixels)", use_container_width=True)

elif ds_tab == "Underwater Crack Dataset":
    st.subheader("🌊 Underwater Concrete & Submerged Structure Cracks")
    st.markdown("""
    - **Origin**: Mendeley Data (`dx.doi.org/10.17632/g962rbm77b/1`), Gao et al.
    - **Subject**: Submerged underwater structures and concrete piers with visible fissure cracking.
    - **Annotation Type**: Binary PNG segmentation masks (Crack = 255, Background = 0).
    - **Standard Dimensions**: 512 &times; 512 &times; 3 RGB.
    """)

    crack_samples = scan_crack_samples(ROOT / "data/raw/underwater_crack", zip_fallback=ROOT / "under-crack-images.zip", limit=100)
    st.metric("Indexed Samples (Showing top 100 of 1,037)", len(crack_samples))

    if crack_samples:
        sample_idx = st.slider("Select Crack Sample Index", 0, len(crack_samples)-1, 0)
        s = crack_samples[sample_idx]
        img, mask = load_image_and_mask(s)

        alpha = st.slider("Crack Mask Overlay Opacity", 0.0, 1.0, 0.45, 0.05)
        color_mask = np.zeros_like(img)
        color_mask[mask > 0] = [255, 107, 107]
        blended = cv2.addWeighted(img, 1.0 - alpha, color_mask, alpha, 0)

        c1, c2 = st.columns(2)
        c1.image(img, caption=f"Underwater Image: {s['image_id']} (512x512)", use_container_width=True)
        c2.image(blended, caption=f"Ground-Truth Crack Mask (Coverage: {(np.sum(mask>0)/mask.size)*100:.2f}%)", use_container_width=True)

else:
    st.subheader("📊 Train / Validation / Test Stratification & Split Manifests")
    st.markdown("""
    To ensure scientific rigor and avoid test set contamination:
    - **Grouping**: Samples are grouped by vessel hull scene (`scene_id`) or video capture sequence.
    - **Proportions**: 70% Training, 15% Validation, 15% Test.
    - **No Data Leakage**: Near-duplicate frames from the same video are strictly confined to a single split partition.
    """)

    manifest_file = ROOT / "data" / "processed" / "split_manifest_mavecodd.json"
    if manifest_file.exists():
        splits = load_split_manifest(manifest_file)
        split_summary = pd.DataFrame([
            {"Partition": "Train (70%)", "Scene Count": len(splits["train"]), "Purpose": "Model parameter optimization and augmentations"},
            {"Partition": "Validation (15%)", "Scene Count": len(splits["val"]), "Purpose": "Hyperparameter tuning, early stopping, and threshold calibration"},
            {"Partition": "Test (15%)", "Scene Count": len(splits["test"]), "Purpose": "Held-out unbiased evaluation and benchmark comparison"},
        ])
        st.dataframe(split_summary, use_container_width=True, hide_index=True)

        fig_splits = px.bar(
            split_summary,
            x="Partition",
            y="Scene Count",
            color="Partition",
            color_discrete_sequence=["#0984e3", "#fdcb6e", "#00b894"],
            title="MaVeCoDD Leakage-Free Split Partition Counts"
        )
        st.plotly_chart(fig_splits, use_container_width=True)
    else:
        st.warning("Split manifest not found. Run `python scripts/prepare_dataset.py` to generate.")
