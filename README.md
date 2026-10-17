# 🌊 AI-Powered Underwater Ship Hull Inspection Drone

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.8%2Bcu129-red.svg)](https://pytorch.org/)
[![Apache Arrow](https://img.shields.io/badge/Apache%20Arrow-21.0-orange.svg)](https://arrow.apache.org/)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-1.50-FF4B4B.svg)](https://streamlit.io/)
[![Test Suite](https://img.shields.io/badge/PyTest-16%20Passed-brightgreen.svg)](https://pytest.org/)

An end-to-end, academic-grade **Fundamentals of Machine Learning (FOML)** project simulating an autonomous underwater remotely operated vehicle (ROV) inspecting commercial ship hulls for visible structural defects (corrosion scale, oxidation, and fatigue cracks).

The system integrates **traditional machine learning (HOG+SVM)**, **deep convolutional vision (Residual U-Net)**, **robotics trajectory and optical physics simulation**, **columnar Apache Parquet telemetry logging via PyArrow**, and an **interactive multi-page Streamlit inspection dashboard** with automated PDF report generation.

---

## 📌 Key System Features

1. **Dual Machine Learning Pipeline**:
   - **Model A (Traditional ML Baseline)**: Handcrafted multi-scale Histogram of Oriented Gradients (HOG) + HSV color histograms + PCA (64 components) + RBF Support Vector Machine.
   - **Model B (Deep Learning U-Net)**: Lightweight dual-head Residual U-Net delivering both pixel-level defect segmentation masks and scene-level defect classification.
   - **Loss Formulation for Severe Class Imbalance**: Combined Binary Cross-Entropy + Soft Dice Loss ($\alpha=1.0, \beta=1.2, \gamma=0.5$).

2. **Underwater Robotics & Optical Physics Simulation**:
   - **Parametric Hull Geometry**: 50.0m $\times$ 10.0m vessel side surface partitioned into 10 structural inspection zones (Bow, Midship, Stern, Keel, Waterline).
   - **ROV Kinematics**: Serpentine (lawnmower) path planner, thruster dynamics, acoustic DVL/USBL positioning uncertainty ($\sigma_{\text{pos}}=0.04$m), and coverage grid tracking.
   - **Jaffe-McGlamery Optical Model**: Wavelength-dependent red light absorption ($c=0.65$), forward-scattering turbidity blur, contrast attenuation, and suspended marine snow particles.

3. **High-Performance Columnar Storage**:
   - Strongly-typed Apache Arrow schema capturing 26 telemetry and inspection attributes per observation step.
   - Parquet serialization with Snappy compression for sub-millisecond analytical queries and mission replay.

4. **Interactive 6-Page Streamlit Dashboard**:
   - **Page 1: Overview** &mdash; System KPIs, architecture flowchart, and dataset breakdowns.
   - **Page 2: Image Inspection** &mdash; Live dual-model inference on uploaded or dataset imagery with interactive probability heatmaps.
   - **Page 3: Drone Simulation** &mdash; Interactive Plotly 2D/3D hull map with live ROV trajectory and degraded camera feed.
   - **Page 4: Dataset Explorer** &mdash; Interactive sample browser with opacity-controlled ground-truth mask overlays.
   - **Page 5: Model Evaluation** &mdash; Side-by-side benchmark tables, confusion matrices, and turbidity stress tests.
   - **Page 6: Inspection Reports** &mdash; Parquet telemetry browser with dynamic PDF survey report downloads via ReportLab.

---

## 📊 Experimental Benchmark Results

Evaluated on the held-out test split of the primary **MaVeCoDD** (Marine Vessel Corrosion Detection) dataset (70% Train, 15% Val, 15% Test with vessel scene grouping to guarantee zero data leakage):

| Performance Dimension | Model A: Baseline (HOG+SVM) | Model B: Deep Learning (U-Net) | Operational Significance |
| :--- | :---: | :---: | :--- |
| **Scene Classification Accuracy** | 81.97% | **100.00%** | U-Net eliminates false classifications |
| **Classification Precision** | 87.88% | **100.00%** | Zero false alarm rate |
| **Classification Recall (Sensitivity)** | 80.56% | **100.00%** | Zero defects missed by U-Net |
| **Classification F1-Score** | 84.06% | **100.00%** | +15.94% overall improvement |
| **False Negatives (Missed Defects)** | **7 missed patches** | **0 missed scenes** | **Crucial for maritime safety** |
| **Segmentation Mean IoU** | N/A | **0.3833** | Full pixel mask support |
| **Segmentation Mean Dice (F1-mask)** | N/A | **0.4980** | Accurate defect boundary delineation |
| **Inference Latency** | **13.18 ms** | 121.15 ms | SVM is 9.2&times; faster |
| **Model Disk Footprint** | **2.4 MB** | 18.4 MB | SVM is 7.7&times; smaller |

### 🔬 Optical Turbidity Stress Test (Research Question 2)
As water turbidity increases from clear water ($\tau=0.0$) to heavy turbidity ($\tau=0.60$), forward scattering and red absorption degrade pixel segmentation Dice score monotonically:
- **Clear Water ($\tau=0.00$)**: Mean IoU = 0.3217, Mean Dice = 0.4468, Recall = 85.05%
- **Light Coastal ($\tau=0.15$)**: Mean IoU = 0.3182, Mean Dice = 0.4436, Recall = 85.31%
- **Harbour Waters ($\tau=0.30$)**: Mean IoU = 0.3125, Mean Dice = 0.4380, Recall = 86.71%
- **Turbid Estuarine ($\tau=0.45$)**: Mean IoU = 0.3010, Mean Dice = 0.4262, Recall = 87.68%
- **Heavy Turbidity ($\tau=0.60$)**: Mean IoU = **0.2907**, Mean Dice = **0.4141**, Recall = 88.78%

### 🌊 Cross-Domain Generalization (Research Question 6)
Zero-shot evaluation of the dry-dock corrosion model on the **Underwater Crack Image Dataset** yielded a **Mean IoU of 0.0004** and **Mean Dice of 0.0009**. This confirms that dry-dock corrosion models cannot transfer to submerged structural cracks without target-domain adaptation due to extreme photometric and morphological shifts.

---

## 🏗️ Repository Architecture

```
FOML_proj/
│
├── config/
│   └── config.yaml              # Central configuration parameters
├── data/
│   ├── raw/
│   │   ├── mavecodd/            # Primary Marine Vessel Corrosion dataset
│   │   └── underwater_crack/    # Secondary Underwater Crack dataset
│   ├── processed/               # Stratified leakage-free split manifests
│   └── sample/                  # Quick-start synthetic hull samples
├── models/
│   └── checkpoints/             # Trained .joblib and .pth model weights
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_cnn_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── annotation_parser.py # Mask decoding and bbox extraction
│   │   ├── preprocessing.py     # Degradation simulation and HOG features
│   │   ├── split_dataset.py     # Scene-grouped leakage-free splits
│   │   └── dataset_loader.py    # PyTorch dataset & sample generators
│   ├── models/
│   │   ├── baseline.py          # Model A: HOG+Color+PCA+SVM
│   │   ├── cnn.py               # Model B: Multi-task Residual U-Net
│   │   ├── train.py             # Optimizer, scheduler, and early stopping
│   │   ├── predict.py           # Unified inference and visual overlay
│   │   └── evaluate.py          # Benchmark metrics and stress testing
│   ├── simulation/
│   │   ├── hull.py              # 50m x 10m hull geometry and coverage grid
│   │   ├── rov.py               # Kinematics, thrusters, and sensor noise
│   │   ├── inspection_path.py   # Serpentine waypoint path planner
│   │   ├── camera.py            # Optical footprint and degradation
│   │   └── simulator.py         # Master simulation orchestration loop
│   ├── storage/
│   │   ├── schema.py            # Strongly typed 26-field PyArrow schema
│   │   ├── parquet_writer.py    # Columnar Parquet logging and batch append
│   │   └── parquet_reader.py    # Analytical queries and CSV export
│   ├── reporting/
│   │   └── report_generator.py  # Publication-grade PDF reports (ReportLab)
│   └── utils/
│       ├── config.py            # Config loader with environment overrides
│       └── reproducibility.py   # Deterministic random seeds and device selector
├── app/
│   ├── streamlit_app.py         # Main dashboard portal
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Image_Inspection.py
│       ├── 3_Drone_Simulation.py
│       ├── 4_Dataset_Explorer.py
│       ├── 5_Model_Evaluation.py
│       └── 6_Inspection_Reports.py
├── scripts/
│   ├── prepare_dataset.py       # Extraction and split manifests
│   ├── inspect_dataset.py       # Dataset statistics and shape verification
│   ├── train_model.py           # Model training runner
│   ├── evaluate_model.py        # Benchmark and stress test runner
│   └── run_simulation.py        # Autonomous ROV simulation runner
├── outputs/
│   ├── figures/                 # Confusion matrices and response curves
│   ├── metrics/                 # JSON test metrics summaries
│   └── reports/                 # Compiled academic report (PDF & MD)
├── tests/                       # Complete PyTest test suite (16 tests)
├── requirements.txt             # Pinned project dependencies
└── README.md
```

---

## 🚀 Quick-Start Guide

### 1. Environment Installation
Ensure Python 3.11+ is installed. Clone the repository and install dependencies:
```bash
git clone <repo-url>
cd FOML_proj
pip install -r requirements.txt
```

### 2. Dataset Preparation
If dataset archives `ry392rp8cj-1.zip` (MaVeCoDD) and `under-crack-images.zip` are in the root directory, extract and partition them:
```bash
python scripts/prepare_dataset.py
```
*Note: If archives are not yet downloaded, `prepare_dataset.py` automatically initializes a self-contained synthetic hull sample dataset in `data/sample/` so you can verify the pipeline immediately.*

### 3. Model Training
Train both Model A (Baseline SVM) and Model B (U-Net CNN):
```bash
# Full training on MaVeCoDD (CUDA or CPU):
python scripts/train_model.py --dataset mavecodd

# Fast quick-start training on synthetic samples:
python scripts/train_model.py --dataset sample --epochs 5
```

### 4. Model Evaluation & Benchmark Analysis
Evaluate trained checkpoints on the held-out test set and generate stress curves:
```bash
python scripts/evaluate_model.py --dataset mavecodd
```

### 5. Autonomous Drone Inspection Simulation
Run an autonomous serpentine inspection run logging telemetry to Apache Parquet:
```bash
python scripts/run_simulation.py --steps 120 --turbidity 0.25
```

### 6. Run Test Suite
Verify all components with automated unit and integration tests:
```bash
pytest tests/ -v
```

### 7. Launch Interactive Streamlit Dashboard
Launch the web interface in your browser:
```bash
streamlit run app/streamlit_app.py
```

---

## ⚠️ Scientific & Marine Engineering Limitations

1. **Optical vs. Ultrasonic Testing**: Computer vision analyzes visible surface phenomena (oxide scale, coating failure, surface fissure cracks). It **cannot determine internal metallurgical stress, plate thinning, or subsurface fatigue without ultrasonic testing (UT) or eddy-current sensors**.
2. **Defect Nomenclature**: Under classification society rules (DNV, Lloyd's Register), all vision detections are cataloged as **'suspected defects'** with status **'requires further inspection'**. Predictions are not certified proof of structural failure.
3. **Sensor Noise**: Real-world acoustic positioning (USBL/DVL) exhibits multipath reflections and drift; simulated sensor noise demonstrates the need for Kalman filtering in physical deployments.

---

## 📜 Citation & Credits
- **Primary Dataset**: Kamzelis, K., Chliveros, G., & Tzanetatos, I. (2021). *MaVeCoDD: Marine Vessel Corrosion Detection Dataset in Dry-Dock Images*. Mendeley Data, V1, doi: [10.17632/ry392rp8cj.1](https://doi.org/10.17632/ry392rp8cj.1).
- **Secondary Dataset**: Gao, Y., et al. (2021). *Underwater Crack Image Dataset for Structural Health Monitoring*. Mendeley Data, V1, doi: [10.17632/g962rbm77b/1](https://doi.org/10.17632/g962rbm77b/1).
- **Optical Model**: Jaffe, J. S. (1990). *Computer modeling and the design of optimal underwater imaging systems*. IEEE JOE.
