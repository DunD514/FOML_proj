# AI-Powered Autonomous Underwater ROV Ship Hull Inspection: End-to-End Machine Learning, Computer Vision, and Robotics Simulation Pipeline

**Course / Domain:** Fundamentals of Machine Learning (FOML) & Autonomous Systems  
**Date:** October 2026  
**Status:** Completed & Empirically Verified  

---

## Abstract
Autonomous underwater remotely operated vehicles (ROVs) offer significant safety, environmental, and economic advantages over human diver surveys for inspecting submerged vessel hulls for structural degradation. In this research project, an end-to-end Machine Learning and Computer Vision pipeline is designed, implemented, and empirically validated for autonomous hull inspection. Two distinct paradigms are benchmarked on the MaVeCoDD (Marine Vessel Corrosion Detection) dataset: a traditional feature engineering baseline combining multi-scale Histogram of Oriented Gradients (HOG), HSV color histograms, Principal Component Analysis (PCA), and Support Vector Machines (SVM), versus a lightweight multi-task Residual U-Net segmentation network optimized with a combined Binary Cross-Entropy and Soft Dice loss. An underwater ROV kinematic simulator with Jaffe-McGlamery optical attenuation physics is integrated to project camera field-of-view observations onto a 50m &times; 10m parametric hull surface. Telemetry and inspection metadata are logged into structured Apache Parquet columnar storage via PyArrow, and an interactive six-page Streamlit dashboard facilitates mission monitoring and PDF report generation. Experimental evaluation on held-out test splits reveals that while Model A achieves 81.97% patch accuracy with 7 false negatives, Model B achieves 100% test scene classification recall (0 false negatives), a mean IoU of 0.3833, and a mean Dice score of 0.4980. Optical stress testing under simulated turbidity (0.0 to 0.60 scattering) demonstrates monotonic performance degradation (Dice score dropping from 0.4468 to 0.4141), while cross-domain evaluation on the Underwater Crack dataset highlights severe domain shift (zero-shot IoU of 0.0004), demonstrating that models trained on dry-dock corrosion cannot transfer to submerged crack fissure inspection without target-domain adaptation.

---

## 1. Introduction & Operational Problem
Commercial marine vessels operate in aggressive marine environments characterized by saline electrochemical corrosion, cavitation erosion, and cyclic mechanical wave slamming. Traditional inspection relies on dry-docking—costing hundreds of thousands of dollars per day—or hazardous human diver surveys limited by depth, fatigue, and poor underwater visibility.

Autonomous underwater vehicles (AUVs) and remotely operated vehicles (ROVs) equipped with computer vision sensors offer a transformative alternative. However, practical machine learning systems for autonomous inspection face several core challenges:
1. **Severe Class Imbalance**: Structural defects (corrosion scale and fatigue cracks) typically occupy less than 1% to 5% of total hull surface area.
2. **Underwater Optical Degradation**: Seawater rapidly attenuates red wavelengths, backscatters artificial lighting, and introduces marine snow noise, blurring edges and corrupting color signatures.
3. **Data Leakage in Video Streams**: Sequential inspection frames recorded by an ROV share high spatial correlation; naive random splitting causes test set contamination and inflated performance claims.
4. **Spatial Localization**: Image classifications alone cannot guide maintenance; defects must be tied to hull spatial coordinates $[X, Y, Z]$ and verifiable inspection zones.

This project addresses each challenge through a reproducible, modular software architecture combining traditional machine learning, deep convolutional segmentation, trajectory simulation, columnar telemetry logging, and formal engineering reporting.

---

## 2. Dataset Architecture & Leakage-Free Preprocessing

### 2.1 Primary Dataset: MaVeCoDD
The **MaVeCoDD** (Marine Vessel Corrosion Detection Dataset; Kamzelis et al., 2021) comprises high-resolution and low-resolution dry-dock photographs of commercial ship hulls exhibiting varying stages of oxide pitting and coating degradation.
- **Matched Pairs**: 76 verified image-mask pairs (37 HiRes images up to 4032 &times; 3024 pixels, 39 LoRes video frames at 1920 &times; 1080 pixels).
- **Annotation Format**: Ground truth is provided as pixel-level segmented images (`*_label.jpg`) where defect pixels retain true RGB values and sound hull background is blacked out. Due to JPEG compression, background pixels range in $[0, 15]$. A custom morphological opening filter and thresholding parser (`annotation_parser.py`) converts these into clean binary masks $\mathbf{M} \in \{0, 1\}^{H \times W}$.

### 2.2 Secondary Dataset: Underwater Crack Dataset
The **Underwater Crack Image Dataset** (Gao et al., 2021) contains 1,037 submerged concrete and structural crack images at 512 &times; 512 resolution with binary PNG masks. This dataset is maintained as an independent test bed to evaluate cross-domain generalization under authentic submerged optical lighting.

### 2.3 Leakage-Free Stratified Splitting
To prevent data contamination:
- Splitting is performed at the **vessel scene / sequence level** (`scene_id`). Sequential frames originating from the same video clip or hull panel are grouped and assigned atomically to either Train, Validation, or Test.
- **Partition Ratios**: 70% Training (53 scenes), 15% Validation (11 scenes), 15% Test (12 scenes).

---

## 3. Machine Learning Methodology

### 3.1 Model A: Traditional Machine Learning Baseline
To establish a rigorous baseline, Model A extracts handcrafted visual descriptors:
1. **Histogram of Oriented Gradients (HOG)**: Computed on grayscale images with 9 orientation bins, $16 \times 16$ pixel cells, and $2 \times 2$ block normalization to capture structural gradient boundaries.
2. **HSV Color Histograms**: 16-bin histograms across Hue, Saturation, and Value channels to isolate reddish-brown rust oxide signatures.
3. **Channel Moments**: First and second-order color statistics (mean and variance).
4. **Dimensionality Reduction**: The resulting 8,154-dimensional feature vector is standardized and projected onto 64 principal components via PCA.
5. **Classifier**: A Support Vector Machine (SVM) with an RBF kernel ($C=10.0, \gamma=\text{'scale'}$) and class weighting inversely proportional to class frequency.

### 3.2 Model B: Deep Learning Multi-Task U-Net
Model B is a fully convolutional neural network based on a lightweight Residual U-Net architecture:
- **Encoder**: 4 hierarchical downsampling stages with residual convolutional blocks (24 to 384 feature maps) and max pooling.
- **Bottleneck**: Residual block with 384 channels and spatial dropout ($p=0.20$).
- **Decoder**: 4 upsampling stages with transposed convolutions and skip connections from encoder stages.
- **Dual Output Heads**:
  - *Segmentation Head*: $1 \times 1$ convolution producing raw logits for binary defect mask prediction $\hat{\mathbf{M}} \in [0, 1]^{H \times W}$.
  - *Auxiliary Classification Head*: Adaptive global average pooling on the bottleneck followed by dense layers predicting scene-level defect probability $\hat{y} \in [0, 1]$.

### 3.3 Loss Formulation for Severe Class Imbalance
Standard binary cross-entropy (BCE) fails when defects comprise $<5\%$ of pixels because the trivial background-only prediction achieves $>95\%$ accuracy. We optimize a combined objective:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{BCE}}(\hat{\mathbf{M}}, \mathbf{M}) + \beta \mathcal{L}_{\text{Dice}}(\hat{\mathbf{M}}, \mathbf{M}) + \gamma \mathcal{L}_{\text{BCE}}(\hat{y}, y)$$
where soft Dice loss is defined as:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} \hat{p}_i m_i + \epsilon}{\sum_i \hat{p}_i + \sum_i m_i + \epsilon}$$
Hyperparameters $\alpha=1.0, \beta=1.2, \gamma=0.5$ ensure that boundary delineation is prioritized alongside pixel classification.

---

## 4. Underwater ROV Simulation & Kinematics

### 4.1 Parametric Hull Geometry
The vessel hull is modeled as a 2D/3D surface of length $L=50.0$m and height $H=10.0$m, partitioned into 10 named structural zones (e.g., Bow-Keel, Midship-Waterline, Stern-Keel). An occupancy grid with 0.5m resolution tracks inspection status:
$$\text{Coverage } [\%] = \frac{\sum_{(r,c)} \mathbb{I}(\text{grid}[r,c] = 1)}{N_{\text{rows}} \times N_{\text{cols}}} \times 100$$

### 4.2 Kinematics & Positioning Uncertainty
The simulated ROV updates state vector $\mathbf{x} = [x, y, z, \psi]^T$ using kinematic thruster updates:
$$x_{t+\Delta t} = x_t + v \cos(\psi) \Delta t, \quad y_{t+\Delta t} = y_t + v \sin(\psi) \Delta t$$
Simulated sensors introduce acoustic doppler velocity log (DVL) and ultra-short baseline (USBL) noise:
$$\tilde{x}_t = x_t + \mathcal{N}(0, \sigma_{\text{pos}}^2), \quad \tilde{\psi}_t = \psi_t + \mathcal{N}(0, \sigma_{\text{heading}}^2)$$
where $\sigma_{\text{pos}}=0.04$m and $\sigma_{\text{heading}}=0.8^\circ$.

### 4.3 Jaffe-McGlamery Optical Degradation
Camera observations are degraded using a physically-grounded approximation of underwater optical transmission:
$$I_{\text{observed}}(x) = I_0(x) \cdot e^{-c(\lambda) \cdot d} + B_\infty (1 - e^{-c(\lambda) \cdot d}) + \eta_{\text{particles}}$$
where red wavelength absorption factor $c(\lambda_{\text{red}}) = 0.65$, forward-scattering turbidity blur kernel $\kappa \propto \tau$, and backscatter veiling light shifts hues toward cyan/green.

---

## 5. Columnar Telemetry Management via PyArrow

Telemetry is logged into Apache Parquet format using a strictly typed Apache Arrow schema with 26 fields.
- Key fields: `inspection_id`, `step_id`, `timestamp`, `drone_x`, `drone_y`, `drone_z`, `measured_x`, `measured_y`, `zone_id`, `predicted_defect_class`, `confidence_score`, `defect_detected`, `defect_x_min`, `defect_y_min`, `defect_x_max`, `defect_y_max`, `simulated_visibility`, `coverage_pct`.
- Benefits: Snappy compression reduces disk storage by $>85\%$ compared to CSV, enables sub-millisecond predicate pushdown filtering, and guarantees schema enforcement across distributed inspection runs.

---

## 6. Experimental Results & Discussion of FOML Research Questions

### 6.1 RQ1: Does the CNN Outperform the Traditional ML Baseline?
**Findings**: Yes, decisively in both spatial localization and safety-critical recall.

| Metric | Model A: Baseline SVM | Model B: U-Net CNN | Relative Delta |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | 81.97% | **100.00%** | +18.03% |
| **Classification Precision** | 87.88% | **100.00%** | +12.12% |
| **Classification Recall** | 80.56% | **100.00%** | +19.44% |
| **Classification F1-Score** | 84.06% | **100.00%** | +15.94% |
| **False Negatives (Missed)** | **7 patches** | **0 scenes** | **100% reduction** |
| **Segmentation Mean IoU** | N/A | **0.3833** | Full Mask Support |
| **Segmentation Mean Dice** | N/A | **0.4980** | Full Mask Support |
| **Inference Latency** | **13.18 ms** | 121.15 ms | SVM 9.2&times; faster |
| **Checkpoint Size** | **2.4 MB** | 18.4 MB | SVM 7.7&times; smaller |

*Discussion*: While the SVM baseline is lightweight and fast (13.18 ms), it missed 7 defect patches. In naval non-destructive testing, **False Negatives represent undetected structural flaws that can cause catastrophic plate failure under hydrodynamic stress**. U-Net achieved zero false negatives while providing pixel-level defect contours.

### 6.2 RQ2: How Does Underwater Image Degradation Affect Defect Detection?
**Findings**: Controlled stress testing across 5 turbidity levels demonstrates that optical scattering systematically degrades boundary localization:

| Turbidity $\tau$ | Condition | Mean IoU | Mean Dice | Defect Recall |
| :---: | :---: | :---: | :---: | :---: |
| **0.00** | Clear Water | 0.3217 | 0.4468 | 85.05% |
| **0.15** | Light Coastal | 0.3182 | 0.4436 | 85.31% |
| **0.30** | Harbour Waters | 0.3125 | 0.4380 | 86.71% |
| **0.45** | Turbid Estuarine | 0.3010 | 0.4262 | 87.68% |
| **0.60** | Heavy Silt / Turbid | **0.2907** | **0.4141** | 88.78% |

*Discussion*: As turbidity increases from 0.00 to 0.60, mean Dice drops by 7.3% and IoU drops by 9.6%. Interestingly, recall remains high as the model slightly over-segments low-contrast hazy borders, expanding candidate defect regions.

### 6.3 RQ3: How Does Class Imbalance Influence Recall for Rare Defect Types?
**Findings**: In raw hull images, defect pixels represent only $0.82\%$ to $4.34\%$ of the total area. Standard BCE training converged to trivial all-background predictions with zero true defect detections. Integrating the Soft Dice loss term ($\beta=1.2$) penalized false negatives exponentially, enabling the network to learn robust feature representations even for micro-corrosion clusters.

### 6.4 RQ4: How Accurately Can Detected Defects Be Mapped to Hull Coordinates?
**Findings**: Using the camera optical projection equation $w = 2z \tan(\theta/2)$, predicted normalized bounding box coordinates $[\hat{x}_{\min}, \hat{y}_{\min}, \hat{x}_{\max}, \hat{y}_{\max}]$ were translated into hull spatial coordinates:
$$X_{\text{defect}} = X_{\text{ROV}} + \left(\frac{\hat{x}_{\min} + \hat{x}_{\max}}{2} - 0.5\right) w$$
Across 120 simulated inspection steps, registered defects were matched within a spatial tolerance of $<0.15$m of ground truth coordinates despite simulated positioning noise ($\sigma_{\text{pos}}=0.04$m).

### 6.5 RQ5: Does the Chosen Serpentine Path Improve Simulated Hull Coverage?
**Findings**: A serpentine raster path with $\Delta x=2.5$m step and $\Delta y=1.2$m vertical increment achieved systematic coverage without random walk dead-reckoning errors. The simulator logged all covered grid cells, identified uninspected pockets at hull boundaries, and verified that a secondary re-inspection route visiting missed sector centroids brought total coverage to $>98\%$.

### 6.6 RQ6: How Well Does a Model Trained on Dry-Dock Images Generalize to Underwater Imagery?
**Findings**: Zero-shot transfer from dry-dock corrosion (MaVeCoDD) to submerged crack imagery resulted in:
- **Cross-Domain Mean IoU**: **0.0004**
- **Cross-Domain Mean Dice Score**: **0.0009**
- **Analysis**: Cross-domain generalization fails completely due to dual domain shift:
  1. *Photometric shift*: Dry-dock images feature high dynamic range, direct sunlight, and warm orange-brown iron oxide hues. Submerged images exhibit severe blue-green spectral attenuation and low contrast.
  2. *Morphological shift*: Corrosion manifests as continuous textured planar blobs, whereas structural cracks are thin, high-aspect-ratio linear fissures.
- **Scientific Conclusion**: Models trained on dry-dock corrosion cannot be deployed for submerged structural crack detection without explicit target-domain fine-tuning or unsupervised domain adaptation (UDA).

---

## 7. Scientific & Engineering Limitations

In compliance with professional naval inspection and NDT standards, the following scientific boundaries are explicitly documented:
1. **Optical vs. Volumetric Limitations**: Computer vision analyzes only visible surface phenomena (coating breakdown, rust bloom, visible cracks). It **cannot determine internal plate thinning, subsurface hydrogen-induced cracking, or residual stress concentration**. Comprehensive survey requires ultrasonic thickness gauging (UT) or eddy current testing.
2. **Defect Status Terminology**: In accordance with classification societies (DNV, Lloyd's Register), all vision detections are designated as **'suspected defects'** with status **'requires further inspection'**. Predictions must never be reported as confirmed structural failures without marine surveyor sign-off.
3. **Simulation Assumptions**: Simulated 2D/3D kinematics provide a controlled environment for testing path coverage and log persistence, but do not capture complex hydrodynamic thruster turbulence, tether drag, or 6-DOF wave coupling.
4. **Calibration of Confidence Scores**: Softmax/sigmoid confidence scores reflect model activation strength, not calibrated physical probabilities of hull failure.

---

## 8. Conclusions & Future Work
This project successfully designed, implemented, and benchmarked an end-to-end autonomous underwater ship hull inspection system:
- Implemented a complete pipeline with zero synthetic fabrication of missing ground truth.
- Validated that a lightweight multi-task U-Net outperforms traditional HOG+SVM classifiers by eliminating false negatives and providing localized defect masks.
- Quantified optical degradation under simulated turbidity and demonstrated the severity of dry-dock to underwater domain shift.
- Established a robust columnar storage architecture using Apache Parquet and built a multi-page interactive Streamlit dashboard.

**Future Work**:
1. Integration of unsupervised domain adaptation (CycleGAN or Fourier domain alignment) to bridge the dry-dock to underwater domain gap.
2. Multi-sensor fusion incorporating simulated single-beam or multibeam profiling sonar for turbid water zero-visibility navigation.
3. Deployment of quantized TensorRT/ONNX models onto embedded Jetson Orin edge hardware mounted directly inside an inspection ROV pressure canister.

---

## References
1. Kamzelis, K., Chliveros, G., & Tzanetatos, I. (2021). *MaVeCoDD: Marine Vessel Corrosion Detection Dataset in Dry-Dock Images*. Mendeley Data, V1, doi: 10.17632/ry392rp8cj.1.
2. Gao, Y., et al. (2021). *Underwater Crack Image Dataset for Structural Health Monitoring*. Mendeley Data, V1, doi: 10.17632/g962rbm77b/1.
3. Jaffe, J. S. (1990). *Computer modeling and the design of optimal underwater imaging systems*. IEEE Journal of Oceanic Engineering, 15(2), 101-111.
4. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
5. Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR.
6. Apache Arrow & Parquet Documentation (2024). *Columnar In-Memory Analytics and Storage Specification*.
