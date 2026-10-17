"""Script to generate educational Jupyter Notebooks for the FOML project."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11.0"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().split("\n")]}


def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.strip().split("\n")]}


# 1. Notebook 01: Dataset Exploration
nb1_cells = [
    md_cell("# Notebook 01: Underwater Hull Dataset Exploration & Annotation Parsing\n\n"
            "This notebook inspects the primary **MaVeCoDD** (Marine Vessel Corrosion) dataset and the secondary **Underwater Crack** dataset. "
            "We verify image dimensions, parse pixel-level masks, extract bounding boxes, and formulate leakage-free splits."),
    code_cell("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
              "from src.utils.config import load_config\n"
              "from src.data.dataset_loader import scan_mavecodd_samples, scan_crack_samples, load_image_and_mask\n"
              "from src.data.annotation_parser import parse_mavecodd_mask, extract_defect_boxes\n"
              "import matplotlib.pyplot as plt\nimport numpy as np\n\n"
              "config = load_config()\nprint('Loaded configuration:', config['project']['name'])"),
    md_cell("## 1. Inspect MaVeCoDD Samples\n\nLet's scan raw images and mask pairs from MaVeCoDD."),
    code_cell("mav_samples = scan_mavecodd_samples('../data/raw/mavecodd', zip_fallback='../ry392rp8cj-1.zip')\n"
              "print(f'Total matched MaVeCoDD pairs: {len(mav_samples)}')\n\n"
              "img, mask = load_image_and_mask(mav_samples[0])\n"
              "print('Sample shape:', img.shape, 'Mask shape:', mask.shape)\n\n"
              "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
              "axes[0].imshow(img)\naxes[0].set_title('Raw Marine Vessel Hull')\naxes[0].axis('off')\n"
              "axes[1].imshow(mask, cmap='hot')\naxes[1].set_title('Parsed Binary Corrosion Mask')\naxes[1].axis('off')\n"
              "plt.tight_layout()\nplt.show()"),
    md_cell("## 2. Bounding Box Extraction via Connected Components\n\n"
            "We can derive object-detection bounding boxes from the segmentation masks without manual box labeling."),
    code_cell("boxes = extract_defect_boxes(mask, defect_type='corrosion')\n"
              "print(f'Extracted {len(boxes)} defect bounding boxes')\n"
              "for b in boxes[:3]:\n    print(b)")
]

with open(NB_DIR / "01_dataset_exploration.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb1_cells), f, indent=2)

# 2. Notebook 02: Baseline Model
nb2_cells = [
    md_cell("# Notebook 02: Traditional ML Baseline (HOG + Color + PCA + SVM)\n\n"
            "In this notebook, we train and evaluate **Model A**, our handcrafted feature baseline. "
            "We extract Histogram of Oriented Gradients (HOG) features and HSV color histograms, reduce dimensions with PCA, and fit an SVM."),
    code_cell("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
              "from src.utils.config import load_config\n"
              "from src.data.split_dataset import load_split_manifest\n"
              "from src.models.baseline import HullDefectBaselineClassifier, prepare_patch_dataset\n"
              "from src.models.evaluate import evaluate_baseline_on_patches\n\n"
              "config = load_config()\nsplits = load_split_manifest('../data/processed/split_manifest_mavecodd.json')\n"
              "print(f'Train scenes: {len(splits[\"train\"])}, Test scenes: {len(splits[\"test\"])}')"),
    md_cell("## 1. Extract Patches and Handcrafted Features"),
    code_cell("train_imgs, train_y = prepare_patch_dataset(splits['train'], patches_per_sample=4)\n"
              "print(f'Total training patches: {len(train_imgs)}')\n\n"
              "model = HullDefectBaselineClassifier(pca_components=64, svm_c=10.0)\n"
              "X_train = model.extract_batch_features(train_imgs)\n"
              "print('Feature matrix shape:', X_train.shape)\n"
              "model.fit(X_train, train_y)\nprint('Baseline model trained successfully!')"),
    md_cell("## 2. Evaluate Baseline on Held-out Test Set"),
    code_cell("eval_res = evaluate_baseline_on_patches(model, splits['test'])\n"
              "for k, v in eval_res.items():\n    print(f'{k}: {v}')")
]

with open(NB_DIR / "02_baseline_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb2_cells), f, indent=2)

# 3. Notebook 03: CNN Training
nb3_cells = [
    md_cell("# Notebook 03: Deep Learning U-Net Training & Segmentation\n\n"
            "Here we train **Model B**, a dual-head lightweight U-Net on MaVeCoDD corrosion segmentation. "
            "We optimize a combined Binary Cross-Entropy + Soft Dice Loss to overcome foreground-background class imbalance."),
    code_cell("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
              "import torch\nfrom src.utils.config import load_config\n"
              "from src.utils.reproducibility import get_device\n"
              "from src.data.split_dataset import load_split_manifest\n"
              "from src.models.cnn import HullUNet\n"
              "from src.models.train import train_unet\n\n"
              "config = load_config()\ndev = get_device('auto')\nprint('Using device:', dev)\n"
              "splits = load_split_manifest('../data/processed/split_manifest_mavecodd.json')"),
    md_cell("## 1. Model Architecture & Parameters"),
    code_cell("unet = HullUNet(in_channels=3, out_channels=1, base_channels=24).to(dev)\n"
              "total_params = sum(p.numel() for p in unet.parameters())\n"
              "print(f'Total trainable parameters: {total_params:,}')"),
    md_cell("## 2. Train U-Net on Training Split"),
    code_cell("config['cnn_model']['epochs'] = 5\n"
              "results = train_unet(splits['train'], splits['val'], config=config, device=dev)\n"
              "print('Training complete! Best val loss:', results['best_val_loss'])")
]

with open(NB_DIR / "03_cnn_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb3_cells), f, indent=2)

# 4. Notebook 04: Model Evaluation
nb4_cells = [
    md_cell("# Notebook 04: Comparative Model Evaluation & Research Question Experiments\n\n"
            "This notebook analyzes the experimental results addressing all 6 FOML research questions: "
            "traditional ML vs CNN, underwater optical degradation, class imbalance, and cross-domain transfer."),
    code_cell("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n\n"
              "import json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
              "with open('../outputs/metrics/test_metrics_mavecodd.json', 'r') as f:\n"
              "    metrics = json.load(f)\n\n"
              "print('Evaluation summary loaded successfully.')"),
    md_cell("## 1. Traditional ML Baseline vs Deep Learning U-Net"),
    code_cell("base = metrics['baseline_model_a']\nunet = metrics['deep_learning_model_b']\n\n"
              "df = pd.DataFrame([\n"
              "    {'Metric': 'Accuracy', 'Baseline SVM': base['accuracy'], 'U-Net': unet['classification_accuracy']},\n"
              "    {'Metric': 'Precision', 'Baseline SVM': base['precision'], 'U-Net': unet['classification_precision']},\n"
              "    {'Metric': 'Recall', 'Baseline SVM': base['recall'], 'U-Net': unet['classification_recall']},\n"
              "    {'Metric': 'F1-Score', 'Baseline SVM': base['f1_score'], 'U-Net': unet['classification_f1']},\n"
              "])\nprint(df.to_string(index=False))"),
    md_cell("## 2. Optical Turbidity Degradation Stress Test"),
    code_cell("deg = pd.DataFrame(metrics['optical_degradation_stress_test'])\n"
              "plt.figure(figsize=(8, 4.5))\n"
              "plt.plot(deg['turbidity'], deg['mean_iou'], marker='o', label='Mean IoU')\n"
              "plt.plot(deg['turbidity'], deg['mean_dice'], marker='s', label='Mean Dice')\n"
              "plt.xlabel('Turbidity Scattering Coefficient')\nplt.ylabel('Score')\n"
              "plt.title('Performance vs Underwater Turbidity')\nplt.legend()\nplt.grid(True)\nplt.show()"),
    md_cell("## 3. Cross-Domain Generalization to Underwater Cracks"),
    code_cell("cross = metrics['cross_domain_generalization']\n"
              "print('Cross-domain metrics:', cross)")
]

with open(NB_DIR / "04_model_evaluation.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb4_cells), f, indent=2)

print("All 4 Jupyter Notebooks generated successfully!")
