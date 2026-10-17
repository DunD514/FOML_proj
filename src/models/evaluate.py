"""Comprehensive model evaluation module comparing Model A and Model B across FOML research questions."""

from typing import Dict, Any, List, Optional
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from src.data.dataset_loader import load_image_and_mask
from src.data.preprocessing import (
    resize_image_and_mask,
    apply_underwater_degradation
)
from src.models.baseline import HullDefectBaselineClassifier, prepare_patch_dataset
from src.models.cnn import HullUNet, compute_segmentation_metrics
from src.models.predict import HullInspectionPredictor


def evaluate_baseline_on_patches(
    baseline_model: HullDefectBaselineClassifier,
    test_samples: List[Dict[str, Any]],
    random_seed: int = 42
) -> Dict[str, Any]:
    """Evaluate Model A (Baseline) on patch classification."""
    test_imgs, y_true = prepare_patch_dataset(test_samples, patches_per_sample=6, random_seed=random_seed)
    if len(test_imgs) == 0:
        return {}

    t0 = time.perf_counter()
    X_test = baseline_model.extract_batch_features(test_imgs)
    y_pred = baseline_model.predict(X_test)
    y_prob = baseline_model.predict_proba(X_test)[:, 1] if baseline_model.classifier.classes_.shape[0] > 1 else y_pred
    total_time = (time.perf_counter() - t0) * 1000.0
    latency_per_img = total_time / len(test_imgs)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None

    return {
        "model_name": "Model A (HOG+Color+PCA+SVM)",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "test_samples": len(test_imgs),
        "latency_ms_per_image": round(latency_per_img, 2),
    }


def evaluate_unet_on_test_set(
    predictor: HullInspectionPredictor,
    test_samples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate Model B (U-Net) on pixel segmentation and scene defect classification."""
    ious, dices, pixel_accs = [], [], []
    precisions, recalls = [], []
    y_true_cls, y_pred_cls, y_prob_cls = [], [], []
    latencies = []

    for s in test_samples:
        img, gt_mask = load_image_and_mask(s)
        # Ground truth defect label
        has_defect_gt = 1 if np.sum(gt_mask > 0) > 30 else 0
        y_true_cls.append(has_defect_gt)

        res = predictor.predict(img)
        latencies.append(res["inference_time_ms"])

        pred_mask = res["binary_mask"]
        pred_label = 1 if res["defect_detected"] else 0
        y_pred_cls.append(pred_label)
        y_prob_cls.append(res["confidence_score"] if pred_label == 1 else 1.0 - res["confidence_score"])

        # Segmentation metrics
        seg_m = compute_segmentation_metrics(pred_mask, gt_mask)
        ious.append(seg_m["iou"])
        dices.append(seg_m["dice"])
        pixel_accs.append(seg_m["pixel_accuracy"])
        precisions.append(seg_m["precision"])
        recalls.append(seg_m["recall"])

    y_true_arr = np.array(y_true_cls)
    y_pred_arr = np.array(y_pred_cls)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_cls)) if len(np.unique(y_true_arr)) > 1 else None

    return {
        "model_name": "Model B (U-Net CNN)",
        "mean_iou": round(float(np.mean(ious)), 4),
        "mean_dice": round(float(np.mean(dices)), 4),
        "mean_pixel_accuracy": round(float(np.mean(pixel_accs)), 4),
        "pixel_precision": round(float(np.mean(precisions)), 4),
        "pixel_recall": round(float(np.mean(recalls)), 4),
        "classification_accuracy": round(acc, 4),
        "classification_precision": round(prec, 4),
        "classification_recall": round(rec, 4),
        "classification_f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "latency_ms_per_image": round(float(np.mean(latencies)), 2),
        "test_samples": len(test_samples)
    }


def evaluate_underwater_degradation_stress_test(
    predictor: HullInspectionPredictor,
    test_samples: List[Dict[str, Any]],
    turbidity_levels: List[float] = [0.0, 0.15, 0.30, 0.45, 0.60]
) -> List[Dict[str, Any]]:
    """Evaluate model degradation under simulated optical turbidity conditions (Research Question 2)."""
    results = []

    for turb in turbidity_levels:
        ious, dices, recalls = [], [], []
        for s in test_samples:
            img, gt_mask = load_image_and_mask(s)
            deg_img = apply_underwater_degradation(img, turbidity=turb, red_attenuation=0.65)
            res = predictor.predict(deg_img)
            m = compute_segmentation_metrics(res["binary_mask"], gt_mask)
            ious.append(m["iou"])
            dices.append(m["dice"])
            recalls.append(m["recall"])

        results.append({
            "turbidity": turb,
            "condition": "Clear Water" if turb == 0.0 else ("Light Coastal" if turb <= 0.2 else ("Harbour Waters" if turb <= 0.4 else "Turbid Estuarine")),
            "mean_iou": round(float(np.mean(ious)), 4),
            "mean_dice": round(float(np.mean(dices)), 4),
            "mean_recall": round(float(np.mean(recalls)), 4)
        })

    return results


def save_evaluation_plots(
    baseline_eval: Dict[str, Any],
    unet_eval: Dict[str, Any],
    degradation_results: List[Dict[str, Any]],
    output_dir: str | Path
) -> None:
    """Generate and save publication-grade comparison figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    cm_a = [[baseline_eval["confusion_matrix"]["TN"], baseline_eval["confusion_matrix"]["FP"]],
            [baseline_eval["confusion_matrix"]["FN"], baseline_eval["confusion_matrix"]["TP"]]]
    cm_b = [[unet_eval["confusion_matrix"]["TN"], unet_eval["confusion_matrix"]["FP"]],
            [unet_eval["confusion_matrix"]["FN"], unet_eval["confusion_matrix"]["TP"]]]

    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                xticklabels=["Sound", "Defect"], yticklabels=["Sound", "Defect"])
    axes[0].set_title("Model A: Baseline (HOG+SVM)\nConfusion Matrix")
    axes[0].set_ylabel("True Hull Condition")
    axes[0].set_xlabel("Predicted Condition")

    sns.heatmap(cm_b, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False,
                xticklabels=["Sound", "Defect"], yticklabels=["Sound", "Defect"])
    axes[1].set_title("Model B: Deep Learning (U-Net)\nConfusion Matrix")
    axes[1].set_ylabel("True Hull Condition")
    axes[1].set_xlabel("Predicted Condition")

    plt.tight_layout()
    plt.savefig(out / "confusion_matrix_comparison.png", dpi=300)
    plt.close()

    # 2. Performance Comparison Bar Chart
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    a_vals = [baseline_eval["accuracy"], baseline_eval["precision"], baseline_eval["recall"], baseline_eval["f1_score"]]
    b_vals = [unet_eval["classification_accuracy"], unet_eval["classification_precision"],
              unet_eval["classification_recall"], unet_eval["classification_f1"]]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, a_vals, width, label="Model A (HOG+SVM)", color="#3498db")
    ax.bar(x + width/2, b_vals, width, label="Model B (U-Net CNN)", color="#2ecc71")
    ax.set_ylabel("Score [0.0 - 1.0]")
    ax.set_title("Defect Classification: Traditional ML vs Deep Learning")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")

    for i in range(len(metrics_names)):
        ax.text(x[i] - width/2, a_vals[i] + 0.02, f"{a_vals[i]:.2f}", ha="center", fontsize=9)
        ax.text(x[i] + width/2, b_vals[i] + 0.02, f"{b_vals[i]:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out / "model_comparison_barchart.png", dpi=300)
    plt.close()

    # 3. Underwater Turbidity Degradation Curve
    df_deg = pd.DataFrame(degradation_results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df_deg["turbidity"], df_deg["mean_iou"], marker="o", linewidth=2.5, color="#e74c3c", label="Mean IoU")
    ax.plot(df_deg["turbidity"], df_deg["mean_dice"], marker="s", linewidth=2.5, color="#9b59b6", label="Mean Dice Score")
    ax.plot(df_deg["turbidity"], df_deg["mean_recall"], marker="^", linewidth=2.5, color="#e67e22", label="Defect Recall")
    ax.set_xlabel("Optical Turbidity Scattering Coefficient")
    ax.set_ylabel("Metric Score")
    ax.set_title("Impact of Underwater Turbidity on Defect Detection Fidelity")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "turbidity_degradation_curve.png", dpi=300)
    plt.close()
