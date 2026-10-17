"""Evaluation runner script comparing Baseline and U-Net models on test splits and stress experiments."""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_device
from src.data.split_dataset import load_split_manifest
from src.models.baseline import HullDefectBaselineClassifier
from src.models.predict import HullInspectionPredictor
from src.models.evaluate import (
    evaluate_baseline_on_patches,
    evaluate_unet_on_test_set,
    evaluate_underwater_degradation_stress_test,
    save_evaluation_plots
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate underwater hull inspection models.")
    parser.add_argument("--dataset", choices=["mavecodd", "crack", "sample"], default="mavecodd",
                        help="Dataset test split to evaluate on (default: mavecodd)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device")
    args = parser.parse_args()

    print("=" * 70)
    print(f"EVALUATION PIPELINE: AI-POWERED UNDERWATER HULL INSPECTION DRONE [{args.dataset.upper()}]")
    print("=" * 70)

    config = load_config()
    set_seed(config.get("random_seed", 42))
    dev = get_device(args.device)

    checkpoints_dir = ROOT / config["paths"]["model_dir"]
    baseline_path = checkpoints_dir / f"baseline_{args.dataset}.joblib"
    unet_path = checkpoints_dir / f"unet_{args.dataset}.pth"

    if not baseline_path.exists() or not unet_path.exists():
        print("Required model checkpoints not found. Running training first...")
        from scripts.train_model import main as train_main
        sys.argv = ["train_model.py", "--dataset", args.dataset, "--device", args.device]
        train_main()

    # Load test split
    manifest_file = ROOT / config["paths"]["processed_dir"] / f"split_manifest_{args.dataset}.json"
    splits = load_split_manifest(manifest_file)
    test_samples = splits["test"]
    print(f"Loaded {len(test_samples)} held-out test scenes.")

    # 1. Evaluate Model A (Baseline SVM)
    print("\n[1/4] Evaluating Model A (HOG+Color+PCA+SVM) on Test Patches...")
    baseline_model = HullDefectBaselineClassifier.load(baseline_path)
    baseline_eval = evaluate_baseline_on_patches(baseline_model, test_samples)
    print(f"  * Accuracy:  {baseline_eval.get('accuracy', 'N/A')}")
    print(f"  * Precision: {baseline_eval.get('precision', 'N/A')}")
    print(f"  * Recall:    {baseline_eval.get('recall', 'N/A')}")
    print(f"  * F1-Score:  {baseline_eval.get('f1_score', 'N/A')}")
    print(f"  * False Negatives: {baseline_eval.get('false_negatives', 'N/A')}")
    print(f"  * Latency per image: {baseline_eval.get('latency_ms_per_image', 'N/A')} ms")

    # 2. Evaluate Model B (U-Net CNN)
    print("\n[2/4] Evaluating Model B (Deep Learning U-Net) on Test Set...")
    predictor = HullInspectionPredictor(
        baseline_path=baseline_path,
        unet_path=unet_path,
        device=dev
    )
    unet_eval = evaluate_unet_on_test_set(predictor, test_samples)
    print(f"  * Mean IoU:          {unet_eval['mean_iou']}")
    print(f"  * Mean Dice Score:   {unet_eval['mean_dice']}")
    print(f"  * Pixel Accuracy:    {unet_eval['mean_pixel_accuracy']}")
    print(f"  * Class Accuracy:    {unet_eval['classification_accuracy']}")
    print(f"  * Class Precision:   {unet_eval['classification_precision']}")
    print(f"  * Class Recall:      {unet_eval['classification_recall']}")
    print(f"  * Class F1-Score:    {unet_eval['classification_f1']}")
    print(f"  * False Negatives:   {unet_eval['false_negatives']}")
    print(f"  * Latency per image: {unet_eval['latency_ms_per_image']} ms")

    # 3. Turbidity Degradation Stress Test (Research Question 2)
    print("\n[3/4] Running Optical Turbidity Degradation Stress Test...")
    degradation_results = evaluate_underwater_degradation_stress_test(
        predictor,
        test_samples,
        turbidity_levels=[0.0, 0.15, 0.30, 0.45, 0.60]
    )
    for d in degradation_results:
        print(f"  * Turbidity {d['turbidity']:.2f} ({d['condition']}): IoU={d['mean_iou']:.4f}, Dice={d['mean_dice']:.4f}, Recall={d['mean_recall']:.4f}")

    # 4. Cross-Domain Generalization Test (Research Question 6)
    crack_manifest = ROOT / config["paths"]["processed_dir"] / "split_manifest_crack.json"
    cross_domain_metrics = None
    if crack_manifest.exists() and args.dataset != "crack":
        print("\n[4/4] Running Cross-Domain Generalization Experiment (Dry-dock to Underwater)...")
        crack_splits = load_split_manifest(crack_manifest)
        crack_test = crack_splits["test"][:25]  # Subset for evaluation
        cross_eval = evaluate_unet_on_test_set(predictor, crack_test)
        print(f"  * Cross-Domain IoU on Underwater Crack: {cross_eval['mean_iou']}")
        print(f"  * Cross-Domain Dice Score: {cross_eval['mean_dice']}")
        cross_domain_metrics = {
            "source_dataset": args.dataset,
            "target_dataset": "underwater_crack",
            "cross_domain_iou": cross_eval["mean_iou"],
            "cross_domain_dice": cross_eval["mean_dice"],
            "cross_domain_accuracy": cross_eval["classification_accuracy"]
        }

    # Save metrics and figures
    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    figures_dir = ROOT / config["paths"]["figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    final_metrics = {
        "dataset": args.dataset,
        "baseline_model_a": baseline_eval,
        "deep_learning_model_b": unet_eval,
        "optical_degradation_stress_test": degradation_results,
        "cross_domain_generalization": cross_domain_metrics
    }

    metrics_out = metrics_dir / f"test_metrics_{args.dataset}.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    save_evaluation_plots(baseline_eval, unet_eval, degradation_results, figures_dir)

    print("\n" + "=" * 70)
    print("EVALUATION FINISHED!")
    print(f"Metrics saved to: {metrics_out}")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
