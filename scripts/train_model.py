"""Training runner script for both Model A (Baseline SVM) and Model B (U-Net CNN)."""

import os
import sys
import argparse
from pathlib import Path
import json

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_device
from src.data.split_dataset import load_split_manifest
from src.models.train import train_baseline, train_unet


def main():
    parser = argparse.ArgumentParser(description="Train underwater hull inspection ML models.")
    parser.add_argument("--dataset", choices=["mavecodd", "crack", "sample"], default="mavecodd",
                        help="Dataset to train on (default: mavecodd)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device")
    args = parser.parse_args()

    print("=" * 70)
    print(f"TRAINING PIPELINE: AI-POWERED UNDERWATER HULL INSPECTION DRONE [{args.dataset.upper()}]")
    print("=" * 70)

    config = load_config()
    set_seed(config.get("random_seed", 42))
    dev = get_device(args.device)
    print(f"Execution Device: {dev}")

    manifest_file = ROOT / config["paths"]["processed_dir"] / f"split_manifest_{args.dataset}.json"
    if not manifest_file.exists():
        print(f"Error: Manifest file {manifest_file} not found. Running prepare_dataset.py first...")
        from scripts.prepare_dataset import main as prep_main
        prep_main()

    splits = load_split_manifest(manifest_file)
    train_samples = splits["train"]
    val_samples = splits["val"]
    print(f"Loaded splits -> Train: {len(train_samples)}, Val: {len(val_samples)}")

    checkpoints_dir = ROOT / config["paths"]["model_dir"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train Model A: Traditional ML Baseline
    baseline_path = checkpoints_dir / f"baseline_{args.dataset}.joblib"
    baseline_results = train_baseline(
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
        save_path=baseline_path
    )

    # 2. Train Model B: Deep Learning U-Net
    if args.epochs is not None:
        config["cnn_model"]["epochs"] = args.epochs

    unet_path = checkpoints_dir / f"unet_{args.dataset}.pth"
    unet_results = train_unet(
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
        save_path=unet_path,
        device=dev
    )

    # Save training history summary
    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "device": str(dev),
        "baseline": {
            "training_time_s": baseline_results["training_time_s"],
            "val_accuracy": baseline_results["val_accuracy"],
            "checkpoint": str(baseline_path.name)
        },
        "unet": {
            "training_time_s": unet_results["training_time_s"],
            "best_val_loss": unet_results["best_val_loss"],
            "checkpoint": str(unet_path.name),
            "final_val_iou": unet_results["history"]["val_iou"][-1] if unet_results["history"]["val_iou"] else None,
            "final_val_dice": unet_results["history"]["val_dice"][-1] if unet_results["history"]["val_dice"] else None
        }
    }

    summary_file = metrics_dir / f"training_summary_{args.dataset}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING SUCCESSFUL!")
    print(f"Baseline saved to: {baseline_path}")
    print(f"U-Net saved to: {unet_path}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
