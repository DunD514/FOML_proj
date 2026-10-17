"""Training routines for both Model A (Baseline SVM) and Model B (U-Net)."""

from typing import Dict, Any, List, Optional
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.reproducibility import get_device, set_seed
from src.data.dataset_loader import HullDefectDataset
from src.models.baseline import HullDefectBaselineClassifier, prepare_patch_dataset
from src.models.cnn import HullUNet, CombinedSegmentationLoss, compute_segmentation_metrics


def train_baseline(
    train_samples: List[Dict[str, Any]],
    val_samples: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str | Path] = None
) -> Dict[str, Any]:
    """Train the traditional ML baseline (HOG + Color + PCA + SVM)."""
    t0 = time.time()
    cfg = config or {}
    base_cfg = cfg.get("baseline_model", {})

    print("\n--- Training Traditional ML Baseline (Model A) ---")
    print(f"Extracting patches from {len(train_samples)} training scenes...")
    train_imgs, train_y = prepare_patch_dataset(
        train_samples,
        patches_per_sample=6,
        patch_size=256,
        random_seed=cfg.get("random_seed", 42)
    )
    print(f"Generated {len(train_imgs)} patches (Defect: {np.sum(train_y == 1)}, Sound: {np.sum(train_y == 0)})")

    model = HullDefectBaselineClassifier(
        orientations=base_cfg.get("hog_orientations", 9),
        pixels_per_cell=tuple(base_cfg.get("hog_pixels_per_cell", [16, 16])),
        cells_per_block=tuple(base_cfg.get("hog_cells_per_block", [2, 2])),
        color_bins=base_cfg.get("color_bins", 16),
        use_pca=base_cfg.get("pca_enabled", True),
        pca_components=base_cfg.get("pca_components", 64),
        svm_c=base_cfg.get("svm_c", 10.0),
        svm_gamma=base_cfg.get("svm_gamma", "scale"),
        svm_kernel=base_cfg.get("svm_kernel", "rbf"),
        random_state=cfg.get("random_seed", 42)
    )

    print("Extracting handcrafted feature matrix...")
    X_train = model.extract_batch_features(train_imgs)
    print(f"Feature matrix shape: {X_train.shape}")

    print("Fitting Scaler, PCA, and Support Vector Machine...")
    model.fit(X_train, train_y)

    val_accuracy = None
    if val_samples:
        val_imgs, val_y = prepare_patch_dataset(val_samples, patches_per_sample=4, random_seed=cfg.get("random_seed", 42))
        if val_imgs:
            X_val = model.extract_batch_features(val_imgs)
            val_preds = model.predict(X_val)
            val_accuracy = float(np.mean(val_preds == val_y))
            print(f"Validation Patch Accuracy: {val_accuracy:.4f}")

    training_time = time.time() - t0
    print(f"Baseline training completed in {training_time:.2f} seconds.")

    if save_path:
        model.save(save_path)
        print(f"Saved baseline checkpoint to {save_path}")

    return {
        "model": model,
        "training_time_s": training_time,
        "n_train_samples": len(train_imgs),
        "val_accuracy": val_accuracy,
        "feature_dim": X_train.shape[1]
    }


def train_unet(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Train Deep Learning U-Net (Model B) for segmentation and detection."""
    cfg = config or {}
    cnn_cfg = cfg.get("cnn_model", {})
    epochs = cnn_cfg.get("epochs", 15)
    batch_size = cnn_cfg.get("batch_size", 8)
    lr = cnn_cfg.get("learning_rate", 1e-3)
    patience = cnn_cfg.get("early_stopping_patience", 5)
    image_size = tuple(cnn_cfg.get("image_size", [256, 256]))

    dev = device or get_device(cnn_cfg.get("device", "auto"))
    print(f"\n--- Training Deep Learning U-Net (Model B) on {dev} ---")

    train_ds = HullDefectDataset(train_samples, image_size=image_size, augment=True, degrade_prob=0.25)
    val_ds = HullDefectDataset(val_samples, image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = HullUNet(in_channels=3, out_channels=1, base_channels=24).to(dev)
    criterion = CombinedSegmentationLoss(alpha=1.0, beta=1.2, gamma=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cnn_cfg.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "lr": []
    }

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss_accum = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(dev)
            masks = batch["mask"].to(dev)
            labels = batch["label"].to(dev)

            optimizer.zero_grad()
            mask_logits, class_logits = model(imgs)
            loss, _ = criterion(mask_logits, masks, class_logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * imgs.size(0)

        train_loss = train_loss_accum / len(train_ds)

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_ious, val_dices = [], []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(dev)
                masks = batch["mask"].to(dev)
                labels = batch["label"].to(dev)

                mask_logits, class_logits = model(imgs)
                loss, _ = criterion(mask_logits, masks, class_logits, labels)
                val_loss_accum += loss.item() * imgs.size(0)

                pred_probs = torch.sigmoid(mask_logits).cpu().numpy()
                target_masks = masks.cpu().numpy()
                for p, t in zip(pred_probs, target_masks):
                    m = compute_segmentation_metrics(p[0], t[0], threshold=0.5)
                    val_ious.append(m["iou"])
                    val_dices.append(m["dice"])

        val_loss = val_loss_accum / len(val_ds)
        mean_iou = float(np.mean(val_ious))
        mean_dice = float(np.mean(val_dices))
        curr_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(mean_iou)
        history["val_dice"].append(mean_dice)
        history["lr"].append(curr_lr)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {mean_iou:.4f} | Val Dice: {mean_dice:.4f} | LR: {curr_lr:.2e}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            no_improve = 0
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": best_weights,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_iou": mean_iou,
                    "val_dice": mean_dice,
                    "config": cnn_cfg
                }, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        scheduler.step()

    total_time = time.time() - t0
    print(f"U-Net training completed in {total_time:.2f} seconds.")

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "training_time_s": total_time,
        "device": str(dev)
    }
