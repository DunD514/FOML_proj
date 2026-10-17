"""Unit tests for ML models: baseline feature extraction, U-Net architecture, and losses."""

import pytest
import numpy as np
import torch

from src.models.baseline import HullDefectBaselineClassifier
from src.models.cnn import HullUNet, CombinedSegmentationLoss, compute_segmentation_metrics


def test_baseline_feature_extraction():
    """Test handcrafted feature vector generation shape and consistency."""
    classifier = HullDefectBaselineClassifier(
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        color_bins=16,
        use_pca=False
    )
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    feat = classifier.extract_features(img)

    assert feat.ndim == 1
    assert len(feat) > 1000
    assert not np.isnan(feat).any()


def test_unet_forward_pass():
    """Test U-Net dual-head output tensor shapes."""
    model = HullUNet(in_channels=3, out_channels=1, base_channels=16)
    model.eval()

    # Batch of 2 RGB images at 256x256
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        mask_logits, class_logits = model(x)

    # Segmentation head: [B, 1, H, W]
    assert mask_logits.shape == (2, 1, 256, 256)
    # Classification head: [B, 1]
    assert class_logits.shape == (2, 1)


def test_combined_loss_backward():
    """Test that combined BCE + Dice loss produces valid gradients."""
    model = HullUNet(in_channels=3, out_channels=1, base_channels=16)
    criterion = CombinedSegmentationLoss()

    x = torch.randn(2, 3, 128, 128)
    target_mask = torch.randint(0, 2, (2, 1, 128, 128)).float()
    target_label = torch.tensor([[1.0], [0.0]])

    mask_logits, class_logits = model(x)
    loss, loss_dict = criterion(mask_logits, target_mask, class_logits, target_label)

    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    loss.backward()

    # Check that parameters received gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grads


def test_segmentation_metrics():
    """Test IoU and Dice metric calculation against known values."""
    pred = np.zeros((10, 10), dtype=np.float32)
    target = np.zeros((10, 10), dtype=np.float32)

    # 4 overlapping pixels out of 6 total
    pred[2:4, 2:4] = 1.0   # 4 pixels
    target[2:4, 2:4] = 1.0 # 4 pixels
    target[2:4, 4:5] = 1.0 # +2 pixels -> 6 total

    metrics = compute_segmentation_metrics(pred, target, threshold=0.5)

    assert 0.6 < metrics["iou"] < 0.7   # 4 / 6 = 0.667
    assert 0.7 < metrics["dice"] < 0.85 # (2*4) / (4+6) = 0.80
    assert metrics["precision"] > 0.95
