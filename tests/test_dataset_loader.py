"""Unit tests for dataset loading, mask parsing, and split hygiene."""

import pytest
import numpy as np
from pathlib import Path

from src.data.annotation_parser import parse_mavecodd_mask, parse_crack_mask, extract_defect_boxes
from src.data.preprocessing import resize_image_and_mask, apply_underwater_degradation
from src.data.split_dataset import create_leakage_free_splits


def test_parse_mavecodd_mask():
    """Test parsing of MaVeCoDD colored corrosion masks into binary format."""
    dummy_mask = np.zeros((100, 100, 3), dtype=np.uint8)
    # Simulate corrosion patch (> 15)
    dummy_mask[20:50, 30:70, :] = [180, 120, 40]
    # Simulate JPEG compression noise (<= 15)
    dummy_mask[0:10, 0:10, :] = [10, 12, 8]

    bin_mask, boxes = parse_mavecodd_mask(dummy_mask, threshold=15)

    assert bin_mask.shape == (100, 100)
    assert bin_mask.dtype == np.uint8
    assert np.all(bin_mask[0:10, 0:10] == 0), "Compression noise should be filtered out"
    assert np.all(bin_mask[25:45, 35:65] == 1), "Defect region should be 1"
    assert len(boxes) >= 1
    assert boxes[0]["defect_type"] == "corrosion"


def test_parse_crack_mask():
    """Test parsing of underwater crack masks."""
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[40:60, 40:60] = 255

    bin_mask, boxes = parse_crack_mask(dummy_mask, threshold=128)

    assert bin_mask.shape == (100, 100)
    assert bin_mask[50, 50] == 1
    assert bin_mask[10, 10] == 0
    assert len(boxes) == 1
    assert boxes[0]["defect_type"] == "crack"


def test_extract_defect_boxes():
    """Test extraction of normalized bounding box coordinates."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:100, 50:150] = 1

    boxes = extract_defect_boxes(mask, defect_type="corrosion", min_area=50)

    assert len(boxes) == 1
    b = boxes[0]
    assert b["bbox_pixel"] == [50, 50, 150, 100]
    assert b["bbox_normalized"] == [0.25, 0.25, 0.75, 0.5]


def test_underwater_degradation():
    """Test underwater optical degradation preserves array shape and range."""
    img = np.full((128, 128, 3), 150, dtype=np.uint8)
    deg = apply_underwater_degradation(img, turbidity=0.4, red_attenuation=0.7)

    assert deg.shape == (128, 128, 3)
    assert deg.dtype == np.uint8
    # Red channel should be attenuated relative to green/blue
    assert np.mean(deg[:, :, 0]) <= np.mean(deg[:, :, 2])


def test_leakage_free_splits():
    """Test that grouped splitting strictly prevents sequence/scene leakage."""
    samples = [
        {"image_id": f"img_{i}", "scene_id": f"scene_{i // 4}"}
        for i in range(24)
    ]
    splits = create_leakage_free_splits(
        samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, group_key="scene_id", random_seed=42
    )

    train_scenes = {s["scene_id"] for s in splits["train"]}
    val_scenes = {s["scene_id"] for s in splits["val"]}
    test_scenes = {s["scene_id"] for s in splits["test"]}

    assert len(train_scenes.intersection(val_scenes)) == 0, "Leakage detected between train and val!"
    assert len(train_scenes.intersection(test_scenes)) == 0, "Leakage detected between train and test!"
    assert len(val_scenes.intersection(test_scenes)) == 0, "Leakage detected between val and test!"
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 24
