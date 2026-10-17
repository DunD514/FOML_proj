"""Dataset loader supporting MaVeCoDD, Underwater Crack, and synthetic sample data."""

from typing import List, Dict, Any, Optional, Tuple, Callable
import os
import io
import zipfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.annotation_parser import parse_mavecodd_mask, parse_crack_mask
from src.data.preprocessing import (
    resize_image_and_mask,
    apply_underwater_degradation,
    extract_handcrafted_features
)


class HullDefectDataset(Dataset):
    """PyTorch Dataset for hull inspection images with binary segmentation masks."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        degrade_prob: float = 0.0,
        transform: Optional[Callable] = None,
        random_seed: int = 42
    ):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.degrade_prob = degrade_prob
        self.transform = transform
        self.rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image, mask = load_image_and_mask(sample)

        # Apply resizing
        image, mask = resize_image_and_mask(image, mask, target_size=self.image_size)

        # Optional simulated underwater degradation
        if self.degrade_prob > 0.0 and self.rng.uniform() < self.degrade_prob:
            turbidity = float(self.rng.uniform(0.1, 0.45))
            image = apply_underwater_degradation(image, turbidity=turbidity)

        # Data augmentation (flips and rotation)
        if self.augment:
            if self.rng.uniform() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if self.rng.uniform() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            k = int(self.rng.integers(0, 4))
            if k > 0:
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()

        # Convert to float PyTorch tensors
        # Image: [C, H, W] normalized to [0, 1]
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_normalized = (img_tensor - mean) / std

        # Mask: [1, H, W] with float 0.0 or 1.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        # Classification label: 1 if any defect pixel exists, else 0
        has_defect = 1.0 if mask_tensor.sum() > 10 else 0.0
        label_tensor = torch.tensor([has_defect], dtype=torch.float32)

        return {
            "image": img_normalized,
            "raw_image": torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0,
            "mask": mask_tensor,
            "label": label_tensor,
            "image_id": sample.get("image_id", f"sample_{idx}"),
            "defect_type": sample.get("defect_type", "unknown"),
            "dataset_source": sample.get("dataset_source", "unknown"),
        }


def load_image_and_mask(sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and binary mask from file path or zip archive reference.

    Args:
        sample: Dictionary containing file paths or zip references.

    Returns:
        Tuple of (RGB uint8 image, binary uint8 mask).
    """
    source = sample.get("dataset_source", "mavecodd")

    if "zip_path" in sample and os.path.exists(sample["zip_path"]):
        with zipfile.ZipFile(sample["zip_path"], "r") as z:
            img_bytes = z.read(sample["image_rel_path"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_arr = np.array(img)

            if "mask_rel_path" in sample and sample["mask_rel_path"]:
                mask_bytes = z.read(sample["mask_rel_path"])
                mask_img = Image.open(io.BytesIO(mask_bytes))
                mask_arr = np.array(mask_img)
            else:
                mask_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)
    else:
        # Load directly from disk
        img_path = sample["image_path"]
        img = Image.open(img_path).convert("RGB")
        img_arr = np.array(img)

        if sample.get("mask_path") and os.path.exists(sample["mask_path"]):
            mask_img = Image.open(sample["mask_path"])
            mask_arr = np.array(mask_img)
        else:
            mask_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)

    # Parse mask based on source
    if source == "mavecodd":
        binary_mask, _ = parse_mavecodd_mask(mask_arr)
    elif source == "underwater_crack":
        binary_mask, _ = parse_crack_mask(mask_arr)
    else:
        # Generic mask
        if mask_arr.ndim == 3:
            binary_mask = (mask_arr[:, :, 0] > 128).astype(np.uint8)
        else:
            binary_mask = (mask_arr > 128).astype(np.uint8)

    return img_arr, binary_mask


def scan_mavecodd_samples(
    raw_dir: str | Path,
    zip_fallback: Optional[str | Path] = None
) -> List[Dict[str, Any]]:
    """Scan and index all valid raw image and mask pairs from MaVeCoDD dataset."""
    raw_path = Path(raw_dir)
    samples: List[Dict[str, Any]] = []

    if raw_path.exists() and (raw_path / "HiRes").exists():
        # Extracted files available
        for res_type in ["HiRes", "LoRes"]:
            raw_folder = raw_path / res_type / "raw"
            lbl_folder = raw_path / res_type / "labeled"
            if not raw_folder.exists() or not lbl_folder.exists():
                continue

            for img_file in sorted(raw_folder.glob("*.jpg")):
                stem = img_file.stem
                # Check for corresponding label
                lbl_candidates = [
                    lbl_folder / f"{stem}_label.jpg",
                    lbl_folder / f"{stem}_labels.jpg",
                    lbl_folder / f"{stem}.jpg"
                ]
                lbl_file = next((c for c in lbl_candidates if c.exists()), None)
                if lbl_file:
                    scene_id = stem.split("_")[0] if res_type == "LoRes" else stem
                    samples.append({
                        "image_id": f"mavecodd_{res_type}_{stem}",
                        "image_path": str(img_file),
                        "mask_path": str(lbl_file),
                        "scene_id": scene_id,
                        "defect_type": "corrosion",
                        "resolution_tier": res_type,
                        "dataset_source": "mavecodd"
                    })
    elif zip_fallback and Path(zip_fallback).exists():
        # Fall back to reading inside ZIP without needing extraction
        with zipfile.ZipFile(zip_fallback, "r") as z:
            namelist = set(z.namelist())
            for res_type in ["HiRes", "LoRes"]:
                raws = [n for n in namelist if n.startswith(f"{res_type}/raw/") and n.endswith(".jpg")]
                for raw_entry in sorted(raws):
                    stem = Path(raw_entry).stem
                    lbl_candidates = [
                        f"{res_type}/labeled/{stem}_label.jpg",
                        f"{res_type}/labeled/{stem}_labels.jpg"
                    ]
                    lbl_entry = next((c for c in lbl_candidates if c in namelist), None)
                    if lbl_entry:
                        scene_id = stem[:12] if res_type == "LoRes" else stem
                        samples.append({
                            "image_id": f"mavecodd_{res_type}_{stem}",
                            "zip_path": str(zip_fallback),
                            "image_rel_path": raw_entry,
                            "mask_rel_path": lbl_entry,
                            "scene_id": scene_id,
                            "defect_type": "corrosion",
                            "resolution_tier": res_type,
                            "dataset_source": "mavecodd"
                        })

    return samples


def scan_crack_samples(
    raw_dir: str | Path,
    zip_fallback: Optional[str | Path] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Scan and index all valid raw image and mask pairs from Underwater Crack dataset."""
    raw_path = Path(raw_dir)
    samples: List[Dict[str, Any]] = []

    if raw_path.exists() and (raw_path / "images").exists():
        img_folder = raw_path / "images"
        lbl_folder = raw_path / "labels"
        for img_file in sorted(img_folder.glob("*.jpg")):
            stem = img_file.stem
            lbl_file = lbl_folder / f"{stem}.png"
            if lbl_file.exists():
                samples.append({
                    "image_id": f"crack_{stem}",
                    "image_path": str(img_file),
                    "mask_path": str(lbl_file),
                    "scene_id": f"scene_{(len(samples) // 20):03d}",
                    "defect_type": "crack",
                    "dataset_source": "underwater_crack"
                })
                if limit and len(samples) >= limit:
                    break
    elif zip_fallback and Path(zip_fallback).exists():
        with zipfile.ZipFile(zip_fallback, "r") as z:
            namelist = set(z.namelist())
            imgs = [n for n in namelist if n.startswith("images/") and n.endswith(".jpg")]
            for img_entry in sorted(imgs):
                stem = Path(img_entry).stem
                lbl_entry = f"labels/{stem}.png"
                if lbl_entry in namelist:
                    samples.append({
                        "image_id": f"crack_{stem}",
                        "zip_path": str(zip_fallback),
                        "image_rel_path": img_entry,
                        "mask_rel_path": lbl_entry,
                        "scene_id": f"scene_{(len(samples) // 20):03d}",
                        "defect_type": "crack",
                        "dataset_source": "underwater_crack"
                    })
                    if limit and len(samples) >= limit:
                        break

    return samples


def generate_synthetic_samples(
    output_dir: str | Path,
    num_samples: int = 40,
    image_size: Tuple[int, int] = (256, 256),
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """Generate synthetic ship hull images with simulated corrosion and crack defects.

    Provides instant verification and self-contained execution without external downloads.
    """
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    masks_dir = out_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    w, h = image_size
    samples = []

    for i in range(num_samples):
        # Base hull plate texture (dark gray / navy / red primer steel)
        base_color = rng.choice([
            [60, 70, 80],    # Weathered steel hull
            [120, 40, 35],   # Red antifouling primer
            [45, 55, 65],    # Dark hull plate
        ])
        base_img = np.full((h, w, 3), base_color, dtype=np.uint8)

        # Add plate texture noise
        noise = rng.normal(0, 12, (h, w, 3)).astype(np.int16)
        base_img = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Weld seam line
        weld_y = int(rng.integers(h // 4, 3 * h // 4))
        weld_color = (int(base_color[0] + 25), int(base_color[1] + 25), int(base_color[2] + 25))
        cv2.line(base_img, (0, weld_y), (w, weld_y), weld_color, 3)

        mask = np.zeros((h, w), dtype=np.uint8)
        defect_type = "sound"

        # 60% defect rate (half corrosion, half crack), 40% sound hull
        r = rng.uniform()
        if r < 0.35:
            # Corrosion defect (rough, textured orange-brown rust patch)
            defect_type = "corrosion"
            cx, cy = int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4))
            rx, ry = int(rng.integers(25, 60)), int(rng.integers(20, 50))
            cv2.ellipse(mask, (cx, cy), (rx, ry), int(rng.integers(0, 180)), 0, 360, 255, -1)
            # Add texture to corrosion
            rust_color = np.array([170, 80, 25], dtype=np.uint8)
            rust_patch = np.full((h, w, 3), rust_color, dtype=np.uint8)
            rust_noise = rng.normal(0, 25, (h, w, 3)).astype(np.int16)
            rust_patch = np.clip(rust_patch.astype(np.int16) + rust_noise, 0, 255).astype(np.uint8)
            mask_bool = mask > 0
            base_img[mask_bool] = rust_patch[mask_bool]
        elif r < 0.65:
            # Crack defect (jagged dark fissure)
            defect_type = "crack"
            start_x, start_y = int(rng.integers(20, w - 80)), int(rng.integers(20, h - 80))
            cur_x, cur_y = start_x, start_y
            num_segments = int(rng.integers(6, 14))
            for _ in range(num_segments):
                next_x = cur_x + int(rng.integers(8, 25))
                next_y = cur_y + int(rng.integers(-10, 12))
                next_x = max(0, min(w - 1, next_x))
                next_y = max(0, min(h - 1, next_y))
                cv2.line(mask, (cur_x, cur_y), (next_x, next_y), 255, 3)
                cv2.line(base_img, (cur_x, cur_y), (next_x, next_y), (15, 15, 20), 3)
                cur_x, cur_y = next_x, next_y

        img_filename = f"synthetic_{i:04d}.jpg"
        mask_filename = f"synthetic_{i:04d}_mask.png"
        img_filepath = images_dir / img_filename
        mask_filepath = masks_dir / mask_filename

        Image.fromarray(base_img).save(img_filepath)
        Image.fromarray(mask).save(mask_filepath)

        samples.append({
            "image_id": f"synthetic_{i:04d}",
            "image_path": str(img_filepath),
            "mask_path": str(mask_filepath),
            "scene_id": f"hull_panel_{(i // 10):02d}",
            "defect_type": defect_type,
            "dataset_source": "synthetic"
        })

    return samples
