"""Traditional ML Baseline: HOG + Color Histograms + PCA + Support Vector Machine."""

import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

from src.data.preprocessing import extract_handcrafted_features, resize_image_and_mask
from src.data.dataset_loader import load_image_and_mask


class HullDefectBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Pipeline combining handcrafted feature extraction, normalization, PCA, and SVM."""

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (16, 16),
        cells_per_block: Tuple[int, int] = (2, 2),
        color_bins: int = 16,
        use_pca: bool = True,
        pca_components: int = 64,
        svm_c: float = 10.0,
        svm_gamma: str = "scale",
        svm_kernel: str = "rbf",
        random_state: int = 42
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.color_bins = color_bins
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.svm_kernel = svm_kernel
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=random_state) if use_pca else None
        self.classifier = SVC(
            C=svm_c,
            gamma=svm_gamma,
            kernel=svm_kernel,
            probability=True,
            random_state=random_state,
            class_weight="balanced"
        )
        self.is_fitted = False

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract 1D handcrafted feature vector from an RGB image."""
        resized_img, _ = resize_image_and_mask(image, target_size=(256, 256))
        return extract_handcrafted_features(
            resized_img,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            color_bins=self.color_bins
        )

    def extract_batch_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features for a list of RGB images."""
        feats = [self.extract_features(img) for img in images]
        return np.vstack(feats)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HullDefectBaselineClassifier":
        """Fit scaler, optional PCA, and SVM on extracted feature matrix X and labels y."""
        X_scaled = self.scaler.fit_transform(X)
        if self.use_pca and self.pca is not None:
            # Adjust pca_components if samples < features
            max_comp = min(X.shape[0] - 1, X.shape[1], self.pca_components)
            if max_comp != self.pca.n_components:
                self.pca = PCA(n_components=max_comp, random_state=self.random_state)
            X_trans = self.pca.fit_transform(X_scaled)
        else:
            X_trans = X_scaled

        self.classifier.fit(X_trans, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary defect labels (0: Sound Hull, 1: Suspected Defect)."""
        X_scaled = self.scaler.transform(X)
        X_trans = self.pca.transform(X_scaled) if self.use_pca and self.pca is not None else X_scaled
        return self.classifier.predict(X_trans)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        X_trans = self.pca.transform(X_scaled) if self.use_pca and self.pca is not None else X_scaled
        return self.classifier.predict_proba(X_trans)

    def predict_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Run single image inference and return class, confidence, and latency."""
        t0 = time.perf_counter()
        feat = self.extract_features(image).reshape(1, -1)
        pred = int(self.predict(feat)[0])
        proba = self.predict_proba(feat)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "prediction": pred,
            "defect_class": "suspected_defect" if pred == 1 else "sound_hull",
            "confidence": float(proba[pred]),
            "probabilities": {
                "sound_hull": float(proba[0]),
                "suspected_defect": float(proba[1]) if len(proba) > 1 else 0.0
            },
            "latency_ms": round(latency_ms, 2)
        }

    def save(self, filepath: str | Path) -> None:
        """Save model bundle to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "HullDefectBaselineClassifier":
        """Load model bundle from disk."""
        return joblib.load(filepath)


def prepare_patch_dataset(
    samples: List[Dict[str, Any]],
    patches_per_sample: int = 4,
    patch_size: int = 256,
    random_seed: int = 42
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Extract sound-hull and defect patches from full-size inspection scenes.

    Generates balanced positive (defect) and negative (sound hull) training examples.
    """
    rng = np.random.default_rng(random_seed)
    images_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for s in samples:
        img, mask = load_image_and_mask(s)
        h, w = img.shape[:2]

        if h < patch_size or w < patch_size:
            resized_img, resized_mask = resize_image_and_mask(img, mask, (patch_size, patch_size))
            has_defect = 1 if np.sum(resized_mask > 0) > 30 else 0
            images_list.append(resized_img)
            labels_list.append(has_defect)
            continue

        # Extract multiple crops: positive crops containing defects, negative crops with zero defects
        mask_bool = (mask > 0)
        has_any_defect = np.sum(mask_bool) > 50

        # Try to find a defect crop
        if has_any_defect:
            defect_ys, defect_xs = np.where(mask_bool)
            for _ in range(max(1, patches_per_sample // 2)):
                idx = rng.integers(0, len(defect_ys))
                cy, cx = defect_ys[idx], defect_xs[idx]
                y1 = max(0, min(h - patch_size, cy - patch_size // 2))
                x1 = max(0, min(w - patch_size, cx - patch_size // 2))
                crop_img = img[y1:y1 + patch_size, x1:x1 + patch_size]
                crop_mask = mask[y1:y1 + patch_size, x1:x1 + patch_size]
                if crop_img.shape[:2] == (patch_size, patch_size):
                    images_list.append(crop_img)
                    labels_list.append(1 if np.sum(crop_mask > 0) > 30 else 0)

        # Try to find a sound hull crop (no defect)
        for _ in range(max(1, patches_per_sample // 2)):
            for attempt in range(10):
                y1 = rng.integers(0, h - patch_size + 1)
                x1 = rng.integers(0, w - patch_size + 1)
                crop_mask = mask[y1:y1 + patch_size, x1:x1 + patch_size]
                if np.sum(crop_mask > 0) < 5:  # Pure sound hull
                    crop_img = img[y1:y1 + patch_size, x1:x1 + patch_size]
                    images_list.append(crop_img)
                    labels_list.append(0)
                    break

    return images_list, np.array(labels_list, dtype=int)
