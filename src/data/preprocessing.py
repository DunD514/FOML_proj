"""Preprocessing and underwater degradation pipeline."""

from typing import Tuple, Optional
import numpy as np
import cv2
from skimage.feature import hog


def resize_image_and_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize image and optional binary mask to target dimensions.

    Args:
        image: RGB image array of shape (H, W, 3).
        mask: Optional binary mask array of shape (H, W).
        target_size: Desired (width, height).

    Returns:
        Resized image and resized mask.
    """
    w, h = target_size
    resized_img = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    if mask is not None:
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        resized_mask = (resized_mask > 0).astype(np.uint8)
    else:
        resized_mask = None

    return resized_img, resized_mask


def apply_underwater_degradation(
    image: np.ndarray,
    turbidity: float = 0.3,
    red_attenuation: float = 0.65,
    contrast_factor: float = 0.8,
    marine_snow_density: float = 0.001,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """Simulate underwater optical physics (Jaffe-McGlamery optical model approximation).

    Simulates:
    1. Selective spectral absorption (red wavelengths attenuate rapidly in water).
    2. Forward scattering / turbidity (haze and blurring).
    3. Contrast loss due to veiling light / backscatter.
    4. Marine snow / suspended particulate matter.

    Args:
        image: RGB image in uint8 [0, 255].
        turbidity: Scattering coefficient in [0.0, 1.0].
        red_attenuation: Factor to suppress red channel [0.0, 1.0].
        contrast_factor: Contrast retention [0.0, 1.0].
        marine_snow_density: Density of suspended particle specks.
        random_seed: Optional seed for reproducibility.

    Returns:
        Degraded RGB image in uint8 [0, 255].
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    img_float = image.astype(np.float32) / 255.0

    # 1. Wavelength-dependent absorption (Red drops heavily, Green/Blue preserved)
    # Channel order: RGB
    img_float[:, :, 0] *= (1.0 - red_attenuation * 0.7)  # Red channel
    img_float[:, :, 1] *= (1.0 - red_attenuation * 0.15) # Green channel
    img_float[:, :, 2] *= 1.0                            # Blue channel

    # 2. Add greenish-blue backscatter veiling light
    veiling_light = np.array([0.05, 0.25, 0.35], dtype=np.float32) * turbidity
    img_float = img_float * (1.0 - turbidity * 0.5) + veiling_light

    # 3. Contrast attenuation
    mean_val = np.mean(img_float, axis=(0, 1), keepdims=True)
    img_float = mean_val + contrast_factor * (img_float - mean_val)

    # 4. Turbidity / forward-scattering blur
    if turbidity > 0.1:
        kernel_size = int(round(turbidity * 6)) * 2 + 1
        if kernel_size >= 3:
            img_float = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigmaX=turbidity * 2.0)

    # 5. Marine snow / particulate noise
    if marine_snow_density > 0:
        h, w = image.shape[:2]
        num_particles = int(h * w * marine_snow_density)
        if num_particles > 0:
            ys = rng.integers(0, h, num_particles)
            xs = rng.integers(0, w, num_particles)
            particle_brightness = rng.uniform(0.6, 1.0, num_particles)
            for y, x, b in zip(ys, xs, particle_brightness):
                radius = rng.integers(1, 3)
                cv2.circle(img_float, (int(x), int(y)), radius, (float(b), float(b), float(b)), -1)

    degraded = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    return degraded


def extract_handcrafted_features(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2),
    color_bins: int = 16
) -> np.ndarray:
    """Extract combined HOG, color histogram, and texture statistics feature vector.

    Used by Model A (Traditional ML Baseline).

    Args:
        image: RGB image in uint8 [0, 255].
        orientations: Number of HOG gradient bins.
        pixels_per_cell: Size of HOG cell in pixels.
        cells_per_block: Number of cells per block for normalization.
        color_bins: Histogram bins per color channel.

    Returns:
        1D numpy array of concatenated features.
    """
    # 1. HOG features on grayscale representation
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_feats = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=False
    )

    # 2. Color Histograms in HSV space (effective for corrosion color signatures)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=color_bins, range=(0, 180), density=True)
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=color_bins, range=(0, 256), density=True)
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=color_bins, range=(0, 256), density=True)

    # 3. Channel statistics (mean, std, skew)
    rgb_means = np.mean(image, axis=(0, 1)) / 255.0
    rgb_stds = np.std(image, axis=(0, 1)) / 255.0

    # 4. Concatenate all handcrafted features
    features = np.concatenate([
        hog_feats,
        h_hist,
        s_hist,
        v_hist,
        rgb_means,
        rgb_stds
    ]).astype(np.float32)

    return features
