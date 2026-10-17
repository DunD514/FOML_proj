"""Annotation parser for underwater hull defect datasets (MaVeCoDD & Underwater Crack)."""

from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
from PIL import Image


def parse_mavecodd_mask(mask_array: np.ndarray, threshold: int = 15) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Parse MaVeCoDD mask into a binary mask and extract defect bounding boxes.

    MaVeCoDD masks preserve color values on corrosion pixels while sound hull background
    is near black (values < 15 due to JPEG compression).

    Args:
        mask_array: RGB or Grayscale numpy array of the labeled image.
        threshold: Pixel value threshold to distinguish defect from sound hull.

    Returns:
        binary_mask: uint8 array (0 for background, 1 for defect).
        annotations: List of defect bounding boxes with attributes.
    """
    if mask_array.ndim == 3:
        # If any channel exceeds threshold, it's a defect pixel
        binary_mask = (np.any(mask_array > threshold, axis=2)).astype(np.uint8)
    else:
        binary_mask = (mask_array > threshold).astype(np.uint8)

    # Clean up minor salt-and-pepper compression artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    annotations = extract_defect_boxes(binary_mask, defect_type="corrosion")
    return binary_mask, annotations


def parse_crack_mask(mask_array: np.ndarray, threshold: int = 128) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Parse underwater crack mask into a binary mask and extract bounding boxes.

    Args:
        mask_array: Grayscale numpy array of the crack mask.
        threshold: Binarization threshold.

    Returns:
        binary_mask: uint8 array (0 for background, 1 for crack).
        annotations: List of defect bounding boxes with attributes.
    """
    if mask_array.ndim == 3:
        mask_gray = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask_array

    binary_mask = (mask_gray > threshold).astype(np.uint8)
    annotations = extract_defect_boxes(binary_mask, defect_type="crack")
    return binary_mask, annotations


def extract_defect_boxes(
    binary_mask: np.ndarray,
    defect_type: str = "defect",
    min_area: int = 30
) -> List[Dict[str, Any]]:
    """Extract connected components as localized bounding boxes with normalized coordinates.

    Args:
        binary_mask: 2D uint8 binary mask (0 or 1).
        defect_type: Label string ('corrosion', 'crack', etc.).
        min_area: Minimum pixel area to consider as a valid defect region.

    Returns:
        List of annotation dictionaries with pixel and normalized bboxes.
    """
    height, width = binary_mask.shape[:2]
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h

        # Calculate normalized coordinates [0.0, 1.0]
        norm_box = [
            round(x_min / width, 4),
            round(y_min / height, 4),
            round(x_max / width, 4),
            round(y_max / height, 4),
        ]

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = round((M["m10"] / M["m00"]) / width, 4)
            cy = round((M["m01"] / M["m00"]) / height, 4)
        else:
            cx = round((x_min + w / 2) / width, 4)
            cy = round((y_min + h / 2) / height, 4)

        boxes.append({
            "defect_id": idx + 1,
            "defect_type": defect_type,
            "bbox_pixel": [x_min, y_min, x_max, y_max],
            "bbox_normalized": norm_box,
            "centroid_normalized": [cx, cy],
            "pixel_area": int(area),
            "area_fraction": round(float(area) / (width * height), 6),
        })

    return boxes
