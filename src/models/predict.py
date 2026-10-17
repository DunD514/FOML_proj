"""Unified inference predictor for hull defect detection and segmentation."""

from typing import Dict, Any, List, Optional, Union, Tuple
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

from src.utils.reproducibility import get_device
from src.data.preprocessing import resize_image_and_mask
from src.data.annotation_parser import extract_defect_boxes
from src.models.baseline import HullDefectBaselineClassifier
from src.models.cnn import HullUNet


class HullInspectionPredictor:
    """Production-ready inference wrapper serving both Model A and Model B."""

    def __init__(
        self,
        baseline_path: Optional[str | Path] = None,
        unet_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.40,
        mask_threshold: float = 0.45
    ):
        self.device = device or get_device("auto")
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold

        # Load Model A (Baseline)
        self.baseline_model = None
        if baseline_path and Path(baseline_path).exists():
            try:
                self.baseline_model = HullDefectBaselineClassifier.load(baseline_path)
            except Exception as e:
                print(f"Warning: Could not load baseline model: {e}")

        # Load Model B (U-Net)
        self.unet_model = None
        if unet_path and Path(unet_path).exists():
            try:
                ckpt = torch.load(unet_path, map_location=self.device)
                self.unet_model = HullUNet(in_channels=3, out_channels=1, base_channels=24)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.unet_model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.unet_model.load_state_dict(ckpt)
                self.unet_model.to(self.device)
                self.unet_model.eval()
            except Exception as e:
                print(f"Warning: Could not load U-Net checkpoint: {e}")

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        defect_hint: str = "corrosion"
    ) -> Dict[str, Any]:
        """Run full defect prediction on input image.

        Args:
            image_input: Image path, PIL Image, or RGB numpy array.
            defect_hint: Defect category hint ('corrosion' or 'crack').

        Returns:
            Dictionary containing prediction labels, confidence, mask, and bounding boxes.
        """
        # Load image into numpy RGB
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
            img_arr = np.array(img)
        elif isinstance(image_input, Image.Image):
            img_arr = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            img_arr = image_input if image_input.ndim == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError("Unsupported image input format.")

        orig_h, orig_w = img_arr.shape[:2]
        t0 = time.perf_counter()

        # 1. Baseline Model A inference (if available)
        baseline_res = None
        if self.baseline_model is not None:
            baseline_res = self.baseline_model.predict_image(img_arr)

        # 2. U-Net Model B inference
        if self.unet_model is not None:
            # Resize for model forward pass
            inp_img, _ = resize_image_and_mask(img_arr, target_size=(256, 256))
            tensor_img = torch.from_numpy(inp_img.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor_norm = ((tensor_img - mean) / std).to(self.device)

            with torch.no_grad():
                mask_logits, class_logits = self.unet_model(tensor_norm)
                prob_map_256 = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
                class_prob = float(torch.sigmoid(class_logits)[0, 0].cpu().item())

            # Upscale probability map back to original dimensions
            prob_map = cv2.resize(prob_map_256, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            bin_mask = (prob_map > self.mask_threshold).astype(np.uint8)

            # Extract defect bounding boxes
            boxes = extract_defect_boxes(bin_mask, defect_type=defect_hint)

            defect_pixels = int(np.sum(bin_mask))
            defect_area_frac = float(defect_pixels / (orig_w * orig_h))

            # Determine defect presence based on mask area and class head
            has_defect = (defect_pixels > 40) or (class_prob > self.confidence_threshold and defect_pixels > 10)
            confidence = max(class_prob, float(np.mean(prob_map[bin_mask > 0])) if defect_pixels > 0 else 1.0 - class_prob)

            if has_defect:
                predicted_class = f"suspected_{defect_hint}"
                status = "requires_further_inspection"
            else:
                predicted_class = "sound_hull"
                status = "clear"
        elif baseline_res is not None:
            # Fall back to baseline alone
            has_defect = (baseline_res["prediction"] == 1)
            confidence = baseline_res["confidence"]
            predicted_class = f"suspected_{defect_hint}" if has_defect else "sound_hull"
            status = "requires_further_inspection" if has_defect else "clear"
            prob_map = np.full((orig_h, orig_w), confidence if has_defect else 0.0, dtype=np.float32)
            bin_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            boxes = []
            defect_area_frac = 0.0
        else:
            raise RuntimeError("No model is loaded (neither U-Net nor Baseline).")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_defect_class": predicted_class,
            "status": status,
            "confidence_score": round(confidence, 4),
            "defect_detected": has_defect,
            "probability_map": prob_map,
            "binary_mask": bin_mask,
            "bounding_boxes": boxes,
            "defect_area_fraction": round(defect_area_frac, 6),
            "inference_time_ms": round(latency_ms, 2),
            "baseline_result": baseline_res,
            "image_shape": (orig_h, orig_w)
        }

    @staticmethod
    def draw_overlay(
        image: np.ndarray,
        pred_dict: Dict[str, Any],
        show_mask: bool = True,
        show_boxes: bool = True,
        defect_color: Tuple[int, int, int] = (255, 60, 50)
    ) -> np.ndarray:
        """Render defect mask overlay and localized bounding boxes on image."""
        vis = image.copy()

        # 1. Semi-transparent mask overlay
        if show_mask and "binary_mask" in pred_dict and pred_dict["binary_mask"] is not None:
            mask = pred_dict["binary_mask"]
            if np.sum(mask) > 0:
                color_mask = np.zeros_like(vis)
                color_mask[mask > 0] = defect_color
                vis = cv2.addWeighted(vis, 0.75, color_mask, 0.25, 0)

        # 2. Draw bounding boxes and confidence tags
        if show_boxes and "bounding_boxes" in pred_dict:
            for b in pred_dict["bounding_boxes"]:
                x1, y1, x2, y2 = b["bbox_pixel"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), defect_color, 2)
                label = f"{b['defect_type']}: {b['area_fraction']*100:.1f}%"
                cv2.putText(vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, defect_color, 1)

        return vis
