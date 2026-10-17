"""Underwater camera projection, realistic ship hull observation generator, and HUD."""

from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from src.data.preprocessing import apply_underwater_degradation
from src.data.dataset_loader import scan_mavecodd_samples, load_image_and_mask


def generate_realistic_hull_plate(
    drone_x: float,
    drone_y: float,
    resolution: Tuple[int, int] = (640, 480),
    random_seed: int = 42
) -> np.ndarray:
    """Synthesize an authentic commercial ship hull steel plate with weld seams, rivets, and draft marks.

    Guarantees that even without external photos, the user sees an authentic ship hull.
    """
    w, h = resolution
    rng = np.random.default_rng(random_seed + int(drone_x * 10 + drone_y))

    # Base marine anti-fouling hull paint (dark red primer oxide)
    # Bottom hull (keel) is redder, waterline has black boot-topping band
    if drone_y > 8.0:
        base_color = np.array([40, 45, 52], dtype=np.uint8)  # Black boot-topping band near waterline
    else:
        base_color = np.array([125, 42, 32], dtype=np.uint8) # Dark oxide red anti-fouling coating

    img = np.full((h, w, 3), base_color, dtype=np.uint8)

    # Steel plate surface micro-roughness & weathering noise
    noise = rng.normal(0, 14, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Horizontal welded hull strake seam (longitudinal plate joint)
    seam_y = h // 2
    cv2.line(img, (0, seam_y), (w, seam_y), (int(base_color[0] + 35), int(base_color[1] + 35), int(base_color[2] + 35)), 4)
    cv2.line(img, (0, seam_y + 2), (w, seam_y + 2), (20, 20, 20), 2)

    # Vertical frame weld joint
    weld_x = w // 3
    cv2.line(img, (weld_x, 0), (weld_x, h), (int(base_color[0] + 25), int(base_color[1] + 25), int(base_color[2] + 25)), 3)
    cv2.line(img, (weld_x + 1, 0), (weld_x + 1, h), (20, 20, 20), 1)

    # Array of steel rivets / structural bolt studs along frame line
    for ry in range(30, h, 40):
        cv2.circle(img, (weld_x, ry), 5, (200, 210, 220), -1)
        cv2.circle(img, (weld_x, ry), 5, (40, 40, 40), 1)

    # Maritime draft mark stencils (e.g. 6M, 8M, 10M) or Frame numbers
    stencil_font = cv2.FONT_HERSHEY_DUPLEX
    frame_text = f"FR-{int(drone_x * 2 + 10)}"
    cv2.putText(img, frame_text, (weld_x + 15, seam_y - 20), stencil_font, 0.6, (220, 220, 220), 2)
    cv2.putText(img, f"PL-0{int(drone_y / 2 + 1)}", (weld_x + 15, seam_y + 35), stencil_font, 0.5, (180, 180, 180), 1)

    # Zinc sacrificial anode block (corrosion prevention plate on vessel hulls)
    if int(drone_x) % 6 == 0:
        anode_x, anode_y = int(w * 0.65), int(h * 0.35)
        cv2.rectangle(img, (anode_x, anode_y), (anode_x + 90, anode_y + 40), (190, 195, 200), -1)
        cv2.rectangle(img, (anode_x, anode_y), (anode_x + 90, anode_y + 40), (60, 65, 70), 2)
        cv2.circle(img, (anode_x + 20, anode_y + 20), 4, (40, 40, 40), -1)
        cv2.circle(img, (anode_x + 70, anode_y + 20), 4, (40, 40, 40), -1)
        cv2.putText(img, "ZINC ANODE", (anode_x + 10, anode_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 30, 30), 1)

    # Water runoff streaks & rust drip staining
    for _ in range(5):
        rx = int(rng.integers(50, w - 50))
        streak_len = int(rng.integers(40, 120))
        cv2.line(img, (rx, seam_y), (rx + int(rng.integers(-5, 5)), seam_y + streak_len), (160, 70, 20), 2)

    return img


class SimulatedUnderwaterCamera:
    """Simulates an ROV-mounted inspection camera with optical physics and heads-up display (HUD)."""

    def __init__(
        self,
        fov_deg: float = 75.0,
        aspect_ratio: float = 4.0 / 3.0,
        turbidity: float = 0.20,
        red_attenuation: float = 0.50,
        contrast_factor: float = 0.88,
        resolution: Tuple[int, int] = (640, 480),
        random_seed: int = 42
    ):
        self.fov_deg = fov_deg
        self.aspect_ratio = aspect_ratio
        self.turbidity = turbidity
        self.red_attenuation = red_attenuation
        self.contrast_factor = contrast_factor
        self.resolution = resolution
        self.rng = np.random.default_rng(random_seed)
        self.frame_count = 0

        # Auto-load real MaVeCoDD ship hull dataset samples
        root = Path(__file__).resolve().parents[2]
        self.real_hull_samples = scan_mavecodd_samples(
            root / "data/raw/mavecodd",
            zip_fallback=root / "ry392rp8cj-1.zip"
        )

    def calculate_footprint(self, standoff_z_m: float) -> Tuple[float, float]:
        """Compute the physical dimensions (width, height) of hull area in camera FOV."""
        half_angle_rad = np.radians(self.fov_deg / 2.0)
        width_m = 2.0 * standoff_z_m * np.tan(half_angle_rad)
        height_m = width_m / self.aspect_ratio
        return round(float(width_m), 3), round(float(height_m), 3)

    def capture_observation(
        self,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        hull_defects: List[Dict[str, Any]],
        sample_pool: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Simulate capturing a hull observation at given 3D coordinates."""
        self.frame_count += 1
        footprint_w, footprint_h = self.calculate_footprint(drone_z)
        x_min = drone_x - footprint_w / 2.0
        x_max = drone_x + footprint_w / 2.0
        y_min = drone_y - footprint_h / 2.0
        y_max = drone_y + footprint_h / 2.0

        # Check for intersection with registered defects
        intersecting_defect = None
        for d in hull_defects:
            dx = d["x_m"] - drone_x
            dy = d["y_m"] - drone_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= (d.get("radius_m", 1.2) + footprint_w / 2.0):
                intersecting_defect = d
                break

        pool = sample_pool if sample_pool else self.real_hull_samples

        # 1. If defect present, load corresponding real ship corrosion image
        if intersecting_defect is not None and intersecting_defect.get("image_sample"):
            raw_img, raw_mask = load_image_and_mask(intersecting_defect["image_sample"])
            image_source = intersecting_defect["image_sample"].get("dataset_source", "mavecodd")
            image_id = intersecting_defect["image_sample"].get("image_id", "sample_defect")
            gt_defect = True
        elif pool and len(pool) > 0:
            # Map coordinates to a deterministic real ship hull photograph from dataset
            idx = int(abs(drone_x * 5 + drone_y * 7)) % len(pool)
            s = pool[idx]
            raw_img, raw_mask = load_image_and_mask(s)
            image_source = s.get("dataset_source", "mavecodd_hull_photo")
            image_id = s.get("image_id", f"hull_frame_{idx}")
            gt_defect = (np.sum(raw_mask > 0) > 40)
        else:
            # Generate authentic steel hull plate with welded seams and draft markings
            raw_img = generate_realistic_hull_plate(drone_x, drone_y, resolution=self.resolution)
            image_source = "synthetic_vessel_plating"
            image_id = f"hull_plate_{int(drone_x)}_{int(drone_y)}"
            gt_defect = False

        # Resize to sensor resolution
        obs_clean = cv2.resize(raw_img, self.resolution, interpolation=cv2.INTER_AREA)

        # Apply realistic optical underwater attenuation (mild so ship details remain vividly visible)
        obs_degraded = apply_underwater_degradation(
            obs_clean,
            turbidity=self.turbidity,
            red_attenuation=self.red_attenuation,
            contrast_factor=self.contrast_factor,
            marine_snow_density=0.0008
        )

        return {
            "image": obs_degraded,
            "clean_image": obs_clean,
            "footprint_bounds": [x_min, x_max, y_min, y_max],
            "footprint_dimensions_m": [footprint_w, footprint_h],
            "ground_truth_defect": gt_defect,
            "defect_metadata": intersecting_defect,
            "dataset_source": image_source,
            "camera_image_id": image_id,
            "simulated_turbidity": self.turbidity,
            "frame_number": self.frame_count
        }

    def render_hud(
        self,
        frame: np.ndarray,
        sensor_data: Optional[Dict[str, Any]] = None,
        ml_result: Optional[Dict[str, Any]] = None,
        zone_id: str = "Midship-Keel"
    ) -> np.ndarray:
        """Render tactical marine Heads-Up Display (HUD) directly on camera frame."""
        hud = frame.copy()
        h, w = hud.shape[:2]

        # Semi-transparent top and bottom telemetry banners
        overlay = hud.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (10, 18, 25), -1)
        cv2.rectangle(overlay, (0, h - 50), (w, h), (10, 18, 25), -1)
        hud = cv2.addWeighted(hud, 0.75, overlay, 0.25, 0)

        # Center Aiming Reticle / Crosshair
        cx, cy = w // 2, h // 2
        reticle_color = (0, 255, 230) # Cyan
        cv2.circle(hud, (cx, cy), 18, reticle_color, 1)
        cv2.line(hud, (cx - 30, cy), (cx - 10, cy), reticle_color, 1)
        cv2.line(hud, (cx + 10, cy), (cx + 30, cy), reticle_color, 1)
        cv2.line(hud, (cx, cy - 25), (cx, cy - 10), reticle_color, 1)
        cv2.line(hud, (cx, cy + 10), (cx, cy + 25), reticle_color, 1)

        # Extract sensor telemetry
        depth_str = "6.50m"
        pressure_str = "1.66 bar"
        lidar_str = "1.50m"
        alert_str = "SAFE"
        alert_color = (46, 204, 113) # Green
        roll_val, pitch_val, heading_val = 0.0, 0.0, 0.0

        if sensor_data:
            if "depth_pressure" in sensor_data:
                dp = sensor_data["depth_pressure"]
                depth_str = f"{dp['submerged_depth_m']:.2f}m"
                pressure_str = f"{dp['hydrostatic_pressure_bar']:.2f} bar"

            if "lidar" in sensor_data:
                lidar_str = f"{sensor_data['lidar']['min_distance_m']:.2f}m"

            if "collision" in sensor_data:
                col = sensor_data["collision"]
                alert_str = col["alert_level"]
                if alert_str == "CRITICAL_COLLISION":
                    alert_color = (50, 50, 235) # Red
                elif alert_str in ["WARNING", "CAUTION"]:
                    alert_color = (0, 165, 255) # Orange
                else:
                    alert_color = (46, 204, 113) # Green

            if "imu" in sensor_data:
                imu = sensor_data["imu"]
                roll_val = imu["roll_deg"]
                pitch_val = imu["pitch_deg"]
                heading_val = imu["yaw_deg"]

        # 1. Top Ribbon
        top_text = f"HDG: {heading_val:03.0f}  PITCH: {pitch_val:+.1f}  ROLL: {roll_val:+.1f}  ZONE: {zone_id}"
        cv2.putText(hud, top_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 240, 255), 1, cv2.LINE_AA)

        # REC LIVE indicator
        cv2.circle(hud, (w - 75, 21), 6, (0, 0, 255), -1)
        cv2.putText(hud, "LIVE", (w - 62, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        # 2. Left HUD Tape: Depth & Pressure
        cv2.rectangle(hud, (10, 55), (155, 115), (15, 25, 35), -1)
        cv2.rectangle(hud, (10, 55), (155, 115), (0, 200, 220), 1)
        cv2.putText(hud, "DEPTH SENSOR", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 220), 1)
        cv2.putText(hud, f"D: {depth_str}", (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(hud, f"P: {pressure_str}", (16, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 230, 255), 1)

        # 3. Right HUD Tape: LiDAR & Collision
        cv2.rectangle(hud, (w - 185, 55), (w - 10, 115), (15, 25, 35), -1)
        cv2.rectangle(hud, (w - 185, 55), (w - 10, 115), alert_color, 1)
        cv2.putText(hud, "LIDAR STANDOFF", (w - 177, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 220), 1)
        cv2.putText(hud, f"Z: {lidar_str}", (w - 177, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(hud, f"[{alert_str}]", (w - 177, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.42, alert_color, 1)

        # 4. AI Vision Overlay
        if ml_result is not None:
            if ml_result.get("bounding_boxes"):
                for b in ml_result["bounding_boxes"]:
                    x1, y1, x2, y2 = b["bbox_pixel"]
                    sx = w / ml_result["image_shape"][1]
                    sy = h / ml_result["image_shape"][0]
                    bx1, by1 = int(x1 * sx), int(y1 * sy)
                    bx2, by2 = int(x2 * sx), int(y2 * sy)
                    cv2.rectangle(hud, (bx1, by1), (bx2, by2), (0, 60, 255), 2)
                    label = f"{b['defect_type'].upper()} ({ml_result['confidence_score']*100:.0f}%)"
                    cv2.putText(hud, label, (bx1, max(45, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 60, 255), 1)

            det_label = ml_result.get("predicted_defect_class", "sound_hull").replace("_", " ").upper()
            status_text = f"AI VISION: {det_label} | CONF: {ml_result['confidence_score']*100:.1f}% | STATUS: {ml_result['status'].upper()}"
            cv2.putText(hud, status_text, (15, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(hud, "AI VISION: LIVE U-NET DEFECT SCREENING ACTIVE", (15, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1)

        return hud
