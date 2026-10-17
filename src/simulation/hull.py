"""Ship hull geometry, coordinate system, inspection zones, and coverage grid."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class ShipHullSurface:
    """Parametric representation of a ship's side hull surface.

    Coordinate system:
    - X-axis: Longitude along the hull (0.0 m at Bow to length_m at Stern).
    - Y-axis: Depth / Height along the hull (0.0 m at Keel / bottom to height_m at Waterline).
    - Z-axis: Standoff distance normal to hull plate (0.0 m at hull surface).
    """

    def __init__(
        self,
        length_m: float = 50.0,
        height_m: float = 10.0,
        resolution_m: float = 0.5,
        zone_count_x: int = 5,
        zone_count_y: int = 2
    ):
        self.length_m = length_m
        self.height_m = height_m
        self.resolution_m = resolution_m
        self.zone_count_x = zone_count_x
        self.zone_count_y = zone_count_y

        # Coverage grid: 0 = uninspected, 1 = inspected, -1 = missed/flagged
        self.grid_cols = int(np.ceil(length_m / resolution_m))
        self.grid_rows = int(np.ceil(height_m / resolution_m))
        self.coverage_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int8)

        # Ground-truth defect locations on hull [x, y, radius, defect_type, sample_ref]
        self.defects: List[Dict[str, Any]] = []
        self._initialize_zones()

    def _initialize_zones(self) -> None:
        """Partition hull surface into named inspection zones."""
        self.zones: Dict[str, Dict[str, float]] = {}
        zone_len = self.length_m / self.zone_count_x
        zone_h = self.height_m / self.zone_count_y

        x_names = ["Bow", "Forward-Midship", "Midship", "Aft-Midship", "Stern"]
        y_names = ["Keel-Bottom", "Upper-Waterline"]

        for iy in range(self.zone_count_y):
            for ix in range(self.zone_count_x):
                x_lbl = x_names[ix] if ix < len(x_names) else f"SecX_{ix}"
                y_lbl = y_names[iy] if iy < len(y_names) else f"SecY_{iy}"
                zone_id = f"{x_lbl}-{y_lbl}"

                self.zones[zone_id] = {
                    "x_min": ix * zone_len,
                    "x_max": (ix + 1) * zone_len,
                    "y_min": iy * zone_h,
                    "y_max": (iy + 1) * zone_h,
                    "zone_id": zone_id
                }

    def add_defect(
        self,
        x_m: float,
        y_m: float,
        radius_m: float = 0.8,
        defect_type: str = "corrosion",
        image_sample: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a ground-truth defect on the hull."""
        zone = self.get_zone_at(x_m, y_m)
        self.defects.append({
            "defect_id": f"DEF_{len(self.defects) + 1:03d}",
            "x_m": round(x_m, 2),
            "y_m": round(y_m, 2),
            "z_m": 0.0,
            "radius_m": radius_m,
            "defect_type": defect_type,
            "zone_id": zone,
            "image_sample": image_sample
        })

    def get_zone_at(self, x_m: float, y_m: float) -> str:
        """Identify zone name containing coordinates (x, y)."""
        for zone_id, bounds in self.zones.items():
            if bounds["x_min"] <= x_m <= bounds["x_max"] and bounds["y_min"] <= y_m <= bounds["y_max"]:
                return zone_id
        return "Unknown-Zone"

    def mark_coverage(self, x_min: float, x_max: float, y_min: float, y_max: float) -> int:
        """Mark cells covered by camera footprint as inspected."""
        c_min = max(0, min(self.grid_cols - 1, int(x_min / self.resolution_m)))
        c_max = max(0, min(self.grid_cols - 1, int(x_max / self.resolution_m)))
        r_min = max(0, min(self.grid_rows - 1, int(y_min / self.resolution_m)))
        r_max = max(0, min(self.grid_rows - 1, int(y_max / self.resolution_m)))

        newly_covered = 0
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if self.coverage_grid[r, c] == 0:
                    self.coverage_grid[r, c] = 1
                    newly_covered += 1
        return newly_covered

    def get_coverage_percentage(self) -> float:
        """Calculate total percentage of hull surface inspected."""
        total_cells = self.grid_cols * self.grid_rows
        inspected = int(np.sum(self.coverage_grid == 1))
        return round((inspected / total_cells) * 100.0, 2)

    def get_missed_regions(self) -> List[Dict[str, float]]:
        """Identify contiguous rectangular patches of uninspected hull surface."""
        missed = []
        uninspected = (self.coverage_grid == 0)
        # Find rows/columns with uninspected cells
        for r in range(0, self.grid_rows, 4):
            for c in range(0, self.grid_cols, 4):
                block = uninspected[r:r+4, c:c+4]
                if np.mean(block) > 0.5:
                    missed.append({
                        "x_center_m": round((c + 2) * self.resolution_m, 2),
                        "y_center_m": round((r + 2) * self.resolution_m, 2),
                        "width_m": 4 * self.resolution_m,
                        "height_m": 4 * self.resolution_m,
                        "zone_id": self.get_zone_at((c + 2) * self.resolution_m, (r + 2) * self.resolution_m)
                    })
        return missed

    def reset_coverage(self) -> None:
        """Reset coverage map for a new inspection run."""
        self.coverage_grid.fill(0)
