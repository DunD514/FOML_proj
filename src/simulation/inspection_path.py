"""Waypoint path planning and serpentine trajectory generation for hull scanning."""

from typing import List, Tuple, Dict, Any
import numpy as np


class SerpentinePathPlanner:
    """Generates continuous lawnmower/serpentine survey paths and re-inspection routes."""

    def __init__(
        self,
        hull_length_m: float = 50.0,
        hull_height_m: float = 10.0,
        standoff_dist_m: float = 1.5,
        step_x_m: float = 2.0,
        step_y_m: float = 1.2
    ):
        self.hull_length_m = hull_length_m
        self.hull_height_m = hull_height_m
        self.standoff_dist_m = standoff_dist_m
        self.step_x_m = step_x_m
        self.step_y_m = step_y_m

    def generate_serpentine_waypoints(self) -> List[Tuple[float, float, float]]:
        """Generate 3D waypoints for complete serpentine coverage.

        Sweeps back and forth horizontally across X, incrementing Y vertically.
        """
        waypoints = []
        y_levels = np.arange(self.step_y_m / 2.0, self.hull_height_m, self.step_y_m)

        moving_right = True
        for y in y_levels:
            if moving_right:
                x_coords = np.arange(1.0, self.hull_length_m, self.step_x_m)
            else:
                x_coords = np.arange(self.hull_length_m - 1.0, 0.0, -self.step_x_m)

            for x in x_coords:
                waypoints.append((round(float(x), 2), round(float(y), 2), self.standoff_dist_m))

            moving_right = not moving_right

        return waypoints

    def generate_reinspection_waypoints(
        self,
        missed_regions: List[Dict[str, float]]
    ) -> List[Tuple[float, float, float]]:
        """Generate targeted path visiting uninspected/missed sectors."""
        if not missed_regions:
            return []

        # Sort missed sectors in serpentine order
        sorted_regions = sorted(missed_regions, key=lambda r: (r["y_center_m"], r["x_center_m"]))
        waypoints = []
        for r in sorted_regions:
            waypoints.append((
                r["x_center_m"],
                r["y_center_m"],
                self.standoff_dist_m
            ))

        return waypoints
