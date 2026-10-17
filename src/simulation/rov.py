"""Underwater ROV kinematics, sensor noise, and state management."""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class UnderwaterROV:
    """Simulated Remotely Operated Vehicle (ROV) inspecting a ship's hull."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 5.0,
        z: float = 1.5,
        heading_deg: float = 0.0,
        speed_mps: float = 0.8,
        position_noise_std: float = 0.04,
        heading_noise_std: float = 0.8,
        random_seed: Optional[int] = 42
    ):
        # Ground truth state
        self.x = x
        self.y = y
        self.z = z  # Standoff distance from hull plate
        self.heading_deg = heading_deg  # Yaw angle in degrees
        self.speed_mps = speed_mps

        # Velocity components
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        # Sensor noise characteristics (simulated vs measured state)
        self.position_noise_std = position_noise_std
        self.heading_noise_std = heading_noise_std
        self.rng = np.random.default_rng(random_seed)

        # Operational metrics
        self.total_distance_m = 0.0
        self.battery_percent = 100.0
        self.is_active = True

    def update_motion(self, target_x: float, target_y: float, target_z: float, dt_s: float) -> Tuple[float, float, float]:
        """Step kinematic motion towards a 3D target coordinate with thruster dynamics."""
        dx = target_x - self.x
        dy = target_y - self.y
        dz = target_z - self.z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if dist < 1e-4:
            self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
            return self.get_measured_pose()

        # Compute unit direction vector
        dir_x, dir_y, dir_z = dx / dist, dy / dist, dz / dist
        step_dist = min(self.speed_mps * dt_s, dist)

        # Update ground-truth position
        prev_x, prev_y, prev_z = self.x, self.y, self.z
        self.x += dir_x * step_dist
        self.y += dir_y * step_dist
        self.z += dir_z * step_dist

        actual_moved = np.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2 + (self.z - prev_z)**2)
        self.total_distance_m += actual_moved
        self.battery_percent = max(0.0, self.battery_percent - (actual_moved * 0.02))

        # Update velocities and heading
        self.vx = (self.x - prev_x) / dt_s if dt_s > 0 else 0.0
        self.vy = (self.y - prev_y) / dt_s if dt_s > 0 else 0.0
        self.vz = (self.z - prev_z) / dt_s if dt_s > 0 else 0.0

        if abs(dx) > 1e-3 or abs(dy) > 1e-3:
            target_heading = np.degrees(np.arctan2(dy, dx))
            # Smooth heading adjustment
            diff = (target_heading - self.heading_deg + 180) % 360 - 180
            self.heading_deg = (self.heading_deg + np.clip(diff, -15.0, 15.0)) % 360

        return self.get_measured_pose()

    def get_measured_pose(self) -> Dict[str, float]:
        """Return simulated sensor measurements including Gaussian positioning uncertainty.

        Distinguishes simulated true ground truth from realistic acoustic/DVL sensor data.
        """
        noise_x = float(self.rng.normal(0, self.position_noise_std))
        noise_y = float(self.rng.normal(0, self.position_noise_std))
        noise_z = float(self.rng.normal(0, self.position_noise_std * 0.5))
        noise_heading = float(self.rng.normal(0, self.heading_noise_std))

        return {
            "true_x": round(self.x, 3),
            "true_y": round(self.y, 3),
            "true_z": round(self.z, 3),
            "true_heading_deg": round(self.heading_deg, 2),
            "measured_x": round(self.x + noise_x, 3),
            "measured_y": round(self.y + noise_y, 3),
            "measured_z": round(self.z + noise_z, 3),
            "measured_heading_deg": round((self.heading_deg + noise_heading) % 360, 2),
            "total_distance_m": round(self.total_distance_m, 2),
            "battery_percent": round(self.battery_percent, 1)
        }
