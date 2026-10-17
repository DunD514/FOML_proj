"""Multi-sensor suite for underwater inspection ROV.

Simulates:
1. Profiling LiDAR / High-frequency acoustic multibeam sounder
2. Hydrostatic pressure and depth sensor
3. Collision detection and proximity warning system
4. 6-DOF Inertial Measurement Unit (IMU)
5. Environmental water quality and temperature sensors
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class LidarSensor:
    """Forward-looking profiling LiDAR / acoustic distance scanner."""

    def __init__(
        self,
        num_beams: int = 19,
        fov_deg: float = 90.0,
        max_range_m: float = 10.0,
        range_noise_std: float = 0.015,
        random_seed: Optional[int] = 42
    ):
        self.num_beams = num_beams
        self.fov_deg = fov_deg
        self.max_range_m = max_range_m
        self.range_noise_std = range_noise_std
        self.rng = np.random.default_rng(random_seed)
        self.angles_deg = np.linspace(-fov_deg / 2.0, fov_deg / 2.0, num_beams)

    def scan(self, drone_x: float, drone_y: float, standoff_z: float, hull_length: float = 50.0) -> Dict[str, Any]:
        """Perform polar LiDAR scan measuring distance to the hull surface plate.

        Args:
            drone_x: Horizontal coordinate along hull.
            drone_y: Vertical coordinate along hull.
            standoff_z: Standoff distance normal to hull plate.
            hull_length: Total hull longitudinal length.

        Returns:
            Dictionary containing beam angles, measured ranges, point cloud, and min distance.
        """
        ranges = []
        point_cloud = []

        for angle in self.angles_deg:
            rad = np.radians(angle)
            # Theoretical geometric distance to planar hull: r = z / cos(angle)
            cos_a = np.cos(rad)
            if cos_a > 1e-3:
                true_dist = standoff_z / cos_a
            else:
                true_dist = self.max_range_m

            # Add hull boundary and slight surface roughness / plate curvature
            surface_noise = float(self.rng.normal(0, self.range_noise_std))
            measured_dist = max(0.1, min(self.max_range_m, true_dist + surface_noise))

            # Point coordinates in ROV local frame (X_local = lateral, Z_local = forward toward hull)
            p_x = measured_dist * np.sin(rad)
            p_z = measured_dist * np.cos(rad)

            ranges.append(round(measured_dist, 3))
            point_cloud.append({
                "angle_deg": round(float(angle), 1),
                "range_m": round(measured_dist, 3),
                "local_x_m": round(float(p_x), 3),
                "local_z_m": round(float(p_z), 3)
            })

        min_range = min(ranges) if ranges else standoff_z
        center_beam = ranges[self.num_beams // 2] if ranges else standoff_z

        return {
            "num_beams": self.num_beams,
            "angles_deg": self.angles_deg.tolist(),
            "ranges_m": ranges,
            "min_distance_m": round(float(min_range), 3),
            "center_standoff_m": round(float(center_beam), 3),
            "point_cloud": point_cloud
        }


class PressureDepthSensor:
    """Hydrostatic pressure and submerged depth sensor."""

    def __init__(
        self,
        waterline_y_m: float = 10.0,
        seawater_density: float = 1025.0, # kg/m^3
        gravity: float = 9.80665,         # m/s^2
        noise_std_bar: float = 0.005,
        random_seed: Optional[int] = 42
    ):
        self.waterline_y_m = waterline_y_m
        self.rho = seawater_density
        self.g = gravity
        self.p_atm_bar = 1.01325
        self.noise_std_bar = noise_std_bar
        self.rng = np.random.default_rng(random_seed)

    def read(self, drone_y_m: float) -> Dict[str, float]:
        """Calculate hydrostatic pressure and submerged depth.

        Coordinate convention: Y=0 is keel/bottom, Y=waterline is sea surface.
        Submerged depth = waterline_y_m - drone_y_m.
        """
        depth_m = max(0.0, self.waterline_y_m - drone_y_m)
        # P = P_atm + rho * g * h (in Pascals, 1 bar = 100,000 Pa)
        hydrostatic_pa = self.rho * self.g * depth_m
        true_pressure_bar = self.p_atm_bar + (hydrostatic_pa / 100000.0)

        # Add sensor noise
        noise = float(self.rng.normal(0, self.noise_std_bar))
        measured_pressure_bar = max(1.0, true_pressure_bar + noise)
        pressure_psi = measured_pressure_bar * 14.5038

        # Simulated water temperature decreases with depth
        temp_c = 18.5 - (depth_m * 0.45) + float(self.rng.normal(0, 0.1))

        return {
            "submerged_depth_m": round(depth_m, 2),
            "hydrostatic_pressure_bar": round(measured_pressure_bar, 3),
            "hydrostatic_pressure_psi": round(pressure_psi, 1),
            "water_temp_c": round(temp_c, 1),
            "seawater_density_kg_m3": self.rho
        }


class CollisionDetectionSystem:
    """Proximity monitoring and real-time collision prevention system."""

    def __init__(
        self,
        safe_standoff_m: float = 1.20,
        warning_standoff_m: float = 0.80,
        critical_standoff_m: float = 0.50
    ):
        self.safe_standoff_m = safe_standoff_m
        self.warning_standoff_m = warning_standoff_m
        self.critical_standoff_m = critical_standoff_m

    def evaluate(self, current_standoff_m: float, approach_velocity_mps: float = 0.0) -> Dict[str, Any]:
        """Evaluate proximity and compute collision alert level and avoidance action."""
        if current_standoff_m < self.critical_standoff_m:
            alert_level = "CRITICAL_COLLISION"
            alert_color = "#e74c3c" # Red
            evasive_action = "EMERGENCY_REVERSE_THRUST"
            auto_brake = True
        elif current_standoff_m < self.warning_standoff_m:
            alert_level = "WARNING"
            alert_color = "#f39c12" # Amber
            evasive_action = "DECELERATE_AND_BACKOFF"
            auto_brake = False
        elif current_standoff_m < self.safe_standoff_m:
            alert_level = "CAUTION"
            alert_color = "#f1c40f" # Yellow
            evasive_action = "MAINTAIN_STANDOFF"
            auto_brake = False
        else:
            alert_level = "SAFE"
            alert_color = "#2ecc71" # Green
            evasive_action = "NOMINAL_SURVEY"
            auto_brake = False

        # Time to impact (TTI)
        tti_s = None
        if approach_velocity_mps > 0.05:
            tti_s = round(current_standoff_m / approach_velocity_mps, 2)

        return {
            "alert_level": alert_level,
            "alert_color": alert_color,
            "evasive_action": evasive_action,
            "auto_brake_active": auto_brake,
            "standoff_distance_m": round(current_standoff_m, 3),
            "time_to_impact_s": tti_s,
            "clearance_margin_m": round(max(0.0, current_standoff_m - self.critical_standoff_m), 3)
        }


class IMUSensor:
    """6-DOF Inertial Measurement Unit with accelerometer and gyroscope."""

    def __init__(self, random_seed: Optional[int] = 42):
        self.rng = np.random.default_rng(random_seed)

    def read(self, heading_deg: float, speed_mps: float) -> Dict[str, float]:
        """Generate 6-DOF IMU telemetry with marine wave surge oscillations."""
        # Simulated wave roll & pitch oscillation
        roll_deg = float(self.rng.normal(0, 1.2))
        pitch_deg = float(self.rng.normal(0, 0.8))
        yaw_deg = float(heading_deg + self.rng.normal(0, 0.5)) % 360.0

        # Angular rates in deg/s
        gyro_x = float(self.rng.normal(0, 0.4))
        gyro_y = float(self.rng.normal(0, 0.3))
        gyro_z = float(self.rng.normal(0, 0.6))

        # Linear acceleration in m/s^2
        accel_x = float(self.rng.normal(0, 0.05))
        accel_y = float(self.rng.normal(0, 0.04))
        accel_z = float(9.81 + self.rng.normal(0, 0.06)) # Gravity

        return {
            "roll_deg": round(roll_deg, 2),
            "pitch_deg": round(pitch_deg, 2),
            "yaw_deg": round(yaw_deg, 2),
            "gyro_x_dps": round(gyro_x, 3),
            "gyro_y_dps": round(gyro_y, 3),
            "gyro_z_dps": round(gyro_z, 3),
            "accel_x_mps2": round(accel_x, 3),
            "accel_y_mps2": round(accel_y, 3),
            "accel_z_mps2": round(accel_z, 3)
        }


class ROVSensorSuite:
    """Aggregated marine telemetry and multi-sensor package."""

    def __init__(
        self,
        waterline_y_m: float = 10.0,
        safe_standoff_m: float = 1.20,
        random_seed: int = 42
    ):
        self.lidar = LidarSensor(num_beams=19, fov_deg=90.0, random_seed=random_seed)
        self.pressure_depth = PressureDepthSensor(waterline_y_m=waterline_y_m, random_seed=random_seed)
        self.collision = CollisionDetectionSystem(safe_standoff_m=safe_standoff_m)
        self.imu = IMUSensor(random_seed=random_seed)

    def sample_all(
        self,
        drone_x: float,
        drone_y: float,
        standoff_z: float,
        heading_deg: float,
        speed_mps: float,
        approach_velocity: float = 0.0
    ) -> Dict[str, Any]:
        """Acquire simultaneous synchronous readings from all sensors."""
        lidar_data = self.lidar.scan(drone_x, drone_y, standoff_z)
        depth_data = self.pressure_depth.read(drone_y)
        collision_data = self.collision.evaluate(lidar_data["min_distance_m"], approach_velocity)
        imu_data = self.imu.read(heading_deg, speed_mps)

        # Environmental salinity & turbidity metrics
        turbidity_ntu = round(float(np.clip(12.0 + standoff_z * 4.5, 5.0, 45.0)), 1)
        salinity_psu = 35.2

        return {
            "lidar": lidar_data,
            "depth_pressure": depth_data,
            "collision": collision_data,
            "imu": imu_data,
            "environment": {
                "turbidity_ntu": turbidity_ntu,
                "salinity_psu": salinity_psu,
                "water_temperature_c": depth_data["water_temp_c"]
            }
        }
