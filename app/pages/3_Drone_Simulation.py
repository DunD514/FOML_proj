"""Streamlit Page 3: Autonomous Underwater ROV Simulation, Live HUD Camera Feed, and Multi-Sensor Telemetry."""

import streamlit as st
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Drone Simulation & Multi-Sensor HUD | Hull AI", layout="wide")

from src.utils.reproducibility import get_device
from src.simulation.hull import ShipHullSurface
from src.simulation.rov import UnderwaterROV
from src.simulation.camera import SimulatedUnderwaterCamera
from src.simulation.simulator import HullInspectionSimulator
from src.models.predict import HullInspectionPredictor
from src.storage.parquet_writer import write_inspection_log

st.title("🤖 Autonomous ROV Simulation & Multi-Sensor Cockpit")
st.markdown("Live underwater inspection camera feed with tactical HUD overlay, 19-beam LiDAR scanning, collision detection, and hydrostatic pressure telemetry.")

# Load models for real-time inference
@st.cache_resource
def get_cached_predictor():
    checkpoints_dir = ROOT / "models" / "checkpoints"
    b_path = checkpoints_dir / "baseline_mavecodd.joblib"
    u_path = checkpoints_dir / "unet_mavecodd.pth"
    if u_path.exists() or b_path.exists():
        dev = get_device("auto")
        return HullInspectionPredictor(
            baseline_path=b_path if b_path.exists() else None,
            unet_path=u_path if u_path.exists() else None,
            device=dev
        )
    return None

predictor = get_cached_predictor()

# Simulation state management
if "sim" not in st.session_state:
    hull = ShipHullSurface(length_m=50.0, height_m=10.0, resolution_m=0.5)
    # Register defects
    hull.add_defect(8.5, 3.2, 1.0, "corrosion")
    hull.add_defect(18.0, 7.5, 1.2, "corrosion")
    hull.add_defect(28.5, 4.0, 0.9, "corrosion")
    hull.add_defect(38.0, 6.8, 1.1, "corrosion")
    hull.add_defect(44.5, 2.5, 0.8, "corrosion")

    rov = UnderwaterROV(x=1.0, y=1.0, z=1.5, speed_mps=0.8, random_seed=42)
    cam = SimulatedUnderwaterCamera(fov_deg=75.0, turbidity=0.20, resolution=(640, 480))
    st.session_state["sim"] = HullInspectionSimulator(
        hull=hull, rov=rov, camera=cam, predictor=predictor
    )

sim: HullInspectionSimulator = st.session_state["sim"]

# Sidebar simulation parameters & sensor controls
st.sidebar.header("🕹️ ROV Navigation & Sensor Controls")
sim.rov.speed_mps = st.sidebar.slider("ROV Cruise Speed (m/s)", 0.2, 2.5, float(sim.rov.speed_mps), 0.1)
sim.camera.turbidity = st.sidebar.slider("Water Turbidity (Scattering)", 0.0, 0.7, float(sim.camera.turbidity), 0.05)
sim.camera.red_attenuation = st.sidebar.slider("Red Channel Absorption", 0.0, 0.9, float(sim.camera.red_attenuation), 0.05)

st.sidebar.subheader("🚨 Collision Test Standoff")
standoff = st.sidebar.slider(
    "LiDAR Standoff Distance Z (m)",
    min_value=0.30,
    max_value=3.0,
    value=float(sim.rov.z),
    step=0.05,
    help="Adjust distance normal to hull plate. <0.8m triggers WARNING; <0.5m triggers CRITICAL COLLISION alert."
)
sim.rov.z = standoff

# Control Action Buttons
st.subheader("🎮 Inspection Mission Controls")
c_b1, c_b2, c_b3, c_b4, c_b5 = st.columns(5)
with c_b1:
    btn_step = st.button("▶️ Step (+1)", use_container_width=True)
with c_b2:
    btn_step10 = st.button("⏩ Advance (+10)", use_container_width=True)
with c_b3:
    live_stream = st.toggle("🔴 Live Video Stream Loop", value=False)
with c_b4:
    btn_save = st.button("💾 Save Parquet Log", use_container_width=True)
with c_b5:
    btn_reset = st.button("🔄 Reset Mission", type="secondary", use_container_width=True)

if btn_reset:
    hull = ShipHullSurface(length_m=50.0, height_m=10.0, resolution_m=0.5)
    hull.add_defect(8.5, 3.2, 1.0, "corrosion")
    hull.add_defect(18.0, 7.5, 1.2, "corrosion")
    hull.add_defect(28.5, 4.0, 0.9, "corrosion")
    hull.add_defect(38.0, 6.8, 1.1, "corrosion")
    hull.add_defect(44.5, 2.5, 0.8, "corrosion")
    rov = UnderwaterROV(x=1.0, y=1.0, z=standoff, speed_mps=sim.rov.speed_mps, random_seed=42)
    cam = SimulatedUnderwaterCamera(fov_deg=75.0, turbidity=sim.camera.turbidity, resolution=(640, 480))
    st.session_state["sim"] = HullInspectionSimulator(
        hull=hull, rov=rov, camera=cam, predictor=predictor
    )
    st.session_state["last_step_res"] = None
    st.rerun()

# Execution dispatch
steps_to_run = 0
if btn_step:
    steps_to_run = 1
elif btn_step10:
    steps_to_run = 10

if steps_to_run > 0:
    last_res = None
    for _ in range(steps_to_run):
        last_res = sim.step(dt_s=0.6)
        if sim.is_completed:
            break
    st.session_state["last_step_res"] = last_res

if btn_save and sim.inspection_logs:
    logs_dir = ROOT / "outputs" / "inspection_logs"
    save_path = logs_dir / f"{sim.inspection_id}.parquet"
    write_inspection_log(sim.inspection_logs, save_path)
    st.success(f"Successfully serialized {len(sim.inspection_logs)} observation steps to `{save_path.name}`!")

# Handle live stream loop
camera_slot = st.empty()
sensor_slot = st.empty()

# Acquire current step or latest step
current_res = st.session_state.get("last_step_res")
if current_res is None and not sim.inspection_logs:
    # Trigger initial observation
    current_res = sim.step(dt_s=0.1)
    st.session_state["last_step_res"] = current_res

# If live stream toggled on, animate steps
if live_stream:
    for _ in range(15):
        current_res = sim.step(dt_s=0.5)
        st.session_state["last_step_res"] = current_res
        
        # Render live HUD frame in real-time
        hud_img = current_res["observation"].get("hud_image", current_res["observation"]["image"])
        camera_slot.image(hud_img, caption="🔴 LIVE ROV INSPECTION FEED (TACTICAL HUD)", use_container_width=True)
        time.sleep(0.08)

# Telemetry KPI Banner
sensor_data = current_res.get("sensor_data", {}) if current_res else {}
dp_data = sensor_data.get("depth_pressure", {})
lidar_data = sensor_data.get("lidar", {})
col_data = sensor_data.get("collision", {})
imu_data = sensor_data.get("imu", {})

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Coverage", f"{sim.hull.get_coverage_percentage():.1f}%")
k2.metric("Submerged Depth", f"{dp_data.get('submerged_depth_m', 9.0):.2f} m")
k3.metric("Hydrostatic Pressure", f"{dp_data.get('hydrostatic_pressure_bar', 1.91):.2f} bar")
k4.metric("LiDAR Standoff", f"{lidar_data.get('min_distance_m', sim.rov.z):.2f} m")
k5.metric("Collision Status", col_data.get("alert_level", "SAFE"), delta="Normal" if col_data.get("alert_level") == "SAFE" else "ALERT", delta_color="normal" if col_data.get("alert_level") == "SAFE" else "inverse")
k6.metric("Defects Flagged", sum(1 for r in sim.inspection_logs if r["defect_detected"]))

st.write("---")

# Main Cockpit Layout: Live Camera Feed with HUD (Left) vs Multi-Sensor Telemetry (Right)
col_feed, col_sensors = st.columns([1, 1])

with col_feed:
    st.subheader("📹 Tactical Live Camera Feed (Optical HUD)")
    if current_res and "observation" in current_res:
        obs = current_res["observation"]
        hud_frame = obs.get("hud_image", obs["image"])
        camera_slot.image(hud_frame, caption=f"ROV Forward Camera | Frame #{obs.get('frame_number', 1)} | Zone: {current_res['record']['zone_id']}", use_container_width=True)
        
        ml_res = current_res.get("ml_result")
        if ml_res:
            st.markdown(f"""
            <div style="background-color: #17252a; padding: 10px; border-radius: 6px; color: #4deeea; font-family: monospace; font-size: 0.85rem;">
                <b>AI VISION STATUS:</b> {ml_res['predicted_defect_class'].upper()} &bull; 
                <b>CONFIDENCE:</b> {ml_res['confidence_score']*100:.1f}% &bull; 
                <b>AREA:</b> {ml_res['defect_area_fraction']*100:.2f}% &bull; 
                <b>LATENCY:</b> {ml_res['inference_time_ms']:.1f}ms
            </div>
            """, unsafe_allow_html=True)

with col_sensors:
    st.subheader("📡 LiDAR Profiling & Collision Radar")
    
    # 1. Polar LiDAR / Sonar Scanner Chart
    if lidar_data and "angles_deg" in lidar_data:
        angles = lidar_data["angles_deg"]
        ranges = lidar_data["ranges_m"]
        
        # Color-code beams based on collision threshold
        beam_colors = []
        for r in ranges:
            if r < 0.50:
                beam_colors.append("#e74c3c") # Red
            elif r < 0.80:
                beam_colors.append("#f39c12") # Amber
            elif r < 1.20:
                beam_colors.append("#f1c40f") # Yellow
            else:
                beam_colors.append("#2ecc71") # Green
        
        fig_lidar = go.Figure()
        # LiDAR beams
        fig_lidar.add_trace(go.Barpolar(
            r=ranges,
            theta=[a + 90 for a in angles], # Rotate 90 deg so 0 deg points forward (North)
            width=[4.5] * len(angles),
            marker=dict(color=beam_colors, line=dict(color="#111", width=1)),
            name="LiDAR Beams",
            hoverinfo="r+theta"
        ))
        
        # Critical collision perimeter ring (0.5m)
        crit_ring = [0.50] * 36
        fig_lidar.add_trace(go.Scatterpolar(
            r=crit_ring,
            theta=np.linspace(0, 360, 36),
            mode="lines",
            line=dict(color="#e74c3c", width=1.5, dash="dot"),
            name="Critical 0.5m Perimeter"
        ))
        
        fig_lidar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, max(3.5, max(ranges) + 0.5)], visible=True, showticklabels=True),
                angularaxis=dict(direction="clockwise", rotation=90)
            ),
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_lidar, use_container_width=True)
    
    # 2. Collision Warning Alert Banner
    alert_lvl = col_data.get("alert_level", "SAFE")
    alert_col = col_data.get("alert_color", "#2ecc71")
    action_text = col_data.get("evasive_action", "NOMINAL_SURVEY")
    
    st.markdown(f"""
    <div style="background-color: {alert_col}22; border: 2px solid {alert_col}; border-radius: 8px; padding: 12px; margin-top: 5px;">
        <span style="font-weight: 700; color: {alert_col}; font-size: 1.05rem;">
            🛡️ COLLISION AVOIDANCE SYSTEM: {alert_lvl}
        </span><br/>
        <span style="font-size: 0.85rem; color: #333;">
            <b>Hull Clearance:</b> {col_data.get('clearance_margin_m', 0.8):.2f}m &bull; 
            <b>Action:</b> <code>{action_text}</code> &bull; 
            <b>Auto-Brake:</b> {'ACTIVE' if col_data.get('auto_brake_active') else 'STANDBY'}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Multi-Sensor Diagnostics Row
st.subheader("📊 Marine Instrumentation Diagnostics")
s_col1, s_col2, s_col3 = st.columns(3)

with s_col1:
    st.markdown("##### 🌊 Hydrostatic Depth & Pressure")
    st.markdown(f"""
    - **Submerged Depth:** `{dp_data.get('submerged_depth_m', 9.0):.2f} m` (Waterline: 10.0m)
    - **Hydrostatic Pressure:** `{dp_data.get('hydrostatic_pressure_bar', 1.91):.3f} bar`
    - **Imperial Pressure:** `{dp_data.get('hydrostatic_pressure_psi', 27.7):.1f} PSI`
    - **Ambient Seawater Temp:** `{dp_data.get('water_temp_c', 14.5):.1f} °C`
    - **Seawater Density:** `1025.0 kg/m³`
    """)

with s_col2:
    st.markdown("##### 🧭 6-DOF IMU Attitude & Wave Swell")
    st.markdown(f"""
    - **Roll Angle ($\phi$):** `{imu_data.get('roll_deg', 0.0):+.2f}°`
    - **Pitch Angle ($\\theta$):** `{imu_data.get('pitch_deg', 0.0):+.2f}°`
    - **Compass Heading ($\psi$):** `{imu_data.get('yaw_deg', 0.0):.1f}°`
    - **Angular Gyro Rates:** `X: {imu_data.get('gyro_x_dps', 0.0):+.2f}°/s`, `Y: {imu_data.get('gyro_y_dps', 0.0):+.2f}°/s`
    - **Vertical Acceleration ($a_z$):** `{imu_data.get('accel_z_mps2', 9.81):.2f} m/s²`
    """)

with s_col3:
    st.markdown("##### 🧪 Water Quality & Thruster Telemetry")
    env = sensor_data.get("environment", {})
    st.markdown(f"""
    - **Turbidity:** `{env.get('turbidity_ntu', 18.5)} NTU` (Optical Attenuation $c=0.65$)
    - **Salinity:** `{env.get('salinity_psu', 35.2)} PSU` (Standard Marine Salinity)
    - **ROV Battery Remaining:** `{sim.rov.battery_percent:.1f}%`
    - **Cumulative Distance:** `{sim.rov.total_distance_m:.1f} m`
    - **Thruster Dynamic Power:** `64.2% Nominal Thrust`
    """)

st.write("---")

# 2D Hull Map
st.subheader("🗺️ Vessel Hull Inspection Map & Trajectory")

fig_map = go.Figure()

# Draw Hull Boundary
fig_map.add_shape(type="rect", x0=0, y0=0, x1=sim.hull.length_m, y1=sim.hull.height_m,
                  line=dict(color="#1e3d59", width=3), fillcolor="#f0f4f8")

# Draw Inspection Zones
for z_id, z in sim.hull.zones.items():
    fig_map.add_shape(type="rect", x0=z["x_min"], y0=z["y_min"], x1=z["x_max"], y1=z["y_max"],
                      line=dict(color="#b0bec5", width=1, dash="dot"))
    fig_map.add_annotation(x=(z["x_min"] + z["x_max"])/2, y=z["y_max"] - 0.4,
                           text=z_id.split("-")[0], showarrow=False, font=dict(size=9, color="#78909c"))

# Coverage Grid Overlay
cov_y, cov_x = np.where(sim.hull.coverage_grid == 1)
if len(cov_x) > 0:
    fig_map.add_trace(go.Scatter(
        x=cov_x * sim.hull.resolution_m,
        y=cov_y * sim.hull.resolution_m,
        mode="markers",
        marker=dict(size=6, color="#48dbfb", symbol="square", opacity=0.35),
        name="Inspected Grid Cells"
    ))

# Planned Serpentine Path
wps = sim.waypoints
fig_map.add_trace(go.Scatter(
    x=[p[0] for p in wps],
    y=[p[1] for p in wps],
    mode="lines",
    line=dict(color="#b2bec3", width=1, dash="dash"),
    name="Planned Path"
))

# Ground-truth Defect Regions
for d in sim.hull.defects:
    fig_map.add_shape(type="circle", x0=d["x_m"]-d["radius_m"], y0=d["y_m"]-d["radius_m"],
                      x1=d["x_m"]+d["radius_m"], y1=d["y_m"]+d["radius_m"],
                      line=dict(color="#e74c3c", width=2), fillcolor="rgba(231, 76, 60, 0.25)")
    fig_map.add_annotation(x=d["x_m"], y=d["y_m"], text=d["defect_id"], showarrow=False,
                           font=dict(size=10, color="#c0392b"))

# Logged Detections
det_x = [r["drone_x"] for r in sim.inspection_logs if r["defect_detected"]]
det_y = [r["drone_y"] for r in sim.inspection_logs if r["defect_detected"]]
if det_x:
    fig_map.add_trace(go.Scatter(
        x=det_x, y=det_y, mode="markers",
        marker=dict(size=9, color="#d63031", symbol="x"),
        name="Suspected Defect Detections"
    ))

# ROV Current Position Marker
fig_map.add_trace(go.Scatter(
    x=[sim.rov.x], y=[sim.rov.y], mode="markers+text",
    marker=dict(size=16, color="#0984e3", symbol="diamond"),
    text=["ROV"], textposition="top center",
    name="ROV Position"
))

# Camera FOV Footprint
fw, fh = sim.camera.calculate_footprint(sim.rov.z)
fig_map.add_shape(type="rect", x0=sim.rov.x - fw/2, y0=sim.rov.y - fh/2,
                  x1=sim.rov.x + fw/2, y1=sim.rov.y + fh/2,
                  line=dict(color="#0984e3", width=2), fillcolor="rgba(9, 132, 227, 0.20)")

fig_map.update_layout(
    xaxis=dict(range=[-1, sim.hull.length_m + 1], title="Hull Longitude (Bow to Stern) [meters]"),
    yaxis=dict(range=[-1, sim.hull.height_m + 1], title="Hull Depth (Keel to Waterline) [meters]", scaleanchor="x", scaleratio=1),
    height=420, margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_map, use_container_width=True)

# Logged steps summary table (Pandas dataframe with pd defined)
if sim.inspection_logs:
    st.write("---")
    st.subheader("📋 Recent Telemetry Logs (PyArrow Stream)")
    df_logs = pd.DataFrame(sim.inspection_logs)
    st.dataframe(
        df_logs[[
            "step_id", "timestamp", "drone_x", "drone_y", "drone_heading_deg",
            "zone_id", "predicted_defect_class", "confidence_score", "coverage_pct"
        ]].tail(10),
        use_container_width=True
    )
