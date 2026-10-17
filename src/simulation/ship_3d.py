"""3D Ship Hull CAD-like Geometry and ROV Digital Twin Visualizer."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import plotly.graph_objects as go


def create_3d_vessel_hull(
    length_m: float = 50.0,
    height_m: float = 10.0,
    beam_m: float = 12.0
) -> Dict[str, Any]:
    """Generate 3D surface mesh vertices and faces for a commercial vessel hull.

    Coordinate system:
    - X: Longitudinal length (0m = Bow, length_m = Stern)
    - Y: Vertical height (0m = Keel / Bottom, height_m = Waterline / Deck)
    - Z: Transverse beam (-beam/2 = Port, +beam/2 = Starboard)
    """
    nx, ny = 35, 18
    xs = np.linspace(0, length_m, nx)
    ys = np.linspace(0, height_m, ny)

    X_grid, Y_grid = np.meshgrid(xs, ys)
    Z_starboard = np.zeros_like(X_grid)
    Z_port = np.zeros_like(X_grid)

    half_beam = beam_m / 2.0

    for ix in range(nx):
        x = xs[ix]
        # Longitudinal hull profile
        # Bow tapering (X < 12m): sharp entry curving out to full beam
        if x < 12.0:
            beam_factor = np.sin((x / 12.0) * (np.pi / 2.0))
        # Stern tapering (X > 38m): tapering down to stern counter
        elif x > 38.0:
            stern_ratio = (x - 38.0) / (length_m - 38.0)
            beam_factor = 1.0 - (0.45 * stern_ratio**1.5)
        else:
            beam_factor = 1.0

        for iy in range(ny):
            y = ys[iy]
            # Vertical section flare / bilge turn (bottom keel y=0 is narrower, flares up to waterline)
            y_ratio = y / height_m
            flare = 0.55 + 0.45 * (y_ratio ** 0.6)

            z_extent = half_beam * beam_factor * flare
            Z_starboard[iy, ix] = z_extent
            Z_port[iy, ix] = -z_extent

    return {
        "x": X_grid,
        "y": Y_grid,
        "z_starboard": Z_starboard,
        "z_port": Z_port,
        "length_m": length_m,
        "height_m": height_m,
        "beam_m": beam_m
    }


def build_3d_digital_twin_figure(
    hull_data: Dict[str, Any],
    rov_x: float,
    rov_y: float,
    rov_z: float,
    rov_heading_deg: float,
    defects: List[Dict[str, Any]],
    camera_standoff_m: float = 1.5,
    camera_fov_deg: float = 75.0
) -> go.Figure:
    """Construct an interactive 3D digital twin of the vessel hull, ROV, camera cone, and defects."""
    fig = go.Figure()

    X = hull_data["x"]
    Y = hull_data["y"]
    Z_s = hull_data["z_starboard"]

    # 1. Ship Hull 3D Surface (Starboard side shell plating undergoing inspection)
    fig.add_trace(go.Surface(
        x=X, y=Z_s, z=Y,  # Swapping Y and Z so Z is vertical (standard CAD convention)
        colorscale=[
            [0.0, "#8B2500"],   # Keel red anti-fouling primer
            [0.4, "#A52A2A"],   # Hull red
            [0.75, "#2C3E50"],  # Black boot-topping waterline band
            [1.0, "#34495E"]    # Dark steel topsides
        ],
        opacity=0.92,
        showscale=False,
        name="Ship Hull Shell Plating"
    ))

    # 2. Waterline Plane (Z_vertical = 10m surface)
    wl_x = [0, hull_data["length_m"], hull_data["length_m"], 0]
    wl_y = [-hull_data["beam_m"], -hull_data["beam_m"], hull_data["beam_m"] + 4, hull_data["beam_m"] + 4]
    wl_z = [hull_data["height_m"]] * 4
    fig.add_trace(go.Mesh3d(
        x=wl_x, y=wl_y, z=wl_z,
        color="#3498db", opacity=0.25,
        name="Sea Surface Waterline"
    ))

    # 3. Ground-truth Structural Defect Markers on Hull
    for d in defects:
        # Find hull surface offset at defect location
        dx = d["x_m"]
        dy = d["y_m"]
        ix = int(np.clip(dx / (hull_data["length_m"] / (X.shape[1] - 1)), 0, X.shape[1] - 1))
        iy = int(np.clip(dy / (hull_data["height_m"] / (Y.shape[0] - 1)), 0, Y.shape[0] - 1))
        surface_z = Z_s[iy, ix] + 0.05

        fig.add_trace(go.Scatter3d(
            x=[dx], y=[surface_z], z=[dy],
            mode="markers+text",
            marker=dict(size=8, color="#e74c3c", symbol="diamond"),
            text=[f"{d['defect_id']} ({d['defect_type']})"],
            textposition="top center",
            name=f"Defect: {d['defect_id']}"
        ))

    # 4. ROV 3D Position and Standoff
    # ROV is inspecting starboard side: its position is at (rov_x, surface_z + camera_standoff_m, rov_y)
    rov_ix = int(np.clip(rov_x / (hull_data["length_m"] / (X.shape[1] - 1)), 0, X.shape[1] - 1))
    rov_iy = int(np.clip(rov_y / (hull_data["height_m"] / (Y.shape[0] - 1)), 0, Y.shape[0] - 1))
    hull_contact_y = Z_s[rov_iy, rov_ix]
    drone_world_y = hull_contact_y + camera_standoff_m

    # ROV Drone Body (Yellow subsea submersible)
    fig.add_trace(go.Scatter3d(
        x=[rov_x], y=[drone_world_y], z=[rov_y],
        mode="markers+text",
        marker=dict(size=14, color="#f1c40f", symbol="square"),
        text=["AUV/ROV"],
        textposition="top center",
        name="Inspection ROV"
    ))

    # 5. 3D Camera Cone / Frustum (Projecting from ROV onto Hull)
    half_angle = np.radians(camera_fov_deg / 2.0)
    w_footprint = 2.0 * camera_standoff_m * np.tan(half_angle)
    h_footprint = w_footprint * 0.75

    # 4 corners of footprint on hull
    c_tl = [rov_x - w_footprint/2, hull_contact_y, rov_y + h_footprint/2]
    c_tr = [rov_x + w_footprint/2, hull_contact_y, rov_y + h_footprint/2]
    c_br = [rov_x + w_footprint/2, hull_contact_y, rov_y - h_footprint/2]
    c_bl = [rov_x - w_footprint/2, hull_contact_y, rov_y - h_footprint/2]

    # Draw pyramid lines from ROV to corners
    for corner in [c_tl, c_tr, c_br, c_bl]:
        fig.add_trace(go.Scatter3d(
            x=[rov_x, corner[0]],
            y=[drone_world_y, corner[1]],
            z=[rov_y, corner[2]],
            mode="lines",
            line=dict(color="#00cec9", width=3, dash="dot"),
            showlegend=False
        ))

    # Draw footprint rectangle on hull
    rect_x = [c_tl[0], c_tr[0], c_br[0], c_bl[0], c_tl[0]]
    rect_y = [c_tl[1], c_tr[1], c_br[1], c_bl[1], c_tl[1]]
    rect_z = [c_tl[2], c_tr[2], c_br[2], c_bl[2], c_tl[2]]
    fig.add_trace(go.Scatter3d(
        x=rect_x, y=rect_y, z=rect_z,
        mode="lines",
        line=dict(color="#00cec9", width=4),
        name="Camera FOV Footprint"
    ))

    # Camera ray line of sight
    fig.add_trace(go.Scatter3d(
        x=[rov_x, rov_x],
        y=[drone_world_y, hull_contact_y],
        z=[rov_y, rov_y],
        mode="lines",
        line=dict(color="#e74c3c", width=3),
        name="LiDAR / Standoff Beam"
    ))

    # Layout: CAD isometric perspective
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Ship Length X (Bow to Stern) [m]", range=[-2, hull_data["length_m"] + 2], backgroundcolor="#111c24"),
            yaxis=dict(title="Beam Y [m]", range=[-hull_data["beam_m"], hull_data["beam_m"] + 5], backgroundcolor="#111c24"),
            zaxis=dict(title="Depth Z (Keel to Waterline) [m]", range=[-1, hull_data["height_m"] + 2], backgroundcolor="#111c24"),
            camera=dict(
                eye=dict(x=-1.5, y=1.8, z=0.9),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode="manual",
            aspectratio=dict(x=3.2, y=1.2, z=1.0)
        ),
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02, bgcolor="rgba(10,20,30,0.8)", font=dict(color="white"))
    )

    return fig
