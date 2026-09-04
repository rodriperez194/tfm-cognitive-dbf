import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_array_layout(array_obj, ax=None, show_indices=False):
    """
    Visualizes the physical layout of the array and the current state of the weights 
    (Magnitude and Phase).

    Visualization Strategy:
    - Position (X, Y): Physical location of the elements in meters.
    - Color: Phase of the complex weights (degrees).
    - Size: Magnitude of the weights (if weight=0, the point disappears).

    Args:
        array_obj: Instance of the Phased_Array_NB class.
        ax (matplotlib.axes.Axes, optional): Axes object to draw on. If None, creates a new figure.
        show_indices (bool): If True, annotates each element with its (n,m) index.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Extract Data
    # Flatten arrays to use scatter() efficiently
    # Physical positions
    X = array_obj.element_positions[:, :, 0].flatten()
    Y = array_obj.element_positions[:, :, 1].flatten()

    # Complex weights
    W = array_obj.W.flatten()

    # Magnitude and Phase
    magnitudes = np.abs(W)
    phases_rad = np.angle(W)
    phases_deg = np.rad2deg(phases_rad)

    # 2. Plot Configuration
    # Normalize marker size for visibility (base_size * magnitude)
    # Adjust '500' based on the figure size or preference
    # We add a small epsilon to avoid size 0 if desired, or let it be 0 to hide off elements
    marker_sizes = 500 * (magnitudes / (np.max(magnitudes) + 1e-9)) 

    # Scatter Plot
    # c=phases_deg: Maps color to phase
    # s=marker_sizes: Maps size to magnitude
    # cmap='hsv': Cyclic colormap (Red at -180 and Red at +180)
    sc = ax.scatter(X, Y, c=phases_deg, s=marker_sizes, 
                    cmap='hsv', alpha=0.8, edgecolors='black', linewidth=0.5,
                    vmin=-180, vmax=180)

    # 3. Decoration and Labels
    # Note: Strings inside the plot (titles/labels) are kept in English as part of the code artifact.
    ax.set_title(f"Array Layout & Weights State\n{array_obj.N}x{array_obj.M} Elements ({array_obj.fc/1e9:.2f} GHz)")
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal') # Crucial to maintain physical geometry

    # Colorbar for Phase
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Weight Phase (deg)")
    # Set ticks to standard angles for readability
    cbar.set_ticks([-180, -90, 0, 90, 180])

    # 4. (Optional) Show indices (n, m) for debugging
    if show_indices:
        for n in range(array_obj.N):
            for m in range(array_obj.M):
                # Recover flat index to find coordinates
                idx = n * array_obj.M + m
                # Only label if the antenna is effectively "active"
                if magnitudes[idx] > 0.1: 
                    ax.text(X[idx], Y[idx], f"{n},{m}", 
                            ha='center', va='center', fontsize=8, color='black', weight='bold')

    return ax

def plot_3d_beampattern(beampattern_data, db_limit=-60, ax=None):
    """
    Plots the full 3D radiation pattern as a 2D Heatmap (Azimuth vs Elevation).
    
    Args:
        beampattern_data (tuple): The tuple (AF_mag, THETA_rad, PHI_rad) returned by 
                                  array.compute_beampattern().
        db_limit (float): The noise floor in dB (min value for the color scale).
        ax (matplotlib.axes.Axes, optional): Target axes.
    """
    # Unpack data
    AF_mag, THETA_rad, PHI_rad = beampattern_data
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to Degrees for plotting
    THETA_deg = np.rad2deg(THETA_rad)
    PHI_deg = np.rad2deg(PHI_rad)

    # Normalize and Convert to dB
    # We add epsilon to avoid log(0)
    AF_mag_norm = AF_mag / np.max(AF_mag)
    with np.errstate(divide='ignore'):
        AF_db = 20 * np.log10(AF_mag_norm + 1e-12)
    
    # Clip values to the dynamic range floor
    AF_db = np.maximum(AF_db, db_limit)

    # Plot Heatmap
    # Note: pcolormesh X argument is Columns (Phi), Y is Rows (Theta)
    mesh = ax.pcolormesh(PHI_deg, THETA_deg, AF_db, 
                         cmap='jet', vmin=db_limit, vmax=0, shading='auto')

    # Decoration
    ax.set_title("3D Beampattern Heatmap")
    ax.set_xlabel("Azimuth (Phi) [deg]")
    ax.set_ylabel("Elevation (Theta) [deg]")
    
    # Set limits explicitly
    ax.set_xlim(np.min(PHI_deg), np.max(PHI_deg))
    ax.set_ylim(np.min(THETA_deg), np.max(THETA_deg))
    
    # Invert Y axis? Usually 0 deg Theta (Zenith) is top in physics, 
    # but in Cartesian plots 0 is bottom. Let's keep 0 bottom to match data matrix.
    
    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Normalized Gain (dB)")
    
    return ax


def plot_2d_beam_cut(beampattern_data, cut_type="azimuth", cut_angle_deg=90, 
                     plot_style="rectangular", db_limit=-40, ax=None):
    """
    Extracts and plots a 2D slice from the pre-computed 3D beampattern data.
    
    Args:
        beampattern_data (tuple): (AF_mag, THETA_rad, PHI_rad).
        cut_type (str): "azimuth" (slices a row, fixed Theta) or 
                        "elevation" (slices a column, fixed Phi).
        cut_angle_deg (float): The angle to freeze.
                               - If Azimuth cut: The fixed Elevation (Theta).
                               - If Elevation cut: The fixed Azimuth (Phi).
        plot_style (str): "rectangular" (Cartesian) or "polar".
        db_limit (float): Minimum dB value to plot.
        ax (matplotlib.axes.Axes, optional): Target axes.
    """
    AF_mag, THETA_rad, PHI_rad = beampattern_data
    
    if ax is None:
        projection = 'polar' if plot_style == 'polar' else None
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'projection': projection})

    # 1. Slice Extraction Logic
    if cut_type.lower() == "azimuth":
        # Fixed Theta -> Varying Phi
        # Find the row index in THETA matrix closest to requested cut_angle_deg
        theta_axis_deg = np.rad2deg(THETA_rad[:, 0])
        idx = np.abs(theta_axis_deg - cut_angle_deg).argmin()
        
        # Extract data
        slice_mag = AF_mag[idx, :]
        angle_axis_rad = PHI_rad[idx, :] # The varying angle
        
        # Meta-data for labels
        xlabel = "Azimuth (Phi) [deg]"
        actual_cut_val = theta_axis_deg[idx]
        fixed_param_name = "Theta"
        
    elif cut_type.lower() == "elevation":
        # Fixed Phi -> Varying Theta
        # Find the col index in PHI matrix closest to requested cut_angle_deg
        phi_axis_deg = np.rad2deg(PHI_rad[0, :])
        idx = np.abs(phi_axis_deg - cut_angle_deg).argmin()
        
        # Extract data
        slice_mag = AF_mag[:, idx]
        angle_axis_rad = THETA_rad[:, idx] # The varying angle
        
        # Meta-data for labels
        xlabel = "Elevation (Theta) [deg]"
        actual_cut_val = phi_axis_deg[idx]
        fixed_param_name = "Phi"
    
    else:
        raise ValueError(f"Unknown cut_type: {cut_type}")

    # 2. dB Conversion
    # Normalize to global max of the slice (usually we want global max of array, 
    # but here let's normalize to global max of the input data provided)
    global_max = np.max(AF_mag)
    slice_mag_norm = slice_mag / global_max
    
    with np.errstate(divide='ignore'):
        slice_db = 20 * np.log10(slice_mag_norm + 1e-12)
    
    slice_db = np.maximum(slice_db, db_limit)
    angle_axis_deg = np.rad2deg(angle_axis_rad)

    # 3. Plotting
    if plot_style == "polar":
        ax.plot(angle_axis_rad, slice_db, linewidth=2)
        ax.set_ylim(db_limit, 0)
        
        # Visual tweak for Elevation cuts in Polar:
        # Standard convention: Zenith (0 deg) is Up.
        if cut_type == "elevation":
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1) # Clockwise (0->90->180)
            
    else: # Rectangular
        ax.plot(angle_axis_deg, slice_db, linewidth=2)
        ax.set_xlim(np.min(angle_axis_deg), np.max(angle_axis_deg))
        ax.set_ylim(db_limit, 2) # A bit of headroom above 0 dB
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized Gain (dB)")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)

    ax.set_title(f"{cut_type.title()} Cut (Fixed {fixed_param_name} ≈ {actual_cut_val:.1f}°)")
    
    return ax

def quick_plot_beam(beampattern_data, cut_type="azimuth", cut_angle_deg=90, 
                    style="rectangular", target_angle=None, db_limit=-40):
    """
    High-level wrapper to plot a radiation cut with a single function call.
    Handles figure creation, projection setup, and target annotation.

    Args:
        beampattern_data: The (AF, THETA, PHI) tuple.
        cut_type (str): "azimuth" or "elevation".
        cut_angle_deg (float): The fixed angle for the slice.
        style (str): "rectangular" or "polar".
        target_angle (float, optional): If provided, draws a reference line at this angle.
        db_limit (float): Floor for the dB scale.

    Returns:
        fig, ax: The figure and axes objects.
    """
    # 1. Automatic Figure and Projection Setup
    projection = 'polar' if style == "polar" else None
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': projection})

    # 2. Call the core plotting function
    plot_2d_beam_cut(beampattern_data, cut_type=cut_type, cut_angle_deg=cut_angle_deg, 
                     plot_style=style, db_limit=db_limit, ax=ax)

    # 3. Automatic Target Annotation
    if target_angle is not None:
        if style == "polar":
            # In polar, we draw a radial line
            target_rad = np.deg2rad(target_angle)
            ax.plot([target_rad, target_rad], [db_limit, 0], 
                    color='red', linestyle=':', linewidth=2, label=f'Target ({target_angle}°)')
        else:
            # In rectangular, we draw a vertical line
            ax.axvline(target_angle, color='red', linestyle=':', linewidth=2, 
                       label=f'Target ({target_angle}°)')
        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig, ax

def doa_spectrum_to_beampattern_data(spatial_spectrum, theta_grid_deg, phi_grid_deg):
    """
    Adapter: converts flat DOA spectrum (K,) into
    (AF_mag, THETA_rad, PHI_rad) compatible with your
    plot_3d_beampattern and quick_plot_beam functions.
    """
    n_theta = len(theta_grid_deg)
    n_phi = len(phi_grid_deg)

    AF_mag = np.asarray(spatial_spectrum).reshape(n_theta, n_phi)

    theta_mesh_deg, phi_mesh_deg = np.meshgrid(
        theta_grid_deg,
        phi_grid_deg,
        indexing="ij"
    )

    THETA_rad = np.deg2rad(theta_mesh_deg)
    PHI_rad   = np.deg2rad(phi_mesh_deg)

    return AF_mag, THETA_rad, PHI_rad

def plot_beamforming_comparison(
    array,
    weight_dict,
    target_direction,
    jammer_direction,
    cut_reference="target",   # "target" or "jammer"
    floor_db=-60.0,
    color_map=None,
    color_target="green",
    color_jammer="red",
    show_legend=True,
    show=True,
):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from tfm.math.narrow_band.array_response import compute_array_factor

    # ------------------------------------------------------------
    # Defaults for colors
    # ------------------------------------------------------------
    if color_map is None:
        color_map = {
            "Steering": "#1f77b4",
            "Null steering": "#ff7f0e",
        }

    # ------------------------------------------------------------
    # Cut selection
    # ------------------------------------------------------------
    if cut_reference not in {"target", "jammer"}:
        raise ValueError("cut_reference must be 'target' or 'jammer'.")

    if cut_reference == "target":
        phi_cut = target_direction[1]
        theta_cut = target_direction[0]
        main_title = "Beamforming Comparison Around Target Direction"
    else:
        phi_cut = jammer_direction[1]
        theta_cut = jammer_direction[0]
        main_title = "Beamforming Comparison Around Jammer Direction"

    theta_vals = np.linspace(0.0, 180.0, 721)
    phi_vals = np.linspace(0.0, 360.0, 721)

    elev_curves = {}
    az_curves_db = {}

    # ------------------------------------------------------------
    # Compute responses
    # ------------------------------------------------------------
    for label, weights in weight_dict.items():
        w_2d = weights.reshape(array.N, array.M)

        af_elev = np.array([
            compute_array_factor(
                weights=w_2d,
                element_positions=array.element_positions,
                wavenumber_k=array.k_num,
                direction=(theta, phi_cut),
            )
            for theta in theta_vals
        ])
        af_elev_db = 20 * np.log10(np.maximum(af_elev / np.max(af_elev), 1e-12))
        elev_curves[label] = af_elev_db

        af_az = np.array([
            compute_array_factor(
                weights=w_2d,
                element_positions=array.element_positions,
                wavenumber_k=array.k_num,
                direction=(theta_cut, phi),
            )
            for phi in phi_vals
        ])
        af_az_db = 20 * np.log10(np.maximum(af_az / np.max(af_az), 1e-12))
        az_curves_db[label] = np.maximum(af_az_db, floor_db)

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "polar"}]],
        vertical_spacing=0.18
    )

    # ---------------- Elevation ----------------
    for label, yvals in elev_curves.items():
        fig.add_trace(
            go.Scatter(
                x=theta_vals,
                y=yvals,
                mode="lines",
                name=f"{label} (elev)",
                showlegend=show_legend,
                line=dict(
                    width=2,
                    color=color_map.get(label, None),
                ),
                hovertemplate="Theta: %{x:.1f}°<br>AF: %{y:.2f} dB<extra></extra>",
            ),
            row=1, col=1
        )

    # DOAs in elevation plot
    fig.add_trace(
        go.Scatter(
            x=[target_direction[0], target_direction[0]],
            y=[floor_db, 5],
            mode="lines",
            name="Target (theta)",
            showlegend=show_legend,
            line=dict(color=color_target, dash="dash"),
            hovertemplate="Target theta: %{x:.1f}°<extra></extra>",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[jammer_direction[0], jammer_direction[0]],
            y=[floor_db, 5],
            mode="lines",
            name="Jammer (theta)",
            showlegend=show_legend,
            line=dict(color=color_jammer, dash="dash"),
            hovertemplate="Jammer theta: %{x:.1f}°<extra></extra>",
        ),
        row=1, col=1
    )

    # ---------------- Azimuth (polar) ----------------
    for label, yvals_db in az_curves_db.items():
        fig.add_trace(
            go.Scatterpolar(
                r=yvals_db - floor_db,
                theta=phi_vals,
                mode="lines",
                name=f"{label} (az)",
                showlegend=show_legend,
                line=dict(
                    width=2,
                    color=color_map.get(label, None),
                ),
                customdata=yvals_db,
                hovertemplate="Phi: %{theta:.1f}°<br>AF: %{customdata:.2f} dB<extra></extra>",
            ),
            row=2, col=1
        )

    # DOAs in azimuth plot
    fig.add_trace(
        go.Scatterpolar(
            r=[0, abs(floor_db)],
            theta=[target_direction[1], target_direction[1]],
            mode="lines",
            name="Target (phi)",
            showlegend=show_legend,
            line=dict(color=color_target, dash="dash"),
            hovertemplate="Target phi: %{theta:.1f}°<extra></extra>",
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[0, abs(floor_db)],
            theta=[jammer_direction[1], jammer_direction[1]],
            mode="lines",
            name="Jammer (phi)",
            showlegend=show_legend,
            line=dict(color=color_jammer, dash="dash"),
            hovertemplate="Jammer phi: %{theta:.1f}°<extra></extra>",
        ),
        row=2, col=1
    )

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------
    fig.update_xaxes(title="Theta (deg)", range=[0, 180], row=1, col=1)
    fig.update_yaxes(title="Normalized AF (dB)", range=[floor_db, 5], row=1, col=1)

    fig.update_polars(
        radialaxis=dict(
            range=[0, abs(floor_db)],
            tickvals=[0, 15, 30, 45, 60],
            ticktext=["-60", "-45", "-30", "-15", "0 dB"],
        ),
        angularaxis=dict(rotation=0, direction="counterclockwise"),
        row=2, col=1
    )

    fig.update_layout(
        title=main_title,
        template="plotly_white",
        height=900,
        width=1100,
        showlegend=show_legend,
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )

    # Subplot titles
    fig.add_annotation(
        text=f"Elevation cut (phi = {phi_cut:.1f}°)",
        x=0.5, y=1.05, xref="paper", yref="paper",
        showarrow=False
    )

    fig.add_annotation(
        text=f"Azimuth cut (theta = {theta_cut:.1f}°)",
        x=0.5, y=0.48, xref="paper", yref="paper",
        showarrow=False
    )

    if show:
        fig.show(config={"scrollZoom": True})

    return fig