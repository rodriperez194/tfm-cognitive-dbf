import numpy as np


def absolute_to_relative_position(
    position_hist: np.ndarray,
    antenna_position: np.ndarray,
) -> np.ndarray:
    """
    Convert an absolute Cartesian trajectory into a Cartesian trajectory
    relative to a fixed antenna.

    Parameters
    ----------
    position_hist : np.ndarray
        Absolute target position history with shape (N, 3).
    antenna_position : np.ndarray
        Fixed antenna position [xa, ya, za] with shape (3,).

    Returns
    -------
    np.ndarray
        Relative Cartesian trajectory with shape (N, 3).
    """
    position_hist = np.asarray(position_hist, dtype=float)
    antenna_position = np.asarray(antenna_position, dtype=float).reshape(3,)

    if position_hist.ndim != 2 or position_hist.shape[1] != 3:
        raise ValueError("position_hist must have shape (N, 3).")

    return position_hist - antenna_position[None, :]


def relative_position_to_spherical(
    relative_position_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a relative Cartesian trajectory into spherical coordinates.

    Angular convention:
        theta : polar angle from +z [deg]
        phi   : azimuth in xy-plane from +x towards +y [deg]

    Parameters
    ----------
    relative_position_hist : np.ndarray
        Relative Cartesian trajectory with shape (N, 3).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        r_hist     : range history, shape (N,)
        theta_hist : polar angle history [deg], shape (N,)
        phi_hist   : azimuth history [deg], shape (N,)
    """
    relative_position_hist = np.asarray(relative_position_hist, dtype=float)

    if (
        relative_position_hist.ndim != 2
        or relative_position_hist.shape[1] != 3
    ):
        raise ValueError("relative_position_hist must have shape (N, 3).")

    x = relative_position_hist[:, 0]
    y = relative_position_hist[:, 1]
    z = relative_position_hist[:, 2]

    r_hist = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r_hist < 1e-12, 1e-12, r_hist)

    theta_hist = np.rad2deg(
        np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    )

    phi_hist = np.rad2deg(np.arctan2(y, x))
    phi_hist = np.mod(phi_hist, 360.0)

    return r_hist, theta_hist, phi_hist


def relative_position_to_unit_vector(
    relative_position_hist: np.ndarray,
) -> np.ndarray:
    """
    Convert a relative Cartesian trajectory into a unit-vector trajectory.

    Parameters
    ----------
    relative_position_hist : np.ndarray
        Relative Cartesian trajectory with shape (N, 3).

    Returns
    -------
    np.ndarray
        Unit-vector history with shape (N, 3).
    """
    relative_position_hist = np.asarray(relative_position_hist, dtype=float)

    if (
        relative_position_hist.ndim != 2
        or relative_position_hist.shape[1] != 3
    ):
        raise ValueError("relative_position_hist must have shape (N, 3).")

    norms = np.linalg.norm(relative_position_hist, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-12, 1e-12, norms)

    return relative_position_hist / norms_safe


def unit_vector_derivatives(
    u_hist: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute unit-vector velocity and acceleration using finite differences.

    Parameters
    ----------
    u_hist : np.ndarray
        Unit-vector history with shape (N, 3).
    dt : float
        Time step [s].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        du_hist   : first derivative of unit vector, shape (N, 3)
        ddu_hist  : second derivative of unit vector, shape (N, 3)
    """
    u_hist = np.asarray(u_hist, dtype=float)

    if u_hist.ndim != 2 or u_hist.shape[1] != 3:
        raise ValueError("u_hist must have shape (N, 3).")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if u_hist.shape[0] < 2:
        raise ValueError("At least two samples are required.")

    num_steps = u_hist.shape[0]

    du_hist = np.zeros_like(u_hist, dtype=float)
    ddu_hist = np.zeros_like(u_hist, dtype=float)

    # ============================================================
    # 1. FIRST DERIVATIVE
    # ============================================================

    du_hist[0] = (u_hist[1] - u_hist[0]) / dt
    du_hist[-1] = (u_hist[-1] - u_hist[-2]) / dt

    if num_steps > 2:
        du_hist[1:-1] = (u_hist[2:] - u_hist[:-2]) / (2.0 * dt)

    # ============================================================
    # 2. SECOND DERIVATIVE
    # ============================================================

    ddu_hist[0] = (du_hist[1] - du_hist[0]) / dt
    ddu_hist[-1] = (du_hist[-1] - du_hist[-2]) / dt

    if num_steps > 2:
        ddu_hist[1:-1] = (du_hist[2:] - du_hist[:-2]) / (2.0 * dt)

    return du_hist, ddu_hist