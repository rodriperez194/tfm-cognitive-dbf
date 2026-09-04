import numpy as np


def wrap_360(angle_deg: float) -> float:
    """
    Wrap angle to [0, 360).
    """
    return angle_deg % 360.0


def wrap_180(angle_deg: float) -> float:
    """
    Wrap angular difference to [-180, 180).
    Useful for computing minimal angular errors.
    """
    return ((angle_deg + 180.0) % 360.0) - 180.0


def normalize_direction(theta_deg: float, phi_deg: float):
    """
    Normalize direction according to the array convention:

    theta ∈ [0,90]
    phi   ∈ [0,360)

    Used only for simulation / ground truth handling.
    """

    theta = theta_deg
    phi = phi_deg

    if theta < 0:
        theta = -theta
        phi += 180

    if theta > 90:
        theta = 90

    phi = wrap_360(phi)

    return theta, phi


# ============================================================
# UNIT VECTOR GEOMETRY
# ============================================================

def angles_to_unit_vector(theta_deg, phi_deg):
    """
    Convert angular coordinates (theta, phi) in degrees into a unit vector.

    Angular convention:
        theta : polar angle from +z
        phi   : azimuth in xy-plane from +x towards +y

    Parameters
    ----------
    theta_deg : float or np.ndarray
        Polar angle(s) in degrees.
    phi_deg : float or np.ndarray
        Azimuth angle(s) in degrees.

    Returns
    -------
    np.ndarray
        If inputs are scalars, returns shape (3,).
        If inputs are 1D arrays of shape (N,), returns shape (N, 3).
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    phi_deg = np.asarray(phi_deg, dtype=float)

    if theta_deg.shape != phi_deg.shape:
        raise ValueError("theta_deg and phi_deg must have the same shape.")

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    ux = np.sin(theta) * np.cos(phi)
    uy = np.sin(theta) * np.sin(phi)
    uz = np.cos(theta)

    u = np.stack((ux, uy, uz), axis=-1)

    return u


def unit_vector_to_angles(u):
    """
    Convert a 3D vector into angular coordinates (theta, phi) in degrees.

    The input vector is normalized internally for numerical safety.

    Angular convention:
        theta : polar angle from +z
        phi   : azimuth in xy-plane from +x towards +y

    Returns:
        tuple: (theta_deg, phi_deg)
    """
    u = np.asarray(u, dtype=float).reshape(3,)
    norm_u = np.linalg.norm(u)

    if norm_u < 1e-12:
        return 0.0, 0.0

    u = u / norm_u

    ux, uy, uz = u
    uz = np.clip(uz, -1.0, 1.0)

    theta_deg = float(np.rad2deg(np.arccos(uz)))
    phi_deg = float(np.rad2deg(np.arctan2(uy, ux)))
    phi_deg = wrap_360(phi_deg)

    return theta_deg, phi_deg


# ============================================================
# ANGULAR VELOCITY -> UNIT VECTOR VELOCITY
# ============================================================

def angular_rates_to_unit_vector_velocity(
    theta_deg: float,
    phi_deg: float,
    theta_dot_deg_s: float,
    phi_dot_deg_s: float,
):
    """
    Convert angular rates (theta_dot, phi_dot) into the velocity of the
    corresponding unit vector.

    Inputs:
        theta_deg        : polar angle [deg]
        phi_deg          : azimuth [deg]
        theta_dot_deg_s  : d(theta)/dt [deg/s]
        phi_dot_deg_s    : d(phi)/dt [deg/s]

    Returns:
        np.ndarray: [dux, duy, duz]
    """
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    theta_dot = np.deg2rad(theta_dot_deg_s)
    phi_dot = np.deg2rad(phi_dot_deg_s)

    dux = (
        np.cos(theta) * np.cos(phi) * theta_dot
        - np.sin(theta) * np.sin(phi) * phi_dot
    )

    duy = (
        np.cos(theta) * np.sin(phi) * theta_dot
        + np.sin(theta) * np.cos(phi) * phi_dot
    )

    duz = -np.sin(theta) * theta_dot

    return np.array([dux, duy, duz], dtype=float)