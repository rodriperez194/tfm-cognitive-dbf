from __future__ import annotations

import numpy as np


# ============================================================
# Angular convention
# ============================================================
#
# theta: polar angle measured from +z axis, in degrees.
#        Visible hemisphere: theta in [0, 90].
#
# phi: azimuth angle measured in the xy-plane from +x toward +y,
#      in degrees.
#      Normalized interval: phi in [0, 360).
#
# Unit vector:
#   u = [ux, uy, uz]
#
#   ux = sin(theta) cos(phi)
#   uy = sin(theta) sin(phi)
#   uz = cos(theta)
#
# ============================================================


def normalize_phi(phi_deg: float) -> float:
    """
    Normalize azimuth angle to the interval [0, 360).

    Parameters
    ----------
    phi_deg : float
        Azimuth angle in degrees.

    Returns
    -------
    float
        Normalized azimuth angle in degrees.
    """
    return float(phi_deg % 360.0)


def enforce_visible_theta(theta_deg: float) -> float:
    """
    Enforce the visible-hemisphere constraint for theta.

    The visible hemisphere is defined as theta in [0, 90] degrees.

    Parameters
    ----------
    theta_deg : float
        Polar angle in degrees.

    Returns
    -------
    float
        Clipped polar angle in degrees.
    """
    return float(np.clip(theta_deg, 0.0, 90.0))


def normalize_angles(
    theta_deg: float,
    phi_deg: float,
    enforce_visible: bool = True,
) -> tuple[float, float]:
    """
    Normalize an angular pair according to the project convention.

    Parameters
    ----------
    theta_deg : float
        Polar angle in degrees, measured from +z.

    phi_deg : float
        Azimuth angle in degrees, measured from +x toward +y.

    enforce_visible : bool, optional
        If True, theta is clipped to the visible hemisphere [0, 90].
        If False, theta is left unchanged.

    Returns
    -------
    tuple[float, float]
        Normalized pair (theta_deg, phi_deg).
    """
    theta_out = enforce_visible_theta(theta_deg) if enforce_visible else float(theta_deg)
    phi_out = normalize_phi(phi_deg)

    return theta_out, phi_out


def angles_to_unit_vector(
    theta_deg: float,
    phi_deg: float,
    enforce_visible: bool = True,
    normalize_output: bool = True,
) -> np.ndarray:
    """
    Convert an angular direction into a 3D unit vector.

    Parameters
    ----------
    theta_deg : float
        Polar angle in degrees, measured from +z.

    phi_deg : float
        Azimuth angle in degrees, measured from +x toward +y.

    enforce_visible : bool, optional
        If True, theta is clipped to [0, 90] before conversion.

    normalize_output : bool, optional
        If True, the resulting vector is explicitly normalized.

    Returns
    -------
    np.ndarray
        Unit vector [ux, uy, uz] with shape (3,).
    """
    theta_deg, phi_deg = normalize_angles(
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        enforce_visible=enforce_visible,
    )

    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)

    ux = np.sin(theta_rad) * np.cos(phi_rad)
    uy = np.sin(theta_rad) * np.sin(phi_rad)
    uz = np.cos(theta_rad)

    u = np.array([ux, uy, uz], dtype=np.float64)

    if normalize_output:
        norm = np.linalg.norm(u)

        if norm == 0.0:
            raise ValueError("Cannot normalize a zero vector.")

        u = u / norm

    return u


def unit_vector_to_angles(
    u: np.ndarray,
    enforce_visible: bool = True,
) -> tuple[float, float]:
    """
    Convert a 3D unit vector into angular coordinates.

    Parameters
    ----------
    u : np.ndarray
        Direction vector with shape (3,).

    enforce_visible : bool, optional
        If True, the resulting theta is clipped to [0, 90].

    Returns
    -------
    tuple[float, float]
        Angular pair (theta_deg, phi_deg).
    """
    u = np.asarray(u, dtype=np.float64)

    if u.shape != (3,):
        raise ValueError(f"Expected vector with shape (3,), got {u.shape}.")

    norm = np.linalg.norm(u)

    if norm == 0.0:
        raise ValueError("Cannot convert a zero vector to angles.")

    u = u / norm

    ux, uy, uz = u

    uz = np.clip(uz, -1.0, 1.0)

    theta_rad = np.arccos(uz)
    phi_rad = np.arctan2(uy, ux)

    theta_deg = float(np.rad2deg(theta_rad))
    phi_deg = float(np.rad2deg(phi_rad))

    theta_deg, phi_deg = normalize_angles(
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        enforce_visible=enforce_visible,
    )

    return theta_deg, phi_deg


def angular_distance_deg(
    u1: np.ndarray,
    u2: np.ndarray,
) -> float:
    """
    Compute the angular distance between two direction vectors.

    Parameters
    ----------
    u1 : np.ndarray
        First direction vector with shape (3,).

    u2 : np.ndarray
        Second direction vector with shape (3,).

    Returns
    -------
    float
        Angular distance in degrees.
    """
    u1 = np.asarray(u1, dtype=np.float64)
    u2 = np.asarray(u2, dtype=np.float64)

    if u1.shape != (3,):
        raise ValueError(f"Expected u1 with shape (3,), got {u1.shape}.")

    if u2.shape != (3,):
        raise ValueError(f"Expected u2 with shape (3,), got {u2.shape}.")

    norm_1 = np.linalg.norm(u1)
    norm_2 = np.linalg.norm(u2)

    if norm_1 == 0.0 or norm_2 == 0.0:
        raise ValueError("Cannot compute angular distance with a zero vector.")

    u1 = u1 / norm_1
    u2 = u2 / norm_2

    dot = float(np.dot(u1, u2))
    dot = np.clip(dot, -1.0, 1.0)

    return float(np.rad2deg(np.arccos(dot)))


def angles_to_unit_vectors(
    angles: np.ndarray,
    enforce_visible: bool = True,
) -> np.ndarray:
    """
    Convert multiple angular pairs into unit vectors.

    Parameters
    ----------
    angles : np.ndarray
        Array with shape (N, 2), where each row is [theta_deg, phi_deg].

    enforce_visible : bool, optional
        If True, theta is clipped to [0, 90] for all directions.

    Returns
    -------
    np.ndarray
        Array of unit vectors with shape (N, 3).
    """
    angles = np.asarray(angles, dtype=np.float64)

    if angles.ndim != 2 or angles.shape[1] != 2:
        raise ValueError(
            f"Expected angles with shape (N, 2), got {angles.shape}."
        )

    unit_vectors = [
        angles_to_unit_vector(
            theta_deg=theta,
            phi_deg=phi,
            enforce_visible=enforce_visible,
        )
        for theta, phi in angles
    ]

    return np.asarray(unit_vectors, dtype=np.float64)


def unit_vectors_to_angles(
    unit_vectors: np.ndarray,
    enforce_visible: bool = True,
) -> np.ndarray:
    """
    Convert multiple unit vectors into angular pairs.

    Parameters
    ----------
    unit_vectors : np.ndarray
        Array with shape (N, 3).

    enforce_visible : bool, optional
        If True, theta is clipped to [0, 90] for all directions.

    Returns
    -------
    np.ndarray
        Array with shape (N, 2), where each row is [theta_deg, phi_deg].
    """
    unit_vectors = np.asarray(unit_vectors, dtype=np.float64)

    if unit_vectors.ndim != 2 or unit_vectors.shape[1] != 3:
        raise ValueError(
            f"Expected unit_vectors with shape (N, 3), got {unit_vectors.shape}."
        )

    angles = [
        unit_vector_to_angles(
            u=u,
            enforce_visible=enforce_visible,
        )
        for u in unit_vectors
    ]

    return np.asarray(angles, dtype=np.float64)