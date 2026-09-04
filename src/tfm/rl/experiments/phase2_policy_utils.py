# ============================================================
# Policy utilities for Phase 2 DRL beam steering agents
# ============================================================

from __future__ import annotations

import numpy as np

from tfm.math.narrow_band.geometry import (
    angles_to_unit_vector,
    unit_vector_to_angles,
)


def wrap_360(phi_deg: float) -> float:
    """
    Wrap an azimuth angle to [0, 360).

    Parameters
    ----------
    phi_deg : float
        Azimuth angle in degrees.

    Returns
    -------
    float
        Wrapped azimuth angle in degrees.
    """

    return float(phi_deg % 360.0)


def apply_output_angle_constraints(
    theta_deg: float,
    phi_deg: float,
) -> tuple[float, float]:
    """
    Apply the project angular convention.

    Convention:
        - theta in [0, 90]
        - phi in [0, 360)
        - if theta < 0, reflect theta and shift phi by 180 deg
        - if theta > 90, saturate theta at 90

    Parameters
    ----------
    theta_deg : float
        Polar angle in degrees.

    phi_deg : float
        Azimuth angle in degrees.

    Returns
    -------
    tuple[float, float]
        Constrained theta and phi in degrees.
    """

    theta = float(theta_deg)
    phi = float(phi_deg)

    if theta < 0.0:
        theta = -theta
        phi = phi + 180.0

    if theta > 90.0:
        theta = 90.0

    phi = wrap_360(phi)

    return theta, phi


def build_phase2_state_from_target(
    target_direction_deg: tuple[float, float],
    observation_mode: str,
    max_jammers: int = 3,
) -> np.ndarray:
    """
    Build a Phase 2 observation vector from a target direction.

    Phase 2 assumes no active jammers, but keeps fixed-size jammer slots
    for compatibility with later phases.

    Supported observation modes:
        - "angles"
        - "unit_vector"

    For observation_mode="angles", the state is:
        [
            theta_target_norm,
            phi_target_norm,
            theta_j1_norm, phi_j1_norm, m1,
            theta_j2_norm, phi_j2_norm, m2,
            theta_j3_norm, phi_j3_norm, m3
        ]

    For observation_mode="unit_vector", the state is:
        [
            u_target_x, u_target_y, u_target_z,
            u_j1_x, u_j1_y, u_j1_z, m1,
            u_j2_x, u_j2_y, u_j2_z, m2,
            u_j3_x, u_j3_y, u_j3_z, m3
        ]

    Parameters
    ----------
    target_direction_deg : tuple[float, float]
        Target direction as (theta_deg, phi_deg).

    observation_mode : str
        Observation representation.

    max_jammers : int
        Number of jammer slots.

    Returns
    -------
    np.ndarray
        Observation vector.
    """

    theta_deg, phi_deg = apply_output_angle_constraints(
        target_direction_deg[0],
        target_direction_deg[1],
    )

    if observation_mode == "angles":
        theta_norm = theta_deg / 90.0
        phi_norm = phi_deg / 360.0

        state = [theta_norm, phi_norm]

        for _ in range(max_jammers):
            state.extend([0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    if observation_mode == "unit_vector":
        u_target = angles_to_unit_vector(
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            enforce_visible=True,
        )

        state = list(u_target)

        for _ in range(max_jammers):
            state.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    raise ValueError(f"Unknown observation_mode: {observation_mode}")


def agent_action_to_direction_deg(
    action_agent: np.ndarray,
    action_mode: str,
) -> tuple[float, float]:
    """
    Convert an agent action into steering angles.

    Supported action modes:
        - "angles":
            action = [theta_norm, phi_norm]

        - "unit_vector":
            action = [u_x, u_y, u_z]

    Parameters
    ----------
    action_agent : np.ndarray
        Raw action returned by the DRL policy.

    action_mode : str
        Action representation.

    Returns
    -------
    tuple[float, float]
        Steering direction as (theta_deg, phi_deg).
    """

    action = np.asarray(action_agent, dtype=np.float32).reshape(-1)

    if action_mode == "angles":
        if action.shape[0] != 2:
            raise ValueError(
                f"Action mode 'angles' expects dimension 2, but got shape {action.shape}."
            )

        theta_norm = float(np.clip(action[0], 0.0, 1.0))
        phi_norm = float(np.clip(action[1], 0.0, 1.0))

        theta_deg = theta_norm * 90.0
        phi_deg = phi_norm * 360.0

        return apply_output_angle_constraints(theta_deg, phi_deg)

    if action_mode == "unit_vector":
        if action.shape[0] != 3:
            raise ValueError(
                f"Action mode 'unit_vector' expects dimension 3, but got shape {action.shape}."
            )

        norm = np.linalg.norm(action)

        if norm < 1e-8:
            u = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            u = action / norm

        # Enforce visible hemisphere: theta in [0, 90]
        if u[2] < 0.0:
            u[2] = abs(u[2])
            u = u / (np.linalg.norm(u) + 1e-8)

        theta_deg, phi_deg = unit_vector_to_angles(
            u,
            enforce_visible=True,
        )

        return apply_output_angle_constraints(theta_deg, phi_deg)

    raise ValueError(f"Unknown action_mode: {action_mode}")


def angular_error_theta(theta_true: float, theta_pred: float) -> float:
    """
    Compute absolute theta error in degrees.
    """

    return float(abs(theta_true - theta_pred))


def angular_error_phi(phi_true: float, phi_pred: float) -> float:
    """
    Compute wrapped phi error in degrees.
    """

    diff = abs(float(phi_true) - float(phi_pred)) % 360.0
    return float(min(diff, 360.0 - diff))