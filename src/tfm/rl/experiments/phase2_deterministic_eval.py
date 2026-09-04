# ============================================================
# Deterministic evaluation utilities for Phase 2 beam steering
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from tfm.physics.narrow_band.phased_array_nb import Phased_Array_NB
from tfm.physics.narrow_band.weights_deterministic_nb import steering_weights

from tfm.math.narrow_band.metrics import (
    compute_sinr,
    compute_directivity,
)

from tfm.math.narrow_band.geometry import (
    angles_to_unit_vector,
    angular_distance_deg,
)

from tfm.rl.experiments.phase2_policy_utils import (
    apply_output_angle_constraints,
    build_phase2_state_from_target,
    agent_action_to_direction_deg,
    angular_error_theta,
    angular_error_phi,
)


def evaluate_phase2_steering_agent(
    model,
    array: Phased_Array_NB,
    observation_mode: str,
    action_mode: str,
    num_samples: int = 5000,
    seed: int = 42,
    target_power: float = 1.0,
    noise_power: float = 1e-3,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 90.0,
    phi_min_deg: float = 0.0,
    phi_max_deg: float = 360.0,
    max_jammers: int = 3,
    directivity_theta_res: float = 2.0,
    directivity_phi_res: float = 2.0,
    compute_directivity_metrics: bool = True,
) -> pd.DataFrame:
    """
    Evaluate a Phase 2 beam steering agent against classical steering.

    This evaluation does not rely on BeamformingEnv. It directly:
        1. Samples random target directions.
        2. Builds the corresponding policy observation.
        3. Queries the trained agent.
        4. Converts the agent action into a steering direction.
        5. Builds deterministic steering weights for both:
            - classical target steering
            - agent-predicted steering
        6. Computes SINR and, optionally, directivity.

    Parameters
    ----------
    model
        Trained Stable-Baselines3 model.

    array : Phased_Array_NB
        Phased array instance.

    observation_mode : str
        Observation representation used by the agent.

    action_mode : str
        Action representation used by the agent.

    num_samples : int
        Number of Monte Carlo samples.

    seed : int
        Random seed.

    target_power : float
        Desired signal power.

    noise_power : float
        Noise power.

    theta_min_deg : float
        Minimum sampled theta.

    theta_max_deg : float
        Maximum sampled theta.

    phi_min_deg : float
        Minimum sampled phi.

    phi_max_deg : float
        Maximum sampled phi.

    max_jammers : int
        Number of jammer slots in the state.

    directivity_theta_res : float
        Theta resolution for directivity computation.

    directivity_phi_res : float
        Phi resolution for directivity computation.

    compute_directivity_metrics : bool
        If True, compute directivity metrics. This is more expensive.

    Returns
    -------
    pd.DataFrame
        Per-sample evaluation results.
    """

    rng = np.random.default_rng(seed)
    records = []

    for idx in range(num_samples):
        # --------------------------------------------------------
        # 1. Sample random target
        # --------------------------------------------------------
        theta_raw = rng.uniform(theta_min_deg, theta_max_deg)
        phi_raw = rng.uniform(phi_min_deg, phi_max_deg)

        theta_target_deg, phi_target_deg = apply_output_angle_constraints(
            theta_raw,
            phi_raw,
        )

        target_direction = (theta_target_deg, phi_target_deg)

        # --------------------------------------------------------
        # 2. Build state and query agent
        # --------------------------------------------------------
        state = build_phase2_state_from_target(
            target_direction_deg=target_direction,
            observation_mode=observation_mode,
            max_jammers=max_jammers,
        )

        action_agent, _ = model.predict(state, deterministic=True)

        theta_agent_deg, phi_agent_deg = agent_action_to_direction_deg(
            action_agent=action_agent,
            action_mode=action_mode,
        )

        agent_direction = (theta_agent_deg, phi_agent_deg)

        # --------------------------------------------------------
        # 3. Build classical and agent steering weights
        # --------------------------------------------------------
        w_steer = steering_weights(
            element_positions=array.element_positions,
            wavenumber_k=array.k_num,
            direction=target_direction,
        )

        w_agent = steering_weights(
            element_positions=array.element_positions,
            wavenumber_k=array.k_num,
            direction=agent_direction,
        )

        # --------------------------------------------------------
        # 4. Evaluate classical steering
        # --------------------------------------------------------
        array.set_weights(w_steer)

        sinr_db_steer = compute_sinr(
            weights=array.W,
            element_positions=array.element_positions,
            wavenumber_k=array.k_num,
            target_direction=target_direction,
            target_power=target_power,
            jammers_directions=[],
            jammers_powers=[],
            noise_power=noise_power,
        )

        if compute_directivity_metrics:
            directivity_db_steer = compute_directivity(
                weights=array.W,
                element_positions=array.element_positions,
                wavenumber_k=array.k_num,
                target_direction=target_direction,
                theta_res=directivity_theta_res,
                phi_res=directivity_phi_res,
            )
        else:
            directivity_db_steer = np.nan

        # --------------------------------------------------------
        # 5. Evaluate agent steering
        # --------------------------------------------------------
        array.set_weights(w_agent)

        sinr_db_agent = compute_sinr(
            weights=array.W,
            element_positions=array.element_positions,
            wavenumber_k=array.k_num,
            target_direction=target_direction,
            target_power=target_power,
            jammers_directions=[],
            jammers_powers=[],
            noise_power=noise_power,
        )

        if compute_directivity_metrics:
            directivity_db_agent = compute_directivity(
                weights=array.W,
                element_positions=array.element_positions,
                wavenumber_k=array.k_num,
                target_direction=target_direction,
                theta_res=directivity_theta_res,
                phi_res=directivity_phi_res,
            )
        else:
            directivity_db_agent = np.nan

        # --------------------------------------------------------
        # 6. Error metrics
        # --------------------------------------------------------
        theta_err = angular_error_theta(theta_target_deg, theta_agent_deg)
        phi_err = angular_error_phi(phi_target_deg, phi_agent_deg)

        u_target = angles_to_unit_vector(
            theta_deg=theta_target_deg,
            phi_deg=phi_target_deg,
            enforce_visible=True,
        )

        u_agent = angles_to_unit_vector(
            theta_deg=theta_agent_deg,
            phi_deg=phi_agent_deg,
            enforce_visible=True,
        )

        angular_error_3d = angular_distance_deg(u_target, u_agent)

        sinr_loss_db = sinr_db_steer - sinr_db_agent
        directivity_loss_db = directivity_db_steer - directivity_db_agent

        records.append(
            {
                "sample_id": idx,
                "theta_target_deg": float(theta_target_deg),
                "phi_target_deg": float(phi_target_deg),
                "theta_agent_deg": float(theta_agent_deg),
                "phi_agent_deg": float(phi_agent_deg),
                "theta_error_deg": float(theta_err),
                "phi_error_deg": float(phi_err),
                "angular_error_3d_deg": float(angular_error_3d),
                "sinr_db_steer": float(sinr_db_steer),
                "sinr_db_agent": float(sinr_db_agent),
                "sinr_loss_db": float(sinr_loss_db),
                "directivity_dbi_steer": float(directivity_db_steer),
                "directivity_dbi_agent": float(directivity_db_agent),
                "directivity_loss_db": float(directivity_loss_db),
            }
        )

    return pd.DataFrame(records)


def summarize_phase2_steering_results(results_df: pd.DataFrame) -> dict:
    """
    Compute aggregate metrics for deterministic Phase 2 steering evaluation.

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-sample evaluation results.

    Returns
    -------
    dict
        Summary metrics.
    """

    required_columns = [
        "theta_error_deg",
        "phi_error_deg",
        "angular_error_3d_deg",
        "sinr_db_agent",
        "sinr_loss_db",
    ]

    for col in required_columns:
        if col not in results_df.columns:
            raise KeyError(f"Missing results column: {col}")

    summary = {}

    # -------------------------------
    # Angular errors
    # -------------------------------
    summary["theta_error_mean"] = results_df["theta_error_deg"].mean()
    summary["theta_error_std"] = results_df["theta_error_deg"].std()

    summary["phi_error_mean"] = results_df["phi_error_deg"].mean()
    summary["phi_error_std"] = results_df["phi_error_deg"].std()

    summary["angular_error_3d_mean"] = results_df["angular_error_3d_deg"].mean()
    summary["angular_error_3d_std"] = results_df["angular_error_3d_deg"].std()
    summary["angular_error_3d_p95"] = results_df["angular_error_3d_deg"].quantile(0.95)
    summary["angular_error_3d_max"] = results_df["angular_error_3d_deg"].max()

    # -------------------------------
    # SINR
    # -------------------------------
    summary["sinr_agent_mean"] = results_df["sinr_db_agent"].mean()
    summary["sinr_agent_std"] = results_df["sinr_db_agent"].std()

    summary["sinr_loss_mean"] = results_df["sinr_loss_db"].mean()
    summary["sinr_loss_std"] = results_df["sinr_loss_db"].std()
    summary["sinr_loss_p95"] = results_df["sinr_loss_db"].quantile(0.95)
    summary["sinr_loss_max"] = results_df["sinr_loss_db"].max()

    # -------------------------------
    # Directivity
    # -------------------------------
    if "directivity_loss_db" in results_df.columns:
        summary["directivity_loss_mean"] = results_df["directivity_loss_db"].mean()
        summary["directivity_loss_std"] = results_df["directivity_loss_db"].std()

    # -------------------------------
    # Success ratios
    # -------------------------------
    summary["pct_sinr_loss_lt_0.1dB"] = (
        np.mean(results_df["sinr_loss_db"] < 0.1) * 100.0
    )
    summary["pct_sinr_loss_lt_0.5dB"] = (
        np.mean(results_df["sinr_loss_db"] < 0.5) * 100.0
    )
    summary["pct_sinr_loss_lt_1dB"] = (
        np.mean(results_df["sinr_loss_db"] < 1.0) * 100.0
    )

    return {key: float(value) for key, value in summary.items()}


def get_worst_cases(
    results_df: pd.DataFrame,
    metric: str = "sinr_loss_db",
    n: int = 20,
) -> pd.DataFrame:
    """
    Return the worst samples according to a selected metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-sample evaluation dataframe.

    metric : str
        Metric used for sorting.

    n : int
        Number of rows to return.

    Returns
    -------
    pd.DataFrame
        Worst-case samples.
    """

    if metric not in results_df.columns:
        raise KeyError(f"Missing metric column: {metric}")

    return results_df.sort_values(metric, ascending=False).head(n).copy()