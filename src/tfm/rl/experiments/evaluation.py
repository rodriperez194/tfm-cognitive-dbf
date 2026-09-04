# ============================================================
# Evaluation utilities for Phase 2 DRL beam steering agents
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from tfm.math.narrow_band.geometry import (
    angles_to_unit_vector,
    angular_distance_deg,
)

from tfm.rl.experiments.phase2_common import build_env


def evaluate_agent(
    model,
    observation_mode: str,
    action_mode: str,
    reward_mode: str,
    reward_config: dict,
    num_eval_samples: int = 5000,
    seed: int = 42,
    array_position: np.ndarray | None = None,
    desired_power: float = 1.0,
    noise_power: float = 1e-3,
    max_jammers: int = 3,
    target_range_m: float = 1000.0,
    num_rows: int = 6,
    num_cols: int = 6,
    carrier_freq: float = 10e9,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate one trained agent against the optimal steering action.

    The optimal action is generated in the same action representation
    expected by the environment:

    - action_mode="angles":
        [theta_norm, phi_norm]

    - action_mode="unit_vector":
        [u_x, u_y, u_z]

    Parameters
    ----------
    model
        Trained Stable-Baselines3 model.

    observation_mode : str
        Observation representation.

    action_mode : str
        Action representation.

    reward_mode : str
        Reward configuration name. Kept for traceability.

    reward_config : dict
        Reward coefficient dictionary used to build the evaluation environment.

    num_eval_samples : int
        Number of Monte Carlo evaluation samples.

    seed : int
        Base random seed.

    array_position : np.ndarray, optional
        Array position in Cartesian coordinates.

    desired_power : float
        Desired signal power.

    noise_power : float
        Thermal noise power.

    max_jammers : int
        Maximum number of jammer slots in the state.

    target_range_m : float
        Target range in meters.

    num_rows : int
        Number of array rows.

    num_cols : int
        Number of array columns.

    carrier_freq : float
        Carrier frequency in Hz.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Evaluation samples dataframe and summary metrics dictionary.
    """

    if array_position is None:
        array_position = np.array([0.0, 0.0, 0.0], dtype=float)

    env_agent = build_env(
        observation_mode=observation_mode,
        action_mode=action_mode,
        reward_config=reward_config,
        array_position=array_position,
        desired_power=desired_power,
        noise_power=noise_power,
        max_jammers=max_jammers,
        target_range_m=target_range_m,
        num_rows=num_rows,
        num_cols=num_cols,
        carrier_freq=carrier_freq,
        monitor=False,
    )

    env_opt = build_env(
        observation_mode=observation_mode,
        action_mode=action_mode,
        reward_config=reward_config,
        array_position=array_position,
        desired_power=desired_power,
        noise_power=noise_power,
        max_jammers=max_jammers,
        target_range_m=target_range_m,
        num_rows=num_rows,
        num_cols=num_cols,
        carrier_freq=carrier_freq,
        monitor=False,
    )

    records = []

    for idx in range(num_eval_samples):
        obs, info = env_agent.reset(seed=seed + idx)

        action_agent, _ = model.predict(obs, deterministic=True)
        action_agent = np.asarray(action_agent, dtype=np.float32)

        _, reward_agent, _, _, info_agent = env_agent.step(action_agent)

        theta_target_deg = float(info_agent["theta_target_deg"])
        phi_target_deg = float(info_agent["phi_target_deg"])

        theta_agent_deg = float(info_agent["theta_steer_deg"])
        phi_agent_deg = float(info_agent["phi_steer_deg"])

        # --------------------------------------------------------
        # Force env_opt to use the exact same target direction
        # --------------------------------------------------------
        env_opt.current_theta_rad = info_agent["theta_target_rad"]
        env_opt.current_phi_rad = info_agent["phi_target_rad"]
        env_opt.current_state = env_opt._build_state(
            env_opt.current_theta_rad,
            env_opt.current_phi_rad,
        )

        # --------------------------------------------------------
        # Optimal action in the correct representation
        # --------------------------------------------------------
        if action_mode == "angles":
            optimal_action = np.array(
                [
                    env_opt._normalize_theta(env_opt.current_theta_rad),
                    env_opt._normalize_phi(env_opt.current_phi_rad),
                ],
                dtype=np.float32,
            )

        elif action_mode == "unit_vector":
            optimal_action = angles_to_unit_vector(
                theta_deg=theta_target_deg,
                phi_deg=phi_target_deg,
                enforce_visible=True,
            ).astype(np.float32)

        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        _, reward_opt, _, _, info_opt = env_opt.step(optimal_action)

        # --------------------------------------------------------
        # Error metrics
        # --------------------------------------------------------
        theta_error = abs(theta_agent_deg - theta_target_deg)

        phi_error_raw = abs(phi_agent_deg - phi_target_deg)
        phi_error = min(phi_error_raw, 360.0 - phi_error_raw)

        total_angular_error = np.sqrt(theta_error**2 + phi_error**2)

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

        sinr_agent = float(info_agent["sinr_db"])
        sinr_opt = float(info_opt["sinr_db"])
        sinr_loss = sinr_opt - sinr_agent

        records.append(
            {
                "sample_id": idx,
                "reward_mode": reward_mode,
                "theta_target_deg": theta_target_deg,
                "phi_target_deg": phi_target_deg,
                "theta_agent_deg": theta_agent_deg,
                "phi_agent_deg": phi_agent_deg,
                "theta_error_deg": float(theta_error),
                "phi_error_deg": float(phi_error),
                "total_angular_error_deg": float(total_angular_error),
                "angular_error_3d_deg": float(angular_error_3d),
                "reward_agent": float(reward_agent),
                "reward_opt": float(reward_opt),
                "sinr_agent_db": sinr_agent,
                "sinr_opt_db": sinr_opt,
                "sinr_loss_db": float(sinr_loss),
                "angle_loss": float(info_agent["angle_loss"]),
            }
        )

    eval_df = pd.DataFrame(records)

    summary = summarize_evaluation(eval_df)

    try:
        env_agent.close()
    except Exception:
        pass

    try:
        env_opt.close()
    except Exception:
        pass

    return eval_df, summary


def summarize_evaluation(eval_df: pd.DataFrame) -> dict:
    """
    Compute summary metrics from an evaluation dataframe.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Evaluation samples dataframe.

    Returns
    -------
    dict
        Summary metrics.
    """

    required_columns = [
        "theta_error_deg",
        "phi_error_deg",
        "total_angular_error_deg",
        "angular_error_3d_deg",
        "sinr_agent_db",
        "sinr_loss_db",
        "reward_agent",
        "reward_opt",
    ]

    for col in required_columns:
        if col not in eval_df.columns:
            raise KeyError(f"Missing evaluation column: {col}")

    summary = {
        "theta_error_mean": eval_df["theta_error_deg"].mean(),
        "theta_error_std": eval_df["theta_error_deg"].std(),
        "phi_error_mean": eval_df["phi_error_deg"].mean(),
        "phi_error_std": eval_df["phi_error_deg"].std(),

        "total_ang_error_mean": eval_df["total_angular_error_deg"].mean(),
        "total_ang_error_std": eval_df["total_angular_error_deg"].std(),
        "total_ang_error_p95": eval_df["total_angular_error_deg"].quantile(0.95),
        "total_ang_error_max": eval_df["total_angular_error_deg"].max(),

        "angular_error_3d_mean": eval_df["angular_error_3d_deg"].mean(),
        "angular_error_3d_std": eval_df["angular_error_3d_deg"].std(),
        "angular_error_3d_p95": eval_df["angular_error_3d_deg"].quantile(0.95),
        "angular_error_3d_max": eval_df["angular_error_3d_deg"].max(),

        "sinr_agent_mean": eval_df["sinr_agent_db"].mean(),
        "sinr_agent_std": eval_df["sinr_agent_db"].std(),

        "sinr_loss_mean": eval_df["sinr_loss_db"].mean(),
        "sinr_loss_std": eval_df["sinr_loss_db"].std(),
        "sinr_loss_p95": eval_df["sinr_loss_db"].quantile(0.95),
        "sinr_loss_max": eval_df["sinr_loss_db"].max(),

        "reward_agent_mean": eval_df["reward_agent"].mean(),
        "reward_opt_mean": eval_df["reward_opt"].mean(),

        "pct_sinr_loss_lt_0.1dB": 100.0 * (eval_df["sinr_loss_db"] < 0.1).mean(),
        "pct_sinr_loss_lt_0.5dB": 100.0 * (eval_df["sinr_loss_db"] < 0.5).mean(),
        "pct_sinr_loss_lt_1dB": 100.0 * (eval_df["sinr_loss_db"] < 1.0).mean(),
    }

    return {key: float(value) for key, value in summary.items()}


def rank_agents(
    comparison_df: pd.DataFrame,
    status_column: str = "status",
) -> pd.DataFrame:
    """
    Rank completed agents using the Phase 2 default criterion.

    Ranking criterion:
        1. Lower mean SINR loss.
        2. Lower P95 SINR loss.
        3. Lower mean 3D angular error.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Global comparison dataframe.

    status_column : str
        Name of the status column.

    Returns
    -------
    pd.DataFrame
        Ranked dataframe.
    """

    if status_column in comparison_df.columns:
        ranked_df = comparison_df[comparison_df[status_column] == "completed"].copy()
    else:
        ranked_df = comparison_df.copy()

    required_columns = [
        "sinr_loss_mean",
        "sinr_loss_p95",
        "angular_error_3d_mean",
    ]

    for col in required_columns:
        if col not in ranked_df.columns:
            raise KeyError(f"Missing ranking column: {col}")

    ranked_df = ranked_df.sort_values(
        by=[
            "sinr_loss_mean",
            "sinr_loss_p95",
            "angular_error_3d_mean",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )

    return ranked_df