# ============================================================
# Common utilities for Phase 2 DRL beam steering experiments
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stable_baselines3 import TD3, SAC, PPO, DDPG, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from tfm.physics.narrow_band.phased_array_nb import Phased_Array_NB
from tfm.rl.envs.beamforming_env import BeamformingEnv


# ============================================================
# Physical setup helpers
# ============================================================

def build_array(
    num_rows: int = 6,
    num_cols: int = 6,
    carrier_freq: float = 10e9,
) -> Phased_Array_NB:
    """
    Create a fresh phased array instance.

    Parameters
    ----------
    num_rows : int
        Number of array rows.

    num_cols : int
        Number of array columns.

    carrier_freq : float
        Carrier frequency in Hz.

    Returns
    -------
    Phased_Array_NB
        Fresh phased array instance.
    """

    return Phased_Array_NB(
        num_rows=num_rows,
        num_cols=num_cols,
        carrier_freq=carrier_freq,
    )


def build_env(
    observation_mode: str,
    action_mode: str,
    reward_config: dict,
    array_position: np.ndarray | None = None,
    desired_power: float = 1.0,
    noise_power: float = 1e-3,
    max_jammers: int = 3,
    target_range_m: float = 1000.0,
    num_rows: int = 6,
    num_cols: int = 6,
    carrier_freq: float = 10e9,
    monitor: bool = True,
) -> BeamformingEnv:
    """
    Build BeamformingEnv with a selected state/action/reward configuration.

    Parameters
    ----------
    observation_mode : str
        Observation representation. Expected values: "angles" or "unit_vector".

    action_mode : str
        Action representation. Expected values: "angles" or "unit_vector".

    reward_config : dict
        Dictionary with effective BeamformingEnv reward coefficients:
        reward_alpha_sinr, reward_beta_sinr_loss and reward_gamma_angle.

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

    monitor : bool
        If True, wrap the environment with Stable-Baselines3 Monitor.

    Returns
    -------
    BeamformingEnv
        Configured environment. If monitor=True, returns a Monitor-wrapped env.
    """

    if array_position is None:
        array_position = np.array([0.0, 0.0, 0.0], dtype=float)

    required_keys = [
        "reward_alpha_sinr",
        "reward_beta_sinr_loss",
        "reward_gamma_angle",
    ]

    for key in required_keys:
        if key not in reward_config:
            raise KeyError(f"Missing reward_config key: {key}")

    env = BeamformingEnv(
        array=build_array(
            num_rows=num_rows,
            num_cols=num_cols,
            carrier_freq=carrier_freq,
        ),
        array_position=array_position,
        desired_power=desired_power,
        noise_power=noise_power,
        max_jammers=max_jammers,
        target_range_m=target_range_m,
        observation_mode=observation_mode,
        action_mode=action_mode,
        reward_alpha_sinr=reward_config["reward_alpha_sinr"],
        reward_beta_sinr_loss=reward_config["reward_beta_sinr_loss"],
        reward_gamma_angle=reward_config["reward_gamma_angle"],
    )

    if monitor:
        env = Monitor(env)

    return env


# ============================================================
# State/action metadata helpers
# ============================================================

def get_state_definition(observation_mode: str) -> list[str]:
    """
    Return a human-readable state definition for metadata.
    """

    if observation_mode == "angles":
        return [
            "theta_target_norm",
            "phi_target_norm",
            "theta_j1_norm",
            "phi_j1_norm",
            "m1",
            "theta_j2_norm",
            "phi_j2_norm",
            "m2",
            "theta_j3_norm",
            "phi_j3_norm",
            "m3",
        ]

    if observation_mode == "unit_vector":
        return [
            "u_target_x",
            "u_target_y",
            "u_target_z",
            "u_j1_x",
            "u_j1_y",
            "u_j1_z",
            "m1",
            "u_j2_x",
            "u_j2_y",
            "u_j2_z",
            "m2",
            "u_j3_x",
            "u_j3_y",
            "u_j3_z",
            "m3",
        ]

    raise ValueError(f"Unknown observation_mode: {observation_mode}")


def get_action_definition(action_mode: str) -> list[str]:
    """
    Return a human-readable action definition for metadata.
    """

    if action_mode == "angles":
        return [
            "theta_steer_norm",
            "phi_steer_norm",
        ]

    if action_mode == "unit_vector":
        return [
            "u_steer_x",
            "u_steer_y",
            "u_steer_z",
        ]

    raise ValueError(f"Unknown action_mode: {action_mode}")


# ============================================================
# Stable-Baselines3 helpers
# ============================================================

def build_action_noise(env) -> NormalActionNoise:
    """
    Create Gaussian action noise for deterministic off-policy methods.
    """

    n_actions = env.action_space.shape[0]

    return NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )


def build_model(
    algorithm: str,
    env,
    seed: int = 42,
    learning_rate: float = 3e-4,
    net_arch: list[int] | None = None,
):
    """
    Create a Stable-Baselines3 model with consistent hyperparameters.

    Parameters
    ----------
    algorithm : str
        One of: "TD3", "SAC", "PPO", "DDPG", "A2C".

    env
        Gymnasium-compatible environment.

    seed : int
        Random seed.

    learning_rate : float
        Optimizer learning rate.

    net_arch : list[int], optional
        MLP architecture. Defaults to [64, 64].

    Returns
    -------
    stable_baselines3 model
        Instantiated DRL model.
    """

    if net_arch is None:
        net_arch = [64, 64]

    policy_kwargs = dict(
        net_arch=net_arch,
    )

    if algorithm == "TD3":
        return TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=build_action_noise(env),
            policy_delay=2,
            target_policy_noise=0.1,
            target_noise_clip=0.3,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
        )

    if algorithm == "SAC":
        return SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
        )

    if algorithm == "PPO":
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
        )

    if algorithm == "DDPG":
        return DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=build_action_noise(env),
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
        )

    if algorithm == "A2C":
        return A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


# ============================================================
# Serialization helpers
# ============================================================

def make_json_safe(obj: Any) -> Any:
    """
    Convert numpy/pandas objects into JSON-serializable Python objects.
    """

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    return obj


def save_agent_metadata(
    agent_id: str,
    algorithm: str,
    observation_mode: str,
    action_mode: str,
    reward_mode: str,
    reward_config: dict,
    env,
    final_summary: dict,
    training_time_seconds: float,
    meta_path: Path,
    training_timesteps: int,
    short_training_timesteps: int,
    additional_training_timesteps: int,
    evaluation_samples: int,
    num_rows: int = 6,
    num_cols: int = 6,
    carrier_freq: float = 10e9,
    desired_power: float = 1.0,
    noise_power: float = 1e-3,
    max_jammers: int = 3,
    target_range_m: float = 1000.0,
    phase: str = "Phase 2 - DRL Beam Steering",
    notes: str | None = None,
) -> dict:
    """
    Save metadata JSON for one trained agent.

    Parameters
    ----------
    agent_id : str
        Agent identifier.

    algorithm : str
        DRL algorithm name.

    observation_mode : str
        Observation representation.

    action_mode : str
        Action representation.

    reward_mode : str
        Reward configuration name.

    reward_config : dict
        Reward coefficient dictionary.

    env
        Environment used for training.

    final_summary : dict
        Evaluation summary metrics.

    training_time_seconds : float
        Training elapsed time.

    meta_path : Path
        Output metadata path.

    training_timesteps : int
        Total training timesteps.

    short_training_timesteps : int
        Initial training timesteps.

    additional_training_timesteps : int
        Additional training timesteps.

    evaluation_samples : int
        Number of final evaluation samples.

    Returns
    -------
    dict
        Metadata dictionary.
    """

    if notes is None:
        notes = (
            "Phase 2 DRL beam steering experiment. Scenario without jammers. "
            "The agent learns beam steering under the selected state/action/reward/model configuration."
        )

    metadata = {
        "agent_id": agent_id,
        "algorithm": algorithm,
        "phase": phase,
        "observation_mode": observation_mode,
        "action_mode": action_mode,
        "reward_mode": reward_mode,
        "observation_dim": int(env.observation_space.shape[0]),
        "action_dim": int(env.action_space.shape[0]),
        "state_definition": get_state_definition(observation_mode),
        "action_definition": get_action_definition(action_mode),

        # Reward configuration
        "reward_definition": reward_config.get("reward_definition", None),
        "reward_alpha_sinr": float(reward_config["reward_alpha_sinr"]),
        "reward_beta_sinr_loss": float(reward_config["reward_beta_sinr_loss"]),
        "reward_gamma_angle": float(reward_config["reward_gamma_angle"]),
        "reward_config": make_json_safe(reward_config),

        # Training setup
        "training_timesteps": int(training_timesteps),
        "short_training_timesteps": int(short_training_timesteps),
        "additional_training_timesteps": int(additional_training_timesteps),
        "evaluation_samples": int(evaluation_samples),

        # Physical setup
        "array": {
            "num_rows": int(num_rows),
            "num_cols": int(num_cols),
            "carrier_freq_hz": float(carrier_freq),
            "desired_power": float(desired_power),
            "noise_power": float(noise_power),
            "max_jammers": int(max_jammers),
            "target_range_m": float(target_range_m),
        },

        # Results
        "final_metrics": make_json_safe(final_summary),
        "training_time_seconds": float(training_time_seconds),
        "notes": notes,
    }

    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(metadata), f, indent=4)

    return metadata