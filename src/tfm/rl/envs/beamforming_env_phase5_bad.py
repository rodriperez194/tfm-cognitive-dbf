from __future__ import annotations

import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tfm.physics.narrow_band.phased_array_nb import Phased_Array_NB
from tfm.physics.narrow_band.weights_deterministic_nb import (
    multi_interference_suppression_weights,
)
from tfm.physics.narrow_band.weights_stochastic_nb import (
    mvdr_weights,
)

from tfm.math.narrow_band.metrics import compute_sinr
from tfm.math.narrow_band.steering_vector import get_steering_vector
from tfm.math.narrow_band.geometry import (
    angles_to_unit_vector,
    unit_vector_to_angles,
)


class BeamformingEnvPhase5(gym.Env):
    """
    Gymnasium environment for Phase 5 cognitive digital beamforming under
    dynamic multi-jammer scenarios with block-based control.

    This version replaces the previous Phase 5 environment by introducing:

    1. Block-based control:
       One RL action is held constant during K internal physical substeps,
       where K = control_interval_steps.

    2. Separated angular reward terms:
       - SOI angular error loss.
       - Jammer angular error loss.

    3. No action-change penalty:
       The smoothness penalty is intentionally removed from the reward.

    Design principle
    ----------------
    The environment does not use ScenarioGenerator internally.

    At reset(), the environment samples:
    - one static signal of interest (SOI) direction,
    - a fixed or randomly selected number of active jammer directions,
    - jammer angular motion parameters.

    At step(action):
    - the action is converted into one SOI action direction and up to three
      jammer action directions,
    - beamforming weights are built once from that action,
    - the same weights are held fixed over control_interval_steps physical
      substeps,
    - jammers move internally inside the block,
    - reward and metrics are averaged/aggregated over the block.

    Observation modes
    -----------------
    observation_mode = "angles"

        state = [
            theta_soi_norm,
            phi_soi_norm,
            theta_j1_norm, phi_j1_norm, m1,
            theta_j2_norm, phi_j2_norm, m2,
            theta_j3_norm, phi_j3_norm, m3
        ]

        Observation dimension: 11.

    observation_mode = "unit_vector"

        state = [
            u_soi_x, u_soi_y, u_soi_z,
            u_j1_x, u_j1_y, u_j1_z, m1,
            u_j2_x, u_j2_y, u_j2_z, m2,
            u_j3_x, u_j3_y, u_j3_z, m3
        ]

        Observation dimension: 15.

    Action modes
    ------------
    action_mode = "angles"

        action = [
            theta_soi_action_norm,
            phi_soi_action_norm,
            theta_j1_action_norm, phi_j1_action_norm,
            theta_j2_action_norm, phi_j2_action_norm,
            theta_j3_action_norm, phi_j3_action_norm
        ]

        Action dimension: 8.

    action_mode = "unit_vector"

        action = [
            u_soi_x, u_soi_y, u_soi_z,
            u_j1_x, u_j1_y, u_j1_z,
            u_j2_x, u_j2_y, u_j2_z,
            u_j3_x, u_j3_y, u_j3_z
        ]

        Action dimension: 12.

    Beamforming modes
    -----------------
    beamforming_mode = "steering"

        Conventional steering weights are generated only from the SOI action
        direction. Jammer action directions are diagnostic and do not affect
        the weights.

    beamforming_mode = "nulling"

        Deterministic multi-jammer nulling weights are generated using the
        SOI action direction and the active jammer action directions.

    beamforming_mode = "mvdr"

        MVDR weights are generated using the SOI action direction and an
        interference-plus-noise covariance matrix constructed from active
        jammer action directions.

    Reward definition
    -----------------
    At each internal physical substep:

        reward =
            alpha * sinr_db
            - beta * clipped_sinr_loss_db
            - gamma_soi * soi_angle_loss
            - gamma_jammer * jammer_angle_loss

    where:

        soi_angle_loss = (soi_angle_error_deg / 180)^2

        jammer_angle_loss =
            mean_j [(jammer_angle_error_j_deg / 180)^2]

    If there are no active jammers, jammer_angle_loss = 0.

    The RL reward returned by step(action) is the mean reward over the
    complete control block:

        reward_rl = mean_k reward_k

    Reference modes
    ---------------
    reference_mode = "steering"

        The SINR-loss reference is perfect conventional steering toward the
        true SOI.

    reference_mode = "same_beamforming_mode"

        The SINR-loss reference uses the same beamforming family as the agent,
        with true SOI and true active jammer directions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        array: Phased_Array_NB,
        array_position: np.ndarray,
        desired_power: float = 1.0,
        noise_power: float = 1e-3,
        max_jammers: int = 3,
        num_active_jammers: int | None = 1,
        active_jammers_choices: list[int] | tuple[int, ...] | None = None,
        jammer_powers: list[float] | None = None,
        theta_limits_rad: tuple[float, float] = (0.0, np.pi / 2.0),
        phi_limits_rad: tuple[float, float] = (0.0, 2.0 * np.pi),
        jammer_theta_limits_rad: tuple[float, float] = (0.0, np.pi / 2.0),
        jammer_phi_limits_rad: tuple[float, float] = (0.0, 2.0 * np.pi),
        min_target_jammer_separation_deg: float = 5.0,
        observation_mode: str = "unit_vector",
        action_mode: str = "unit_vector",
        beamforming_mode: str = "steering",
        enforce_visible_hemisphere: bool = True,
        reward_alpha_sinr: float = 0.0,
        reward_beta_sinr_loss: float = 1.0,
        reward_gamma_soi_angle: float = 0.0,
        reward_gamma_jammer_angle: float = 0.0,
        max_sinr_loss_db: float = 60.0,
        reference_mode: str = "same_beamforming_mode",
        mvdr_diagonal_loading: float = 1e-4,
        nulling_diagonal_loading: float = 1e-8,
        invalid_sinr_db: float = -120.0,
        invalid_value_penalty: float = -1_000.0,
        episode_length: int = 50,
        control_interval_steps: int = 1,
        jammer_motion_mode: str = "linear_angular",
        jammer_theta_velocity_limits_deg_per_step: tuple[float, float] = (-0.5, 0.5),
        jammer_phi_velocity_limits_deg_per_step: tuple[float, float] = (-2.0, 2.0),
        jammer_random_walk_std_deg_per_step: float = 1.0,
        maneuver_interval_steps: int = 10,
        dynamic_jammer_boundary_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.array = array
        self.array_position = np.asarray(array_position, dtype=float).reshape(3)

        self.desired_power = float(desired_power)
        self.noise_power = float(noise_power)

        self.max_jammers = int(max_jammers)
        self.active_jammers_choices = (
            None
            if active_jammers_choices is None
            else [int(value) for value in active_jammers_choices]
        )

        if num_active_jammers is None:
            if self.active_jammers_choices is None:
                raise ValueError(
                    "num_active_jammers can only be None when "
                    "active_jammers_choices is provided."
                )
            self.num_active_jammers = int(self.active_jammers_choices[0])
        else:
            self.num_active_jammers = int(num_active_jammers)

        self.theta_min = float(theta_limits_rad[0])
        self.theta_max = float(theta_limits_rad[1])
        self.phi_min = float(phi_limits_rad[0])
        self.phi_max = float(phi_limits_rad[1])

        self.jammer_theta_min = float(jammer_theta_limits_rad[0])
        self.jammer_theta_max = float(jammer_theta_limits_rad[1])
        self.jammer_phi_min = float(jammer_phi_limits_rad[0])
        self.jammer_phi_max = float(jammer_phi_limits_rad[1])

        self.min_target_jammer_separation_deg = float(
            min_target_jammer_separation_deg
        )

        self.observation_mode = str(observation_mode)
        self.action_mode = str(action_mode)
        self.beamforming_mode = str(beamforming_mode)
        self.enforce_visible_hemisphere = bool(enforce_visible_hemisphere)

        self.reward_alpha_sinr = float(reward_alpha_sinr)
        self.reward_beta_sinr_loss = float(reward_beta_sinr_loss)
        self.reward_gamma_soi_angle = float(reward_gamma_soi_angle)
        self.reward_gamma_jammer_angle = float(reward_gamma_jammer_angle)

        self.max_sinr_loss_db = float(max_sinr_loss_db)
        self.reference_mode = str(reference_mode)

        self.mvdr_diagonal_loading = float(mvdr_diagonal_loading)
        self.nulling_diagonal_loading = float(nulling_diagonal_loading)

        self.invalid_sinr_db = float(invalid_sinr_db)
        self.invalid_value_penalty = float(invalid_value_penalty)

        self.episode_length = int(episode_length)
        self.control_interval_steps = int(control_interval_steps)

        self.jammer_motion_mode = str(jammer_motion_mode)
        self.jammer_theta_velocity_limits_rad_per_step = (
            float(np.deg2rad(jammer_theta_velocity_limits_deg_per_step[0])),
            float(np.deg2rad(jammer_theta_velocity_limits_deg_per_step[1])),
        )
        self.jammer_phi_velocity_limits_rad_per_step = (
            float(np.deg2rad(jammer_phi_velocity_limits_deg_per_step[0])),
            float(np.deg2rad(jammer_phi_velocity_limits_deg_per_step[1])),
        )
        self.jammer_random_walk_std_rad_per_step = float(
            np.deg2rad(jammer_random_walk_std_deg_per_step)
        )
        self.maneuver_interval_steps = int(maneuver_interval_steps)
        self.dynamic_jammer_boundary_mode = str(dynamic_jammer_boundary_mode)

        if self.max_jammers != 3:
            raise ValueError(
                "This environment currently requires max_jammers=3 "
                "to match the fixed roadmap state and action format."
            )

        if self.num_active_jammers < 0:
            raise ValueError("num_active_jammers must be non-negative.")

        if self.num_active_jammers > self.max_jammers:
            raise ValueError("num_active_jammers cannot exceed max_jammers.")

        if self.active_jammers_choices is not None:
            if len(self.active_jammers_choices) == 0:
                raise ValueError("active_jammers_choices cannot be empty.")

            for value in self.active_jammers_choices:
                if value < 0 or value > self.max_jammers:
                    raise ValueError(
                        "All active_jammers_choices values must be between "
                        "0 and max_jammers."
                    )

        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive.")

        if self.control_interval_steps <= 0:
            raise ValueError("control_interval_steps must be positive.")

        if self.maneuver_interval_steps <= 0:
            raise ValueError("maneuver_interval_steps must be positive.")

        if self.beamforming_mode not in ["steering", "nulling", "mvdr"]:
            raise ValueError(
                "Unknown beamforming_mode. Expected one of: "
                "'steering', 'nulling', 'mvdr'."
            )

        if self.observation_mode not in ["angles", "unit_vector"]:
            raise ValueError(
                "Unknown observation_mode. Expected one of: "
                "'angles', 'unit_vector'."
            )

        if self.action_mode not in ["angles", "unit_vector"]:
            raise ValueError(
                "Unknown action_mode. Expected one of: "
                "'angles', 'unit_vector'."
            )

        if self.reference_mode not in ["steering", "same_beamforming_mode"]:
            raise ValueError(
                "Unknown reference_mode. Expected one of: "
                "'steering', 'same_beamforming_mode'."
            )

        if self.jammer_motion_mode not in [
            "linear_angular",
            "random_walk",
            "maneuvering",
        ]:
            raise ValueError(
                "Unknown jammer_motion_mode. Expected one of: "
                "'linear_angular', 'random_walk', 'maneuvering'."
            )

        if self.dynamic_jammer_boundary_mode not in ["reflect", "wrap"]:
            raise ValueError(
                "Unknown dynamic_jammer_boundary_mode. Expected one of: "
                "'reflect', 'wrap'."
            )

        self.jammer_powers_config = (
            None if jammer_powers is None else [float(power) for power in jammer_powers]
        )
        self.jammer_powers = self._build_jammer_powers_for_active_count(
            self.num_active_jammers
        )

        if self.observation_mode == "angles":
            self.observation_dim = 2 + 3 * self.max_jammers

            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.observation_dim,),
                dtype=np.float32,
            )

        elif self.observation_mode == "unit_vector":
            self.observation_dim = 3 + 4 * self.max_jammers

            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.observation_dim,),
                dtype=np.float32,
            )

        if self.action_mode == "angles":
            self.action_dim = 2 + 2 * self.max_jammers

            self.action_space = spaces.Box(
                low=np.zeros(self.action_dim, dtype=np.float32),
                high=np.ones(self.action_dim, dtype=np.float32),
                shape=(self.action_dim,),
                dtype=np.float32,
            )

        elif self.action_mode == "unit_vector":
            self.action_dim = 3 + 3 * self.max_jammers

            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.action_dim,),
                dtype=np.float32,
            )

        self.current_theta_rad: float | None = None
        self.current_phi_rad: float | None = None

        self.current_jammer_thetas_rad: list[float] = []
        self.current_jammer_phis_rad: list[float] = []

        self.jammer_theta_velocities_rad_per_step: list[float] = []
        self.jammer_phi_velocities_rad_per_step: list[float] = []

        self.current_state: np.ndarray | None = None
        self.current_step: int = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment and sample a new dynamic angular scenario.
        """

        super().reset(seed=seed)

        self._sample_num_active_jammers_for_episode()
        self.jammer_powers = self._build_jammer_powers_for_active_count(
            self.num_active_jammers
        )

        theta_rad, phi_rad = self._sample_target_doa()

        jammer_thetas_rad, jammer_phis_rad = self._sample_jammer_doas(
            theta_target_rad=theta_rad,
            phi_target_rad=phi_rad,
        )

        self.current_theta_rad = theta_rad
        self.current_phi_rad = phi_rad

        self.current_jammer_thetas_rad = jammer_thetas_rad
        self.current_jammer_phis_rad = jammer_phis_rad

        (
            self.jammer_theta_velocities_rad_per_step,
            self.jammer_phi_velocities_rad_per_step,
        ) = self._sample_jammer_velocities()

        self.current_step = 0

        state = self._build_state(
            theta_target_rad=theta_rad,
            phi_target_rad=phi_rad,
            jammer_thetas_rad=jammer_thetas_rad,
            jammer_phis_rad=jammer_phis_rad,
        )

        self.current_state = state

        info = self._build_reset_info()

        return state, info

    def step(self, action: np.ndarray):
        """
        Evaluate one RL action over a complete block of physical substeps.

        The action is held constant for control_interval_steps. Beamforming
        weights are built once from the action and then reused during the
        whole block. Jammers evolve internally between physical substeps.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized.")

        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        state_before_block = self.current_state.copy()
        step_index = int(self.current_step)

        (
            theta_soi_action_rad,
            phi_soi_action_rad,
            jammer_action_directions_rad,
        ) = self._action_to_angles(action)

        numerical_error = False

        try:
            weights = self._build_beamforming_weights(
                theta_soi_action_rad=theta_soi_action_rad,
                phi_soi_action_rad=phi_soi_action_rad,
                jammer_action_directions_rad=jammer_action_directions_rad,
            )
        except Exception:
            weights = self._build_safe_fallback_weights()
            numerical_error = True

        weights_are_finite = bool(np.all(np.isfinite(weights)))

        if not weights_are_finite:
            weights = self._build_safe_fallback_weights()
            numerical_error = True

        self.array.set_weights(weights)

        substep_metrics: list[dict] = []

        for block_substep_index in range(self.control_interval_steps):
            metrics = self._evaluate_current_scene_with_fixed_weights(
                weights=weights,
                theta_soi_action_rad=theta_soi_action_rad,
                phi_soi_action_rad=phi_soi_action_rad,
                jammer_action_directions_rad=jammer_action_directions_rad,
                block_substep_index=block_substep_index,
            )

            if metrics["numerical_error"]:
                numerical_error = True

            substep_metrics.append(metrics)

            self._update_dynamic_jammers()

        block_summary = self._build_block_summary(substep_metrics)

        reward = float(block_summary["block_reward_mean"])

        reward_is_finite = bool(np.isfinite(reward))
        if not reward_is_finite:
            reward = self.invalid_value_penalty
            numerical_error = True

        self.current_step += 1

        terminated = bool(self.current_step >= self.episode_length)
        truncated = False

        next_state = self._build_state(
            theta_target_rad=self.current_theta_rad,
            phi_target_rad=self.current_phi_rad,
            jammer_thetas_rad=self.current_jammer_thetas_rad,
            jammer_phis_rad=self.current_jammer_phis_rad,
        )

        self.current_state = next_state

        jammer_action_directions_deg = [
            (
                float(np.rad2deg(theta_rad)),
                float(np.rad2deg(phi_rad)),
            )
            for theta_rad, phi_rad in jammer_action_directions_rad
        ]

        info = {
            "reward": reward,
            "reward_alpha_sinr": self.reward_alpha_sinr,
            "reward_beta_sinr_loss": self.reward_beta_sinr_loss,
            "reward_gamma_soi_angle": self.reward_gamma_soi_angle,
            "reward_gamma_jammer_angle": self.reward_gamma_jammer_angle,
            "beamforming_mode": self.beamforming_mode,
            "reference_mode": self.reference_mode,
            "jammer_motion_mode": self.jammer_motion_mode,
            "dynamic_jammer_boundary_mode": self.dynamic_jammer_boundary_mode,
            "step_index": step_index,
            "episode_length": self.episode_length,
            "control_interval_steps": self.control_interval_steps,
            "num_active_jammers": self.num_active_jammers,
            "state_before_block": state_before_block,
            "next_state": next_state.copy(),
            "weights": self.array.W.copy(),
            "weights_are_finite": weights_are_finite,
            "reward_is_finite": reward_is_finite,
            "numerical_error": numerical_error,
            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,
            "theta_target_deg": float(np.rad2deg(self.current_theta_rad)),
            "phi_target_deg": float(np.rad2deg(self.current_phi_rad)),
            "theta_soi_action_rad": theta_soi_action_rad,
            "phi_soi_action_rad": phi_soi_action_rad,
            "theta_soi_action_deg": float(np.rad2deg(theta_soi_action_rad)),
            "phi_soi_action_deg": float(np.rad2deg(phi_soi_action_rad)),
            "theta_steer_rad": theta_soi_action_rad,
            "phi_steer_rad": phi_soi_action_rad,
            "theta_steer_deg": float(np.rad2deg(theta_soi_action_rad)),
            "phi_steer_deg": float(np.rad2deg(phi_soi_action_rad)),
            "jammer_action_directions_rad": jammer_action_directions_rad.copy(),
            "jammer_action_directions_deg": jammer_action_directions_deg,
            "jammer_thetas_rad": self.current_jammer_thetas_rad.copy(),
            "jammer_phis_rad": self.current_jammer_phis_rad.copy(),
            "jammer_thetas_deg": [
                float(np.rad2deg(theta))
                for theta in self.current_jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(phi))
                for phi in self.current_jammer_phis_rad
            ],
            "jammers_directions_deg": self._get_current_jammer_directions_deg(),
            "jammers_powers": self.jammer_powers.copy(),
            "jammer_theta_velocities_rad_per_step": (
                self.jammer_theta_velocities_rad_per_step.copy()
            ),
            "jammer_phi_velocities_rad_per_step": (
                self.jammer_phi_velocities_rad_per_step.copy()
            ),
            "jammer_theta_velocities_deg_per_step": [
                float(np.rad2deg(value))
                for value in self.jammer_theta_velocities_rad_per_step
            ],
            "jammer_phi_velocities_deg_per_step": [
                float(np.rad2deg(value))
                for value in self.jammer_phi_velocities_rad_per_step
            ],
            "substep_metrics": substep_metrics,
        }

        info.update(block_summary)

        return next_state, reward, terminated, truncated, info

    def _evaluate_current_scene_with_fixed_weights(
        self,
        weights: np.ndarray,
        theta_soi_action_rad: float,
        phi_soi_action_rad: float,
        jammer_action_directions_rad: list[tuple[float, float]],
        block_substep_index: int,
    ) -> dict:
        """
        Evaluate one physical substep using fixed beamforming weights.
        """

        numerical_error = False
        sinr_is_finite = True
        reference_sinr_is_finite = True
        reward_is_finite = True

        target_direction_deg = (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.rad2deg(self.current_phi_rad)),
        )

        jammer_directions_deg = self._get_current_jammer_directions_deg()

        try:
            sinr_db = compute_sinr(
                weights=weights,
                element_positions=self.array.element_positions,
                wavenumber_k=self.array.k_num,
                target_direction=target_direction_deg,
                target_power=self.desired_power,
                jammers_directions=jammer_directions_deg,
                jammers_powers=self.jammer_powers,
                noise_power=self.noise_power,
            )
            sinr_db = float(sinr_db)
        except Exception:
            sinr_db = self.invalid_sinr_db
            numerical_error = True

        sinr_is_finite = bool(np.isfinite(sinr_db))

        if not sinr_is_finite:
            sinr_db = self.invalid_sinr_db
            numerical_error = True

        soi_angle_error_deg = self._compute_angular_error_deg(
            theta_a_rad=self.current_theta_rad,
            phi_a_rad=self.current_phi_rad,
            theta_b_rad=theta_soi_action_rad,
            phi_b_rad=phi_soi_action_rad,
        )

        soi_angle_loss = self._compute_angle_loss(soi_angle_error_deg)

        jammer_action_errors_deg = self._compute_jammer_action_errors_deg(
            jammer_action_directions_rad=jammer_action_directions_rad,
        )

        jammer_angle_loss = self._compute_jammer_angle_loss(
            jammer_action_errors_deg=jammer_action_errors_deg,
        )

        jammer_action_error_mean_deg = self._safe_nanmean(jammer_action_errors_deg)
        jammer_action_error_max_deg = self._safe_nanmax(jammer_action_errors_deg)

        try:
            reference_sinr_db = self._compute_reference_sinr_db()
        except Exception:
            reference_sinr_db = self.invalid_sinr_db
            numerical_error = True

        reference_sinr_is_finite = bool(np.isfinite(reference_sinr_db))

        if not reference_sinr_is_finite:
            reference_sinr_db = self.invalid_sinr_db
            numerical_error = True

        sinr_loss_db = reference_sinr_db - sinr_db

        if not np.isfinite(sinr_loss_db):
            sinr_loss_db = self.max_sinr_loss_db
            numerical_error = True

        sinr_loss_db = max(0.0, float(sinr_loss_db))
        clipped_sinr_loss_db = min(sinr_loss_db, self.max_sinr_loss_db)

        reward = self._compute_reward(
            sinr_db=sinr_db,
            clipped_sinr_loss_db=clipped_sinr_loss_db,
            soi_angle_loss=soi_angle_loss,
            jammer_angle_loss=jammer_angle_loss,
        )

        reward_is_finite = bool(np.isfinite(reward))

        if not reward_is_finite:
            reward = self.invalid_value_penalty
            numerical_error = True

        return {
            "block_substep_index": int(block_substep_index),
            "reward": float(reward),
            "sinr_db": float(sinr_db),
            "reference_sinr_db": float(reference_sinr_db),
            "sinr_loss_db": float(sinr_loss_db),
            "clipped_sinr_loss_db": float(clipped_sinr_loss_db),
            "soi_angle_error_deg": float(soi_angle_error_deg),
            "soi_angle_loss": float(soi_angle_loss),
            "jammer_action_errors_deg": jammer_action_errors_deg,
            "jammer_action_error_mean_deg": jammer_action_error_mean_deg,
            "jammer_action_error_max_deg": jammer_action_error_max_deg,
            "jammer_angle_loss": float(jammer_angle_loss),
            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,
            "theta_target_deg": float(np.rad2deg(self.current_theta_rad)),
            "phi_target_deg": float(np.rad2deg(self.current_phi_rad)),
            "jammer_thetas_rad": self.current_jammer_thetas_rad.copy(),
            "jammer_phis_rad": self.current_jammer_phis_rad.copy(),
            "jammer_thetas_deg": [
                float(np.rad2deg(theta))
                for theta in self.current_jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(phi))
                for phi in self.current_jammer_phis_rad
            ],
            "jammers_directions_deg": jammer_directions_deg,
            "numerical_error": numerical_error,
            "sinr_is_finite": sinr_is_finite,
            "reference_sinr_is_finite": reference_sinr_is_finite,
            "reward_is_finite": reward_is_finite,
        }

    def _build_block_summary(self, substep_metrics: list[dict]) -> dict:
        """
        Aggregate substep metrics into block-level metrics.
        """

        rewards = self._extract_metric_array(substep_metrics, "reward")
        sinr_values = self._extract_metric_array(substep_metrics, "sinr_db")
        reference_sinr_values = self._extract_metric_array(
            substep_metrics,
            "reference_sinr_db",
        )
        sinr_loss_values = self._extract_metric_array(substep_metrics, "sinr_loss_db")
        clipped_sinr_loss_values = self._extract_metric_array(
            substep_metrics,
            "clipped_sinr_loss_db",
        )
        soi_errors = self._extract_metric_array(
            substep_metrics,
            "soi_angle_error_deg",
        )
        soi_losses = self._extract_metric_array(substep_metrics, "soi_angle_loss")
        jammer_errors_mean = self._extract_metric_array(
            substep_metrics,
            "jammer_action_error_mean_deg",
        )
        jammer_errors_max = self._extract_metric_array(
            substep_metrics,
            "jammer_action_error_max_deg",
        )
        jammer_losses = self._extract_metric_array(
            substep_metrics,
            "jammer_angle_loss",
        )

        numerical_error = any(
            bool(metric.get("numerical_error", False)) for metric in substep_metrics
        )

        return {
            "block_reward_mean": self._safe_array_mean(rewards),
            "block_reward_min": self._safe_array_min(rewards),
            "block_reward_max": self._safe_array_max(rewards),
            "sinr_db": self._safe_array_mean(sinr_values),
            "sinr_db_block_mean": self._safe_array_mean(sinr_values),
            "sinr_db_block_min": self._safe_array_min(sinr_values),
            "sinr_db_block_max": self._safe_array_max(sinr_values),
            "reference_sinr_db": self._safe_array_mean(reference_sinr_values),
            "reference_sinr_db_block_mean": self._safe_array_mean(
                reference_sinr_values
            ),
            "sinr_loss_db": self._safe_array_mean(sinr_loss_values),
            "sinr_loss_db_block_mean": self._safe_array_mean(sinr_loss_values),
            "sinr_loss_db_block_max": self._safe_array_max(sinr_loss_values),
            "clipped_sinr_loss_db": self._safe_array_mean(
                clipped_sinr_loss_values
            ),
            "clipped_sinr_loss_db_block_mean": self._safe_array_mean(
                clipped_sinr_loss_values
            ),
            "clipped_sinr_loss_db_block_max": self._safe_array_max(
                clipped_sinr_loss_values
            ),
            "soi_angle_error_deg": self._safe_array_mean(soi_errors),
            "soi_angle_error_block_mean_deg": self._safe_array_mean(soi_errors),
            "soi_angle_error_block_max_deg": self._safe_array_max(soi_errors),
            "soi_angle_loss": self._safe_array_mean(soi_losses),
            "soi_angle_loss_block_mean": self._safe_array_mean(soi_losses),
            "jammer_action_error_mean_deg": self._safe_array_mean(
                jammer_errors_mean
            ),
            "jammer_action_error_max_deg": self._safe_array_max(
                jammer_errors_max
            ),
            "jammer_angle_error_block_mean_deg": self._safe_array_mean(
                jammer_errors_mean
            ),
            "jammer_angle_error_block_max_deg": self._safe_array_max(
                jammer_errors_max
            ),
            "jammer_angle_loss": self._safe_array_mean(jammer_losses),
            "jammer_angle_loss_block_mean": self._safe_array_mean(jammer_losses),
            "block_numerical_error": numerical_error,
        }

    def _extract_metric_array(self, metrics: list[dict], key: str) -> np.ndarray:
        """
        Extract a numeric metric from all substeps.
        """

        values = [metric.get(key, np.nan) for metric in metrics]
        return np.asarray(values, dtype=float)

    def _build_reset_info(self) -> dict:
        """
        Build reset info dictionary.
        """

        jammer_directions_deg = self._get_current_jammer_directions_deg()

        return {
            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,
            "theta_target_deg": float(np.rad2deg(self.current_theta_rad)),
            "phi_target_deg": float(np.rad2deg(self.current_phi_rad)),
            "num_active_jammers": self.num_active_jammers,
            "jammer_thetas_rad": self.current_jammer_thetas_rad.copy(),
            "jammer_phis_rad": self.current_jammer_phis_rad.copy(),
            "jammer_thetas_deg": [
                float(np.rad2deg(theta))
                for theta in self.current_jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(phi))
                for phi in self.current_jammer_phis_rad
            ],
            "jammers_directions_deg": jammer_directions_deg,
            "jammers_powers": self.jammer_powers.copy(),
            "jammer_motion_mode": self.jammer_motion_mode,
            "dynamic_jammer_boundary_mode": self.dynamic_jammer_boundary_mode,
            "reference_mode": self.reference_mode,
            "beamforming_mode": self.beamforming_mode,
            "control_interval_steps": self.control_interval_steps,
            "jammer_theta_velocities_rad_per_step": (
                self.jammer_theta_velocities_rad_per_step.copy()
            ),
            "jammer_phi_velocities_rad_per_step": (
                self.jammer_phi_velocities_rad_per_step.copy()
            ),
            "jammer_theta_velocities_deg_per_step": [
                float(np.rad2deg(value))
                for value in self.jammer_theta_velocities_rad_per_step
            ],
            "jammer_phi_velocities_deg_per_step": [
                float(np.rad2deg(value))
                for value in self.jammer_phi_velocities_rad_per_step
            ],
            "episode_length": self.episode_length,
            "current_step": self.current_step,
        }

    def _sample_num_active_jammers_for_episode(self) -> None:
        """
        Sample the number of active jammers for the current episode when
        active_jammers_choices is configured.
        """

        if self.active_jammers_choices is None:
            return

        self.num_active_jammers = int(
            self.np_random.choice(self.active_jammers_choices)
        )

    def _build_jammer_powers_for_active_count(self, active_count: int) -> list[float]:
        """
        Build the jammer power list for the selected number of active jammers.

        If jammer_powers was not provided, all active jammers use unit power.
        If one power was provided, it is repeated for all active jammers.
        If max_jammers powers were provided, the first active_count values are used.
        Otherwise, the provided list must match the active_count.
        """

        active_count = int(active_count)

        if active_count < 0 or active_count > self.max_jammers:
            raise ValueError("active_count must be between 0 and max_jammers.")

        if active_count == 0:
            return []

        if self.jammer_powers_config is None:
            return [1.0] * active_count

        if len(self.jammer_powers_config) == 1:
            return [float(self.jammer_powers_config[0])] * active_count

        if len(self.jammer_powers_config) == self.max_jammers:
            return [float(value) for value in self.jammer_powers_config[:active_count]]

        if len(self.jammer_powers_config) == active_count:
            return [float(value) for value in self.jammer_powers_config]

        raise ValueError(
            "jammer_powers must be None, length 1, length max_jammers, "
            "or match the current active jammer count."
        )

    def _sample_target_doa(self) -> tuple[float, float]:
        """
        Sample a random static SOI DOA.
        """

        theta_rad = float(self.np_random.uniform(self.theta_min, self.theta_max))
        phi_rad = float(self.np_random.uniform(self.phi_min, self.phi_max))

        return theta_rad, phi_rad

    def _sample_jammer_doas(
        self,
        theta_target_rad: float,
        phi_target_rad: float,
    ) -> tuple[list[float], list[float]]:
        """
        Sample random initial jammer DOAs with minimum angular separation from
        the SOI and from each other.
        """

        jammer_thetas_rad: list[float] = []
        jammer_phis_rad: list[float] = []

        if self.num_active_jammers == 0:
            return jammer_thetas_rad, jammer_phis_rad

        for _ in range(self.num_active_jammers):
            theta_jam_rad, phi_jam_rad = self._sample_single_jammer_doa(
                theta_target_rad=theta_target_rad,
                phi_target_rad=phi_target_rad,
                existing_jammer_thetas_rad=jammer_thetas_rad,
                existing_jammer_phis_rad=jammer_phis_rad,
            )

            jammer_thetas_rad.append(theta_jam_rad)
            jammer_phis_rad.append(phi_jam_rad)

        return jammer_thetas_rad, jammer_phis_rad

    def _sample_single_jammer_doa(
        self,
        theta_target_rad: float,
        phi_target_rad: float,
        existing_jammer_thetas_rad: list[float],
        existing_jammer_phis_rad: list[float],
        max_attempts: int = 1000,
    ) -> tuple[float, float]:
        """
        Sample one random jammer DOA satisfying angular separation constraints.
        """

        for _ in range(max_attempts):
            theta_jam_rad = float(
                self.np_random.uniform(
                    self.jammer_theta_min,
                    self.jammer_theta_max,
                )
            )

            phi_jam_rad = float(
                self.np_random.uniform(
                    self.jammer_phi_min,
                    self.jammer_phi_max,
                )
            )

            separation_to_target_deg = self._compute_angular_error_deg(
                theta_a_rad=theta_target_rad,
                phi_a_rad=phi_target_rad,
                theta_b_rad=theta_jam_rad,
                phi_b_rad=phi_jam_rad,
            )

            if separation_to_target_deg < self.min_target_jammer_separation_deg:
                continue

            valid_separation = True

            for theta_existing_rad, phi_existing_rad in zip(
                existing_jammer_thetas_rad,
                existing_jammer_phis_rad,
            ):
                separation_to_existing_deg = self._compute_angular_error_deg(
                    theta_a_rad=theta_existing_rad,
                    phi_a_rad=phi_existing_rad,
                    theta_b_rad=theta_jam_rad,
                    phi_b_rad=phi_jam_rad,
                )

                if separation_to_existing_deg < self.min_target_jammer_separation_deg:
                    valid_separation = False
                    break

            if valid_separation:
                return theta_jam_rad, phi_jam_rad

        raise RuntimeError(
            "Could not sample a valid jammer DOA. "
            "Try reducing min_target_jammer_separation_deg."
        )

    def _sample_jammer_velocities(self) -> tuple[list[float], list[float]]:
        """
        Sample one angular velocity pair per active jammer.
        """

        theta_velocities_rad: list[float] = []
        phi_velocities_rad: list[float] = []

        for _ in range(self.num_active_jammers):
            theta_velocity_rad = float(
                self.np_random.uniform(
                    self.jammer_theta_velocity_limits_rad_per_step[0],
                    self.jammer_theta_velocity_limits_rad_per_step[1],
                )
            )

            phi_velocity_rad = float(
                self.np_random.uniform(
                    self.jammer_phi_velocity_limits_rad_per_step[0],
                    self.jammer_phi_velocity_limits_rad_per_step[1],
                )
            )

            theta_velocities_rad.append(theta_velocity_rad)
            phi_velocities_rad.append(phi_velocity_rad)

        return theta_velocities_rad, phi_velocities_rad

    def _update_dynamic_jammers(self) -> None:
        """
        Update jammer DOAs according to the configured motion model.
        """

        if self.num_active_jammers == 0:
            return

        if self.jammer_motion_mode == "maneuvering":
            physical_step_index = (
                self.current_step * self.control_interval_steps
            )
            if physical_step_index % self.maneuver_interval_steps == 0:
                (
                    self.jammer_theta_velocities_rad_per_step,
                    self.jammer_phi_velocities_rad_per_step,
                ) = self._sample_jammer_velocities()

        new_thetas_rad: list[float] = []
        new_phis_rad: list[float] = []
        new_theta_velocities_rad: list[float] = []
        new_phi_velocities_rad: list[float] = []

        for jammer_idx in range(self.num_active_jammers):
            theta_rad = self.current_jammer_thetas_rad[jammer_idx]
            phi_rad = self.current_jammer_phis_rad[jammer_idx]

            theta_velocity_rad = self.jammer_theta_velocities_rad_per_step[
                jammer_idx
            ]
            phi_velocity_rad = self.jammer_phi_velocities_rad_per_step[jammer_idx]

            if self.jammer_motion_mode == "random_walk":
                theta_increment_rad = float(
                    self.np_random.normal(
                        loc=0.0,
                        scale=self.jammer_random_walk_std_rad_per_step,
                    )
                )
                phi_increment_rad = float(
                    self.np_random.normal(
                        loc=0.0,
                        scale=self.jammer_random_walk_std_rad_per_step,
                    )
                )
            else:
                theta_increment_rad = theta_velocity_rad
                phi_increment_rad = phi_velocity_rad

            theta_next_rad = theta_rad + theta_increment_rad
            phi_next_rad = phi_rad + phi_increment_rad

            (
                theta_next_rad,
                phi_next_rad,
                theta_velocity_rad,
                phi_velocity_rad,
            ) = self._apply_jammer_boundary_conditions(
                theta_rad=theta_next_rad,
                phi_rad=phi_next_rad,
                theta_velocity_rad=theta_velocity_rad,
                phi_velocity_rad=phi_velocity_rad,
            )

            new_thetas_rad.append(theta_next_rad)
            new_phis_rad.append(phi_next_rad)
            new_theta_velocities_rad.append(theta_velocity_rad)
            new_phi_velocities_rad.append(phi_velocity_rad)

        self.current_jammer_thetas_rad = new_thetas_rad
        self.current_jammer_phis_rad = new_phis_rad
        self.jammer_theta_velocities_rad_per_step = new_theta_velocities_rad
        self.jammer_phi_velocities_rad_per_step = new_phi_velocities_rad

    def _apply_jammer_boundary_conditions(
        self,
        theta_rad: float,
        phi_rad: float,
        theta_velocity_rad: float,
        phi_velocity_rad: float,
    ) -> tuple[float, float, float, float]:
        """
        Keep dynamic jammer angles inside their configured angular limits.
        """

        if self.dynamic_jammer_boundary_mode == "wrap":
            theta_rad = self._wrap_value(
                value=theta_rad,
                minimum=self.jammer_theta_min,
                maximum=self.jammer_theta_max,
            )
            phi_rad = self._wrap_value(
                value=phi_rad,
                minimum=self.jammer_phi_min,
                maximum=self.jammer_phi_max,
            )

            return theta_rad, phi_rad, theta_velocity_rad, phi_velocity_rad

        if self.dynamic_jammer_boundary_mode == "reflect":
            (
                theta_rad,
                theta_velocity_rad,
            ) = self._reflect_value_and_velocity(
                value=theta_rad,
                velocity=theta_velocity_rad,
                minimum=self.jammer_theta_min,
                maximum=self.jammer_theta_max,
            )

            phi_rad = self._wrap_value(
                value=phi_rad,
                minimum=self.jammer_phi_min,
                maximum=self.jammer_phi_max,
            )

            return theta_rad, phi_rad, theta_velocity_rad, phi_velocity_rad

        raise RuntimeError("Invalid dynamic_jammer_boundary_mode.")

    def _wrap_value(self, value: float, minimum: float, maximum: float) -> float:
        """
        Wrap a scalar value into [minimum, maximum].
        """

        width = maximum - minimum

        if width <= 0.0:
            raise ValueError("Invalid wrapping interval.")

        return float(minimum + np.mod(value - minimum, width))

    def _reflect_value_and_velocity(
        self,
        value: float,
        velocity: float,
        minimum: float,
        maximum: float,
    ) -> tuple[float, float]:
        """
        Reflect a scalar value at interval boundaries and flip its velocity.
        """

        if maximum <= minimum:
            raise ValueError("Invalid reflection interval.")

        reflected_value = float(value)
        reflected_velocity = float(velocity)

        for _ in range(8):
            if reflected_value < minimum:
                reflected_value = minimum + (minimum - reflected_value)
                reflected_velocity = -reflected_velocity
                continue

            if reflected_value > maximum:
                reflected_value = maximum - (reflected_value - maximum)
                reflected_velocity = -reflected_velocity
                continue

            break

        reflected_value = float(np.clip(reflected_value, minimum, maximum))

        return reflected_value, reflected_velocity

    def _build_state(
        self,
        theta_target_rad: float,
        phi_target_rad: float,
        jammer_thetas_rad: list[float],
        jammer_phis_rad: list[float],
    ) -> np.ndarray:
        """
        Build the fixed-size state representation.
        """

        if self.observation_mode == "angles":
            theta_norm = self._normalize_theta(theta_target_rad)
            phi_norm = self._normalize_phi(phi_target_rad)

            state = [theta_norm, phi_norm]

            for jammer_idx in range(self.max_jammers):
                if jammer_idx < self.num_active_jammers:
                    theta_jam_norm = self._normalize_jammer_theta(
                        jammer_thetas_rad[jammer_idx]
                    )
                    phi_jam_norm = self._normalize_jammer_phi(
                        jammer_phis_rad[jammer_idx]
                    )
                    mask = 1.0
                else:
                    theta_jam_norm = 0.0
                    phi_jam_norm = 0.0
                    mask = 0.0

                state.extend([theta_jam_norm, phi_jam_norm, mask])

            return np.array(state, dtype=np.float32)

        if self.observation_mode == "unit_vector":
            theta_target_deg = float(np.rad2deg(theta_target_rad))
            phi_target_deg = float(np.rad2deg(phi_target_rad))

            u_target = angles_to_unit_vector(
                theta_deg=theta_target_deg,
                phi_deg=phi_target_deg,
                enforce_visible=self.enforce_visible_hemisphere,
            )

            state = list(u_target)

            for jammer_idx in range(self.max_jammers):
                if jammer_idx < self.num_active_jammers:
                    theta_jam_deg = float(
                        np.rad2deg(jammer_thetas_rad[jammer_idx])
                    )
                    phi_jam_deg = float(
                        np.rad2deg(jammer_phis_rad[jammer_idx])
                    )

                    u_jammer = angles_to_unit_vector(
                        theta_deg=theta_jam_deg,
                        phi_deg=phi_jam_deg,
                        enforce_visible=self.enforce_visible_hemisphere,
                    )

                    mask = 1.0
                    state.extend([u_jammer[0], u_jammer[1], u_jammer[2], mask])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0])

            return np.array(state, dtype=np.float32)

        raise RuntimeError("Invalid observation_mode.")

    def _action_to_angles(
        self,
        action: np.ndarray,
    ) -> tuple[float, float, list[tuple[float, float]]]:
        """
        Convert the selected action representation into one SOI action
        direction and max_jammers jammer action directions in radians.
        """

        if self.action_mode == "angles":
            action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
            action = np.clip(action, self.action_space.low, self.action_space.high)

            theta_soi_action_rad = self._denormalize_theta(float(action[0]))
            phi_soi_action_rad = self._denormalize_phi(float(action[1]))

            jammer_action_directions_rad: list[tuple[float, float]] = []

            offset = 2

            for jammer_idx in range(self.max_jammers):
                theta_norm = float(action[offset + 2 * jammer_idx])
                phi_norm = float(action[offset + 2 * jammer_idx + 1])

                theta_jammer_action_rad = self._denormalize_jammer_theta(theta_norm)
                phi_jammer_action_rad = self._denormalize_jammer_phi(phi_norm)

                jammer_action_directions_rad.append(
                    (
                        theta_jammer_action_rad,
                        phi_jammer_action_rad,
                    )
                )

            return (
                theta_soi_action_rad,
                phi_soi_action_rad,
                jammer_action_directions_rad,
            )

        if self.action_mode == "unit_vector":
            action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
            action = np.clip(action, self.action_space.low, self.action_space.high)

            u_soi_action = self._normalize_action_unit_vector(action[0:3])

            theta_soi_deg, phi_soi_deg = unit_vector_to_angles(
                u_soi_action,
                enforce_visible=self.enforce_visible_hemisphere,
            )

            theta_soi_action_rad = float(np.deg2rad(theta_soi_deg))
            phi_soi_action_rad = float(np.deg2rad(phi_soi_deg))

            jammer_action_directions_rad: list[tuple[float, float]] = []

            offset = 3

            for jammer_idx in range(self.max_jammers):
                start_idx = offset + 3 * jammer_idx
                end_idx = start_idx + 3

                u_jammer_action = self._normalize_action_unit_vector(
                    action[start_idx:end_idx]
                )

                theta_jammer_deg, phi_jammer_deg = unit_vector_to_angles(
                    u_jammer_action,
                    enforce_visible=self.enforce_visible_hemisphere,
                )

                theta_jammer_action_rad = float(np.deg2rad(theta_jammer_deg))
                phi_jammer_action_rad = float(np.deg2rad(phi_jammer_deg))

                jammer_action_directions_rad.append(
                    (
                        theta_jammer_action_rad,
                        phi_jammer_action_rad,
                    )
                )

            return (
                theta_soi_action_rad,
                phi_soi_action_rad,
                jammer_action_directions_rad,
            )

        raise RuntimeError("Invalid action_mode.")

    def _normalize_action_unit_vector(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize a raw 3D action vector and project it to the visible hemisphere
        if required.
        """

        action = np.asarray(action, dtype=np.float32).reshape(3)
        norm = np.linalg.norm(action)

        if norm < 1e-8:
            u_action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            u_action = action / norm

        if self.enforce_visible_hemisphere and u_action[2] < 0.0:
            u_action[2] = abs(u_action[2])
            u_action = u_action / (np.linalg.norm(u_action) + 1e-8)

        return u_action.astype(np.float32)

    def _build_beamforming_weights(
        self,
        theta_soi_action_rad: float,
        phi_soi_action_rad: float,
        jammer_action_directions_rad: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Build beamforming weights from the selected multi-directional action.
        """

        active_jammer_action_directions_rad = jammer_action_directions_rad[
            : self.num_active_jammers
        ]

        if self.beamforming_mode == "steering":
            return self._build_steering_weights(
                theta_rad=theta_soi_action_rad,
                phi_rad=phi_soi_action_rad,
            )

        if self.beamforming_mode == "nulling":
            return self._build_nulling_weights(
                theta_soi_action_rad=theta_soi_action_rad,
                phi_soi_action_rad=phi_soi_action_rad,
                jammer_action_directions_rad=active_jammer_action_directions_rad,
            )

        if self.beamforming_mode == "mvdr":
            return self._build_mvdr_weights(
                theta_soi_action_rad=theta_soi_action_rad,
                phi_soi_action_rad=phi_soi_action_rad,
                jammer_action_directions_rad=active_jammer_action_directions_rad,
            )

        raise RuntimeError("Invalid beamforming_mode.")

    def _build_steering_weights(self, theta_rad: float, phi_rad: float) -> np.ndarray:
        """
        Build conventional steering weights for a given steering direction.
        """

        theta_deg = float(np.rad2deg(theta_rad))
        phi_deg = float(np.rad2deg(phi_rad))

        weights_flat = get_steering_vector(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            direction=(theta_deg, phi_deg),
        )

        return weights_flat.reshape(self.array.N, self.array.M)

    def _build_nulling_weights(
        self,
        theta_soi_action_rad: float,
        phi_soi_action_rad: float,
        jammer_action_directions_rad: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Build deterministic multi-jammer nulling weights using the SOI action
        direction and all active jammer action directions.
        """

        target_direction = (
            float(np.rad2deg(theta_soi_action_rad)),
            float(np.rad2deg(phi_soi_action_rad)),
        )

        jammer_directions = [
            (
                float(np.rad2deg(theta_jammer_rad)),
                float(np.rad2deg(phi_jammer_rad)),
            )
            for theta_jammer_rad, phi_jammer_rad in jammer_action_directions_rad
        ]

        weights_flat = multi_interference_suppression_weights(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            target_direction=target_direction,
            jammer_directions=jammer_directions,
            diagonal_loading=self.nulling_diagonal_loading,
        )

        return weights_flat.reshape(self.array.N, self.array.M)

    def _build_mvdr_weights(
        self,
        theta_soi_action_rad: float,
        phi_soi_action_rad: float,
        jammer_action_directions_rad: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Build MVDR weights using the SOI action direction and a covariance
        matrix built from all active jammer action directions.
        """

        target_direction = (
            float(np.rad2deg(theta_soi_action_rad)),
            float(np.rad2deg(phi_soi_action_rad)),
        )

        jammer_directions = [
            (
                float(np.rad2deg(theta_jammer_rad)),
                float(np.rad2deg(phi_jammer_rad)),
            )
            for theta_jammer_rad, phi_jammer_rad in jammer_action_directions_rad
        ]

        R_xx = self._build_interference_noise_covariance(
            jammer_directions=jammer_directions,
        )

        weights_flat = mvdr_weights(
            R_xx=R_xx,
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            target_direction=target_direction,
        )

        return weights_flat.reshape(self.array.N, self.array.M)

    def _build_interference_noise_covariance(
        self,
        jammer_directions: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Build an interference-plus-noise covariance matrix from active jammer
        action directions.
        """

        num_elements = self.array.N * self.array.M

        R_xx = self.noise_power * np.eye(num_elements, dtype=np.complex128)

        for jammer_idx, jammer_direction in enumerate(jammer_directions):
            if jammer_idx >= len(self.jammer_powers):
                break

            jammer_power = float(self.jammer_powers[jammer_idx])

            jammer_sv = get_steering_vector(
                element_positions=self.array.element_positions,
                wavenumber_k=self.array.k_num,
                direction=jammer_direction,
            ).astype(np.complex128).reshape(num_elements)

            R_xx += jammer_power * np.outer(
                jammer_sv,
                np.conj(jammer_sv),
            )

        if self.mvdr_diagonal_loading > 0.0:
            R_xx += self.mvdr_diagonal_loading * np.eye(
                num_elements,
                dtype=np.complex128,
            )

        return R_xx

    def _build_safe_fallback_weights(self) -> np.ndarray:
        """
        Build safe fallback conventional steering weights.

        If the current SOI is available, steer towards it. Otherwise, steer
        towards broadside.
        """

        if self.current_theta_rad is not None and self.current_phi_rad is not None:
            return self._build_steering_weights(
                theta_rad=self.current_theta_rad,
                phi_rad=self.current_phi_rad,
            )

        return self._build_steering_weights(
            theta_rad=0.0,
            phi_rad=0.0,
        )

    def _compute_reference_sinr_db(self) -> float:
        """
        Compute the SINR reference according to reference_mode.
        """

        if self.reference_mode == "steering":
            reference_weights = self._build_reference_steering_weights()
        elif self.reference_mode == "same_beamforming_mode":
            reference_weights = self._build_reference_same_mode_weights()
        else:
            raise RuntimeError("Invalid reference_mode.")

        return self._compute_sinr_for_weights(reference_weights)

    def _build_reference_steering_weights(self) -> np.ndarray:
        """
        Build perfect conventional steering weights toward the true SOI.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Environment must be reset before computing reference weights."
            )

        return self._build_steering_weights(
            theta_rad=self.current_theta_rad,
            phi_rad=self.current_phi_rad,
        )

    def _build_reference_same_mode_weights(self) -> np.ndarray:
        """
        Build perfect reference weights using the same beamforming family as
        the agent and the true SOI/jammer directions.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Environment must be reset before computing reference weights."
            )

        true_jammer_directions_rad = list(
            zip(self.current_jammer_thetas_rad, self.current_jammer_phis_rad)
        )

        if self.beamforming_mode == "steering":
            return self._build_steering_weights(
                theta_rad=self.current_theta_rad,
                phi_rad=self.current_phi_rad,
            )

        if self.beamforming_mode == "nulling":
            return self._build_nulling_weights(
                theta_soi_action_rad=self.current_theta_rad,
                phi_soi_action_rad=self.current_phi_rad,
                jammer_action_directions_rad=true_jammer_directions_rad,
            )

        if self.beamforming_mode == "mvdr":
            return self._build_mvdr_weights(
                theta_soi_action_rad=self.current_theta_rad,
                phi_soi_action_rad=self.current_phi_rad,
                jammer_action_directions_rad=true_jammer_directions_rad,
            )

        raise RuntimeError("Invalid beamforming_mode.")

    def _compute_reference_steering_sinr_db(self) -> float:
        """
        Compute SINR using conventional steering exactly towards the true SOI.

        This method is kept for backward compatibility.
        """

        reference_weights = self._build_reference_steering_weights()
        return self._compute_sinr_for_weights(reference_weights)

    def _compute_sinr_for_weights(self, weights: np.ndarray) -> float:
        """
        Compute SINR for the current true SOI/jammer scene and a given weight
        matrix.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Environment must be reset before computing SINR.")

        target_direction_deg = (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.rad2deg(self.current_phi_rad)),
        )

        jammer_directions_deg = self._get_current_jammer_directions_deg()

        sinr_db = compute_sinr(
            weights=weights,
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            target_direction=target_direction_deg,
            target_power=self.desired_power,
            jammers_directions=jammer_directions_deg,
            jammers_powers=self.jammer_powers,
            noise_power=self.noise_power,
        )

        return float(sinr_db)

    def _compute_reward(
        self,
        sinr_db: float,
        clipped_sinr_loss_db: float,
        soi_angle_loss: float,
        jammer_angle_loss: float,
    ) -> float:
        """
        Compute the weighted reward robustly.

        Only terms with non-zero coefficients are accumulated. This prevents
        numerical contamination from expressions such as 0 * NaN.
        """

        reward = 0.0
        invalid_term_detected = False

        if self.reward_alpha_sinr != 0.0:
            if np.isfinite(sinr_db):
                reward += self.reward_alpha_sinr * float(sinr_db)
            else:
                invalid_term_detected = True

        if self.reward_beta_sinr_loss != 0.0:
            if np.isfinite(clipped_sinr_loss_db):
                reward -= self.reward_beta_sinr_loss * float(clipped_sinr_loss_db)
            else:
                invalid_term_detected = True

        if self.reward_gamma_soi_angle != 0.0:
            if np.isfinite(soi_angle_loss):
                reward -= self.reward_gamma_soi_angle * float(soi_angle_loss)
            else:
                invalid_term_detected = True

        if self.reward_gamma_jammer_angle != 0.0:
            if np.isfinite(jammer_angle_loss):
                reward -= self.reward_gamma_jammer_angle * float(jammer_angle_loss)
            else:
                invalid_term_detected = True

        if invalid_term_detected:
            reward += self.invalid_value_penalty

        if not np.isfinite(reward):
            reward = self.invalid_value_penalty

        return float(reward)

    def _get_current_jammer_directions_deg(self) -> list[tuple[float, float]]:
        """
        Return current true jammer directions in degrees.
        """

        jammer_directions_deg = []

        for theta_rad, phi_rad in zip(
            self.current_jammer_thetas_rad,
            self.current_jammer_phis_rad,
        ):
            jammer_directions_deg.append(
                (
                    float(np.rad2deg(theta_rad)),
                    float(np.rad2deg(phi_rad)),
                )
            )

        return jammer_directions_deg

    def _compute_jammer_action_errors_deg(
        self,
        jammer_action_directions_rad: list[tuple[float, float]],
    ) -> list[float]:
        """
        Compute permutation-invariant angular errors between true active
        jammers and predicted active jammer action directions.

        Inactive jammer slots are ignored.

        Why this correction is needed
        -----------------------------
        The old implementation compared jammer directions slot by slot:

            true_jammer_1 vs action_jammer_1
            true_jammer_2 vs action_jammer_2
            true_jammer_3 vs action_jammer_3

        That can be misleading for multi-jammer nulling and MVDR because the
        physical set of jammer directions is permutation-invariant. If the
        agent predicts the correct jammer directions but swaps two action
        slots, the beamformer can still place nulls in the correct angular
        directions, but a slot-wise angular loss penalizes the action as if it
        were wrong.

        This corrected implementation:

        1. Uses only active jammer slots.
        2. Builds the full pairwise angular-error matrix between true jammers
           and predicted jammer-action directions.
        3. Finds the minimum-error assignment over all permutations.
        4. Returns the matched errors under that best assignment.

        Since max_jammers is fixed to 3 in this environment, exhaustive
        permutation search is deterministic, dependency-free, and cheap.
        """

        active_count = int(self.num_active_jammers)

        if active_count == 0:
            return []

        if len(self.current_jammer_thetas_rad) < active_count:
            return [float("nan")] * active_count

        if len(self.current_jammer_phis_rad) < active_count:
            return [float("nan")] * active_count

        if len(jammer_action_directions_rad) < active_count:
            return [float("nan")] * active_count

        true_jammer_directions_rad = [
            (
                self.current_jammer_thetas_rad[jammer_idx],
                self.current_jammer_phis_rad[jammer_idx],
            )
            for jammer_idx in range(active_count)
        ]

        predicted_jammer_directions_rad = jammer_action_directions_rad[
            :active_count
        ]

        error_matrix = self._build_jammer_assignment_error_matrix_deg(
            true_jammer_directions_rad=true_jammer_directions_rad,
            predicted_jammer_directions_rad=predicted_jammer_directions_rad,
        )

        best_assignment = self._find_minimum_jammer_assignment(
            error_matrix_deg=error_matrix,
        )

        matched_errors_deg = [
            float(error_matrix[true_idx, pred_idx])
            for true_idx, pred_idx in enumerate(best_assignment)
        ]

        return matched_errors_deg

    def _build_jammer_assignment_error_matrix_deg(
        self,
        true_jammer_directions_rad: list[tuple[float, float]],
        predicted_jammer_directions_rad: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Build the pairwise angular-error matrix for jammer assignment.

        Rows correspond to true active jammers.
        Columns correspond to predicted active jammer directions.
        """

        active_count = len(true_jammer_directions_rad)

        if active_count != len(predicted_jammer_directions_rad):
            raise ValueError(
                "true_jammer_directions_rad and "
                "predicted_jammer_directions_rad must have the same length."
            )

        error_matrix = np.zeros(
            (active_count, active_count),
            dtype=float,
        )

        for true_idx, (theta_true_rad, phi_true_rad) in enumerate(
            true_jammer_directions_rad
        ):
            for pred_idx, (theta_pred_rad, phi_pred_rad) in enumerate(
                predicted_jammer_directions_rad
            ):
                error_matrix[true_idx, pred_idx] = self._compute_angular_error_deg(
                    theta_a_rad=theta_true_rad,
                    phi_a_rad=phi_true_rad,
                    theta_b_rad=theta_pred_rad,
                    phi_b_rad=phi_pred_rad,
                )

        return error_matrix

    def _find_minimum_jammer_assignment(
        self,
        error_matrix_deg: np.ndarray,
    ) -> tuple[int, ...]:
        """
        Find the minimum-mean-error assignment between true jammers and
        predicted jammer directions.

        Returns
        -------
        tuple[int, ...]
            assignment[true_idx] = predicted_idx
        """

        error_matrix_deg = np.asarray(error_matrix_deg, dtype=float)

        if error_matrix_deg.ndim != 2:
            raise ValueError("error_matrix_deg must be a 2D array.")

        num_true, num_pred = error_matrix_deg.shape

        if num_true != num_pred:
            raise ValueError(
                "error_matrix_deg must be square for jammer assignment."
            )

        if num_true == 0:
            return tuple()

        best_assignment: tuple[int, ...] | None = None
        best_cost = float("inf")

        for assignment in itertools.permutations(range(num_pred)):
            assignment_errors = np.array(
                [
                    error_matrix_deg[true_idx, pred_idx]
                    for true_idx, pred_idx in enumerate(assignment)
                ],
                dtype=float,
            )

            if np.all(np.isfinite(assignment_errors)):
                cost = float(np.mean(assignment_errors))
            else:
                cost = float("inf")

            if cost < best_cost:
                best_cost = cost
                best_assignment = tuple(int(index) for index in assignment)

        if best_assignment is None:
            return tuple(range(num_true))

        return best_assignment

    def _compute_angular_error_deg(
        self,
        theta_a_rad: float,
        phi_a_rad: float,
        theta_b_rad: float,
        phi_b_rad: float,
    ) -> float:
        """
        Compute the 3D angular separation between two directions.
        """

        u_a = self._angles_rad_to_unit_vector(theta_a_rad, phi_a_rad)
        u_b = self._angles_rad_to_unit_vector(theta_b_rad, phi_b_rad)

        dot_product = float(np.dot(u_a, u_b))
        dot_product = np.clip(dot_product, -1.0, 1.0)

        angle_error_rad = np.arccos(dot_product)

        return float(np.rad2deg(angle_error_rad))

    def _compute_angle_loss(self, angle_error_deg: float) -> float:
        """
        Compute normalized squared angular loss.
        """

        if not np.isfinite(angle_error_deg):
            return 1.0

        normalized_angle_error = float(angle_error_deg) / 180.0

        return float(normalized_angle_error**2)

    def _compute_jammer_angle_loss(
        self,
        jammer_action_errors_deg: list[float],
    ) -> float:
        """
        Compute the mean normalized squared angular loss for active jammers.
        """

        if self.num_active_jammers == 0:
            return 0.0

        if len(jammer_action_errors_deg) == 0:
            return 1.0

        losses = []

        for error_deg in jammer_action_errors_deg:
            if np.isfinite(error_deg):
                losses.append((float(error_deg) / 180.0) ** 2)
            else:
                losses.append(1.0)

        if len(losses) == 0:
            return 1.0

        return float(np.mean(losses))

    def _angles_rad_to_unit_vector(
        self,
        theta_rad: float,
        phi_rad: float,
    ) -> np.ndarray:
        """
        Convert angular coordinates in radians to a 3D unit vector.
        """

        ux = np.sin(theta_rad) * np.cos(phi_rad)
        uy = np.sin(theta_rad) * np.sin(phi_rad)
        uz = np.cos(theta_rad)

        return np.array([ux, uy, uz], dtype=float)

    def _safe_nanmean(self, values: list[float]) -> float:
        """
        Compute nanmean safely.
        """

        if len(values) == 0:
            return float("nan")

        values_array = np.asarray(values, dtype=float)

        if np.all(np.isnan(values_array)):
            return float("nan")

        return float(np.nanmean(values_array))

    def _safe_nanmax(self, values: list[float]) -> float:
        """
        Compute nanmax safely.
        """

        if len(values) == 0:
            return float("nan")

        values_array = np.asarray(values, dtype=float)

        if np.all(np.isnan(values_array)):
            return float("nan")

        return float(np.nanmax(values_array))

    def _safe_array_mean(self, values: np.ndarray) -> float:
        """
        Compute finite/nan-safe array mean.
        """

        values = np.asarray(values, dtype=float)

        if values.size == 0:
            return float("nan")

        if np.all(np.isnan(values)):
            return float("nan")

        return float(np.nanmean(values))

    def _safe_array_min(self, values: np.ndarray) -> float:
        """
        Compute finite/nan-safe array minimum.
        """

        values = np.asarray(values, dtype=float)

        if values.size == 0:
            return float("nan")

        if np.all(np.isnan(values)):
            return float("nan")

        return float(np.nanmin(values))

    def _safe_array_max(self, values: np.ndarray) -> float:
        """
        Compute finite/nan-safe array maximum.
        """

        values = np.asarray(values, dtype=float)

        if values.size == 0:
            return float("nan")

        if np.all(np.isnan(values)):
            return float("nan")

        return float(np.nanmax(values))

    def _normalize_theta(self, theta_rad: float) -> float:
        return (theta_rad - self.theta_min) / (self.theta_max - self.theta_min)

    def _normalize_phi(self, phi_rad: float) -> float:
        return (phi_rad - self.phi_min) / (self.phi_max - self.phi_min)

    def _denormalize_theta(self, theta_norm: float) -> float:
        return self.theta_min + theta_norm * (self.theta_max - self.theta_min)

    def _denormalize_phi(self, phi_norm: float) -> float:
        return self.phi_min + phi_norm * (self.phi_max - self.phi_min)

    def _normalize_jammer_theta(self, theta_rad: float) -> float:
        return (
            (theta_rad - self.jammer_theta_min)
            / (self.jammer_theta_max - self.jammer_theta_min)
        )

    def _normalize_jammer_phi(self, phi_rad: float) -> float:
        return (
            (phi_rad - self.jammer_phi_min)
            / (self.jammer_phi_max - self.jammer_phi_min)
        )

    def _denormalize_jammer_theta(self, theta_norm: float) -> float:
        return self.jammer_theta_min + theta_norm * (
            self.jammer_theta_max - self.jammer_theta_min
        )

    def _denormalize_jammer_phi(self, phi_norm: float) -> float:
        return self.jammer_phi_min + phi_norm * (
            self.jammer_phi_max - self.jammer_phi_min
        )