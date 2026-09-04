from __future__ import annotations

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
    Gymnasium environment for Phase 5 cognitive beamforming under dynamic
    multi-jammer scenarios.

    This environment is intended for the final DRL training phase before
    MTT/perception integration.

    Design principle
    ----------------
    The environment does not use ScenarioGenerator internally.

    At every reset(), the environment samples:
    - one random static signal of interest (SOI) direction,
    - either a fixed or randomly selected number of active jammer directions,
    - a dynamic angular motion model for the active jammers.

    Compared with Phase 4, this environment introduces one main change:

    1. Multi-step episodes:
       the SOI remains static during the episode, while jammer DOAs evolve
       over time according to a configurable motion model.

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

        Conventional steering weights are generated only from the SOI
        action direction. Jammer action directions are stored in info but
        are not used to build the weights.

    beamforming_mode = "nulling"

        Deterministic multi-jammer nulling weights are generated using
        the SOI action direction and all active jammer action directions.

    beamforming_mode = "mvdr"

        MVDR weights are generated using the SOI action direction and an
        interference-plus-noise covariance matrix constructed from all
        active jammer action directions.

    Jammer motion modes
    -------------------
    jammer_motion_mode = "linear_angular"

        Each jammer has a sampled angular velocity. Velocities remain
        constant during the episode, except for boundary reflection.

    jammer_motion_mode = "random_walk"

        Jammer DOAs are updated using angular Gaussian increments at every
        step.

    jammer_motion_mode = "maneuvering"

        Jammer velocities are resampled every maneuver_interval_steps.

    Reward definition
    -----------------
    Conceptually:

        reward = alpha * sinr_db
                 - beta * clipped_sinr_loss_db
                 - gamma * angle_loss
                 - delta * action_change_penalty

    The SINR-loss reference can be configured with reference_mode:
    - "steering": perfect conventional steering toward the true SOI.
    - "same_beamforming_mode": perfect reference using the same beamforming
      family as the agent, with true SOI and true active jammer directions.

    Only reward terms with non-zero coefficients are accumulated. This avoids
    numerical contamination from expressions such as 0 * NaN.

    Notes
    -----
    This is a multi-step environment:

        reset() -> sample static SOI and dynamic jammer initial conditions
        step()  -> evaluate action -> update jammer DOAs -> next state

    The action still parameterizes directions, not direct complex weights.
    Direct weight control is intentionally outside Phase 5 scope.
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
        observation_mode: str = "angles",
        action_mode: str = "angles",
        beamforming_mode: str = "steering",
        enforce_visible_hemisphere: bool = True,
        reward_alpha_sinr: float = 1.0,
        reward_beta_sinr_loss: float = 0.0,
        reward_gamma_angle: float = 0.0,
        reward_delta_action_change: float = 0.0,
        max_sinr_loss_db: float = 60.0,
        reference_mode: str = "same_beamforming_mode",
        mvdr_diagonal_loading: float = 1e-4,
        nulling_diagonal_loading: float = 1e-8,
        invalid_sinr_db: float = -120.0,
        invalid_value_penalty: float = -1_000.0,
        episode_length: int = 50,
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
        self.reward_gamma_angle = float(reward_gamma_angle)
        self.reward_delta_action_change = float(reward_delta_action_change)
        self.max_sinr_loss_db = float(max_sinr_loss_db)
        self.reference_mode = str(reference_mode)

        self.mvdr_diagonal_loading = float(mvdr_diagonal_loading)
        self.nulling_diagonal_loading = float(nulling_diagonal_loading)

        self.invalid_sinr_db = float(invalid_sinr_db)
        self.invalid_value_penalty = float(invalid_value_penalty)

        self.episode_length = int(episode_length)
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
        self.previous_action: np.ndarray | None = None

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
        self.previous_action = None

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
        Evaluate one action at the current time step, then update jammer DOAs.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized.")

        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        state_before_update = self.current_state.copy()
        step_index = int(self.current_step)

        (
            theta_soi_action_rad,
            phi_soi_action_rad,
            jammer_action_directions_rad,
        ) = self._action_to_angles(action)

        numerical_error = False
        weights_are_finite = True
        sinr_is_finite = True
        reference_sinr_is_finite = True
        reward_is_finite = True

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

        target_direction_deg = (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.rad2deg(self.current_phi_rad)),
        )

        jammer_directions_deg = self._get_current_jammer_directions_deg()

        try:
            sinr_db = compute_sinr(
                weights=self.array.W,
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

        angle_error_deg = soi_angle_error_deg
        angle_loss = self._compute_angle_loss(angle_error_deg)

        jammer_action_errors_deg = self._compute_jammer_action_errors_deg(
            jammer_action_directions_rad=jammer_action_directions_rad,
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

        action_change_penalty = self._compute_action_change_penalty(action)

        reward = self._compute_reward(
            sinr_db=sinr_db,
            clipped_sinr_loss_db=clipped_sinr_loss_db,
            angle_loss=angle_loss,
            action_change_penalty=action_change_penalty,
        )

        reward_is_finite = bool(np.isfinite(reward))

        if not reward_is_finite:
            reward = self.invalid_value_penalty
            numerical_error = True

        reward = float(reward)

        self.previous_action = action.copy()

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
            "reward_gamma_angle": self.reward_gamma_angle,
            "reward_delta_action_change": self.reward_delta_action_change,
            "beamforming_mode": self.beamforming_mode,
            "reference_mode": self.reference_mode,
            "jammer_motion_mode": self.jammer_motion_mode,
            "dynamic_jammer_boundary_mode": self.dynamic_jammer_boundary_mode,
            "step_index": step_index,
            "episode_length": self.episode_length,
            "sinr_db": sinr_db,
            "reference_sinr_db": reference_sinr_db,
            "sinr_loss_db": sinr_loss_db,
            "clipped_sinr_loss_db": clipped_sinr_loss_db,
            "angle_error_deg": angle_error_deg,
            "soi_angle_error_deg": soi_angle_error_deg,
            "jammer_action_errors_deg": jammer_action_errors_deg,
            "jammer_action_error_mean_deg": jammer_action_error_mean_deg,
            "jammer_action_error_max_deg": jammer_action_error_max_deg,
            "angle_loss": angle_loss,
            "action_change_penalty": action_change_penalty,
            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,
            "theta_target_deg": float(np.rad2deg(self.current_theta_rad)),
            "phi_target_deg": float(np.rad2deg(self.current_phi_rad)),
            "theta_steer_rad": theta_soi_action_rad,
            "phi_steer_rad": phi_soi_action_rad,
            "theta_steer_deg": float(np.rad2deg(theta_soi_action_rad)),
            "phi_steer_deg": float(np.rad2deg(phi_soi_action_rad)),
            "theta_soi_action_rad": theta_soi_action_rad,
            "phi_soi_action_rad": phi_soi_action_rad,
            "theta_soi_action_deg": float(np.rad2deg(theta_soi_action_rad)),
            "phi_soi_action_deg": float(np.rad2deg(phi_soi_action_rad)),
            "jammer_action_directions_rad": jammer_action_directions_rad.copy(),
            "jammer_action_directions_deg": jammer_action_directions_deg,
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
            "jammers_powers": self.jammer_powers.copy(),
            "weights": self.array.W.copy(),
            "state_before_update": state_before_update,
            "numerical_error": numerical_error,
            "weights_are_finite": weights_are_finite,
            "sinr_is_finite": sinr_is_finite,
            "reference_sinr_is_finite": reference_sinr_is_finite,
            "reward_is_finite": reward_is_finite,
        }

        self.current_step += 1

        terminated = bool(self.current_step >= self.episode_length)
        truncated = False

        if not terminated:
            self._update_dynamic_jammers()

            next_state = self._build_state(
                theta_target_rad=self.current_theta_rad,
                phi_target_rad=self.current_phi_rad,
                jammer_thetas_rad=self.current_jammer_thetas_rad,
                jammer_phis_rad=self.current_jammer_phis_rad,
            )
        else:
            next_state = self.current_state.copy()

        self.current_state = next_state

        info["next_jammer_thetas_rad"] = self.current_jammer_thetas_rad.copy()
        info["next_jammer_phis_rad"] = self.current_jammer_phis_rad.copy()
        info["next_jammer_thetas_deg"] = [
            float(np.rad2deg(theta)) for theta in self.current_jammer_thetas_rad
        ]
        info["next_jammer_phis_deg"] = [
            float(np.rad2deg(phi)) for phi in self.current_jammer_phis_rad
        ]
        info["next_state"] = next_state.copy()

        return next_state, reward, terminated, truncated, info

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
            if self.current_step % self.maneuver_interval_steps == 0:
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

        raise RuntimeError("Invalid dynamic jammer boundary mode.")

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

        for _ in range(4):
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

        raise RuntimeError("Invalid observation mode.")

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

        raise RuntimeError("Invalid action mode.")

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

    def _compute_action_change_penalty(self, action: np.ndarray) -> float:
        """
        Compute an action-domain smoothness penalty.
        """

        if self.previous_action is None:
            return 0.0

        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        previous_action = np.asarray(self.previous_action, dtype=np.float32).reshape(
            self.action_dim
        )

        delta_action = action - previous_action

        return float(np.mean(delta_action**2))

    def _build_beamforming_weights(
        self,
        theta_soi_action_rad: float,
        phi_soi_action_rad: float,
        jammer_action_directions_rad: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Build beamforming weights from the multi-jammer action.
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

        raise RuntimeError("Invalid beamforming mode.")

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
        Build an interference-plus-noise covariance matrix from all active
        jammer action directions.
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
            raise RuntimeError("Invalid reference mode.")

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

        raise RuntimeError("Invalid beamforming mode.")

    def _compute_reference_steering_sinr_db(self) -> float:
        """
        Compute SINR using conventional steering exactly towards the true SOI.

        This method is kept for backward compatibility. The main step() path
        uses _compute_reference_sinr_db(), which respects reference_mode.
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
        angle_loss: float,
        action_change_penalty: float,
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

        if self.reward_gamma_angle != 0.0:
            if np.isfinite(angle_loss):
                reward -= self.reward_gamma_angle * float(angle_loss)
            else:
                invalid_term_detected = True

        if self.reward_delta_action_change != 0.0:
            if np.isfinite(action_change_penalty):
                reward -= self.reward_delta_action_change * float(
                    action_change_penalty
                )
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
        Compute angular errors between true active jammers and predicted
        jammer action directions.

        Inactive jammer slots are ignored.
        """

        errors_deg: list[float] = []

        for jammer_idx in range(self.num_active_jammers):
            if jammer_idx >= len(self.current_jammer_thetas_rad):
                errors_deg.append(float("nan"))
                continue

            if jammer_idx >= len(jammer_action_directions_rad):
                errors_deg.append(float("nan"))
                continue

            theta_true_rad = self.current_jammer_thetas_rad[jammer_idx]
            phi_true_rad = self.current_jammer_phis_rad[jammer_idx]

            theta_action_rad, phi_action_rad = jammer_action_directions_rad[
                jammer_idx
            ]

            error_deg = self._compute_angular_error_deg(
                theta_a_rad=theta_true_rad,
                phi_a_rad=phi_true_rad,
                theta_b_rad=theta_action_rad,
                phi_b_rad=phi_action_rad,
            )

            errors_deg.append(error_deg)

        return errors_deg

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