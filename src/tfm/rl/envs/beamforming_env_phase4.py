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


class BeamformingEnvPhase4(gym.Env):
    """
    Gymnasium environment for Phase 4 cognitive beamforming under static
    multi-jammer scenarios.

    This environment is intended only for DRL training.

    Design principle
    ----------------
    The environment does not use ScenarioGenerator internally.

    At every reset(), the environment samples:
    - one random signal of interest (SOI) direction,
    - a configurable number of static jammer directions.

    Compared with Phase 3, this environment introduces two main changes:

    1. Multi-jammer action support:
       the agent outputs one SOI direction and up to max_jammers jammer
       directions.

    2. Numerical robustness:
       reward computation avoids 0 * NaN propagation, and invalid numerical
       values in weights, SINR or reward are converted into finite penalized
       outputs.

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

    Reward definition
    -----------------
    Conceptually:

        reward = alpha * sinr_db
                 - beta * clipped_sinr_loss_db
                 - gamma_soi * soi_angle_loss
                 - gamma_jammer * jammer_angle_loss
                 + milestone_bonus

    where:

        soi_angle_loss = (soi_angle_error_deg / 180)^2

        jammer_angle_loss =
            mean_j [(jammer_angle_error_j_deg / 180)^2]

    If there are no active jammers, jammer_angle_loss = 0.

    Optional milestone bonuses can be enabled to add bounded positive rewards
    when the SOI error, the multi-jammer error, or the SINR loss are below
    configured thresholds. By default all bonus coefficients are zero, so the
    environment behaves as a purely continuous-reward environment unless
    explicitly configured.

    Only reward terms with non-zero coefficients are accumulated. This avoids
    numerical contamination from expressions such as 0 * NaN.

    Reference modes
    ---------------
    reference_mode = "steering"

        The SINR-loss reference is perfect conventional steering toward the
        true SOI.

    reference_mode = "same_beamforming_mode"

        The SINR-loss reference uses the same beamforming family as the agent,
        with true SOI and true active jammer directions.

    Notes
    -----
    This is a one-step environment:

        reset() -> sample SOI and jammer DOAs
        step()  -> evaluate one multi-jammer action -> terminated = True
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
        reward_alpha_sinr: float = 0.0,
        reward_beta_sinr_loss: float = 1.0,
        reward_gamma_soi_angle: float = 0.0,
        reward_gamma_jammer_angle: float = 0.0,
        reward_gamma_angle: float | None = None,
        reward_bonus_good_soi: float = 0.0,
        reward_bonus_good_jammer: float = 0.0,
        reward_bonus_good_sinr_loss: float = 0.0,
        soi_error_bonus_threshold_deg: float = 3.0,
        jammer_error_bonus_threshold_deg: float = 5.0,
        sinr_loss_bonus_threshold_db: float = 1.0,
        max_sinr_loss_db: float = 60.0,
        reference_mode: str = "same_beamforming_mode",
        mvdr_diagonal_loading: float = 1e-4,
        nulling_diagonal_loading: float = 1e-8,
        invalid_sinr_db: float = -120.0,
        invalid_value_penalty: float = -1_000.0,
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

        # Backward-compatible alias: old notebooks may still use
        # reward_gamma_angle for the SOI angular loss. New notebooks should
        # use reward_gamma_soi_angle explicitly.
        if reward_gamma_angle is not None:
            self.reward_gamma_soi_angle = float(reward_gamma_angle)
        else:
            self.reward_gamma_soi_angle = float(reward_gamma_soi_angle)

        self.reward_gamma_jammer_angle = float(reward_gamma_jammer_angle)

        self.reward_bonus_good_soi = float(reward_bonus_good_soi)
        self.reward_bonus_good_jammer = float(reward_bonus_good_jammer)
        self.reward_bonus_good_sinr_loss = float(reward_bonus_good_sinr_loss)
        self.soi_error_bonus_threshold_deg = float(soi_error_bonus_threshold_deg)
        self.jammer_error_bonus_threshold_deg = float(jammer_error_bonus_threshold_deg)
        self.sinr_loss_bonus_threshold_db = float(sinr_loss_bonus_threshold_db)

        self.max_sinr_loss_db = float(max_sinr_loss_db)
        self.reference_mode = str(reference_mode)

        self.mvdr_diagonal_loading = float(mvdr_diagonal_loading)
        self.nulling_diagonal_loading = float(nulling_diagonal_loading)

        self.invalid_sinr_db = float(invalid_sinr_db)
        self.invalid_value_penalty = float(invalid_value_penalty)

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

        if self.soi_error_bonus_threshold_deg < 0.0:
            raise ValueError("soi_error_bonus_threshold_deg must be non-negative.")

        if self.jammer_error_bonus_threshold_deg < 0.0:
            raise ValueError("jammer_error_bonus_threshold_deg must be non-negative.")

        if self.sinr_loss_bonus_threshold_db < 0.0:
            raise ValueError("sinr_loss_bonus_threshold_db must be non-negative.")

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

        self.jammer_powers_config = (
            None if jammer_powers is None else [float(power) for power in jammer_powers]
        )
        self.jammer_powers = self._build_jammer_powers_for_active_count(
            self.num_active_jammers
        )

        # ============================================================
        # Observation space
        # ============================================================

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

        # ============================================================
        # Action space
        # ============================================================

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

        # ============================================================
        # Episode variables
        # ============================================================

        self.current_theta_rad: float | None = None
        self.current_phi_rad: float | None = None

        self.current_jammer_thetas_rad: list[float] = []
        self.current_jammer_phis_rad: list[float] = []

        self.current_state: np.ndarray | None = None

    # ============================================================
    # Gym API
    # ============================================================

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment and sample a new random static angular scenario.
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

        state = self._build_state(
            theta_target_rad=theta_rad,
            phi_target_rad=phi_rad,
            jammer_thetas_rad=jammer_thetas_rad,
            jammer_phis_rad=jammer_phis_rad,
        )

        self.current_state = state

        jammer_directions_deg = self._get_current_jammer_directions_deg()

        info = {
            "theta_target_rad": theta_rad,
            "phi_target_rad": phi_rad,
            "theta_target_deg": float(np.rad2deg(theta_rad)),
            "phi_target_deg": float(np.rad2deg(phi_rad)),
            "num_active_jammers": self.num_active_jammers,
            "jammer_thetas_rad": jammer_thetas_rad.copy(),
            "jammer_phis_rad": jammer_phis_rad.copy(),
            "jammer_thetas_deg": [
                float(np.rad2deg(theta)) for theta in jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(phi)) for phi in jammer_phis_rad
            ],
            "jammers_directions_deg": jammer_directions_deg,
            "jammers_powers": self.jammer_powers.copy(),
            "reference_mode": self.reference_mode,
            "beamforming_mode": self.beamforming_mode,
            "reward_bonus_good_soi": self.reward_bonus_good_soi,
            "reward_bonus_good_jammer": self.reward_bonus_good_jammer,
            "reward_bonus_good_sinr_loss": self.reward_bonus_good_sinr_loss,
            "soi_error_bonus_threshold_deg": self.soi_error_bonus_threshold_deg,
            "jammer_error_bonus_threshold_deg": self.jammer_error_bonus_threshold_deg,
            "sinr_loss_bonus_threshold_db": self.sinr_loss_bonus_threshold_db,
        }

        return state, info

    def step(self, action: np.ndarray):
        """
        Evaluate one multi-jammer action and terminate the episode.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized.")

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
        soi_angle_loss = self._compute_angle_loss(soi_angle_error_deg)
        angle_loss = soi_angle_loss

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

        milestone_bonus, milestone_info = self._compute_milestone_bonus(
            soi_angle_error_deg=soi_angle_error_deg,
            jammer_action_error_mean_deg=jammer_action_error_mean_deg,
            jammer_action_error_max_deg=jammer_action_error_max_deg,
            clipped_sinr_loss_db=clipped_sinr_loss_db,
        )

        reward = self._compute_reward(
            sinr_db=sinr_db,
            clipped_sinr_loss_db=clipped_sinr_loss_db,
            soi_angle_loss=soi_angle_loss,
            jammer_angle_loss=jammer_angle_loss,
            milestone_bonus=milestone_bonus,
        )

        reward_is_finite = bool(np.isfinite(reward))

        if not reward_is_finite:
            reward = self.invalid_value_penalty
            numerical_error = True

        reward = float(reward)

        terminated = True
        truncated = False

        next_state = self.current_state.copy()

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
            "reward_gamma_angle": self.reward_gamma_soi_angle,
            "reward_bonus_good_soi": self.reward_bonus_good_soi,
            "reward_bonus_good_jammer": self.reward_bonus_good_jammer,
            "reward_bonus_good_sinr_loss": self.reward_bonus_good_sinr_loss,
            "soi_error_bonus_threshold_deg": self.soi_error_bonus_threshold_deg,
            "jammer_error_bonus_threshold_deg": self.jammer_error_bonus_threshold_deg,
            "sinr_loss_bonus_threshold_db": self.sinr_loss_bonus_threshold_db,
            "milestone_bonus": float(milestone_bonus),
            **milestone_info,
            "beamforming_mode": self.beamforming_mode,
            "reference_mode": self.reference_mode,
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
            "soi_angle_loss": soi_angle_loss,
            "jammer_angle_loss": jammer_angle_loss,
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
            "jammers_powers": self.jammer_powers.copy(),
            "weights": self.array.W.copy(),
            "numerical_error": numerical_error,
            "weights_are_finite": weights_are_finite,
            "sinr_is_finite": sinr_is_finite,
            "reference_sinr_is_finite": reference_sinr_is_finite,
            "reward_is_finite": reward_is_finite,
        }

        return next_state, reward, terminated, truncated, info

    # ============================================================
    # Internal helpers: random DOA generation
    # ============================================================

    def _sample_num_active_jammers_for_episode(self) -> None:
        """
        Sample the number of active jammers for the current one-step episode
        when active_jammers_choices is configured.
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
        Sample a random SOI DOA.
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
        Sample random jammer DOAs with minimum angular separation from
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

    # ============================================================
    # Internal helpers: state and action
    # ============================================================

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

        Returns
        -------
        theta_soi_action_rad, phi_soi_action_rad, jammer_action_directions_rad
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

    # ============================================================
    # Internal helpers: beamforming and reward
    # ============================================================

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

    def _compute_milestone_bonus(
        self,
        soi_angle_error_deg: float,
        jammer_action_error_mean_deg: float,
        jammer_action_error_max_deg: float,
        clipped_sinr_loss_db: float,
    ) -> tuple[float, dict]:
        """
        Compute optional bounded milestone bonuses.

        For the multi-jammer case, the main jammer milestone is deliberately
        strict: all active jammers must be below the configured angular
        threshold, which is equivalent to checking the maximum matched jammer
        error. The mean condition is also reported for diagnostics.
        """

        good_soi = bool(
            np.isfinite(soi_angle_error_deg)
            and float(soi_angle_error_deg) <= self.soi_error_bonus_threshold_deg
        )

        if self.num_active_jammers == 0:
            good_jammer_mean = False
            good_jammer_all = False
        else:
            good_jammer_mean = bool(
                np.isfinite(jammer_action_error_mean_deg)
                and float(jammer_action_error_mean_deg)
                <= self.jammer_error_bonus_threshold_deg
            )

            good_jammer_all = bool(
                np.isfinite(jammer_action_error_max_deg)
                and float(jammer_action_error_max_deg)
                <= self.jammer_error_bonus_threshold_deg
            )

        good_sinr_loss = bool(
            np.isfinite(clipped_sinr_loss_db)
            and float(clipped_sinr_loss_db) <= self.sinr_loss_bonus_threshold_db
        )

        bonus = 0.0

        if good_soi:
            bonus += self.reward_bonus_good_soi

        if good_jammer_all:
            bonus += self.reward_bonus_good_jammer

        if good_sinr_loss:
            bonus += self.reward_bonus_good_sinr_loss

        info = {
            "milestone_good_soi": good_soi,
            "milestone_good_jammer": good_jammer_all,
            "milestone_good_jammer_mean": good_jammer_mean,
            "milestone_good_jammer_all": good_jammer_all,
            "milestone_good_sinr_loss": good_sinr_loss,
        }

        return float(bonus), info

    def _compute_reward(
        self,
        sinr_db: float,
        clipped_sinr_loss_db: float,
        soi_angle_loss: float,
        jammer_angle_loss: float,
        milestone_bonus: float = 0.0,
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

        if milestone_bonus != 0.0:
            if np.isfinite(milestone_bonus):
                reward += float(milestone_bonus)
            else:
                invalid_term_detected = True

        if invalid_term_detected:
            reward += self.invalid_value_penalty

        if not np.isfinite(reward):
            reward = self.invalid_value_penalty

        return float(reward)

    # ============================================================
    # Internal helpers: geometry
    # ============================================================

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

        This is needed because the physical set of jammer directions is
        permutation-invariant: if the agent predicts the correct jammer
        directions but swaps two action slots, nulling/MVDR can still place
        nulls in the right directions. A slot-wise loss would penalize that
        physically valid solution.
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

    # ============================================================
    # Internal helpers: normalization
    # ============================================================

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