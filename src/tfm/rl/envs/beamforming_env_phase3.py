from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tfm.physics.narrow_band.phased_array_nb import Phased_Array_NB
from tfm.physics.narrow_band.weights_deterministic_nb import (
    interference_suppression_weights,
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


class BeamformingEnvPhase3(gym.Env):
    """
    Gymnasium environment for Phase 3 cognitive digital beamforming under
    one static jammer.

    This corrected Phase 3 environment keeps the original Phase 3 structure:

        reset() -> sample one static SOI and one static jammer
        step()  -> evaluate one dual action and terminate the episode

    but incorporates the key corrections later introduced in Phase 5:

    1. Separate angular loss terms for the SOI and jammer.
    2. Optional same-mode reference SINR for nulling/MVDR training.
    3. Robust reward computation that avoids 0 * NaN contamination.
    4. Safe fallback weights when a beamforming computation becomes invalid.

    Observation modes
    -----------------
    observation_mode = "angles"

        state = [
            theta_soi_norm,
            phi_soi_norm,
            theta_j1_norm, phi_j1_norm, m1,
            theta_j2_norm, phi_j2_norm, m2,
            theta_j3_norm, phi_j3_norm, m3,
        ]

        Observation dimension: 11.

    observation_mode = "unit_vector"

        state = [
            u_soi_x, u_soi_y, u_soi_z,
            u_j1_x, u_j1_y, u_j1_z, m1,
            u_j2_x, u_j2_y, u_j2_z, m2,
            u_j3_x, u_j3_y, u_j3_z, m3,
        ]

        Observation dimension: 15.

    Action modes
    ------------
    action_mode = "angles"

        action = [
            theta_soi_action_norm,
            phi_soi_action_norm,
            theta_jammer_action_norm,
            phi_jammer_action_norm,
        ]

        Action dimension: 4.

    action_mode = "unit_vector"

        action = [
            u_soi_action_x,
            u_soi_action_y,
            u_soi_action_z,
            u_jammer_action_x,
            u_jammer_action_y,
            u_jammer_action_z,
        ]

        Action dimension: 6.

    Beamforming modes
    -----------------
    beamforming_mode = "steering"
        Conventional steering weights are generated only from the SOI action.
        The jammer action is diagnostic and does not affect the weights.

    beamforming_mode = "nulling"
        Deterministic interference suppression weights are generated from the
        SOI action and jammer action.

    beamforming_mode = "mvdr"
        MVDR weights are generated from the SOI action and an
        interference-plus-noise covariance built from the jammer action.

    Reward definition
    -----------------
    reward = alpha * sinr_db
             - beta * clipped_sinr_loss_db
             - gamma_soi * soi_angle_loss
             - gamma_jammer * jammer_angle_loss
             + milestone_bonus

    where:

        soi_angle_loss = (soi_angle_error_deg / 180)^2
        jammer_angle_loss = (jammer_action_error_deg / 180)^2

    If there is no active jammer, jammer_angle_loss = 0.

    Optional milestone bonuses can be enabled to add bounded positive rewards
    when the SOI error, jammer error or SINR loss are below configured
    thresholds. By default all bonuses are zero, so the environment behaves as
    a purely continuous-reward environment unless explicitly configured.

    Reference modes
    ---------------
    reference_mode = "steering"
        The SINR-loss reference is perfect conventional steering toward the
        true SOI.

    reference_mode = "same_beamforming_mode"
        The SINR-loss reference uses the same beamforming family as the agent,
        with true SOI and true jammer directions.

    Notes
    -----
    This environment is intentionally static and one-step. Dynamic jammers,
    block control and multi-jammer actions belong to later phases.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        array: Phased_Array_NB,
        array_position: np.ndarray,
        desired_power: float = 1.0,
        noise_power: float = 1e-3,
        max_jammers: int = 3,
        num_active_jammers: int = 1,
        jammer_powers: list[float] | None = None,
        theta_limits_rad: tuple[float, float] = (0.0, np.pi / 2.0),
        phi_limits_rad: tuple[float, float] = (0.0, 2.0 * np.pi),
        jammer_theta_limits_rad: tuple[float, float] = (0.0, np.pi / 2.0),
        jammer_phi_limits_rad: tuple[float, float] = (0.0, 2.0 * np.pi),
        min_target_jammer_separation_deg: float = 5.0,
        observation_mode: str = "unit_vector",
        action_mode: str = "unit_vector",
        beamforming_mode: str = "nulling",
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
        invalid_sinr_db: float = -120.0,
        invalid_value_penalty: float = -1_000.0,
    ) -> None:
        super().__init__()

        self.array = array
        self.array_position = np.asarray(array_position, dtype=float).reshape(3)

        self.desired_power = float(desired_power)
        self.noise_power = float(noise_power)

        self.max_jammers = int(max_jammers)
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
        self.reference_mode = str(reference_mode)
        self.enforce_visible_hemisphere = bool(enforce_visible_hemisphere)

        self.reward_alpha_sinr = float(reward_alpha_sinr)
        self.reward_beta_sinr_loss = float(reward_beta_sinr_loss)

        # Backward-compatible alias: old notebooks used reward_gamma_angle for
        # the SOI angular loss. New notebooks should use reward_gamma_soi_angle.
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

        self.mvdr_diagonal_loading = float(mvdr_diagonal_loading)
        self.invalid_sinr_db = float(invalid_sinr_db)
        self.invalid_value_penalty = float(invalid_value_penalty)

        if self.max_jammers != 3:
            raise ValueError(
                "This environment currently requires max_jammers=3 "
                "to match the fixed roadmap state format."
            )

        if self.num_active_jammers < 0:
            raise ValueError("num_active_jammers must be non-negative.")

        if self.num_active_jammers > self.max_jammers:
            raise ValueError("num_active_jammers cannot exceed max_jammers.")

        if self.num_active_jammers > 1:
            raise ValueError(
                "BeamformingEnvPhase3 is intended for at most one active jammer. "
                "Use a later-phase environment for multi-jammer training."
            )

        if self.beamforming_mode not in ["steering", "nulling", "mvdr"]:
            raise ValueError(
                "Unknown beamforming_mode. Expected one of: "
                "'steering', 'nulling', 'mvdr'."
            )

        if self.reference_mode not in ["steering", "same_beamforming_mode"]:
            raise ValueError(
                "Unknown reference_mode. Expected one of: "
                "'steering', 'same_beamforming_mode'."
            )

        if self.observation_mode not in ["angles", "unit_vector"]:
            raise ValueError(
                "Unknown observation_mode. Expected one of: 'angles', 'unit_vector'."
            )

        if self.action_mode not in ["angles", "unit_vector"]:
            raise ValueError(
                "Unknown action_mode. Expected one of: 'angles', 'unit_vector'."
            )

        if self.soi_error_bonus_threshold_deg < 0.0:
            raise ValueError("soi_error_bonus_threshold_deg must be non-negative.")

        if self.jammer_error_bonus_threshold_deg < 0.0:
            raise ValueError("jammer_error_bonus_threshold_deg must be non-negative.")

        if self.sinr_loss_bonus_threshold_db < 0.0:
            raise ValueError("sinr_loss_bonus_threshold_db must be non-negative.")

        if jammer_powers is None:
            self.jammer_powers = [1.0] * self.num_active_jammers
        else:
            self.jammer_powers = [float(power) for power in jammer_powers]

        if len(self.jammer_powers) != self.num_active_jammers:
            raise ValueError("Length of jammer_powers must match num_active_jammers.")

        if self.observation_mode == "angles":
            self.observation_dim = 2 + 3 * self.max_jammers
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.observation_dim,),
                dtype=np.float32,
            )
        else:
            self.observation_dim = 3 + 4 * self.max_jammers
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.observation_dim,),
                dtype=np.float32,
            )

        if self.action_mode == "angles":
            self.action_dim = 4
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(self.action_dim,),
                dtype=np.float32,
            )
        else:
            self.action_dim = 6
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
        self.current_state: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment and sample a new static angular scene."""

        super().reset(seed=seed)

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

        info = self._build_reset_info()
        return state, info

    def step(self, action: np.ndarray):
        """Evaluate one dual action and terminate the episode."""

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized.")

        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        (
            theta_soi_action_rad,
            phi_soi_action_rad,
            theta_jammer_action_rad,
            phi_jammer_action_rad,
        ) = self._action_to_angles(action)

        numerical_error = False

        try:
            weights = self._build_beamforming_weights(
                theta_soi_action_rad=theta_soi_action_rad,
                phi_soi_action_rad=phi_soi_action_rad,
                theta_jammer_action_rad=theta_jammer_action_rad,
                phi_jammer_action_rad=phi_jammer_action_rad,
            )
        except Exception:
            weights = self._build_safe_fallback_weights()
            numerical_error = True

        weights_are_finite = bool(np.all(np.isfinite(weights)))
        if not weights_are_finite:
            weights = self._build_safe_fallback_weights()
            numerical_error = True

        self.array.set_weights(weights)

        sinr_db = self._compute_sinr_for_weights(weights)
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

        jammer_action_error_deg = self._compute_first_jammer_action_error_deg(
            theta_jammer_action_rad=theta_jammer_action_rad,
            phi_jammer_action_rad=phi_jammer_action_rad,
        )
        jammer_angle_loss = self._compute_jammer_angle_loss(jammer_action_error_deg)

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
            jammer_action_error_deg=jammer_action_error_deg,
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

        terminated = True
        truncated = False
        next_state = self.current_state.copy()

        jammer_directions_deg = self._get_current_jammer_directions_deg()

        info = {
            "reward": float(reward),
            "reward_alpha_sinr": self.reward_alpha_sinr,
            "reward_beta_sinr_loss": self.reward_beta_sinr_loss,
            "reward_gamma_soi_angle": self.reward_gamma_soi_angle,
            "reward_gamma_jammer_angle": self.reward_gamma_jammer_angle,
            "reward_bonus_good_soi": self.reward_bonus_good_soi,
            "reward_bonus_good_jammer": self.reward_bonus_good_jammer,
            "reward_bonus_good_sinr_loss": self.reward_bonus_good_sinr_loss,
            "soi_error_bonus_threshold_deg": self.soi_error_bonus_threshold_deg,
            "jammer_error_bonus_threshold_deg": self.jammer_error_bonus_threshold_deg,
            "sinr_loss_bonus_threshold_db": self.sinr_loss_bonus_threshold_db,
            "milestone_bonus": float(milestone_bonus),
            **milestone_info,
            # Backward-compatible key used by old notebooks.
            "reward_gamma_angle": self.reward_gamma_soi_angle,
            "beamforming_mode": self.beamforming_mode,
            "reference_mode": self.reference_mode,
            "sinr_db": float(sinr_db),
            "reference_sinr_db": float(reference_sinr_db),
            "reference_steering_sinr_db": float(self._compute_reference_steering_sinr_db()),
            "sinr_loss_db": float(sinr_loss_db),
            "clipped_sinr_loss_db": float(clipped_sinr_loss_db),
            "angle_error_deg": float(soi_angle_error_deg),
            "soi_angle_error_deg": float(soi_angle_error_deg),
            "jammer_action_error_deg": float(jammer_action_error_deg),
            "soi_angle_loss": float(soi_angle_loss),
            "jammer_angle_loss": float(jammer_angle_loss),
            # Backward-compatible key used by old notebooks.
            "angle_loss": float(soi_angle_loss),
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
            "theta_jammer_action_rad": theta_jammer_action_rad,
            "phi_jammer_action_rad": phi_jammer_action_rad,
            "theta_jammer_action_deg": float(np.rad2deg(theta_jammer_action_rad)),
            "phi_jammer_action_deg": float(np.rad2deg(phi_jammer_action_rad)),
            "num_active_jammers": self.num_active_jammers,
            "jammer_thetas_rad": self.current_jammer_thetas_rad.copy(),
            "jammer_phis_rad": self.current_jammer_phis_rad.copy(),
            "jammer_thetas_deg": [
                float(np.rad2deg(theta)) for theta in self.current_jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(phi)) for phi in self.current_jammer_phis_rad
            ],
            "jammers_directions_deg": jammer_directions_deg,
            "jammers_powers": self.jammer_powers.copy(),
            "weights": self.array.W.copy(),
            "weights_are_finite": weights_are_finite,
            "sinr_is_finite": sinr_is_finite,
            "reference_sinr_is_finite": reference_sinr_is_finite,
            "reward_is_finite": reward_is_finite,
            "numerical_error": numerical_error,
        }

        return next_state, float(reward), terminated, truncated, info

    def _sample_target_doa(self) -> tuple[float, float]:
        """Sample a random SOI DOA."""

        theta_rad = float(self.np_random.uniform(self.theta_min, self.theta_max))
        phi_rad = float(self.np_random.uniform(self.phi_min, self.phi_max))
        return theta_rad, phi_rad

    def _sample_jammer_doas(
        self,
        theta_target_rad: float,
        phi_target_rad: float,
    ) -> tuple[list[float], list[float]]:
        """Sample jammer DOAs with a minimum angular separation."""

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
        """Sample one valid jammer DOA."""

        for _ in range(max_attempts):
            theta_jam_rad = float(
                self.np_random.uniform(self.jammer_theta_min, self.jammer_theta_max)
            )
            phi_jam_rad = float(
                self.np_random.uniform(self.jammer_phi_min, self.jammer_phi_max)
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

    def _build_state(
        self,
        theta_target_rad: float,
        phi_target_rad: float,
        jammer_thetas_rad: list[float],
        jammer_phis_rad: list[float],
    ) -> np.ndarray:
        """Build the fixed-size state representation."""

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
                    theta_jam_deg = float(np.rad2deg(jammer_thetas_rad[jammer_idx]))
                    phi_jam_deg = float(np.rad2deg(jammer_phis_rad[jammer_idx]))
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
    ) -> tuple[float, float, float, float]:
        """Convert a dual action into SOI and jammer angles in radians."""

        if self.action_mode == "angles":
            action = np.asarray(action, dtype=np.float32).reshape(4)
            action = np.clip(action, self.action_space.low, self.action_space.high)

            theta_soi_action_rad = self._denormalize_theta(float(action[0]))
            phi_soi_action_rad = self._denormalize_phi(float(action[1]))
            theta_jammer_action_rad = self._denormalize_jammer_theta(float(action[2]))
            phi_jammer_action_rad = self._denormalize_jammer_phi(float(action[3]))

            return (
                theta_soi_action_rad,
                phi_soi_action_rad,
                theta_jammer_action_rad,
                phi_jammer_action_rad,
            )

        if self.action_mode == "unit_vector":
            action = np.asarray(action, dtype=np.float32).reshape(6)
            action = np.clip(action, self.action_space.low, self.action_space.high)

            u_soi_action = self._normalize_action_unit_vector(action[:3])
            u_jammer_action = self._normalize_action_unit_vector(action[3:6])

            theta_soi_deg, phi_soi_deg = unit_vector_to_angles(
                u_soi_action,
                enforce_visible=self.enforce_visible_hemisphere,
            )
            theta_jammer_deg, phi_jammer_deg = unit_vector_to_angles(
                u_jammer_action,
                enforce_visible=self.enforce_visible_hemisphere,
            )

            return (
                float(np.deg2rad(theta_soi_deg)),
                float(np.deg2rad(phi_soi_deg)),
                float(np.deg2rad(theta_jammer_deg)),
                float(np.deg2rad(phi_jammer_deg)),
            )

        raise RuntimeError("Invalid action_mode.")

    def _normalize_action_unit_vector(self, action: np.ndarray) -> np.ndarray:
        """Normalize a raw 3D action and project it to the visible hemisphere."""

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
        theta_jammer_action_rad: float,
        phi_jammer_action_rad: float,
    ) -> np.ndarray:
        """Build beamforming weights from the dual action."""

        if self.beamforming_mode == "steering":
            return self._build_steering_weights(
                theta_rad=theta_soi_action_rad,
                phi_rad=phi_soi_action_rad,
            )

        if self.beamforming_mode == "nulling":
            return self._build_nulling_weights(
                theta_soi_action_rad=theta_soi_action_rad,
                phi_soi_action_rad=phi_soi_action_rad,
                theta_jammer_action_rad=theta_jammer_action_rad,
                phi_jammer_action_rad=phi_jammer_action_rad,
            )

        if self.beamforming_mode == "mvdr":
            return self._build_mvdr_weights(
                theta_soi_action_rad=theta_soi_action_rad,
                phi_soi_action_rad=phi_soi_action_rad,
                theta_jammer_action_rad=theta_jammer_action_rad,
                phi_jammer_action_rad=phi_jammer_action_rad,
            )

        raise RuntimeError("Invalid beamforming_mode.")

    def _build_steering_weights(self, theta_rad: float, phi_rad: float) -> np.ndarray:
        """Build conventional steering weights for a given direction."""

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
        theta_jammer_action_rad: float,
        phi_jammer_action_rad: float,
    ) -> np.ndarray:
        """Build deterministic nulling weights from SOI and jammer actions."""

        target_direction = (
            float(np.rad2deg(theta_soi_action_rad)),
            float(np.rad2deg(phi_soi_action_rad)),
        )
        jammer_direction = (
            float(np.rad2deg(theta_jammer_action_rad)),
            float(np.rad2deg(phi_jammer_action_rad)),
        )

        weights_flat = interference_suppression_weights(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            target_direction=target_direction,
            jammer_direction=jammer_direction,
        )

        return weights_flat.reshape(self.array.N, self.array.M)

    def _build_mvdr_weights(
        self,
        theta_soi_action_rad: float,
        phi_soi_action_rad: float,
        theta_jammer_action_rad: float,
        phi_jammer_action_rad: float,
    ) -> np.ndarray:
        """Build MVDR weights from SOI action and jammer-action covariance."""

        target_direction = (
            float(np.rad2deg(theta_soi_action_rad)),
            float(np.rad2deg(phi_soi_action_rad)),
        )
        jammer_direction = (
            float(np.rad2deg(theta_jammer_action_rad)),
            float(np.rad2deg(phi_jammer_action_rad)),
        )

        R_xx = self._build_interference_noise_covariance(
            jammer_direction=jammer_direction,
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
        jammer_direction: tuple[float, float],
    ) -> np.ndarray:
        """Build interference-plus-noise covariance from the jammer direction."""

        num_elements = self.array.N * self.array.M
        R_xx = self.noise_power * np.eye(num_elements, dtype=np.complex128)

        if self.num_active_jammers > 0:
            jammer_power = float(self.jammer_powers[0])
            jammer_sv = get_steering_vector(
                element_positions=self.array.element_positions,
                wavenumber_k=self.array.k_num,
                direction=jammer_direction,
            ).astype(np.complex128).reshape(num_elements)

            R_xx += jammer_power * np.outer(jammer_sv, np.conj(jammer_sv))

        if self.mvdr_diagonal_loading > 0.0:
            R_xx += self.mvdr_diagonal_loading * np.eye(
                num_elements,
                dtype=np.complex128,
            )

        return R_xx

    def _build_safe_fallback_weights(self) -> np.ndarray:
        """Build safe fallback conventional steering weights."""

        if self.current_theta_rad is not None and self.current_phi_rad is not None:
            return self._build_steering_weights(
                theta_rad=self.current_theta_rad,
                phi_rad=self.current_phi_rad,
            )

        return self._build_steering_weights(theta_rad=0.0, phi_rad=0.0)

    def _compute_reference_sinr_db(self) -> float:
        """Compute the reference SINR according to reference_mode."""

        if self.reference_mode == "steering":
            reference_weights = self._build_reference_steering_weights()
        elif self.reference_mode == "same_beamforming_mode":
            reference_weights = self._build_reference_same_mode_weights()
        else:
            raise RuntimeError("Invalid reference_mode.")

        return self._compute_sinr_for_weights(reference_weights)

    def _build_reference_steering_weights(self) -> np.ndarray:
        """Build perfect conventional steering weights toward the true SOI."""

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Environment must be reset before computing reference.")

        return self._build_steering_weights(
            theta_rad=self.current_theta_rad,
            phi_rad=self.current_phi_rad,
        )

    def _build_reference_same_mode_weights(self) -> np.ndarray:
        """Build perfect same-family reference weights using true directions."""

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Environment must be reset before computing reference.")

        if self.num_active_jammers > 0:
            theta_jammer_rad = self.current_jammer_thetas_rad[0]
            phi_jammer_rad = self.current_jammer_phis_rad[0]
        else:
            theta_jammer_rad = 0.0
            phi_jammer_rad = 0.0

        if self.beamforming_mode == "steering":
            return self._build_steering_weights(
                theta_rad=self.current_theta_rad,
                phi_rad=self.current_phi_rad,
            )

        if self.beamforming_mode == "nulling":
            return self._build_nulling_weights(
                theta_soi_action_rad=self.current_theta_rad,
                phi_soi_action_rad=self.current_phi_rad,
                theta_jammer_action_rad=theta_jammer_rad,
                phi_jammer_action_rad=phi_jammer_rad,
            )

        if self.beamforming_mode == "mvdr":
            return self._build_mvdr_weights(
                theta_soi_action_rad=self.current_theta_rad,
                phi_soi_action_rad=self.current_phi_rad,
                theta_jammer_action_rad=theta_jammer_rad,
                phi_jammer_action_rad=phi_jammer_rad,
            )

        raise RuntimeError("Invalid beamforming_mode.")

    def _compute_reference_steering_sinr_db(self) -> float:
        """Backward-compatible method for perfect-steering reference SINR."""

        reference_weights = self._build_reference_steering_weights()
        return self._compute_sinr_for_weights(reference_weights)

    def _compute_sinr_for_weights(self, weights: np.ndarray) -> float:
        """Compute SINR for the current true SOI/jammer scene."""

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
        jammer_action_error_deg: float,
        clipped_sinr_loss_db: float,
    ) -> tuple[float, dict]:
        """
        Compute optional bounded milestone bonuses.

        The continuous reward remains the main learning signal. These bonuses
        are disabled by default because all bonus coefficients default to zero.
        """

        good_soi = bool(
            np.isfinite(soi_angle_error_deg)
            and float(soi_angle_error_deg) <= self.soi_error_bonus_threshold_deg
        )

        if self.num_active_jammers == 0:
            good_jammer = False
        else:
            good_jammer = bool(
                np.isfinite(jammer_action_error_deg)
                and float(jammer_action_error_deg)
                <= self.jammer_error_bonus_threshold_deg
            )

        good_sinr_loss = bool(
            np.isfinite(clipped_sinr_loss_db)
            and float(clipped_sinr_loss_db) <= self.sinr_loss_bonus_threshold_db
        )

        bonus = 0.0

        if good_soi:
            bonus += self.reward_bonus_good_soi

        if good_jammer:
            bonus += self.reward_bonus_good_jammer

        if good_sinr_loss:
            bonus += self.reward_bonus_good_sinr_loss

        info = {
            "milestone_good_soi": good_soi,
            "milestone_good_jammer": good_jammer,
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
        """Compute reward robustly, accumulating only active finite terms."""

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

    def _get_current_jammer_directions_deg(self) -> list[tuple[float, float]]:
        """Return current true jammer directions in degrees."""

        jammer_directions_deg = []

        for theta_rad, phi_rad in zip(
            self.current_jammer_thetas_rad,
            self.current_jammer_phis_rad,
        ):
            jammer_directions_deg.append(
                (float(np.rad2deg(theta_rad)), float(np.rad2deg(phi_rad)))
            )

        return jammer_directions_deg

    def _compute_first_jammer_action_error_deg(
        self,
        theta_jammer_action_rad: float,
        phi_jammer_action_rad: float,
    ) -> float:
        """Compute angular error between the true jammer and jammer action."""

        if self.num_active_jammers < 1:
            return float("nan")

        if len(self.current_jammer_thetas_rad) < 1:
            return float("nan")

        return self._compute_angular_error_deg(
            theta_a_rad=self.current_jammer_thetas_rad[0],
            phi_a_rad=self.current_jammer_phis_rad[0],
            theta_b_rad=theta_jammer_action_rad,
            phi_b_rad=phi_jammer_action_rad,
        )

    def _compute_jammer_angle_loss(self, jammer_action_error_deg: float) -> float:
        """Compute normalized squared angular loss for the active jammer."""

        if self.num_active_jammers == 0:
            return 0.0

        if not np.isfinite(jammer_action_error_deg):
            return 1.0

        return float((float(jammer_action_error_deg) / 180.0) ** 2)

    def _compute_angular_error_deg(
        self,
        theta_a_rad: float,
        phi_a_rad: float,
        theta_b_rad: float,
        phi_b_rad: float,
    ) -> float:
        """Compute the 3D angular separation between two directions."""

        u_a = self._angles_rad_to_unit_vector(theta_a_rad, phi_a_rad)
        u_b = self._angles_rad_to_unit_vector(theta_b_rad, phi_b_rad)
        dot_product = float(np.dot(u_a, u_b))
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_error_rad = np.arccos(dot_product)
        return float(np.rad2deg(angle_error_rad))

    def _compute_angle_loss(self, angle_error_deg: float) -> float:
        """Compute normalized squared angular loss."""

        if not np.isfinite(angle_error_deg):
            return 1.0

        return float((float(angle_error_deg) / 180.0) ** 2)

    def _angles_rad_to_unit_vector(
        self,
        theta_rad: float,
        phi_rad: float,
    ) -> np.ndarray:
        """Convert angular coordinates in radians to a 3D unit vector."""

        ux = np.sin(theta_rad) * np.cos(phi_rad)
        uy = np.sin(theta_rad) * np.sin(phi_rad)
        uz = np.cos(theta_rad)
        return np.array([ux, uy, uz], dtype=float)

    def _build_reset_info(self) -> dict:
        """Build reset info dictionary."""

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
                float(np.rad2deg(theta)) for theta in self.current_jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(phi)) for phi in self.current_jammer_phis_rad
            ],
            "jammers_directions_deg": jammer_directions_deg,
            "jammers_powers": self.jammer_powers.copy(),
            "beamforming_mode": self.beamforming_mode,
            "reference_mode": self.reference_mode,
            "reward_bonus_good_soi": self.reward_bonus_good_soi,
            "reward_bonus_good_jammer": self.reward_bonus_good_jammer,
            "reward_bonus_good_sinr_loss": self.reward_bonus_good_sinr_loss,
            "soi_error_bonus_threshold_deg": self.soi_error_bonus_threshold_deg,
            "jammer_error_bonus_threshold_deg": self.jammer_error_bonus_threshold_deg,
            "sinr_loss_bonus_threshold_db": self.sinr_loss_bonus_threshold_db,
        }

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
