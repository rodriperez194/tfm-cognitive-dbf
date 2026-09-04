from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tfm.physics.narrow_band.phased_array_nb import Phased_Array_NB
from tfm.physics.narrow_band.weights_stochastic_nb import mvdr_weights
from tfm.physics.narrow_band.weights_deterministic_nb import (
    target_or_zero_weights,
)
from tfm.math.narrow_band.metrics import compute_sinr
from tfm.math.narrow_band.steering_vector import get_steering_vector
from tfm.math.narrow_band.geometry import angles_to_unit_vector

from tfm.scenario.scenario_generator import ScenarioGenerator
from tfm.targets.aircraft import AircraftTarget
from tfm.targets.drone import DroneTarget
from tfm.targets.dummy import Dummy
from tfm.targets.static import StaticTarget
from tfm.targets.truck import TruckRoadTarget

class BeamformingEnvPhase9(gym.Env):
    """
    Gymnasium environment for Phase 9 coefficient-space beamforming.

    The observation, scenario generation, physical evaluation, reward, MVDR
    reference and K-step control cadence follow the validated Phase 7 pipeline.
    The action representation changes completely: the agent no longer controls
    the 36 complex element weights directly.

    Instead, the action contains real and imaginary parts of complex
    coefficients applied to a scenario-dependent electromagnetic basis:

        delta_w = B(s) c
        w_raw = w_point + delta_w
        w_final = normalize(w_raw)

    For ``coefficient_jammer_slots = J`` the basis contains one SOI steering
    vector and five columns per jammer slot: jammer centre plus four fixed
    cross4 virtual directions. Therefore:

        num_complex_coefficients = 1 + 5J
        action_dim = 2 * (1 + 5J)

    For the first one-jammer specialist, J=1 and the action dimension is 12.
    The semantic ordering is:

        [c_S, c_J, c_v0, c_v90, c_v180, c_v270]

    with all real parts first and all imaginary parts second.

    Virtual slots never disappear. If one virtual direction is outside the
    visible hemisphere, lies too close to the SOI, or is removed by the same
    deduplication rule used in the validated Phase 7 diagnostic, the
    corresponding basis column is exactly zero and its active-mask value is 0.

    The deterministic point base is ``target_or_zero_point``. The frozen
    soft-null teachers are:

        0j -> target_or_zero_point
        1j -> soft_cross4_r05_vr0p01
        2j -> soft_cross4_r02_vr0p005
        3j -> soft_cross4_r1p5_vr0p003

    Final power normalization is delegated to ``Phased_Array_NB.set_weights``
    exactly as in the validated direct-weight environment.
    """

    metadata = {"render_modes": []}

    ADAPTIVE_TEACHER_BY_JAMMER_COUNT = {
        0: {
            "candidate_id": "target_or_zero_point",
            "candidate_type": "point_teacher",
            "pattern": None,
            "radius_deg": 0.0,
            "virtual_total_power_ratio": 0.0,
        },
        1: {
            "candidate_id": "soft_cross4_r05_vr0p01",
            "candidate_type": "soft_mvdr",
            "pattern": "cross4",
            "radius_deg": 5.0,
            "virtual_total_power_ratio": 0.01,
        },
        2: {
            "candidate_id": "soft_cross4_r02_vr0p005",
            "candidate_type": "soft_mvdr",
            "pattern": "cross4",
            "radius_deg": 2.0,
            "virtual_total_power_ratio": 0.005,
        },
        3: {
            "candidate_id": "soft_cross4_r1p5_vr0p003",
            "candidate_type": "soft_mvdr",
            "pattern": "cross4",
            "radius_deg": 1.5,
            "virtual_total_power_ratio": 0.003,
        },
    }

    SOFT_NULL_CENTER_POWER_SCALE = 1.0
    SOFT_NULL_DIAGONAL_LOADING = 1e-4
    SOFT_NULL_FALLBACK_DIAGONAL_LOADING = 1e-2
    SOFT_NULL_DENOMINATOR_EPSILON = 1e-12

    MIN_VIRTUAL_DIRECTION_SOI_SEPARATION_DEG = 8.0
    DIRECTION_DEDUPLICATION_DEG = 0.25
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
        observation_mode: str = "unit_vector",
        complex_weight_mode: str = "real_imag",
        weight_hold_steps: int = 1,
        episode_length_physical_steps: int = 200,
        dt: float = 1.0,
        target_position_x_limits_m: tuple[float, float] = (-1000.0, 1000.0),
        target_position_y_limits_m: tuple[float, float] = (-1000.0, 1000.0),
        target_position_z_limits_m: tuple[float, float] = (50.0, 1000.0),
        jammer_target_types: list[str] | tuple[str, ...] = (
            "aircraft",
            "drone",
            "dummy",
            "static",
            "truck",
        ),
        theta_limits_rad: tuple[float, float] = (0.0, np.pi / 2.0),
        phi_limits_rad: tuple[float, float] = (0.0, 2.0 * np.pi),
        jammer_theta_limits_rad: tuple[float, float] = (0.0, np.pi / 2.0),
        jammer_phi_limits_rad: tuple[float, float] = (0.0, 2.0 * np.pi),
        min_source_distance_m: float = 50.0,
        min_target_jammer_separation_deg: float = 5.0,
        enforce_visible_hemisphere: bool = True,
        reward_failure_penalty: float = -10.0,
        reward_soi_max_gain_loss_db: float = 3.0,
        reward_jammer_max_mean_leakage: float = 0.01,
        reward_sinr_scale_db: float = 30.0,
        reward_valid_min: float = -2.0,
        reward_valid_max: float = 2.0,
        reward_teacher_similarity_weight: float = 1.0,
        reward_jammer_leakage_penalty_weight: float = 0.0,
        reward_jammer_leakage_penalty_scale: float = 0.05,
        reward_jammer_leakage_penalty_clip: float = 5.0,
        reward_sinr_loss_bonus_steps: (
            list[tuple[float, float]] | tuple[tuple[float, float], ...] | np.ndarray | None
        ) = None,
        reward_soi_gain_loss_bonus_steps: (
            list[tuple[float, float]] | tuple[tuple[float, float], ...] | np.ndarray | None
        ) = None,
        reward_jammer_leakage_bonus_steps: (
            list[tuple[float, float]] | tuple[tuple[float, float], ...] | np.ndarray | None
        ) = None,
        reward_teacher_similarity_bonus_steps: (
            list[tuple[float, float]] | tuple[tuple[float, float], ...] | np.ndarray | None
        ) = None,
        teacher_diagonal_loading: float = 1e-8,
        teacher_use_pinv: bool = False,
        teacher_similarity_epsilon: float = 1e-12,
        direct_weight_min_power: float = 1e-12,
        max_sinr_loss_db: float = 60.0,
        max_soi_gain_loss_db: float = 60.0,
        max_jammer_leakage_loss: float = 30.0,
        mvdr_diagonal_loading: float = 1e-4,
        invalid_sinr_db: float = -120.0,
        invalid_value_penalty: float = -1_000.0,
        max_scenario_sampling_attempts: int = 200,
        coefficient_jammer_slots: int = 1,
        coefficient_scales: (
            list[float] | tuple[float, ...] | np.ndarray | None
        ) = None,
        coefficient_basis_mode: str = "raw",
        coefficient_basis_orthogonalization_tol: float = 1e-10,
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

        self.jammer_powers_config = (
            None if jammer_powers is None else [float(power) for power in jammer_powers]
        )
        self.jammer_powers = self._build_jammer_powers_for_active_count(
            self.num_active_jammers
        )

        self.observation_mode = str(observation_mode)
        self.complex_weight_mode = str(complex_weight_mode)

        self.weight_hold_steps = int(weight_hold_steps)
        self.episode_length_physical_steps = int(episode_length_physical_steps)
        self.dt = float(dt)

        self.target_position_x_limits_m = tuple(
            float(value) for value in target_position_x_limits_m
        )
        self.target_position_y_limits_m = tuple(
            float(value) for value in target_position_y_limits_m
        )
        self.target_position_z_limits_m = tuple(
            float(value) for value in target_position_z_limits_m
        )

        self.jammer_target_types = [str(value) for value in jammer_target_types]

        self.theta_min = float(theta_limits_rad[0])
        self.theta_max = float(theta_limits_rad[1])
        self.phi_min = float(phi_limits_rad[0])
        self.phi_max = float(phi_limits_rad[1])

        self.jammer_theta_min = float(jammer_theta_limits_rad[0])
        self.jammer_theta_max = float(jammer_theta_limits_rad[1])
        self.jammer_phi_min = float(jammer_phi_limits_rad[0])
        self.jammer_phi_max = float(jammer_phi_limits_rad[1])

        self.min_source_distance_m = float(min_source_distance_m)
        self.min_target_jammer_separation_deg = float(
            min_target_jammer_separation_deg
        )
        self.enforce_visible_hemisphere = bool(enforce_visible_hemisphere)

        self.reward_failure_penalty = float(reward_failure_penalty)
        self.reward_soi_max_gain_loss_db = float(reward_soi_max_gain_loss_db)
        self.reward_jammer_max_mean_leakage = float(
            reward_jammer_max_mean_leakage
        )
        self.reward_sinr_scale_db = float(reward_sinr_scale_db)
        self.reward_valid_min = float(reward_valid_min)
        self.reward_valid_max = float(reward_valid_max)
        self.reward_teacher_similarity_weight = float(
            reward_teacher_similarity_weight
        )
        self.reward_jammer_leakage_penalty_weight = float(
            reward_jammer_leakage_penalty_weight
        )
        self.reward_jammer_leakage_penalty_scale = float(
            reward_jammer_leakage_penalty_scale
        )
        self.reward_jammer_leakage_penalty_clip = float(
            reward_jammer_leakage_penalty_clip
        )

        self.reward_sinr_loss_bonus_steps = self._prepare_bonus_steps(
            reward_sinr_loss_bonus_steps,
            parameter_name="reward_sinr_loss_bonus_steps",
        )
        self.reward_soi_gain_loss_bonus_steps = self._prepare_bonus_steps(
            reward_soi_gain_loss_bonus_steps,
            parameter_name="reward_soi_gain_loss_bonus_steps",
        )
        self.reward_jammer_leakage_bonus_steps = self._prepare_bonus_steps(
            reward_jammer_leakage_bonus_steps,
            parameter_name="reward_jammer_leakage_bonus_steps",
        )
        self.reward_teacher_similarity_bonus_steps = self._prepare_bonus_steps(
            reward_teacher_similarity_bonus_steps,
            parameter_name="reward_teacher_similarity_bonus_steps",
        )

        self.teacher_diagonal_loading = float(teacher_diagonal_loading)
        self.teacher_use_pinv = bool(teacher_use_pinv)
        self.teacher_similarity_epsilon = float(
            teacher_similarity_epsilon
        )
        self.direct_weight_min_power = float(direct_weight_min_power)

        self.max_sinr_loss_db = float(max_sinr_loss_db)
        self.max_soi_gain_loss_db = float(max_soi_gain_loss_db)
        self.max_jammer_leakage_loss = float(max_jammer_leakage_loss)

        self.mvdr_diagonal_loading = float(mvdr_diagonal_loading)
        self.invalid_sinr_db = float(invalid_sinr_db)
        self.invalid_value_penalty = float(invalid_value_penalty)
        self.max_scenario_sampling_attempts = int(max_scenario_sampling_attempts)

        self.coefficient_jammer_slots = int(coefficient_jammer_slots)
        self.coefficient_basis_mode = str(coefficient_basis_mode)
        self.coefficient_basis_orthogonalization_tol = float(
            coefficient_basis_orthogonalization_tol
        )

        self.num_elements = int(self.array.N * self.array.M)
        self.num_complex_coefficients = 1 + 5 * self.coefficient_jammer_slots

        if coefficient_scales is None:
            self.coefficient_scales = np.ones(
                self.num_complex_coefficients,
                dtype=float,
            )
        else:
            self.coefficient_scales = np.asarray(
                coefficient_scales,
                dtype=float,
            ).reshape(self.num_complex_coefficients)

        self.basis_column_labels = self._build_basis_column_labels()

        self._validate_configuration()

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
        else:
            raise RuntimeError("Invalid observation_mode.")

        self.action_dim = 2 * self.num_complex_coefficients
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.current_scenario: dict | None = None
        self.current_physical_step: int = 0

        self.current_theta_rad: float | None = None
        self.current_phi_rad: float | None = None
        self.current_jammer_thetas_rad: list[float] = []
        self.current_jammer_phis_rad: list[float] = []

        self.current_state: np.ndarray | None = None
        self.last_final_weights: np.ndarray | None = None
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment and generate a new dynamic scenario."""

        super().reset(seed=seed)

        self._sample_num_active_jammers_for_episode()
        self.jammer_powers = self._build_jammer_powers_for_active_count(
            self.num_active_jammers
        )

        scenario = self._sample_valid_scenario()

        self.current_scenario = scenario
        self.current_physical_step = 0
        self._load_current_directions_from_scenario(step_idx=0)

        state = self._build_state(
            theta_target_rad=self.current_theta_rad,
            phi_target_rad=self.current_phi_rad,
            jammer_thetas_rad=self.current_jammer_thetas_rad,
            jammer_phis_rad=self.current_jammer_phis_rad,
        )
        self.current_state = state
        self.last_final_weights = None

        info = {
            "num_active_jammers": self.num_active_jammers,
            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,
            "theta_target_deg": float(np.rad2deg(self.current_theta_rad)),
            "phi_target_deg": float(np.rad2deg(self.current_phi_rad)),
            "jammer_thetas_rad": self.current_jammer_thetas_rad.copy(),
            "jammer_phis_rad": self.current_jammer_phis_rad.copy(),
            "jammer_thetas_deg": [
                float(np.rad2deg(theta)) for theta in self.current_jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(phi)) for phi in self.current_jammer_phis_rad
            ],
            "jammers_directions_deg": self._get_current_jammer_directions_deg(),
            "jammers_powers": self.jammer_powers.copy(),
            "observation_mode": self.observation_mode,
            "complex_weight_mode": self.complex_weight_mode,
            "action_type": self._get_action_type(),
            "weight_hold_steps": self.weight_hold_steps,
            "teacher_update_mode": "block_start_hold",
            "episode_length_physical_steps": self.episode_length_physical_steps,
            "dt": self.dt,
            "scenario_metadata": scenario.get("metadata", {}),
            "array_normalize_power": bool(self.array.normalize_power),
            "reward_failure_penalty": self.reward_failure_penalty,
            "reward_soi_max_gain_loss_db": self.reward_soi_max_gain_loss_db,
            "reward_jammer_max_mean_leakage": (
                self.reward_jammer_max_mean_leakage
            ),
            "reward_sinr_scale_db": self.reward_sinr_scale_db,
            "reward_valid_min": self.reward_valid_min,
            "reward_valid_max": self.reward_valid_max,
            "reward_teacher_similarity_weight": (
                self.reward_teacher_similarity_weight
            ),
            "reward_jammer_leakage_penalty_weight": (
                self.reward_jammer_leakage_penalty_weight
            ),
            "reward_jammer_leakage_penalty_scale": (
                self.reward_jammer_leakage_penalty_scale
            ),
            "reward_jammer_leakage_penalty_clip": (
                self.reward_jammer_leakage_penalty_clip
            ),
            "reward_sinr_loss_bonus_steps": (
                self.reward_sinr_loss_bonus_steps.copy()
            ),
            "reward_soi_gain_loss_bonus_steps": (
                self.reward_soi_gain_loss_bonus_steps.copy()
            ),
            "reward_jammer_leakage_bonus_steps": (
                self.reward_jammer_leakage_bonus_steps.copy()
            ),
            "reward_teacher_similarity_bonus_steps": (
                self.reward_teacher_similarity_bonus_steps.copy()
            ),
            "teacher_diagonal_loading": self.teacher_diagonal_loading,
            "teacher_use_pinv": self.teacher_use_pinv,
            "teacher_similarity_epsilon": (
                self.teacher_similarity_epsilon
            ),
            "direct_weight_min_power": self.direct_weight_min_power,
            "coefficient_jammer_slots": self.coefficient_jammer_slots,
            "num_complex_coefficients": self.num_complex_coefficients,
            "coefficient_scales": self.coefficient_scales.copy(),
            "basis_column_labels": self.basis_column_labels.copy(),
            "coefficient_basis_mode": self.coefficient_basis_mode,
            "coefficient_basis_orthogonalization_tol": (
                self.coefficient_basis_orthogonalization_tol
            ),
        }

        return state, info
    def step(self, action: np.ndarray):
        """Apply one native coefficient-space action for one control block."""

        if self.current_scenario is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized.")

        numerical_error = False
        invalid_coefficient_action = False

        block_start_step_idx = self.current_physical_step
        self._load_current_directions_from_scenario(
            step_idx=block_start_step_idx
        )

        try:
            proposed_weights, action_info = self._action_to_coefficient_weights(
                action
            )
            invalid_coefficient_action = bool(
                action_info["invalid_coefficient_action"]
            )
        except Exception:
            proposed_weights = self._build_safe_fallback_weights()
            action_info = self._build_invalid_action_info()
            invalid_coefficient_action = True
            numerical_error = True

        if not np.all(np.isfinite(proposed_weights)):
            proposed_weights = self._build_safe_fallback_weights()
            action_info = self._build_invalid_action_info()
            invalid_coefficient_action = True
            numerical_error = True

        self.array.set_weights(proposed_weights)
        fixed_weights = self.array.W.copy()
        weights_are_finite = bool(np.all(np.isfinite(fixed_weights)))

        if not weights_are_finite:
            fixed_weights = self._build_safe_fallback_weights()
            self.array.set_weights(fixed_weights)
            fixed_weights = self.array.W.copy()
            invalid_coefficient_action = True
            numerical_error = True

        action_info = self._finalize_action_info(
            action_info=action_info,
            normalized_weights=fixed_weights,
            invalid_coefficient_action=invalid_coefficient_action,
        )
        self.last_final_weights = fixed_weights.copy()

        remaining_steps = (
            self.episode_length_physical_steps - self.current_physical_step
        )
        num_block_steps = min(self.weight_hold_steps, remaining_steps)
        block_metrics: list[dict] = []

        teacher_weights_are_valid = True
        try:
            fixed_teacher_weights = self._build_teacher_weights()
        except Exception:
            fixed_teacher_weights = np.zeros(
                (self.array.N, self.array.M),
                dtype=np.complex128,
            )
            teacher_weights_are_valid = False
            numerical_error = True

        for block_offset in range(num_block_steps):
            step_idx = self.current_physical_step + block_offset
            self._load_current_directions_from_scenario(step_idx=step_idx)

            instant_metrics = self._evaluate_fixed_weights_at_current_step(
                weights=fixed_weights,
                teacher_weights=fixed_teacher_weights,
                teacher_weights_are_valid=teacher_weights_are_valid,
                invalid_coefficient_action=invalid_coefficient_action,
            )
            if bool(instant_metrics["numerical_error"]):
                numerical_error = True
            block_metrics.append(instant_metrics)

        reward = self._safe_mean_metric(block_metrics, "reward")
        if not np.isfinite(reward):
            reward = self.invalid_value_penalty
            numerical_error = True
        reward = float(reward)

        self.current_physical_step += num_block_steps
        terminated = bool(
            self.current_physical_step >= self.episode_length_physical_steps
        )
        truncated = False

        next_step_idx = min(
            self.current_physical_step,
            self.episode_length_physical_steps - 1,
        )
        self._load_current_directions_from_scenario(step_idx=next_step_idx)

        next_state = self._build_state(
            theta_target_rad=self.current_theta_rad,
            phi_target_rad=self.current_phi_rad,
            jammer_thetas_rad=self.current_jammer_thetas_rad,
            jammer_phis_rad=self.current_jammer_phis_rad,
        )
        self.current_state = next_state

        info = self._build_block_info(
            block_metrics=block_metrics,
            reward=reward,
            numerical_error=numerical_error,
            weights_are_finite=weights_are_finite,
            fixed_weights=fixed_weights,
            fixed_teacher_weights=fixed_teacher_weights,
            teacher_weights_are_valid=teacher_weights_are_valid,
            action_info=action_info,
            num_block_steps=num_block_steps,
            terminated=terminated,
        )

        return next_state, reward, terminated, truncated, info
    @staticmethod
    def _prepare_bonus_steps(
        bonus_steps: (
            list[tuple[float, float]]
            | tuple[tuple[float, float], ...]
            | np.ndarray
            | None
        ),
        parameter_name: str,
    ) -> np.ndarray:
        """
        Convert one stepped-bonus configuration into a validated Nx2 array.

        Column 0 stores thresholds and column 1 stores non-negative bonuses.
        An empty configuration is represented as an array with shape (0, 2).
        """

        if bonus_steps is None:
            return np.empty((0, 2), dtype=float)

        array = np.asarray(bonus_steps, dtype=float)

        if array.size == 0:
            return np.empty((0, 2), dtype=float)

        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                f"{parameter_name} must be an array-like object "
                "with shape (num_steps, 2)."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{parameter_name} must contain only finite values."
            )

        if np.any(array[:, 1] < 0.0):
            raise ValueError(
                f"{parameter_name} bonuses must be non-negative."
            )

        return np.asarray(array, dtype=float).copy()


    @staticmethod
    def _compute_stepped_bonus(
        metric_value: float,
        bonus_steps: np.ndarray,
        lower_is_better: bool,
    ) -> dict:
        """
        Evaluate one stepped-bonus schedule.

        When several rows are satisfied, the largest configured bonus is
        selected. This avoids accidental accumulation inside one component.
        """

        steps = np.asarray(bonus_steps, dtype=float).reshape(-1, 2)

        if (
            steps.shape[0] == 0
            or not np.isfinite(metric_value)
        ):
            return {
                "bonus": 0.0,
                "matched": False,
                "matched_threshold": float("nan"),
                "matched_row_index": -1,
                "num_satisfied_steps": 0,
            }

        thresholds = steps[:, 0]
        bonuses = steps[:, 1]

        if lower_is_better:
            satisfied_mask = float(metric_value) <= thresholds
        else:
            satisfied_mask = float(metric_value) >= thresholds

        satisfied_indices = np.flatnonzero(satisfied_mask)

        if satisfied_indices.size == 0:
            return {
                "bonus": 0.0,
                "matched": False,
                "matched_threshold": float("nan"),
                "matched_row_index": -1,
                "num_satisfied_steps": 0,
            }

        satisfied_bonuses = bonuses[satisfied_indices]
        local_best_idx = int(np.argmax(satisfied_bonuses))
        matched_row_index = int(satisfied_indices[local_best_idx])

        return {
            "bonus": float(bonuses[matched_row_index]),
            "matched": True,
            "matched_threshold": float(thresholds[matched_row_index]),
            "matched_row_index": matched_row_index,
            "num_satisfied_steps": int(satisfied_indices.size),
        }


    def _validate_configuration(self) -> None:
        """Validate constructor configuration."""

        if self.coefficient_jammer_slots < 1 or self.coefficient_jammer_slots > 3:
            raise ValueError(
                "coefficient_jammer_slots must be between 1 and 3."
            )
        if self.num_active_jammers > self.coefficient_jammer_slots:
            raise ValueError(
                "num_active_jammers cannot exceed coefficient_jammer_slots."
            )
        if self.active_jammers_choices is not None:
            for value in self.active_jammers_choices:
                if value > self.coefficient_jammer_slots:
                    raise ValueError(
                        "active_jammers_choices cannot request more jammers "
                        "than coefficient_jammer_slots."
                    )
        if self.coefficient_basis_mode not in [
            "raw",
            "orthonormal_mgs",
        ]:
            raise ValueError(
                "Unknown coefficient_basis_mode. Expected 'raw' or "
                "'orthonormal_mgs'."
            )
        if (
            not np.isfinite(
                self.coefficient_basis_orthogonalization_tol
            )
            or self.coefficient_basis_orthogonalization_tol <= 0.0
        ):
            raise ValueError(
                "coefficient_basis_orthogonalization_tol must be finite "
                "and positive."
            )

        if self.coefficient_scales.shape != (self.num_complex_coefficients,):
            raise ValueError(
                "coefficient_scales has an invalid shape."
            )
        if (
            not np.all(np.isfinite(self.coefficient_scales))
            or np.any(self.coefficient_scales <= 0.0)
        ):
            raise ValueError(
                "coefficient_scales must contain finite positive values."
            )

        if self.max_jammers != 3:
            raise ValueError(
                "This environment currently requires max_jammers=3 "
                "to match the fixed roadmap state format."
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

        if self.observation_mode not in ["angles", "unit_vector"]:
            raise ValueError(
                "Unknown observation_mode. Expected 'angles' or 'unit_vector'."
            )
        if self.complex_weight_mode != "real_imag":
            raise ValueError(
                "Phase 9 coefficient control currently requires "
                "complex_weight_mode='real_imag'."
            )
        if self.weight_hold_steps <= 0:
            raise ValueError("weight_hold_steps must be a positive integer.")
        if self.episode_length_physical_steps <= 0:
            raise ValueError(
                "episode_length_physical_steps must be a positive integer."
            )
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.min_source_distance_m < 0.0:
            raise ValueError("min_source_distance_m must be non-negative.")
        if self.min_target_jammer_separation_deg < 0.0:
            raise ValueError(
                "min_target_jammer_separation_deg must be non-negative."
            )
        if len(self.jammer_target_types) == 0:
            raise ValueError("jammer_target_types cannot be empty.")

        valid_target_types = {"aircraft", "drone", "dummy", "static", "truck"}
        for target_type in self.jammer_target_types:
            if target_type not in valid_target_types:
                raise ValueError(
                    f"Unknown jammer target type: {target_type}. "
                    f"Expected one of: {sorted(valid_target_types)}."
                )

        if self.reward_failure_penalty >= 0.0:
            raise ValueError("reward_failure_penalty must be negative.")
        if self.reward_soi_max_gain_loss_db < 0.0:
            raise ValueError("reward_soi_max_gain_loss_db must be non-negative.")
        if self.reward_jammer_max_mean_leakage < 0.0:
            raise ValueError(
                "reward_jammer_max_mean_leakage must be non-negative."
            )
        if self.reward_sinr_scale_db <= 0.0:
            raise ValueError("reward_sinr_scale_db must be positive.")
        if self.reward_valid_min > self.reward_valid_max:
            raise ValueError("reward_valid_min cannot exceed reward_valid_max.")
        if self.reward_teacher_similarity_weight < 0.0:
            raise ValueError(
                "reward_teacher_similarity_weight must be non-negative."
            )
        if self.reward_jammer_leakage_penalty_weight < 0.0:
            raise ValueError(
                "reward_jammer_leakage_penalty_weight must be non-negative."
            )
        if self.reward_jammer_leakage_penalty_scale <= 0.0:
            raise ValueError(
                "reward_jammer_leakage_penalty_scale must be positive."
            )
        if self.reward_jammer_leakage_penalty_clip < 0.0:
            raise ValueError(
                "reward_jammer_leakage_penalty_clip must be non-negative."
            )
        if self.teacher_diagonal_loading < 0.0:
            raise ValueError(
                "teacher_diagonal_loading must be non-negative."
            )
        if self.teacher_similarity_epsilon <= 0.0:
            raise ValueError(
                "teacher_similarity_epsilon must be positive."
            )
        if self.direct_weight_min_power < 0.0:
            raise ValueError("direct_weight_min_power must be non-negative.")
        if self.max_sinr_loss_db <= 0.0:
            raise ValueError("max_sinr_loss_db must be positive.")
        if self.max_soi_gain_loss_db <= 0.0:
            raise ValueError("max_soi_gain_loss_db must be positive.")
        if self.max_jammer_leakage_loss <= 0.0:
            raise ValueError("max_jammer_leakage_loss must be positive.")
        if self.max_scenario_sampling_attempts <= 0:
            raise ValueError(
                "max_scenario_sampling_attempts must be a positive integer."
            )

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

    def _sample_valid_scenario(self) -> dict:
        """
        Sample a valid ScenarioGenerator scenario.
        """

        last_error: Exception | None = None

        for _ in range(self.max_scenario_sampling_attempts):
            try:
                desired_source_position = self._sample_source_position_xyz()

                jammers = self._build_random_jammers_for_episode(
                    desired_source_position=desired_source_position,
                )

                scenario_generator = ScenarioGenerator(
                    desired_source_position=desired_source_position,
                    jammers=jammers,
                    array_position=self.array_position,
                    num_steps=self.episode_length_physical_steps,
                    dt=self.dt,
                    desired_power=self.desired_power,
                    jammer_powers=self.jammer_powers,
                    noise_power=self.noise_power,
                )

                scenario = scenario_generator.generate()

                if self._scenario_has_valid_initial_geometry(scenario):
                    return scenario

            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                "Could not sample a valid Phase 9 coefficient-space scenario."
            ) from last_error

        raise RuntimeError("Could not sample a valid Phase 9 coefficient-space scenario.")

    def _build_random_jammers_for_episode(
        self,
        desired_source_position: np.ndarray,
    ) -> list:
        """
        Build the random jammer target objects for the current episode.
        """

        jammers = []

        for jammer_idx in range(self.num_active_jammers):
            target_type = str(self.np_random.choice(self.jammer_target_types))

            x0, y0, z0 = self._sample_source_position_xyz()

            attempts = 0

            while attempts < 100:
                separation_deg = self._compute_angular_error_from_positions_deg(
                    position_a=desired_source_position,
                    position_b=np.array([x0, y0, z0], dtype=float),
                )

                if separation_deg >= self.min_target_jammer_separation_deg:
                    break

                x0, y0, z0 = self._sample_source_position_xyz()
                attempts += 1

            seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))

            jammer = self._build_single_jammer_target(
                target_type=target_type,
                x0=x0,
                y0=y0,
                z0=z0,
                seed=seed,
            )

            jammers.append(jammer)

        return jammers

    def _build_single_jammer_target(
        self,
        target_type: str,
        x0: float,
        y0: float,
        z0: float,
        seed: int,
    ):
        """
        Build one jammer target object.
        """

        if target_type == "aircraft":
            return AircraftTarget(
                x0=x0,
                y0=y0,
                z0=z0,
                dt=self.dt,
                num_steps=self.episode_length_physical_steps,
                seed=seed,
            )

        if target_type == "drone":
            return DroneTarget(
                x0=x0,
                y0=y0,
                z0=z0,
                dt=self.dt,
                num_steps=self.episode_length_physical_steps,
                seed=seed,
            )

        if target_type == "dummy":
            return Dummy(
                x0=x0,
                y0=y0,
                z0=z0,
                dt=self.dt,
                num_steps=self.episode_length_physical_steps,
            )

        if target_type == "static":
            return StaticTarget(
                x0=x0,
                y0=y0,
                z0=z0,
                dt=self.dt,
                num_steps=self.episode_length_physical_steps,
            )

        if target_type == "truck":
            return TruckRoadTarget(
                x0=x0,
                y0=y0,
                z0=z0,
                dt=self.dt,
                num_steps=self.episode_length_physical_steps,
                seed=seed,
            )

        raise RuntimeError("Invalid target_type.")

    def _sample_source_position_xyz(self) -> np.ndarray:
        """
        Sample one Cartesian source position.
        """

        for _ in range(1000):
            x = float(
                self.np_random.uniform(
                    self.target_position_x_limits_m[0],
                    self.target_position_x_limits_m[1],
                )
            )
            y = float(
                self.np_random.uniform(
                    self.target_position_y_limits_m[0],
                    self.target_position_y_limits_m[1],
                )
            )
            z = float(
                self.np_random.uniform(
                    self.target_position_z_limits_m[0],
                    self.target_position_z_limits_m[1],
                )
            )

            position = np.array([x, y, z], dtype=float)

            distance = float(np.linalg.norm(position - self.array_position))

            if distance >= self.min_source_distance_m:
                return position

        raise RuntimeError(
            "Could not sample a valid source position. "
            "Try reducing min_source_distance_m."
        )

    def _scenario_has_valid_initial_geometry(self, scenario: dict) -> bool:
        """
        Check initial SOI-jammer and jammer-jammer angular separations.
        """

        theta_soi = float(scenario["desired"]["doa"]["theta"][0])
        phi_soi = float(scenario["desired"]["doa"]["phi"][0])

        jammer_thetas = [
            float(jammer["doa"]["theta"][0]) for jammer in scenario["jammers"]
        ]
        jammer_phis = [
            float(jammer["doa"]["phi"][0]) for jammer in scenario["jammers"]
        ]

        for theta_jam, phi_jam in zip(jammer_thetas, jammer_phis):
            separation_deg = self._compute_angular_error_deg(
                theta_a_rad=theta_soi,
                phi_a_rad=phi_soi,
                theta_b_rad=theta_jam,
                phi_b_rad=phi_jam,
            )

            if separation_deg < self.min_target_jammer_separation_deg:
                return False

        for i in range(len(jammer_thetas)):
            for j in range(i + 1, len(jammer_thetas)):
                separation_deg = self._compute_angular_error_deg(
                    theta_a_rad=jammer_thetas[i],
                    phi_a_rad=jammer_phis[i],
                    theta_b_rad=jammer_thetas[j],
                    phi_b_rad=jammer_phis[j],
                )

                if separation_deg < self.min_target_jammer_separation_deg:
                    return False

        return True

    def _load_current_directions_from_scenario(self, step_idx: int) -> None:
        """
        Load true SOI and jammer DOAs from the generated scenario.
        """

        if self.current_scenario is None:
            raise RuntimeError("No scenario is currently loaded.")

        step_idx = int(step_idx)

        self.current_theta_rad = float(
            self.current_scenario["desired"]["doa"]["theta"][step_idx]
        )
        self.current_phi_rad = float(
            self.current_scenario["desired"]["doa"]["phi"][step_idx]
        )

        self.current_jammer_thetas_rad = []
        self.current_jammer_phis_rad = []

        for jammer in self.current_scenario["jammers"]:
            self.current_jammer_thetas_rad.append(
                float(jammer["doa"]["theta"][step_idx])
            )
            self.current_jammer_phis_rad.append(
                float(jammer["doa"]["phi"][step_idx])
            )

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

                    state.extend(
                        [
                            float(u_jammer[0]),
                            float(u_jammer[1]),
                            float(u_jammer[2]),
                            1.0,
                        ]
                    )
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0])

            return np.array(state, dtype=np.float32)

        raise RuntimeError("Invalid observation mode.")

    def _build_basis_column_labels(self) -> list[str]:
        """Build deterministic semantic labels for the coefficient basis."""

        if self.coefficient_jammer_slots == 1:
            return [
                "soi",
                "jammer_center",
                "virtual_alpha_0",
                "virtual_alpha_90",
                "virtual_alpha_180",
                "virtual_alpha_270",
            ]

        labels = ["soi"]
        for jammer_idx in range(self.coefficient_jammer_slots):
            prefix = f"jammer_{jammer_idx + 1}"
            labels.extend(
                [
                    f"{prefix}_center",
                    f"{prefix}_virtual_alpha_0",
                    f"{prefix}_virtual_alpha_90",
                    f"{prefix}_virtual_alpha_180",
                    f"{prefix}_virtual_alpha_270",
                ]
            )
        return labels

    def _normalize_weights_like_environment(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Normalize deterministic weights with the array power convention."""

        weights_flat = np.asarray(
            weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        power = float(np.sum(np.abs(weights_flat) ** 2))

        if (
            not np.isfinite(power)
            or power <= 1e-15
            or not np.all(np.isfinite(weights_flat))
        ):
            raise ValueError("Invalid deterministic weight vector.")

        if self.array.normalize_power:
            weights_flat = weights_flat * np.sqrt(
                self.num_elements / power
            )

        return weights_flat.reshape(self.array.N, self.array.M)

    def _current_soi_direction_deg(self) -> tuple[float, float]:
        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Current SOI direction is not available.")

        return (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.mod(np.rad2deg(self.current_phi_rad), 360.0)),
        )

    @staticmethod
    def _angles_deg_to_unit_vector(
        theta_deg: float,
        phi_deg: float,
    ) -> np.ndarray:
        theta_rad = np.deg2rad(float(theta_deg))
        phi_rad = np.deg2rad(float(phi_deg))

        return np.array(
            [
                np.sin(theta_rad) * np.cos(phi_rad),
                np.sin(theta_rad) * np.sin(phi_rad),
                np.cos(theta_rad),
            ],
            dtype=float,
        )

    @staticmethod
    def _unit_vector_to_angles_deg(unit_vector: np.ndarray) -> tuple[float, float]:
        vector = np.asarray(unit_vector, dtype=float).reshape(3)
        vector = vector / np.linalg.norm(vector)

        theta_deg = float(
            np.rad2deg(
                np.arccos(
                    np.clip(vector[2], -1.0, 1.0)
                )
            )
        )
        phi_deg = float(
            np.mod(
                np.rad2deg(np.arctan2(vector[1], vector[0])),
                360.0,
            )
        )

        return theta_deg, phi_deg

    @staticmethod
    def _angular_separation_vectors_deg(
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        vector_a = np.asarray(vector_a, dtype=float).reshape(3)
        vector_b = np.asarray(vector_b, dtype=float).reshape(3)

        vector_a = vector_a / np.linalg.norm(vector_a)
        vector_b = vector_b / np.linalg.norm(vector_b)

        cosine = float(
            np.clip(
                np.dot(vector_a, vector_b),
                -1.0,
                1.0,
            )
        )

        return float(np.rad2deg(np.arccos(cosine)))

    @staticmethod
    def _build_tangent_basis_numpy(
        center_vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        center_vector = np.asarray(center_vector, dtype=float).reshape(3)
        center_vector = center_vector / np.linalg.norm(center_vector)

        reference = np.array([0.0, 0.0, 1.0], dtype=float)

        if abs(float(np.dot(reference, center_vector))) > 0.90:
            reference = np.array([1.0, 0.0, 0.0], dtype=float)

        tangent_1 = np.cross(reference, center_vector)
        tangent_1 = tangent_1 / np.linalg.norm(tangent_1)

        tangent_2 = np.cross(center_vector, tangent_1)
        tangent_2 = tangent_2 / np.linalg.norm(tangent_2)

        return tangent_1, tangent_2

    def _generate_cross4_directions_deg(
        self,
        center_direction_deg: tuple[float, float],
        radius_deg: float,
    ) -> list[tuple[float, float]]:
        """Generate the visible cross4 directions used by the soft teacher."""

        center_vector = self._angles_deg_to_unit_vector(
            *center_direction_deg
        )
        tangent_1, tangent_2 = self._build_tangent_basis_numpy(center_vector)
        radius_rad = np.deg2rad(float(radius_deg))

        directions = []

        for alpha_rad in np.linspace(
            0.0,
            2.0 * np.pi,
            num=4,
            endpoint=False,
        ):
            tangent_direction = (
                np.cos(alpha_rad) * tangent_1
                + np.sin(alpha_rad) * tangent_2
            )

            virtual_vector = (
                np.cos(radius_rad) * center_vector
                + np.sin(radius_rad) * tangent_direction
            )
            virtual_vector = virtual_vector / np.linalg.norm(virtual_vector)

            if virtual_vector[2] < 0.0:
                continue

            directions.append(
                self._unit_vector_to_angles_deg(virtual_vector)
            )

        return directions

    def _generate_cross4_semantic_slots_deg(
        self,
        center_direction_deg: tuple[float, float],
        radius_deg: float,
        soi_direction_deg: tuple[float, float],
    ) -> list[tuple[float, float] | None]:
        """Generate four fixed cross4 semantic slots exactly as in notebook 56."""

        center_vector = self._angles_deg_to_unit_vector(
            *center_direction_deg
        )
        soi_vector = self._angles_deg_to_unit_vector(
            *soi_direction_deg
        )
        tangent_1, tangent_2 = self._build_tangent_basis_numpy(center_vector)
        radius_rad = np.deg2rad(float(radius_deg))

        slots: list[tuple[float, float] | None] = []
        retained_vectors = []

        for alpha_rad in [
            0.0,
            0.5 * np.pi,
            np.pi,
            1.5 * np.pi,
        ]:
            tangent_direction = (
                np.cos(alpha_rad) * tangent_1
                + np.sin(alpha_rad) * tangent_2
            )

            virtual_vector = (
                np.cos(radius_rad) * center_vector
                + np.sin(radius_rad) * tangent_direction
            )
            virtual_vector = virtual_vector / np.linalg.norm(virtual_vector)

            if virtual_vector[2] < 0.0:
                slots.append(None)
                continue

            separation_to_soi_deg = self._angular_separation_vectors_deg(
                virtual_vector,
                soi_vector,
            )

            if (
                separation_to_soi_deg
                < self.MIN_VIRTUAL_DIRECTION_SOI_SEPARATION_DEG
            ):
                slots.append(None)
                continue

            duplicate = False

            for retained_vector in retained_vectors:
                separation_deg = self._angular_separation_vectors_deg(
                    virtual_vector,
                    retained_vector,
                )

                if separation_deg < self.DIRECTION_DEDUPLICATION_DEG:
                    duplicate = True
                    break

            if duplicate:
                slots.append(None)
                continue

            retained_vectors.append(virtual_vector)
            slots.append(self._unit_vector_to_angles_deg(virtual_vector))

        if len(slots) != 4:
            raise RuntimeError("cross4 semantic slot construction failed.")

        return slots

    def _deduplicate_directions_deg(
        self,
        directions_deg: list[tuple[float, float]],
        tolerance_deg: float,
    ) -> list[tuple[float, float]]:
        unique_directions = []

        for direction_deg in directions_deg:
            direction_vector = self._angles_deg_to_unit_vector(
                *direction_deg
            )
            duplicate = False

            for existing_direction in unique_directions:
                existing_vector = self._angles_deg_to_unit_vector(
                    *existing_direction
                )

                if (
                    self._angular_separation_vectors_deg(
                        direction_vector,
                        existing_vector,
                    )
                    <= float(tolerance_deg)
                ):
                    duplicate = True
                    break

            if not duplicate:
                unique_directions.append(
                    (float(direction_deg[0]), float(direction_deg[1]))
                )

        return unique_directions

    def _steering_vector_for_direction(
        self,
        direction_deg: tuple[float, float],
    ) -> np.ndarray:
        return get_steering_vector(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            direction=direction_deg,
        ).astype(np.complex128).reshape(self.num_elements)

    def _build_virtual_directions_by_jammer(
        self,
        candidate_spec: dict,
    ) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]], dict]:
        """Build the filtered virtual directions used by the frozen soft teacher."""

        jammer_centres = [
            (float(theta_deg), float(phi_deg))
            for theta_deg, phi_deg in self._get_current_jammer_directions_deg()
        ]

        diagnostics = {
            "num_jammer_centres": len(jammer_centres),
            "num_virtual_requested": 0,
            "num_virtual_visible": 0,
            "num_virtual_removed_near_soi": 0,
            "num_virtual_after_filtering": 0,
        }

        if (
            candidate_spec["candidate_type"] == "point_teacher"
            or float(candidate_spec["virtual_total_power_ratio"]) <= 0.0
        ):
            return (
                jammer_centres,
                [[] for _ in jammer_centres],
                diagnostics,
            )

        soi_vector = self._angles_deg_to_unit_vector(
            *self._current_soi_direction_deg()
        )

        virtual_groups = []

        for jammer_direction in jammer_centres:
            diagnostics["num_virtual_requested"] += 4

            generated = self._generate_cross4_directions_deg(
                center_direction_deg=jammer_direction,
                radius_deg=float(candidate_spec["radius_deg"]),
            )

            diagnostics["num_virtual_visible"] += len(generated)

            retained = []

            for virtual_direction in generated:
                virtual_vector = self._angles_deg_to_unit_vector(
                    *virtual_direction
                )

                separation_to_soi_deg = self._angular_separation_vectors_deg(
                    virtual_vector,
                    soi_vector,
                )

                if (
                    separation_to_soi_deg
                    < self.MIN_VIRTUAL_DIRECTION_SOI_SEPARATION_DEG
                ):
                    diagnostics["num_virtual_removed_near_soi"] += 1
                    continue

                retained.append(virtual_direction)

            retained = self._deduplicate_directions_deg(
                retained,
                self.DIRECTION_DEDUPLICATION_DEG,
            )

            diagnostics["num_virtual_after_filtering"] += len(retained)
            virtual_groups.append(retained)

        return jammer_centres, virtual_groups, diagnostics

    def _build_point_base_weights(self) -> np.ndarray:
        """Build the frozen ``target_or_zero_point`` deterministic base."""

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Current SOI direction is not available for point-base weights."
            )

        target_direction_deg = self._current_soi_direction_deg()

        weights_flat = target_or_zero_weights(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            target_directions=[target_direction_deg],
            zero_directions=self._get_current_jammer_directions_deg(),
            diagonal_loading=self.teacher_diagonal_loading,
            use_pinv=self.teacher_use_pinv,
        ).astype(np.complex128).reshape(self.num_elements)

        return self._normalize_weights_like_environment(weights_flat)

    def _orthonormalize_semantic_basis_mgs(
        self,
        basis: np.ndarray,
        active_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Orthonormalize the semantic basis with deterministic modified
        Gram-Schmidt while preserving the fixed column ordering.

        Columns that are inactive in the semantic basis remain exactly zero.
        Columns that become numerically dependent on earlier active columns
        also become zero and are marked inactive.

        No arbitrary phase normalization is applied: each surviving column
        inherits the phase convention of the original semantic steering
        vector after orthogonal projection.
        """

        basis = np.asarray(
            basis,
            dtype=np.complex128,
        )

        active_mask = np.asarray(
            active_mask,
            dtype=np.float32,
        ).reshape(
            self.num_complex_coefficients
        )

        expected_shape = (
            self.num_elements,
            self.num_complex_coefficients,
        )

        if basis.shape != expected_shape:
            raise ValueError(
                "Unexpected semantic basis shape for orthonormalization."
            )

        orthonormal_basis = np.zeros_like(
            basis,
            dtype=np.complex128,
        )

        orthonormal_active_mask = np.zeros(
            self.num_complex_coefficients,
            dtype=np.float32,
        )

        for column_idx in range(
            self.num_complex_coefficients
        ):
            if active_mask[
                column_idx
            ] <= 0.5:
                continue

            original_column = basis[
                :,
                column_idx
            ].copy()

            original_norm = float(
                np.linalg.norm(
                    original_column
                )
            )

            if (
                not np.isfinite(
                    original_norm
                )
                or original_norm <= 0.0
            ):
                continue

            vector = original_column.copy()

            # Two deterministic MGS passes improve numerical orthogonality
            # without changing the semantic ordering.
            for _ in range(2):
                for previous_idx in range(
                    column_idx
                ):
                    if orthonormal_active_mask[
                        previous_idx
                    ] <= 0.5:
                        continue

                    previous_column = (
                        orthonormal_basis[
                            :,
                            previous_idx
                        ]
                    )

                    vector = (
                        vector
                        - previous_column
                        * np.vdot(
                            previous_column,
                            vector,
                        )
                    )

            vector_norm = float(
                np.linalg.norm(
                    vector
                )
            )

            dependency_threshold = (
                self.coefficient_basis_orthogonalization_tol
                * max(
                    original_norm,
                    1.0,
                )
            )

            if (
                not np.isfinite(
                    vector_norm
                )
                or vector_norm <= dependency_threshold
            ):
                continue

            orthonormal_basis[
                :,
                column_idx
            ] = (
                vector
                / vector_norm
            )

            orthonormal_active_mask[
                column_idx
            ] = 1.0

        return (
            orthonormal_basis,
            orthonormal_active_mask,
        )


    def _build_coefficient_basis(self) -> dict:
        """Build the fixed-semantic scenario-dependent coefficient basis."""

        soi_direction = self._current_soi_direction_deg()
        jammer_directions = [
            (float(theta_deg), float(phi_deg))
            for theta_deg, phi_deg in self._get_current_jammer_directions_deg()
        ]

        candidate_spec = self.ADAPTIVE_TEACHER_BY_JAMMER_COUNT[
            int(self.num_active_jammers)
        ]

        columns = [
            self._steering_vector_for_direction(soi_direction)
        ]
        active_mask = [1.0]
        virtual_slots_by_jammer = []

        for jammer_slot_idx in range(self.coefficient_jammer_slots):
            if jammer_slot_idx >= len(jammer_directions):
                columns.extend(
                    [
                        np.zeros(self.num_elements, dtype=np.complex128)
                        for _ in range(5)
                    ]
                )
                active_mask.extend([0.0] * 5)
                virtual_slots_by_jammer.append([None, None, None, None])
                continue

            jammer_direction = jammer_directions[jammer_slot_idx]
            columns.append(
                self._steering_vector_for_direction(jammer_direction)
            )
            active_mask.append(1.0)

            semantic_slots = self._generate_cross4_semantic_slots_deg(
                center_direction_deg=jammer_direction,
                radius_deg=float(candidate_spec["radius_deg"]),
                soi_direction_deg=soi_direction,
            )
            virtual_slots_by_jammer.append(semantic_slots)

            for direction in semantic_slots:
                if direction is None:
                    columns.append(
                        np.zeros(self.num_elements, dtype=np.complex128)
                    )
                    active_mask.append(0.0)
                else:
                    columns.append(
                        self._steering_vector_for_direction(direction)
                    )
                    active_mask.append(1.0)

        semantic_basis = np.column_stack(columns)
        semantic_active_mask = np.asarray(
            active_mask,
            dtype=np.float32,
        )

        expected_shape = (
            self.num_elements,
            self.num_complex_coefficients,
        )
        if semantic_basis.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected coefficient basis shape: {semantic_basis.shape}; "
                f"expected {expected_shape}."
            )

        if semantic_active_mask.shape != (
            self.num_complex_coefficients,
        ):
            raise RuntimeError(
                "Unexpected coefficient basis mask shape."
            )

        semantic_basis_rank = int(
            np.linalg.matrix_rank(
                semantic_basis
            )
        )

        if self.coefficient_basis_mode == "raw":
            decoder_basis = semantic_basis.copy()
            decoder_active_mask = semantic_active_mask.copy()

        elif self.coefficient_basis_mode == "orthonormal_mgs":
            (
                decoder_basis,
                decoder_active_mask,
            ) = self._orthonormalize_semantic_basis_mgs(
                basis=semantic_basis,
                active_mask=semantic_active_mask,
            )

        else:
            raise RuntimeError(
                "Invalid coefficient_basis_mode after validation."
            )

        decoder_basis_rank = int(
            np.linalg.matrix_rank(
                decoder_basis
            )
        )

        active_decoder_columns = (
            decoder_basis[
                :,
                decoder_active_mask > 0.5,
            ]
        )

        if active_decoder_columns.shape[1] <= 1:
            decoder_gram_condition_number = 1.0

        else:
            decoder_gram = (
                active_decoder_columns.conj().T
                @ active_decoder_columns
            )

            decoder_gram_condition_number = float(
                np.linalg.cond(
                    decoder_gram
                )
            )

        active_semantic_columns = (
            semantic_basis[
                :,
                semantic_active_mask > 0.5,
            ]
        )

        if active_semantic_columns.shape[1] <= 1:
            semantic_gram_condition_number = 1.0

        else:
            semantic_gram = (
                active_semantic_columns.conj().T
                @ active_semantic_columns
            )

            semantic_gram_condition_number = float(
                np.linalg.cond(
                    semantic_gram
                )
            )

        return {
            "basis_matrix": decoder_basis,
            "basis_active_mask": decoder_active_mask,
            "basis_column_labels": self.basis_column_labels.copy(),
            "basis_rank": decoder_basis_rank,
            "coefficient_basis_mode": self.coefficient_basis_mode,
            "semantic_basis_matrix": semantic_basis,
            "semantic_basis_active_mask": semantic_active_mask,
            "semantic_basis_rank": semantic_basis_rank,
            "semantic_gram_condition_number": (
                semantic_gram_condition_number
            ),
            "decoder_gram_condition_number": (
                decoder_gram_condition_number
            ),
            "soi_direction": soi_direction,
            "jammer_directions": jammer_directions,
            "virtual_slots_by_jammer": virtual_slots_by_jammer,
        }

    def _decode_coefficient_action(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Decode one normalized coefficient action into raw complex weights."""

        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        real_action = action[: self.num_complex_coefficients].astype(float)
        imag_action = action[self.num_complex_coefficients :].astype(float)

        normalized_complex_coefficients = real_action + 1j * imag_action
        complex_coefficients = (
            normalized_complex_coefficients * self.coefficient_scales
        )

        basis_info = self._build_coefficient_basis()
        active_mask = basis_info["basis_active_mask"].astype(float)
        complex_coefficients = complex_coefficients * active_mask

        residual_flat = (
            basis_info["basis_matrix"] @ complex_coefficients
        )

        base_weights = self._build_point_base_weights()
        base_flat = np.asarray(
            base_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        raw_weights_flat = base_flat + residual_flat
        raw_weight_power = float(np.sum(np.abs(raw_weights_flat) ** 2))

        invalid_coefficient_action = bool(
            not np.isfinite(raw_weight_power)
            or raw_weight_power <= self.direct_weight_min_power
            or not np.all(np.isfinite(raw_weights_flat))
            or not np.all(np.isfinite(complex_coefficients))
        )

        action_info = {
            "raw_action": action.copy(),
            "coefficient_action": action.copy(),
            "coefficient_real_action": real_action.copy(),
            "coefficient_imag_action": imag_action.copy(),
            "complex_coefficients": complex_coefficients.copy(),
            "coefficient_scales": self.coefficient_scales.copy(),
            "basis_matrix": basis_info["basis_matrix"].copy(),
            "basis_active_mask": basis_info["basis_active_mask"].copy(),
            "basis_column_labels": basis_info["basis_column_labels"].copy(),
            "basis_rank": int(basis_info["basis_rank"]),
            "coefficient_basis_mode": (
                basis_info[
                    "coefficient_basis_mode"
                ]
            ),
            "semantic_basis_matrix": (
                basis_info[
                    "semantic_basis_matrix"
                ].copy()
            ),
            "semantic_basis_active_mask": (
                basis_info[
                    "semantic_basis_active_mask"
                ].copy()
            ),
            "semantic_basis_rank": int(
                basis_info[
                    "semantic_basis_rank"
                ]
            ),
            "semantic_gram_condition_number": float(
                basis_info[
                    "semantic_gram_condition_number"
                ]
            ),
            "decoder_gram_condition_number": float(
                basis_info[
                    "decoder_gram_condition_number"
                ]
            ),
            "residual_weights": residual_flat.reshape(
                self.array.N,
                self.array.M,
            ).copy(),
            "residual_norm": float(np.linalg.norm(residual_flat)),
            "base_weights": np.asarray(
                base_weights,
                dtype=np.complex128,
            ).copy(),
            "raw_weights_before_normalization": raw_weights_flat.reshape(
                self.array.N,
                self.array.M,
            ).copy(),
            "raw_weight_power_before_normalization": raw_weight_power,
            "invalid_coefficient_action": invalid_coefficient_action,
            "virtual_slots_by_jammer": basis_info[
                "virtual_slots_by_jammer"
            ],
        }

        return (
            raw_weights_flat.reshape(self.array.N, self.array.M),
            action_info,
        )

    def _action_to_coefficient_weights(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        return self._decode_coefficient_action(action)

    def _build_invalid_action_info(self) -> dict:
        """Build logging information for an invalid coefficient action."""

        try:
            basis_info = self._build_coefficient_basis()
            basis_matrix = basis_info["basis_matrix"]
            basis_active_mask = basis_info["basis_active_mask"]
            basis_rank = basis_info["basis_rank"]
            semantic_basis_matrix = basis_info[
                "semantic_basis_matrix"
            ]
            semantic_basis_active_mask = basis_info[
                "semantic_basis_active_mask"
            ]
            semantic_basis_rank = basis_info[
                "semantic_basis_rank"
            ]
            semantic_gram_condition_number = basis_info[
                "semantic_gram_condition_number"
            ]
            decoder_gram_condition_number = basis_info[
                "decoder_gram_condition_number"
            ]
            virtual_slots_by_jammer = basis_info["virtual_slots_by_jammer"]
        except Exception:
            basis_matrix = np.zeros(
                (self.num_elements, self.num_complex_coefficients),
                dtype=np.complex128,
            )
            basis_active_mask = np.zeros(
                self.num_complex_coefficients,
                dtype=np.float32,
            )
            basis_rank = 0
            semantic_basis_matrix = basis_matrix.copy()
            semantic_basis_active_mask = basis_active_mask.copy()
            semantic_basis_rank = 0
            semantic_gram_condition_number = float("inf")
            decoder_gram_condition_number = float("inf")
            virtual_slots_by_jammer = []

        try:
            base_weights = self._build_point_base_weights()
        except Exception:
            base_weights = np.zeros(
                (self.array.N, self.array.M),
                dtype=np.complex128,
            )

        return {
            "raw_action": np.zeros(self.action_dim, dtype=np.float32),
            "coefficient_action": np.zeros(self.action_dim, dtype=np.float32),
            "coefficient_real_action": np.zeros(
                self.num_complex_coefficients,
                dtype=float,
            ),
            "coefficient_imag_action": np.zeros(
                self.num_complex_coefficients,
                dtype=float,
            ),
            "complex_coefficients": np.zeros(
                self.num_complex_coefficients,
                dtype=np.complex128,
            ),
            "coefficient_scales": self.coefficient_scales.copy(),
            "basis_matrix": basis_matrix.copy(),
            "basis_active_mask": basis_active_mask.copy(),
            "basis_column_labels": self.basis_column_labels.copy(),
            "basis_rank": int(basis_rank),
            "coefficient_basis_mode": self.coefficient_basis_mode,
            "semantic_basis_matrix": (
                semantic_basis_matrix.copy()
            ),
            "semantic_basis_active_mask": (
                semantic_basis_active_mask.copy()
            ),
            "semantic_basis_rank": int(
                semantic_basis_rank
            ),
            "semantic_gram_condition_number": float(
                semantic_gram_condition_number
            ),
            "decoder_gram_condition_number": float(
                decoder_gram_condition_number
            ),
            "residual_weights": np.zeros(
                (self.array.N, self.array.M),
                dtype=np.complex128,
            ),
            "residual_norm": 0.0,
            "base_weights": np.asarray(
                base_weights,
                dtype=np.complex128,
            ).copy(),
            "raw_weights_before_normalization": np.asarray(
                base_weights,
                dtype=np.complex128,
            ).copy(),
            "raw_weight_power_before_normalization": float(
                np.sum(np.abs(np.asarray(base_weights).reshape(-1)) ** 2)
            ),
            "invalid_coefficient_action": True,
            "virtual_slots_by_jammer": virtual_slots_by_jammer,
        }

    def _finalize_action_info(
        self,
        action_info: dict,
        normalized_weights: np.ndarray,
        invalid_coefficient_action: bool,
    ) -> dict:
        """Add normalized-weight diagnostics to coefficient action information."""

        normalized_weights = np.asarray(
            normalized_weights,
            dtype=np.complex128,
        ).reshape(self.array.N, self.array.M)
        normalized_flat = normalized_weights.reshape(self.num_elements)

        result = dict(action_info)
        result.update(
            {
                "weights": normalized_weights.copy(),
                "final_magnitude": np.abs(normalized_flat).copy(),
                "final_phase_rad": np.angle(normalized_flat).copy(),
                "final_phase_norm": np.angle(normalized_flat).copy() / np.pi,
                "final_weight_power": float(
                    np.sum(np.abs(normalized_flat) ** 2)
                ),
                "invalid_coefficient_action": bool(
                    invalid_coefficient_action
                ),
            }
        )
        return result

    def _solve_distortionless_mvdr(
        self,
        covariance: np.ndarray,
        target_steering: np.ndarray,
    ) -> tuple[np.ndarray, str, bool]:
        covariance = np.asarray(covariance, dtype=np.complex128)
        target_steering = np.asarray(
            target_steering,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        solver_name = "solve"
        used_fallback = False

        try:
            solved = np.linalg.solve(covariance, target_steering)
        except np.linalg.LinAlgError:
            solver_name = "pinv"
            used_fallback = True
            solved = np.linalg.pinv(covariance) @ target_steering

        denominator = np.vdot(target_steering, solved)

        if (
            not np.isfinite(denominator)
            or abs(denominator) <= self.SOFT_NULL_DENOMINATOR_EPSILON
        ):
            solver_name = "pinv_fallback"
            used_fallback = True
            solved = np.linalg.pinv(covariance) @ target_steering
            denominator = np.vdot(target_steering, solved)

        if (
            not np.isfinite(denominator)
            or abs(denominator) <= self.SOFT_NULL_DENOMINATOR_EPSILON
        ):
            raise RuntimeError("Invalid MVDR distortionless denominator.")

        weights = solved / denominator

        if not np.all(np.isfinite(weights)):
            raise RuntimeError("Soft-null MVDR weights are not finite.")

        return weights, solver_name, used_fallback

    def _build_soft_mvdr_weights(
        self,
        candidate_spec: dict,
    ) -> tuple[np.ndarray, dict]:
        target_direction = self._current_soi_direction_deg()
        target_steering = self._steering_vector_for_direction(target_direction)

        (
            jammer_centres,
            virtual_groups,
            diagnostics,
        ) = self._build_virtual_directions_by_jammer(candidate_spec)

        identity = np.eye(self.num_elements, dtype=np.complex128)
        covariance = (
            float(self.noise_power) * identity
            + float(self.SOFT_NULL_DIAGONAL_LOADING) * identity
        )

        total_virtual_power = 0.0

        for jammer_index, jammer_direction in enumerate(jammer_centres):
            jammer_power = (
                float(self.jammer_powers[jammer_index])
                if jammer_index < len(self.jammer_powers)
                else 1.0
            )

            center_power = (
                jammer_power * float(self.SOFT_NULL_CENTER_POWER_SCALE)
            )
            center_steering = self._steering_vector_for_direction(
                jammer_direction
            )

            covariance += center_power * np.outer(
                center_steering,
                np.conj(center_steering),
            )

            retained_virtuals = virtual_groups[jammer_index]
            total_virtual_power_for_jammer = (
                center_power
                * float(candidate_spec["virtual_total_power_ratio"])
            )

            if retained_virtuals:
                virtual_power_per_direction = (
                    total_virtual_power_for_jammer / len(retained_virtuals)
                )
            else:
                virtual_power_per_direction = 0.0
                total_virtual_power_for_jammer = 0.0

            for virtual_direction in retained_virtuals:
                virtual_steering = self._steering_vector_for_direction(
                    virtual_direction
                )
                covariance += virtual_power_per_direction * np.outer(
                    virtual_steering,
                    np.conj(virtual_steering),
                )

            total_virtual_power += total_virtual_power_for_jammer

        covariance = 0.5 * (
            covariance + np.conj(covariance.T)
        )
        condition_number = float(np.linalg.cond(covariance))

        try:
            (
                weights_flat,
                solver_name,
                used_fallback,
            ) = self._solve_distortionless_mvdr(
                covariance,
                target_steering,
            )
        except Exception:
            fallback_covariance = (
                covariance
                + float(self.SOFT_NULL_FALLBACK_DIAGONAL_LOADING) * identity
            )
            (
                weights_flat,
                solver_name,
                _,
            ) = self._solve_distortionless_mvdr(
                fallback_covariance,
                target_steering,
            )
            solver_name = f"{solver_name}_loaded_fallback"
            used_fallback = True
            condition_number = float(np.linalg.cond(fallback_covariance))

        weights = self._normalize_weights_like_environment(weights_flat)

        diagnostics.update(
            {
                "solver": solver_name,
                "used_fallback": bool(used_fallback),
                "matrix_condition_number": condition_number,
                "total_virtual_covariance_power": float(total_virtual_power),
                "final_weight_power": float(
                    np.sum(np.abs(np.asarray(weights).reshape(-1)) ** 2)
                ),
            }
        )

        return weights, diagnostics

    def _build_adaptive_teacher_weights(
        self,
        jammer_count: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        if jammer_count is None:
            jammer_count = int(self.num_active_jammers)

        jammer_count = int(jammer_count)
        candidate_spec = self.ADAPTIVE_TEACHER_BY_JAMMER_COUNT[jammer_count]

        if candidate_spec["candidate_type"] == "point_teacher":
            weights = self._build_point_base_weights()
            diagnostics = {
                "candidate_id": candidate_spec["candidate_id"],
                "candidate_type": "point_teacher",
                "used_fallback": False,
                "num_virtual_after_filtering": 0,
            }
            return weights, diagnostics

        weights, diagnostics = self._build_soft_mvdr_weights(candidate_spec)
        diagnostics["candidate_id"] = candidate_spec["candidate_id"]
        diagnostics["candidate_type"] = candidate_spec["candidate_type"]

        return weights, diagnostics

    def _build_teacher_weights(self) -> np.ndarray:
        """Build the frozen Phase 9 teacher for the current jammer count."""

        weights, _ = self._build_adaptive_teacher_weights(
            jammer_count=self.num_active_jammers
        )
        return np.asarray(weights, dtype=np.complex128).reshape(
            self.array.N,
            self.array.M,
        )


    def _compute_teacher_weight_similarity(
        self,
        agent_weights: np.ndarray,
        teacher_weights: np.ndarray,
    ) -> dict:
        """
        Compute phase-invariant normalized similarity between complex weights.
        """

        agent_flat = np.asarray(
            agent_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        teacher_flat = np.asarray(
            teacher_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        agent_norm = float(np.linalg.norm(agent_flat))
        teacher_norm = float(np.linalg.norm(teacher_flat))
        denominator = agent_norm * teacher_norm

        if (
            not np.isfinite(denominator)
            or denominator <= self.teacher_similarity_epsilon
            or not np.all(np.isfinite(agent_flat))
            or not np.all(np.isfinite(teacher_flat))
        ):
            raise RuntimeError(
                "Cannot compute teacher similarity from invalid weights."
            )

        inner_product = np.vdot(teacher_flat, agent_flat)

        similarity = float(
            np.clip(
                np.abs(inner_product) / denominator,
                0.0,
                1.0,
            )
        )

        global_phase_offset_rad = float(np.angle(inner_product))

        aligned_agent_unit = (
            agent_flat
            * np.exp(-1j * global_phase_offset_rad)
            / agent_norm
        )
        teacher_unit = teacher_flat / teacher_norm

        aligned_complex_mse = float(
            np.mean(
                np.abs(aligned_agent_unit - teacher_unit) ** 2
            )
        )

        return {
            "teacher_weight_similarity": similarity,
            "teacher_weight_loss": float(1.0 - similarity),
            "teacher_global_phase_offset_rad": global_phase_offset_rad,
            "teacher_global_phase_offset_deg": float(
                np.rad2deg(global_phase_offset_rad)
            ),
            "teacher_aligned_complex_mse": aligned_complex_mse,
        }


    def _build_invalid_teacher_metrics(self) -> dict:
        """Return penalized teacher metrics for invalid numerical cases."""

        return {
            "teacher_weight_similarity": 0.0,
            "teacher_weight_loss": 1.0,
            "teacher_global_phase_offset_rad": 0.0,
            "teacher_global_phase_offset_deg": 0.0,
            "teacher_aligned_complex_mse": 1.0,
        }


    def _evaluate_fixed_weights_at_current_step(
        self,
        weights: np.ndarray,
        teacher_weights: np.ndarray,
        teacher_weights_are_valid: bool,
        invalid_coefficient_action: bool,
    ) -> dict:
        """
        Evaluate fixed direct weights at the current physical step.

        ``teacher_weights`` are computed once at the beginning of the current
        control block and remain unchanged throughout the block. This makes
        the teacher obey the same K-step update constraint as the agent.
        """

        numerical_error = False

        try:
            sinr_db = self._compute_sinr_for_weights(weights)
        except Exception:
            sinr_db = self.invalid_sinr_db
            numerical_error = True
        if not np.isfinite(sinr_db):
            sinr_db = self.invalid_sinr_db
            numerical_error = True

        try:
            reference_weights = self._build_mvdr_reference_weights()
            reference_sinr_db = self._compute_sinr_for_weights(reference_weights)
        except Exception:
            reference_sinr_db = self.invalid_sinr_db
            numerical_error = True
        if not np.isfinite(reference_sinr_db):
            reference_sinr_db = self.invalid_sinr_db
            numerical_error = True

        sinr_loss_db = reference_sinr_db - sinr_db
        if not np.isfinite(sinr_loss_db):
            sinr_loss_db = self.max_sinr_loss_db
            numerical_error = True
        sinr_loss_db = max(0.0, float(sinr_loss_db))
        clipped_sinr_loss_db = min(sinr_loss_db, self.max_sinr_loss_db)

        try:
            soi_gain_metrics = self._compute_soi_gain_metrics(weights=weights)
        except Exception:
            soi_gain_metrics = self._build_invalid_soi_gain_metrics()
            numerical_error = True

        try:
            jammer_leakage_metrics = self._compute_jammer_leakage_metrics(
                weights=weights,
                soi_gain_linear=soi_gain_metrics["soi_gain_linear"],
            )
        except Exception:
            jammer_leakage_metrics = self._build_invalid_jammer_leakage_metrics()
            numerical_error = True

        if teacher_weights_are_valid:
            try:
                teacher_metrics = self._compute_teacher_weight_similarity(
                    agent_weights=weights,
                    teacher_weights=teacher_weights,
                )
            except Exception:
                teacher_metrics = self._build_invalid_teacher_metrics()
                numerical_error = True
        else:
            teacher_metrics = self._build_invalid_teacher_metrics()
            numerical_error = True

        reward, reward_info = self._compute_threshold_reward(
            sinr_db=sinr_db,
            sinr_loss_db=sinr_loss_db,
            soi_gain_loss_db=soi_gain_metrics["soi_gain_loss_db"],
            jammer_mean_leakage=jammer_leakage_metrics["jammer_leakage_loss"],
            teacher_weight_similarity=teacher_metrics[
                "teacher_weight_similarity"
            ],
            invalid_coefficient_action=invalid_coefficient_action,
            numerical_error=numerical_error,
        )

        metrics = {
            "reward": float(reward),
            "sinr_db": float(sinr_db),
            "reference_sinr_db": float(reference_sinr_db),
            "sinr_loss_db": float(sinr_loss_db),
            "clipped_sinr_loss_db": float(clipped_sinr_loss_db),
            "numerical_error": bool(numerical_error),
            "invalid_coefficient_action": bool(invalid_coefficient_action),
            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,
            "theta_target_deg": float(np.rad2deg(self.current_theta_rad)),
            "phi_target_deg": float(np.rad2deg(self.current_phi_rad)),
            "jammer_thetas_rad": self.current_jammer_thetas_rad.copy(),
            "jammer_phis_rad": self.current_jammer_phis_rad.copy(),
            "jammer_thetas_deg": [
                float(np.rad2deg(theta)) for theta in self.current_jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(phi)) for phi in self.current_jammer_phis_rad
            ],
            "jammers_directions_deg": self._get_current_jammer_directions_deg(),
            "num_active_jammers": self.num_active_jammers,
            "jammers_powers": self.jammer_powers.copy(),
            "teacher_weights": teacher_weights.copy(),
            **teacher_metrics,
            **reward_info,
            **soi_gain_metrics,
            **jammer_leakage_metrics,
        }
        return metrics


    def _compute_sinr_for_weights(self, weights: np.ndarray) -> float:
        """
        Compute SINR for the current true SOI/jammer scene and a given
        weight matrix.
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

    def _build_mvdr_reference_weights(self) -> np.ndarray:
        """
        Build instantaneous MVDR reference weights using true SOI and true
        jammer DOAs at the current physical step.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Environment must be reset before computing reference weights."
            )

        target_direction = (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.rad2deg(self.current_phi_rad)),
        )

        jammer_directions = self._get_current_jammer_directions_deg()

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
        Build interference-plus-noise covariance matrix from true active
        jammer directions.
        """

        num_elements = self.num_elements

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

    def _compute_soi_gain_metrics(self, weights: np.ndarray) -> dict:
        """
        Compute gain and gain-loss metrics toward the SOI.
        """

        target_direction_deg = (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.rad2deg(self.current_phi_rad)),
        )

        soi_gain_linear = self._compute_array_response_gain_linear(
            weights=weights,
            direction_deg=target_direction_deg,
        )

        reference_weights = self._build_steering_weights(
            theta_rad=self.current_theta_rad,
            phi_rad=self.current_phi_rad,
        )

        reference_soi_gain_linear = self._compute_array_response_gain_linear(
            weights=reference_weights,
            direction_deg=target_direction_deg,
        )

        soi_gain_db = self._linear_to_db(soi_gain_linear)
        reference_soi_gain_db = self._linear_to_db(reference_soi_gain_linear)

        soi_gain_loss_db = reference_soi_gain_db - soi_gain_db

        if not np.isfinite(soi_gain_loss_db):
            soi_gain_loss_db = self.max_soi_gain_loss_db

        soi_gain_loss_db = max(0.0, float(soi_gain_loss_db))
        clipped_soi_gain_loss_db = min(
            soi_gain_loss_db,
            self.max_soi_gain_loss_db,
        )

        return {
            "soi_gain_linear": float(soi_gain_linear),
            "soi_gain_db": float(soi_gain_db),
            "reference_soi_gain_linear": float(reference_soi_gain_linear),
            "reference_soi_gain_db": float(reference_soi_gain_db),
            "soi_gain_loss_db": float(soi_gain_loss_db),
            "clipped_soi_gain_loss_db": float(clipped_soi_gain_loss_db),
        }

    def _compute_jammer_leakage_metrics(
        self,
        weights: np.ndarray,
        soi_gain_linear: float,
    ) -> dict:
        """
        Compute jammer leakage metrics.

        The main jammer loss is the mean ratio between jammer-direction gain
        and SOI-direction gain:

            leakage_j = G_jammer_j / (G_soi + eps)

        Lower is better.
        """

        if self.num_active_jammers == 0:
            return {
                "jammer_gains_linear": [],
                "jammer_gains_db": [],
                "jammer_leakage_values": [],
                "jammer_leakage_loss": 0.0,
                "clipped_jammer_leakage_loss": 0.0,
                "jammer_leakage_loss_db": float("-inf"),
                "jammer_leakage_max": 0.0,
            }

        eps = 1e-12
        denominator = max(float(soi_gain_linear), eps)

        jammer_gains_linear = []
        jammer_gains_db = []
        leakage_values = []

        for theta_jam_rad, phi_jam_rad in zip(
            self.current_jammer_thetas_rad,
            self.current_jammer_phis_rad,
        ):
            jammer_direction_deg = (
                float(np.rad2deg(theta_jam_rad)),
                float(np.rad2deg(phi_jam_rad)),
            )

            jammer_gain_linear = self._compute_array_response_gain_linear(
                weights=weights,
                direction_deg=jammer_direction_deg,
            )

            leakage = float(jammer_gain_linear) / denominator

            jammer_gains_linear.append(float(jammer_gain_linear))
            jammer_gains_db.append(float(self._linear_to_db(jammer_gain_linear)))
            leakage_values.append(float(leakage))

        if len(leakage_values) == 0:
            jammer_leakage_loss = 0.0
            jammer_leakage_max = 0.0
        else:
            jammer_leakage_loss = float(np.mean(leakage_values))
            jammer_leakage_max = float(np.max(leakage_values))

        if not np.isfinite(jammer_leakage_loss):
            jammer_leakage_loss = self.max_jammer_leakage_loss

        clipped_jammer_leakage_loss = min(
            max(0.0, jammer_leakage_loss),
            self.max_jammer_leakage_loss,
        )

        jammer_leakage_loss_db = self._linear_to_db(jammer_leakage_loss)

        return {
            "jammer_gains_linear": jammer_gains_linear,
            "jammer_gains_db": jammer_gains_db,
            "jammer_leakage_values": leakage_values,
            "jammer_leakage_loss": float(jammer_leakage_loss),
            "clipped_jammer_leakage_loss": float(clipped_jammer_leakage_loss),
            "jammer_leakage_loss_db": float(jammer_leakage_loss_db),
            "jammer_leakage_max": float(jammer_leakage_max),
        }

    def _compute_array_response_gain_linear(
        self,
        weights: np.ndarray,
        direction_deg: tuple[float, float],
    ) -> float:
        """
        Compute |w^H a(theta, phi)|^2 for a given direction.
        """

        weights_flat = np.asarray(weights, dtype=np.complex128).reshape(
            self.num_elements
        )

        steering_vector = get_steering_vector(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            direction=direction_deg,
        ).astype(np.complex128).reshape(self.num_elements)

        response = np.vdot(weights_flat, steering_vector)
        gain = np.abs(response) ** 2

        return float(np.real(gain))

    def _build_steering_weights(self, theta_rad: float, phi_rad: float) -> np.ndarray:
        """
        Build conventional phase-only steering weights for a given direction.
        """

        theta_deg = float(np.rad2deg(theta_rad))
        phi_deg = float(np.rad2deg(phi_rad))

        weights_flat = get_steering_vector(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            direction=(theta_deg, phi_deg),
        )

        weights_flat = np.asarray(weights_flat, dtype=np.complex128).reshape(
            self.num_elements
        )

        weights_flat = np.exp(1j * np.angle(weights_flat))

        return weights_flat.reshape(self.array.N, self.array.M)

    def _build_safe_fallback_weights(self) -> np.ndarray:
        """Build safe fallback weights for invalid coefficient actions."""

        if self.current_theta_rad is not None and self.current_phi_rad is not None:
            try:
                return self._build_point_base_weights()
            except Exception:
                return self._build_steering_weights(
                    theta_rad=self.current_theta_rad,
                    phi_rad=self.current_phi_rad,
                )

        return np.ones((self.array.N, self.array.M), dtype=np.complex128)

    def _build_invalid_soi_gain_metrics(self) -> dict:
        """
        Return penalized SOI-gain metrics for invalid numerical cases.
        """

        return {
            "soi_gain_linear": 0.0,
            "soi_gain_db": float("-inf"),
            "reference_soi_gain_linear": 0.0,
            "reference_soi_gain_db": float("-inf"),
            "soi_gain_loss_db": self.max_soi_gain_loss_db,
            "clipped_soi_gain_loss_db": self.max_soi_gain_loss_db,
        }

    def _build_invalid_jammer_leakage_metrics(self) -> dict:
        """
        Return penalized jammer-leakage metrics for invalid numerical cases.
        """

        return {
            "jammer_gains_linear": [],
            "jammer_gains_db": [],
            "jammer_leakage_values": [],
            "jammer_leakage_loss": self.max_jammer_leakage_loss,
            "clipped_jammer_leakage_loss": self.max_jammer_leakage_loss,
            "jammer_leakage_loss_db": self._linear_to_db(
                self.max_jammer_leakage_loss
            ),
            "jammer_leakage_max": self.max_jammer_leakage_loss,
        }

    def _compute_threshold_reward(
        self,
        sinr_db: float,
        sinr_loss_db: float,
        soi_gain_loss_db: float,
        jammer_mean_leakage: float,
        teacher_weight_similarity: float,
        invalid_coefficient_action: bool,
        numerical_error: bool,
    ) -> tuple[float, dict]:
        """
        Compute the Phase 7 reward with gates, dense shaping and stepped bonuses.

        For a valid transition:

            reward =
                clipped_normalized_sinr
                + teacher_similarity_component
                - dense_jammer_leakage_penalty
                + sinr_loss_bonus
                + soi_gain_loss_bonus
                + jammer_leakage_bonus
                + teacher_similarity_bonus

        Stepped bonuses from different components accumulate. Within a single
        component, only the largest bonus among all satisfied rows is applied.
        """

        normalized_jammer_leakage = 0.0
        jammer_leakage_penalty = 0.0
        jammer_leakage_reward_component = 0.0

        sinr_bonus_info = self._compute_stepped_bonus(
            metric_value=sinr_loss_db,
            bonus_steps=self.reward_sinr_loss_bonus_steps,
            lower_is_better=True,
        )

        soi_bonus_info = self._compute_stepped_bonus(
            metric_value=soi_gain_loss_db,
            bonus_steps=self.reward_soi_gain_loss_bonus_steps,
            lower_is_better=True,
        )

        if self.num_active_jammers == 0:
            jammer_bonus_info = {
                "bonus": 0.0,
                "matched": False,
                "matched_threshold": float("nan"),
                "matched_row_index": -1,
                "num_satisfied_steps": 0,
            }
        else:
            jammer_bonus_info = self._compute_stepped_bonus(
                metric_value=jammer_mean_leakage,
                bonus_steps=self.reward_jammer_leakage_bonus_steps,
                lower_is_better=True,
            )

        teacher_bonus_info = self._compute_stepped_bonus(
            metric_value=teacher_weight_similarity,
            bonus_steps=self.reward_teacher_similarity_bonus_steps,
            lower_is_better=False,
        )

        reward_sinr_loss_bonus = 0.0
        reward_soi_gain_loss_bonus = 0.0
        reward_jammer_leakage_bonus = 0.0
        reward_teacher_similarity_bonus = 0.0
        reward_total_stepped_bonus = 0.0

        if invalid_coefficient_action:
            reward = self.invalid_value_penalty
            failure_reason = "invalid_coefficient_action"
            soi_condition_met = False
            jammer_condition_met = False
            all_conditions_met = False
            valid_objective = 0.0

        elif numerical_error or not np.isfinite(sinr_db):
            reward = self.invalid_value_penalty
            failure_reason = "numerical_error"
            soi_condition_met = False
            jammer_condition_met = False
            all_conditions_met = False
            valid_objective = 0.0

        else:
            soi_condition_met = bool(
                np.isfinite(soi_gain_loss_db)
                and float(soi_gain_loss_db)
                <= self.reward_soi_max_gain_loss_db
            )

            if self.num_active_jammers == 0:
                jammer_condition_met = True
            else:
                jammer_condition_met = bool(
                    np.isfinite(jammer_mean_leakage)
                    and float(jammer_mean_leakage)
                    <= self.reward_jammer_max_mean_leakage
                )

            all_conditions_met = bool(
                soi_condition_met and jammer_condition_met
            )

            valid_objective = float(sinr_db) / self.reward_sinr_scale_db

            if not soi_condition_met:
                reward = self.reward_failure_penalty
                failure_reason = "soi_constraint"

            elif not jammer_condition_met:
                reward = self.reward_failure_penalty
                failure_reason = "jammer_constraint"

            else:
                clipped_valid_objective = float(
                    np.clip(
                        valid_objective,
                        self.reward_valid_min,
                        self.reward_valid_max,
                    )
                )

                if self.num_active_jammers > 0:
                    normalized_jammer_leakage = float(
                        np.clip(
                            float(jammer_mean_leakage)
                            / self.reward_jammer_leakage_penalty_scale,
                            0.0,
                            self.reward_jammer_leakage_penalty_clip,
                        )
                    )

                    jammer_leakage_penalty = float(
                        self.reward_jammer_leakage_penalty_weight
                        * normalized_jammer_leakage
                    )

                    jammer_leakage_reward_component = float(
                        -jammer_leakage_penalty
                    )

                reward_sinr_loss_bonus = float(
                    sinr_bonus_info["bonus"]
                )
                reward_soi_gain_loss_bonus = float(
                    soi_bonus_info["bonus"]
                )
                reward_jammer_leakage_bonus = float(
                    jammer_bonus_info["bonus"]
                )
                reward_teacher_similarity_bonus = float(
                    teacher_bonus_info["bonus"]
                )

                reward_total_stepped_bonus = float(
                    reward_sinr_loss_bonus
                    + reward_soi_gain_loss_bonus
                    + reward_jammer_leakage_bonus
                    + reward_teacher_similarity_bonus
                )

                reward = float(
                    clipped_valid_objective
                    + jammer_leakage_reward_component
                    + reward_total_stepped_bonus
                )
                failure_reason = "none"

        teacher_reward_component = (
            self.reward_teacher_similarity_weight
            * float(np.clip(teacher_weight_similarity, 0.0, 1.0))
        )

        if (
            not invalid_coefficient_action
            and not numerical_error
            and np.isfinite(reward)
        ):
            reward += teacher_reward_component
        else:
            teacher_reward_component = 0.0
            normalized_jammer_leakage = 0.0
            jammer_leakage_penalty = 0.0
            jammer_leakage_reward_component = 0.0
            reward_sinr_loss_bonus = 0.0
            reward_soi_gain_loss_bonus = 0.0
            reward_jammer_leakage_bonus = 0.0
            reward_teacher_similarity_bonus = 0.0
            reward_total_stepped_bonus = 0.0

        soi_constraint_violation_db = (
            max(
                0.0,
                float(soi_gain_loss_db)
                - self.reward_soi_max_gain_loss_db,
            )
            if np.isfinite(soi_gain_loss_db)
            else self.max_soi_gain_loss_db
        )

        if self.num_active_jammers == 0:
            jammer_constraint_violation = 0.0
        elif np.isfinite(jammer_mean_leakage):
            jammer_constraint_violation = max(
                0.0,
                float(jammer_mean_leakage)
                - self.reward_jammer_max_mean_leakage,
            )
        else:
            jammer_constraint_violation = self.max_jammer_leakage_loss

        info = {
            "reward_soi_condition_met": bool(soi_condition_met),
            "reward_jammer_condition_met": bool(jammer_condition_met),
            "reward_all_conditions_met": bool(all_conditions_met),
            "reward_failure_applied": bool(not all_conditions_met),
            "reward_failure_reason": failure_reason,

            "reward_valid_objective": float(valid_objective),
            "reward_valid_objective_clipped": float(
                np.clip(
                    valid_objective,
                    self.reward_valid_min,
                    self.reward_valid_max,
                )
            ),

            "reward_teacher_similarity_weight": float(
                self.reward_teacher_similarity_weight
            ),
            "reward_teacher_component": float(
                teacher_reward_component
            ),

            "reward_jammer_leakage_penalty_weight": float(
                self.reward_jammer_leakage_penalty_weight
            ),
            "reward_jammer_leakage_penalty_scale": float(
                self.reward_jammer_leakage_penalty_scale
            ),
            "reward_jammer_leakage_penalty_clip": float(
                self.reward_jammer_leakage_penalty_clip
            ),
            "reward_jammer_leakage_normalized": float(
                normalized_jammer_leakage
            ),
            "reward_jammer_leakage_penalty": float(
                jammer_leakage_penalty
            ),
            "reward_jammer_leakage_component": float(
                jammer_leakage_reward_component
            ),

            "reward_sinr_loss_bonus": float(
                reward_sinr_loss_bonus
            ),
            "reward_soi_gain_loss_bonus": float(
                reward_soi_gain_loss_bonus
            ),
            "reward_jammer_leakage_bonus": float(
                reward_jammer_leakage_bonus
            ),
            "reward_teacher_similarity_bonus": float(
                reward_teacher_similarity_bonus
            ),
            "reward_total_stepped_bonus": float(
                reward_total_stepped_bonus
            ),

            "reward_sinr_loss_bonus_matched": bool(
                sinr_bonus_info["matched"]
            ),
            "reward_sinr_loss_bonus_threshold": float(
                sinr_bonus_info["matched_threshold"]
            ),
            "reward_sinr_loss_bonus_row_index": int(
                sinr_bonus_info["matched_row_index"]
            ),
            "reward_sinr_loss_bonus_num_satisfied_steps": int(
                sinr_bonus_info["num_satisfied_steps"]
            ),

            "reward_soi_gain_loss_bonus_matched": bool(
                soi_bonus_info["matched"]
            ),
            "reward_soi_gain_loss_bonus_threshold": float(
                soi_bonus_info["matched_threshold"]
            ),
            "reward_soi_gain_loss_bonus_row_index": int(
                soi_bonus_info["matched_row_index"]
            ),
            "reward_soi_gain_loss_bonus_num_satisfied_steps": int(
                soi_bonus_info["num_satisfied_steps"]
            ),

            "reward_jammer_leakage_bonus_matched": bool(
                jammer_bonus_info["matched"]
            ),
            "reward_jammer_leakage_bonus_threshold": float(
                jammer_bonus_info["matched_threshold"]
            ),
            "reward_jammer_leakage_bonus_row_index": int(
                jammer_bonus_info["matched_row_index"]
            ),
            "reward_jammer_leakage_bonus_num_satisfied_steps": int(
                jammer_bonus_info["num_satisfied_steps"]
            ),

            "reward_teacher_similarity_bonus_matched": bool(
                teacher_bonus_info["matched"]
            ),
            "reward_teacher_similarity_bonus_threshold": float(
                teacher_bonus_info["matched_threshold"]
            ),
            "reward_teacher_similarity_bonus_row_index": int(
                teacher_bonus_info["matched_row_index"]
            ),
            "reward_teacher_similarity_bonus_num_satisfied_steps": int(
                teacher_bonus_info["num_satisfied_steps"]
            ),

            "soi_constraint_violation_db": float(
                soi_constraint_violation_db
            ),
            "jammer_constraint_violation": float(
                jammer_constraint_violation
            ),
        }

        return float(reward), info


    def _build_block_info(
        self,
        block_metrics: list[dict],
        reward: float,
        numerical_error: bool,
        weights_are_finite: bool,
        fixed_weights: np.ndarray,
        fixed_teacher_weights: np.ndarray,
        teacher_weights_are_valid: bool,
        action_info: dict,
        num_block_steps: int,
        terminated: bool,
    ) -> dict:
        """Build aggregated information for one Phase 9 control block."""

        last_metrics = block_metrics[-1] if len(block_metrics) > 0 else {}

        info = {
            "reward": float(reward),
            "block_reward_mean": float(reward),
            "num_block_steps": int(num_block_steps),
            "weight_hold_steps": self.weight_hold_steps,
            "teacher_update_mode": "block_start_hold",
            "teacher_hold_steps": int(num_block_steps),
            "teacher_weights_are_valid": bool(teacher_weights_are_valid),
            "fixed_teacher_weights": np.asarray(
                fixed_teacher_weights,
                dtype=np.complex128,
            ).copy(),
            "current_physical_step": int(self.current_physical_step),
            "episode_length_physical_steps": self.episode_length_physical_steps,
            "terminated": bool(terminated),
            "observation_mode": self.observation_mode,
            "complex_weight_mode": self.complex_weight_mode,
            "coefficient_jammer_slots": self.coefficient_jammer_slots,
            "num_complex_coefficients": self.num_complex_coefficients,
            "action_type": self._get_action_type(),
            "num_active_jammers": self.num_active_jammers,
            "jammers_powers": self.jammer_powers.copy(),
            "weights": fixed_weights.copy(),
            "raw_action": action_info["raw_action"].copy(),
            "coefficient_action": action_info["coefficient_action"].copy(),
            "coefficient_real_action": action_info["coefficient_real_action"].copy(),
            "coefficient_imag_action": action_info["coefficient_imag_action"].copy(),
            "complex_coefficients": action_info["complex_coefficients"].copy(),
            "coefficient_scales": action_info["coefficient_scales"].copy(),
            "basis_matrix": action_info["basis_matrix"].copy(),
            "basis_active_mask": action_info["basis_active_mask"].copy(),
            "basis_column_labels": action_info["basis_column_labels"].copy(),
            "basis_rank": int(action_info["basis_rank"]),
            "coefficient_basis_mode": str(
                action_info[
                    "coefficient_basis_mode"
                ]
            ),
            "semantic_basis_matrix": action_info[
                "semantic_basis_matrix"
            ].copy(),
            "semantic_basis_active_mask": action_info[
                "semantic_basis_active_mask"
            ].copy(),
            "semantic_basis_rank": int(
                action_info[
                    "semantic_basis_rank"
                ]
            ),
            "semantic_gram_condition_number": float(
                action_info[
                    "semantic_gram_condition_number"
                ]
            ),
            "decoder_gram_condition_number": float(
                action_info[
                    "decoder_gram_condition_number"
                ]
            ),
            "residual_weights": action_info["residual_weights"].copy(),
            "residual_norm": float(action_info["residual_norm"]),
            "base_weights": action_info["base_weights"].copy(),
            "raw_weights_before_normalization": action_info[
                "raw_weights_before_normalization"
            ].copy(),
            "raw_weight_power_before_normalization": float(
                action_info["raw_weight_power_before_normalization"]
            ),
            "virtual_slots_by_jammer": action_info[
                "virtual_slots_by_jammer"
            ],
            "final_magnitude": action_info["final_magnitude"].copy(),
            "final_phase_rad": action_info["final_phase_rad"].copy(),
            "final_phase_norm": action_info["final_phase_norm"].copy(),
            "final_weight_power": float(action_info["final_weight_power"]),
            "invalid_coefficient_action": bool(
                action_info["invalid_coefficient_action"]
            ),
            "numerical_error": bool(numerical_error),
            "weights_are_finite": bool(weights_are_finite),
            "array_normalize_power": bool(self.array.normalize_power),
            "reward_failure_penalty": self.reward_failure_penalty,
            "reward_soi_max_gain_loss_db": self.reward_soi_max_gain_loss_db,
            "reward_jammer_max_mean_leakage": (
                self.reward_jammer_max_mean_leakage
            ),
            "reward_sinr_scale_db": self.reward_sinr_scale_db,
            "reward_valid_min": self.reward_valid_min,
            "reward_valid_max": self.reward_valid_max,
            "reward_teacher_similarity_weight": (
                self.reward_teacher_similarity_weight
            ),
            "reward_jammer_leakage_penalty_weight": (
                self.reward_jammer_leakage_penalty_weight
            ),
            "reward_jammer_leakage_penalty_scale": (
                self.reward_jammer_leakage_penalty_scale
            ),
            "reward_jammer_leakage_penalty_clip": (
                self.reward_jammer_leakage_penalty_clip
            ),
            "reward_sinr_loss_bonus_steps": (
                self.reward_sinr_loss_bonus_steps.copy()
            ),
            "reward_soi_gain_loss_bonus_steps": (
                self.reward_soi_gain_loss_bonus_steps.copy()
            ),
            "reward_jammer_leakage_bonus_steps": (
                self.reward_jammer_leakage_bonus_steps.copy()
            ),
            "reward_teacher_similarity_bonus_steps": (
                self.reward_teacher_similarity_bonus_steps.copy()
            ),
            "teacher_diagonal_loading": self.teacher_diagonal_loading,
            "teacher_use_pinv": self.teacher_use_pinv,
            "teacher_similarity_epsilon": (
                self.teacher_similarity_epsilon
            ),
            "direct_weight_min_power": self.direct_weight_min_power,
            "coefficient_scales_config": self.coefficient_scales.copy(),
            "substep_metrics": block_metrics,
        }

        aggregate_keys = [
            "sinr_db",
            "reference_sinr_db",
            "sinr_loss_db",
            "clipped_sinr_loss_db",
            "soi_gain_db",
            "reference_soi_gain_db",
            "soi_gain_loss_db",
            "clipped_soi_gain_loss_db",
            "jammer_leakage_loss",
            "clipped_jammer_leakage_loss",
            "jammer_leakage_max",
            "reward_valid_objective",
            "reward_valid_objective_clipped",
            "reward_teacher_component",
            "reward_jammer_leakage_normalized",
            "reward_jammer_leakage_penalty",
            "reward_jammer_leakage_component",
            "reward_sinr_loss_bonus",
            "reward_soi_gain_loss_bonus",
            "reward_jammer_leakage_bonus",
            "reward_teacher_similarity_bonus",
            "reward_total_stepped_bonus",
            "reward_sinr_loss_bonus_matched",
            "reward_soi_gain_loss_bonus_matched",
            "reward_jammer_leakage_bonus_matched",
            "reward_teacher_similarity_bonus_matched",
            "teacher_weight_similarity",
            "teacher_weight_loss",
            "teacher_global_phase_offset_rad",
            "teacher_global_phase_offset_deg",
            "teacher_aligned_complex_mse",
            "soi_constraint_violation_db",
            "jammer_constraint_violation",
            "reward_soi_condition_met",
            "reward_jammer_condition_met",
            "reward_all_conditions_met",
            "reward_failure_applied",
            "invalid_coefficient_action",
        ]

        for key in aggregate_keys:
            info[f"{key}_mean"] = self._safe_mean_metric(block_metrics, key)
            info[f"{key}_last"] = self._safe_last_metric(block_metrics, key)

        info["sinr_db"] = info["sinr_db_mean"]
        info["reference_sinr_db"] = info["reference_sinr_db_mean"]
        info["sinr_loss_db"] = info["sinr_loss_db_mean"]
        info["clipped_sinr_loss_db"] = info["clipped_sinr_loss_db_mean"]
        info["soi_gain_loss_db"] = info["soi_gain_loss_db_mean"]
        info["jammer_leakage_loss"] = info["jammer_leakage_loss_mean"]
        info["teacher_weight_similarity"] = info[
            "teacher_weight_similarity_mean"
        ]
        info["teacher_weight_loss"] = info[
            "teacher_weight_loss_mean"
        ]
        info["reward_teacher_component"] = info[
            "reward_teacher_component_mean"
        ]
        info["reward_jammer_leakage_normalized"] = info[
            "reward_jammer_leakage_normalized_mean"
        ]
        info["reward_jammer_leakage_penalty"] = info[
            "reward_jammer_leakage_penalty_mean"
        ]
        info["reward_jammer_leakage_component"] = info[
            "reward_jammer_leakage_component_mean"
        ]
        info["reward_sinr_loss_bonus"] = info[
            "reward_sinr_loss_bonus_mean"
        ]
        info["reward_soi_gain_loss_bonus"] = info[
            "reward_soi_gain_loss_bonus_mean"
        ]
        info["reward_jammer_leakage_bonus"] = info[
            "reward_jammer_leakage_bonus_mean"
        ]
        info["reward_teacher_similarity_bonus"] = info[
            "reward_teacher_similarity_bonus_mean"
        ]
        info["reward_total_stepped_bonus"] = info[
            "reward_total_stepped_bonus_mean"
        ]
        info["soi_condition_met_fraction"] = info[
            "reward_soi_condition_met_mean"
        ]
        info["jammer_condition_met_fraction"] = info[
            "reward_jammer_condition_met_mean"
        ]
        info["all_conditions_met_fraction"] = info[
            "reward_all_conditions_met_mean"
        ]
        info["failure_penalty_fraction"] = info[
            "reward_failure_applied_mean"
        ]

        failure_reasons = [
            item.get("reward_failure_reason", "unknown") for item in block_metrics
        ]
        info["reward_failure_reason_last"] = (
            failure_reasons[-1] if failure_reasons else "unknown"
        )
        info["reward_failure_reasons"] = failure_reasons

        info.update(
            {
                "theta_target_rad": last_metrics.get("theta_target_rad"),
                "phi_target_rad": last_metrics.get("phi_target_rad"),
                "theta_target_deg": last_metrics.get("theta_target_deg"),
                "phi_target_deg": last_metrics.get("phi_target_deg"),
                "jammer_thetas_rad": last_metrics.get("jammer_thetas_rad", []),
                "jammer_phis_rad": last_metrics.get("jammer_phis_rad", []),
                "jammer_thetas_deg": last_metrics.get("jammer_thetas_deg", []),
                "jammer_phis_deg": last_metrics.get("jammer_phis_deg", []),
                "jammers_directions_deg": last_metrics.get(
                    "jammers_directions_deg", []
                ),
                "jammer_gains_linear": last_metrics.get(
                    "jammer_gains_linear", []
                ),
                "jammer_gains_db": last_metrics.get("jammer_gains_db", []),
                "jammer_leakage_values": last_metrics.get(
                    "jammer_leakage_values", []
                ),
                "teacher_weights": np.asarray(
                    fixed_teacher_weights,
                    dtype=np.complex128,
                ).copy(),
            }
        )
        return info


    def _safe_mean_metric(self, metrics: list[dict], key: str) -> float:
        """
        Safely compute mean of a scalar metric over a block.
        """

        values = []

        for item in metrics:
            value = item.get(key, np.nan)

            if isinstance(value, (list, tuple, dict, np.ndarray)):
                continue

            if np.isfinite(value):
                values.append(float(value))

        if len(values) == 0:
            return float("nan")

        return float(np.mean(values))

    def _safe_last_metric(self, metrics: list[dict], key: str) -> float:
        """
        Safely return the last scalar metric in a block.
        """

        if len(metrics) == 0:
            return float("nan")

        value = metrics[-1].get(key, np.nan)

        if isinstance(value, (list, tuple, dict, np.ndarray)):
            return float("nan")

        if not np.isfinite(value):
            return float("nan")

        return float(value)

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

    def _compute_angular_error_from_positions_deg(
        self,
        position_a: np.ndarray,
        position_b: np.ndarray,
    ) -> float:
        """
        Compute angular separation between two Cartesian positions as seen
        from the array.
        """

        relative_a = (
            np.asarray(position_a, dtype=float).reshape(3) - self.array_position
        )
        relative_b = (
            np.asarray(position_b, dtype=float).reshape(3) - self.array_position
        )

        norm_a = np.linalg.norm(relative_a)
        norm_b = np.linalg.norm(relative_b)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        u_a = relative_a / norm_a
        u_b = relative_b / norm_b

        dot_product = float(np.dot(u_a, u_b))
        dot_product = np.clip(dot_product, -1.0, 1.0)

        return float(np.rad2deg(np.arccos(dot_product)))

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

    def _build_jammer_powers_for_active_count(self, active_count: int) -> list[float]:
        """
        Build jammer power list for the selected number of active jammers.
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

    def _normalize_theta(self, theta_rad: float) -> float:
        return (theta_rad - self.theta_min) / (self.theta_max - self.theta_min)

    def _normalize_phi(self, phi_rad: float) -> float:
        return (phi_rad - self.phi_min) / (self.phi_max - self.phi_min)

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

    @staticmethod
    def _linear_to_db(value: float, eps: float = 1e-12) -> float:
        """
        Convert a non-negative linear value to dB safely.
        """

        safe_value = max(float(value), eps)

        return float(10.0 * np.log10(safe_value))

    @staticmethod
    def _wrap_phase_rad(phase_rad: np.ndarray) -> np.ndarray:
        """
        Wrap phase to [-pi, pi).
        """

        return (np.asarray(phase_rad, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi

    def _get_action_type(self) -> str:
        """Return the native Phase 9 action type used in logs."""

        return (
            "coefficient_real_imag"
            f"_{self.coefficient_basis_mode}"
        )

