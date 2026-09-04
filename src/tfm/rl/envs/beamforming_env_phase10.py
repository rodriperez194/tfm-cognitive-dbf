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


class BeamformingEnvPhase10(gym.Env):
    """
    Gymnasium environment for Phase 10 symmetric complex-weight beamforming.

    The observation architecture is unchanged from Phase 7. The agent observes
    the SOI direction and up to three jammer directions, using either normalized
    angles or 3D unit vectors with jammer-presence masks.

    The physical array still contains E = array.N * array.M complex weights, but
    the policy controls only E/2 independent complex coefficients:

        action = [
            Re(c_1), ..., Re(c_{E/2}),
            Im(c_1), ..., Im(c_{E/2}),
        ]

    Each independent coefficient is copied to the element related by central
    array symmetry (180-degree rotation of the N x M weight matrix). Therefore,

        c[n, m] = c[N - 1 - n, M - 1 - m].

    The symmetric coefficients modulate the conventional steering vector toward
    the current SOI:

        W = W_steering * C_symmetric

    where ``*`` denotes element-wise complex multiplication. The action is not
    an additive residual correction: it directly parameterizes the symmetric
    modulation C_symmetric. For a 6 x 6 array this reduces the action from
    72 real components to 36 real components (18 complex coefficients).

    Power normalization is delegated to ``Phased_Array_NB.set_weights``. With
    ``array.normalize_power=True``, the final weights satisfy

        sum_n |w_n|^2 = E.

    Reward
    ------
    This Phase 10 variant uses a physics-aligned continuous reward derived from
    the objective that produced the successful physics-aware supervised actor.

    For every numerically valid action, the per-step loss is

        L = lambda_j * L_jammer
            + lambda_soi * L_soi
            + lambda_dir * L_direction

    and the reinforcement-learning reward is

        reward = -L.

    The jammer term is

        L_jammer = log1p(mean_jammer_leakage / leakage_scale),

    where mean_jammer_leakage is the same mean jammer-to-SOI gain ratio already
    computed by Phase 10. This signal remains continuous below the historical
    0.01 diagnostic threshold, so 1e-2, 1e-3, 1e-4 and 1e-5 are explicitly
    distinguished by the reward. For zero-jammer episodes L_jammer = 0.

    The SOI term is

        L_soi = (max(0, soi_gain_loss_db - soi_gate_db) / soi_gate_db)^2.

    Therefore the SOI receives no penalty while it remains inside the configured
    3 dB operating region, matching the successful supervised physics loss.

    The direction term is the complex projective loss

        L_direction = 1 - |c_agent^H c_teacher|^2
                            / (||c_agent||^2 ||c_teacher||^2 + eps),

    where c_teacher is the exact maximum-SINR solution inside the same Phase 10
    central-symmetric subspace. The teacher is computed at the beginning of each
    control block and held for weight_hold_steps, just like the agent action.
    The loss is invariant to common complex phase and scale.

    The historical SOI and jammer thresholds are still reported as diagnostics,
    but neither is a reward gate. No SINR base term, stepped bonus, legacy dense
    leakage penalty or full-array teacher-similarity reward is added in this
    physics-aligned mode. Invalid direct actions and numerical failures retain
    the strong invalid_value_penalty.

    The original full-array target-or-zero teacher and MVDR reference are retained
    for diagnostics/evaluation only and do not define the physics reward.
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
        reward_physics_jammer_weight: float = 1.0,
        reward_physics_soi_weight: float = 1.0,
        reward_physics_teacher_direction_weight: float = 0.1,
        reward_physics_jammer_leakage_scale: float = 0.01,
        reward_physics_soi_gate_db: float = 3.0,
        reward_physics_epsilon: float = 1e-12,
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

        self.reward_physics_jammer_weight = float(
            reward_physics_jammer_weight
        )
        self.reward_physics_soi_weight = float(
            reward_physics_soi_weight
        )
        self.reward_physics_teacher_direction_weight = float(
            reward_physics_teacher_direction_weight
        )
        self.reward_physics_jammer_leakage_scale = float(
            reward_physics_jammer_leakage_scale
        )
        self.reward_physics_soi_gate_db = float(
            reward_physics_soi_gate_db
        )
        self.reward_physics_epsilon = float(
            reward_physics_epsilon
        )
        self.reward_mode = "physics_aligned"

        self.num_elements = int(self.array.N * self.array.M)

        # Phase 10 central-symmetry parameterization.
        # The mapping is constructed on the N x M weight matrix so that each
        # independent element is paired with its 180-degree rotated partner:
        # (n, m) <-> (N - 1 - n, M - 1 - m).
        flat_indices = np.arange(self.num_elements, dtype=int).reshape(
            self.array.N,
            self.array.M,
        )
        mirrored_indices = np.flip(flat_indices, axis=(0, 1))
        primary_mask = flat_indices < mirrored_indices

        self.symmetry_primary_flat_indices = flat_indices[
            primary_mask
        ].reshape(-1)
        self.symmetry_secondary_flat_indices = mirrored_indices[
            primary_mask
        ].reshape(-1)
        self.num_independent_weights = int(
            self.symmetry_primary_flat_indices.size
        )
        self.symmetry_mode = "central_180_steering_modulation"

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

        self.action_dim = 2 * self.num_independent_weights
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
            "action_dim": self.action_dim,
            "num_elements": self.num_elements,
            "num_independent_weights": self.num_independent_weights,
            "symmetry_mode": self.symmetry_mode,
            "symmetry_primary_flat_indices": (
                self.symmetry_primary_flat_indices.copy()
            ),
            "symmetry_secondary_flat_indices": (
                self.symmetry_secondary_flat_indices.copy()
            ),
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
            "reward_jammer_gate_mode": "diagnostic_only",
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
            "reward_mode": self.reward_mode,
            "reward_physics_jammer_weight": (
                self.reward_physics_jammer_weight
            ),
            "reward_physics_soi_weight": (
                self.reward_physics_soi_weight
            ),
            "reward_physics_teacher_direction_weight": (
                self.reward_physics_teacher_direction_weight
            ),
            "reward_physics_jammer_leakage_scale": (
                self.reward_physics_jammer_leakage_scale
            ),
            "reward_physics_soi_gate_db": (
                self.reward_physics_soi_gate_db
            ),
            "reward_physics_epsilon": self.reward_physics_epsilon,
        }

        return state, info
    def step(self, action: np.ndarray):
        """Apply one symmetric complex-weight action for one control block."""

        if self.current_scenario is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized.")

        numerical_error = False
        invalid_direct_action = False

        try:
            proposed_weights, action_info = self._action_to_direct_weights(action)
            invalid_direct_action = bool(action_info["invalid_direct_weight_action"])
        except Exception:
            proposed_weights = self._build_safe_fallback_weights()
            action_info = self._build_invalid_action_info()
            invalid_direct_action = True
            numerical_error = True

        if not np.all(np.isfinite(proposed_weights)):
            proposed_weights = self._build_safe_fallback_weights()
            action_info = self._build_invalid_action_info()
            invalid_direct_action = True
            numerical_error = True

        self.array.set_weights(proposed_weights)
        fixed_weights = self.array.W.copy()
        weights_are_finite = bool(np.all(np.isfinite(fixed_weights)))

        if not weights_are_finite:
            fixed_weights = self._build_safe_fallback_weights()
            self.array.set_weights(fixed_weights)
            fixed_weights = self.array.W.copy()
            invalid_direct_action = True
            numerical_error = True

        action_info = self._finalize_action_info(
            action_info=action_info,
            normalized_weights=fixed_weights,
            invalid_direct_action=invalid_direct_action,
        )
        self.last_final_weights = fixed_weights.copy()

        remaining_steps = (
            self.episode_length_physical_steps - self.current_physical_step
        )
        num_block_steps = min(self.weight_hold_steps, remaining_steps)
        block_metrics: list[dict] = []

        # The teacher uses exactly the same update cadence as the agent.
        # It is computed from the geometry available at the beginning of the
        # control block and held fixed for all physical substeps in the block.
        block_start_step_idx = self.current_physical_step
        self._load_current_directions_from_scenario(
            step_idx=block_start_step_idx
        )

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

        physics_teacher_is_valid = True
        try:
            fixed_physics_teacher_coefficients = (
                self._build_symmetric_max_sinr_teacher_coefficients()
            )
        except Exception:
            fixed_physics_teacher_coefficients = np.zeros(
                self.num_independent_weights,
                dtype=np.complex128,
            )
            physics_teacher_is_valid = False
            numerical_error = True

        for block_offset in range(num_block_steps):
            step_idx = self.current_physical_step + block_offset
            self._load_current_directions_from_scenario(step_idx=step_idx)

            instant_metrics = self._evaluate_fixed_weights_at_current_step(
                weights=fixed_weights,
                teacher_weights=fixed_teacher_weights,
                teacher_weights_are_valid=teacher_weights_are_valid,
                physics_teacher_coefficients=(
                    fixed_physics_teacher_coefficients
                ),
                physics_teacher_is_valid=physics_teacher_is_valid,
                invalid_direct_action=invalid_direct_action,
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
            fixed_physics_teacher_coefficients=(
                fixed_physics_teacher_coefficients
            ),
            physics_teacher_is_valid=physics_teacher_is_valid,
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
                "Phase 10 symmetric control currently requires "
                "complex_weight_mode='real_imag'."
            )
        if self.num_elements % 2 != 0:
            raise ValueError(
                "Phase 10 central symmetry requires an even number of "
                "array elements."
            )
        if 2 * self.num_independent_weights != self.num_elements:
            raise ValueError(
                "Invalid Phase 10 central-symmetry mapping."
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
        if self.reward_physics_jammer_weight < 0.0:
            raise ValueError(
                "reward_physics_jammer_weight must be non-negative."
            )
        if self.reward_physics_soi_weight < 0.0:
            raise ValueError(
                "reward_physics_soi_weight must be non-negative."
            )
        if self.reward_physics_teacher_direction_weight < 0.0:
            raise ValueError(
                "reward_physics_teacher_direction_weight must be non-negative."
            )
        if self.reward_physics_jammer_leakage_scale <= 0.0:
            raise ValueError(
                "reward_physics_jammer_leakage_scale must be positive."
            )
        if self.reward_physics_soi_gate_db <= 0.0:
            raise ValueError(
                "reward_physics_soi_gate_db must be positive."
            )
        if self.reward_physics_epsilon <= 0.0:
            raise ValueError(
                "reward_physics_epsilon must be positive."
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
                "Could not sample a valid Phase 10 direct-weight scenario."
            ) from last_error

        raise RuntimeError("Could not sample a valid Phase 10 direct-weight scenario.")

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

    def _action_to_direct_weights(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Convert the reduced real/imaginary action into symmetric final weights.

        The agent generates ``num_independent_weights`` complex coefficients.
        They are expanded by central symmetry and then multiplied element-wise
        by conventional steering weights toward the current SOI.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Current SOI direction is required to build Phase 10 weights."
            )

        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        real_action = action[: self.num_independent_weights].astype(float)
        imag_action = action[self.num_independent_weights :].astype(float)
        independent_coefficients = real_action + 1j * imag_action

        symmetric_modulation_flat = np.zeros(
            self.num_elements,
            dtype=np.complex128,
        )
        symmetric_modulation_flat[
            self.symmetry_primary_flat_indices
        ] = independent_coefficients
        symmetric_modulation_flat[
            self.symmetry_secondary_flat_indices
        ] = independent_coefficients

        steering_weights = self._build_steering_weights(
            theta_rad=self.current_theta_rad,
            phi_rad=self.current_phi_rad,
        )
        steering_flat = np.asarray(
            steering_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        weights_flat = steering_flat * symmetric_modulation_flat

        direct_weight_power_before_normalization = float(
            np.sum(np.abs(weights_flat) ** 2)
        )
        invalid_direct_action = bool(
            not np.isfinite(direct_weight_power_before_normalization)
            or direct_weight_power_before_normalization
            <= self.direct_weight_min_power
            or not np.all(np.isfinite(independent_coefficients))
            or not np.all(np.isfinite(symmetric_modulation_flat))
            or not np.all(np.isfinite(weights_flat))
        )

        if invalid_direct_action:
            weights = self._build_safe_fallback_weights()
        else:
            weights = weights_flat.reshape(self.array.N, self.array.M)

        action_info = {
            "raw_action": action.copy(),
            "direct_real_action": real_action.copy(),
            "direct_imag_action": imag_action.copy(),
            "independent_complex_action": independent_coefficients.copy(),
            "symmetric_modulation_weights": (
                symmetric_modulation_flat.reshape(
                    self.array.N,
                    self.array.M,
                ).copy()
            ),
            "steering_weights_before_modulation": steering_weights.copy(),
            "direct_weights_before_normalization": weights_flat.reshape(
                self.array.N,
                self.array.M,
            ).copy(),
            "direct_weight_power_before_normalization": (
                direct_weight_power_before_normalization
            ),
            "invalid_direct_weight_action": invalid_direct_action,
        }

        return np.asarray(weights, dtype=np.complex128), action_info

    def _build_invalid_action_info(self) -> dict:
        """Build logging information for an invalid Phase 10 action."""

        return {
            "raw_action": np.zeros(self.action_dim, dtype=float),
            "direct_real_action": np.zeros(
                self.num_independent_weights,
                dtype=float,
            ),
            "direct_imag_action": np.zeros(
                self.num_independent_weights,
                dtype=float,
            ),
            "independent_complex_action": np.zeros(
                self.num_independent_weights,
                dtype=np.complex128,
            ),
            "symmetric_modulation_weights": np.zeros(
                (self.array.N, self.array.M),
                dtype=np.complex128,
            ),
            "steering_weights_before_modulation": (
                self._build_safe_fallback_weights().copy()
            ),
            "direct_weights_before_normalization": np.zeros(
                (self.array.N, self.array.M),
                dtype=np.complex128,
            ),
            "direct_weight_power_before_normalization": 0.0,
            "invalid_direct_weight_action": True,
        }

    def _finalize_action_info(
        self,
        action_info: dict,
        normalized_weights: np.ndarray,
        invalid_direct_action: bool,
    ) -> dict:
        """Add final normalized-weight diagnostics to the action information."""

        normalized_weights = np.asarray(
            normalized_weights, dtype=np.complex128
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
                "invalid_direct_weight_action": bool(invalid_direct_action),
            }
        )
        return result


    def _build_phase10_reduced_matrix(self) -> np.ndarray:
        """
        Build the reduced Phase 10 matrix G such that

            w_raw = G @ c

        for the current SOI steering direction and the 18 independent
        central-symmetric complex coefficients c.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Current SOI direction is required for the reduced matrix."
            )

        steering_weights = np.asarray(
            self._build_steering_weights(
                theta_rad=self.current_theta_rad,
                phi_rad=self.current_phi_rad,
            ),
            dtype=np.complex128,
        ).reshape(self.num_elements)

        duplication_matrix = np.zeros(
            (self.num_elements, self.num_independent_weights),
            dtype=np.complex128,
        )

        for coefficient_idx, (primary_idx, secondary_idx) in enumerate(
            zip(
                self.symmetry_primary_flat_indices,
                self.symmetry_secondary_flat_indices,
            )
        ):
            duplication_matrix[int(primary_idx), coefficient_idx] = 1.0
            duplication_matrix[int(secondary_idx), coefficient_idx] = 1.0

        return steering_weights[:, None] * duplication_matrix


    def _build_symmetric_max_sinr_teacher_coefficients(self) -> np.ndarray:
        """
        Compute the exact maximum-SINR direction in the Phase 10 symmetric
        subspace, matching the teacher used by supervised pretraining.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Current SOI direction is required for the physics teacher."
            )

        target_direction_deg = (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.rad2deg(self.current_phi_rad)),
        )

        soi_vector = np.asarray(
            get_steering_vector(
                element_positions=self.array.element_positions,
                wavenumber_k=self.array.k_num,
                direction=target_direction_deg,
            ),
            dtype=np.complex128,
        ).reshape(self.num_elements)

        covariance = self.noise_power * np.eye(
            self.num_elements,
            dtype=np.complex128,
        )

        for jammer_idx, jammer_direction in enumerate(
            self._get_current_jammer_directions_deg()
        ):
            jammer_vector = np.asarray(
                get_steering_vector(
                    element_positions=self.array.element_positions,
                    wavenumber_k=self.array.k_num,
                    direction=jammer_direction,
                ),
                dtype=np.complex128,
            ).reshape(self.num_elements)

            jammer_power = float(self.jammer_powers[jammer_idx])

            covariance += jammer_power * np.outer(
                jammer_vector,
                np.conj(jammer_vector),
            )

        reduced_matrix = self._build_phase10_reduced_matrix()

        reduced_covariance = (
            reduced_matrix.conj().T
            @ covariance
            @ reduced_matrix
        )

        reduced_soi_vector = (
            reduced_matrix.conj().T
            @ soi_vector
        )

        try:
            coefficients = np.linalg.solve(
                reduced_covariance,
                reduced_soi_vector,
            )
        except np.linalg.LinAlgError:
            coefficients = (
                np.linalg.pinv(reduced_covariance)
                @ reduced_soi_vector
            )

        coefficients = np.asarray(
            coefficients,
            dtype=np.complex128,
        ).reshape(self.num_independent_weights)

        coefficient_norm = float(np.linalg.norm(coefficients))

        if (
            not np.all(np.isfinite(coefficients))
            or not np.isfinite(coefficient_norm)
            or coefficient_norm <= self.reward_physics_epsilon
        ):
            raise RuntimeError(
                "Symmetric maximum-SINR teacher coefficients are invalid."
            )

        return coefficients


    def _extract_symmetric_coefficients_from_weights(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Recover the independent symmetric modulation direction from final
        physical weights. Global array-power normalization only introduces a
        common positive scale and therefore does not alter the projective loss.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Current SOI direction is required to recover coefficients."
            )

        weights_flat = np.asarray(
            weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        steering_flat = np.asarray(
            self._build_steering_weights(
                theta_rad=self.current_theta_rad,
                phi_rad=self.current_phi_rad,
            ),
            dtype=np.complex128,
        ).reshape(self.num_elements)

        if (
            not np.all(np.isfinite(weights_flat))
            or not np.all(np.isfinite(steering_flat))
            or np.any(np.abs(steering_flat) <= self.reward_physics_epsilon)
        ):
            raise RuntimeError(
                "Cannot recover symmetric coefficients from invalid weights."
            )

        modulation_flat = weights_flat / steering_flat

        coefficients = modulation_flat[
            self.symmetry_primary_flat_indices
        ]

        return np.asarray(
            coefficients,
            dtype=np.complex128,
        ).reshape(self.num_independent_weights)


    def _compute_physics_teacher_direction_metrics(
        self,
        agent_weights: np.ndarray,
        teacher_coefficients: np.ndarray,
    ) -> dict:
        """
        Compute the same projective complex-direction loss used in the
        successful physics-aware supervised training.
        """

        agent_coefficients = self._extract_symmetric_coefficients_from_weights(
            agent_weights
        )

        teacher_coefficients = np.asarray(
            teacher_coefficients,
            dtype=np.complex128,
        ).reshape(self.num_independent_weights)

        agent_norm_sq = float(
            np.sum(np.abs(agent_coefficients) ** 2)
        )
        teacher_norm_sq = float(
            np.sum(np.abs(teacher_coefficients) ** 2)
        )

        denominator = agent_norm_sq * teacher_norm_sq

        if (
            not np.isfinite(denominator)
            or denominator <= self.reward_physics_epsilon
            or not np.all(np.isfinite(agent_coefficients))
            or not np.all(np.isfinite(teacher_coefficients))
        ):
            raise RuntimeError(
                "Cannot compute physics teacher direction loss."
            )

        inner_product = np.vdot(
            agent_coefficients,
            teacher_coefficients,
        )

        similarity_sq = float(
            np.clip(
                (np.abs(inner_product) ** 2)
                / (denominator + self.reward_physics_epsilon),
                0.0,
                1.0,
            )
        )

        direction_loss = float(1.0 - similarity_sq)

        return {
            "physics_teacher_direction_similarity_sq": similarity_sq,
            "physics_teacher_direction_loss": direction_loss,
            "physics_agent_coefficients": agent_coefficients.copy(),
            "physics_teacher_coefficients": teacher_coefficients.copy(),
        }


    def _build_teacher_weights(self) -> np.ndarray:
        """
        Build deterministic teacher weights for the current scene.

        The current SOI is imposed as a unity-response target and the current
        jammer directions are imposed as zero-response constraints.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Current SOI direction is not available for teacher weights."
            )

        target_direction_deg = (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.rad2deg(self.current_phi_rad)),
        )

        teacher_flat = target_or_zero_weights(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            target_directions=[target_direction_deg],
            zero_directions=self._get_current_jammer_directions_deg(),
            diagonal_loading=self.teacher_diagonal_loading,
            use_pinv=self.teacher_use_pinv,
        ).astype(np.complex128).reshape(self.num_elements)

        teacher_power = float(np.sum(np.abs(teacher_flat) ** 2))

        if (
            not np.isfinite(teacher_power)
            or teacher_power <= self.teacher_similarity_epsilon
            or not np.all(np.isfinite(teacher_flat))
        ):
            raise RuntimeError("Teacher weights are numerically invalid.")

        if self.array.normalize_power:
            teacher_flat = teacher_flat * np.sqrt(
                self.num_elements / teacher_power
            )

        return teacher_flat.reshape(self.array.N, self.array.M)


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
        physics_teacher_coefficients: np.ndarray,
        physics_teacher_is_valid: bool,
        invalid_direct_action: bool,
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

        if physics_teacher_is_valid:
            try:
                physics_teacher_metrics = (
                    self._compute_physics_teacher_direction_metrics(
                        agent_weights=weights,
                        teacher_coefficients=physics_teacher_coefficients,
                    )
                )
            except Exception:
                physics_teacher_metrics = {
                    "physics_teacher_direction_similarity_sq": 0.0,
                    "physics_teacher_direction_loss": 1.0,
                    "physics_agent_coefficients": np.zeros(
                        self.num_independent_weights,
                        dtype=np.complex128,
                    ),
                    "physics_teacher_coefficients": np.asarray(
                        physics_teacher_coefficients,
                        dtype=np.complex128,
                    ).copy(),
                }
                numerical_error = True
        else:
            physics_teacher_metrics = {
                "physics_teacher_direction_similarity_sq": 0.0,
                "physics_teacher_direction_loss": 1.0,
                "physics_agent_coefficients": np.zeros(
                    self.num_independent_weights,
                    dtype=np.complex128,
                ),
                "physics_teacher_coefficients": np.asarray(
                    physics_teacher_coefficients,
                    dtype=np.complex128,
                ).copy(),
            }
            numerical_error = True

        reward, reward_info = self._compute_physics_aligned_reward(
            soi_gain_loss_db=soi_gain_metrics["soi_gain_loss_db"],
            jammer_mean_leakage=jammer_leakage_metrics["jammer_leakage_loss"],
            physics_teacher_direction_loss=physics_teacher_metrics[
                "physics_teacher_direction_loss"
            ],
            invalid_direct_action=invalid_direct_action,
            numerical_error=numerical_error,
        )

        metrics = {
            "reward": float(reward),
            "sinr_db": float(sinr_db),
            "reference_sinr_db": float(reference_sinr_db),
            "sinr_loss_db": float(sinr_loss_db),
            "clipped_sinr_loss_db": float(clipped_sinr_loss_db),
            "numerical_error": bool(numerical_error),
            "invalid_direct_weight_action": bool(invalid_direct_action),
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
            **physics_teacher_metrics,
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
        """
        Build safe fallback weights.
        """

        if self.current_theta_rad is not None and self.current_phi_rad is not None:
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

    def _compute_physics_aligned_reward(
        self,
        soi_gain_loss_db: float,
        jammer_mean_leakage: float,
        physics_teacher_direction_loss: float,
        invalid_direct_action: bool,
        numerical_error: bool,
    ) -> tuple[float, dict]:
        """
        Compute the Phase 10 physics-aligned reward.

        For valid transitions:

            reward = -(
                lambda_j * log1p(mean_leakage / leakage_scale)
                + lambda_soi * (relu(soi_loss_db - soi_gate_db) / soi_gate_db)^2
                + lambda_dir * projective_teacher_direction_loss
            )

        This is the per-sample counterpart of the successful supervised 005/006
        physics-aware loss. The historical SOI/jammer thresholds are retained
        only as diagnostics and never gate a numerically valid reward.
        """

        # Diagnostic conditions retained for evaluation compatibility.
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

        physics_jammer_loss = 0.0
        physics_soi_loss = 0.0
        physics_direction_loss = 0.0
        physics_total_loss = 0.0

        if invalid_direct_action:
            reward = float(self.invalid_value_penalty)
            failure_reason = "invalid_direct_action"
            failure_applied = True

        elif numerical_error:
            reward = float(self.invalid_value_penalty)
            failure_reason = "numerical_error"
            failure_applied = True

        elif (
            not np.isfinite(soi_gain_loss_db)
            or not np.isfinite(jammer_mean_leakage)
            or not np.isfinite(physics_teacher_direction_loss)
        ):
            reward = float(self.invalid_value_penalty)
            failure_reason = "nonfinite_physics_metric"
            failure_applied = True

        else:
            if self.num_active_jammers == 0:
                physics_jammer_loss = 0.0
            else:
                safe_leakage = max(0.0, float(jammer_mean_leakage))
                physics_jammer_loss = float(
                    np.log1p(
                        safe_leakage
                        / self.reward_physics_jammer_leakage_scale
                    )
                )

            soi_excess_db = max(
                0.0,
                float(soi_gain_loss_db)
                - self.reward_physics_soi_gate_db,
            )

            physics_soi_loss = float(
                (
                    soi_excess_db
                    / self.reward_physics_soi_gate_db
                )
                ** 2
            )

            physics_direction_loss = float(
                np.clip(
                    physics_teacher_direction_loss,
                    0.0,
                    1.0,
                )
            )

            physics_total_loss = float(
                self.reward_physics_jammer_weight
                * physics_jammer_loss
                + self.reward_physics_soi_weight
                * physics_soi_loss
                + self.reward_physics_teacher_direction_weight
                * physics_direction_loss
            )

            reward = float(-physics_total_loss)
            failure_reason = "none"
            failure_applied = False

        if not np.isfinite(reward):
            reward = float(self.invalid_value_penalty)
            failure_reason = "nonfinite_reward"
            failure_applied = True

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

        # Keep the legacy reward-info keys as neutral values so existing
        # evaluation code can still read this environment without confusing
        # those legacy components with the new physics reward.
        info = {
            "reward_mode": self.reward_mode,
            "reward_soi_condition_met": bool(soi_condition_met),
            "reward_jammer_condition_met": bool(jammer_condition_met),
            "reward_all_conditions_met": bool(all_conditions_met),
            "reward_failure_applied": bool(failure_applied),
            "reward_failure_reason": failure_reason,
            "reward_soft_jammer_violation": bool(
                self.num_active_jammers > 0
                and not jammer_condition_met
                and not failure_applied
            ),
            "reward_jammer_gate_mode": "diagnostic_only",

            "reward_physics_jammer_weight": float(
                self.reward_physics_jammer_weight
            ),
            "reward_physics_soi_weight": float(
                self.reward_physics_soi_weight
            ),
            "reward_physics_teacher_direction_weight": float(
                self.reward_physics_teacher_direction_weight
            ),
            "reward_physics_jammer_leakage_scale": float(
                self.reward_physics_jammer_leakage_scale
            ),
            "reward_physics_soi_gate_db": float(
                self.reward_physics_soi_gate_db
            ),
            "reward_physics_jammer_loss": float(physics_jammer_loss),
            "reward_physics_soi_loss": float(physics_soi_loss),
            "reward_physics_teacher_direction_loss": float(
                physics_direction_loss
            ),
            "reward_physics_total_loss": float(physics_total_loss),

            "reward_valid_objective": 0.0,
            "reward_valid_objective_clipped": 0.0,
            "reward_teacher_component": 0.0,
            "reward_jammer_leakage_normalized": 0.0,
            "reward_jammer_leakage_penalty": 0.0,
            "reward_jammer_leakage_component": 0.0,
            "reward_sinr_loss_bonus": 0.0,
            "reward_soi_gain_loss_bonus": 0.0,
            "reward_jammer_leakage_bonus": 0.0,
            "reward_teacher_similarity_bonus": 0.0,
            "reward_total_stepped_bonus": 0.0,
            "reward_sinr_loss_bonus_matched": False,
            "reward_soi_gain_loss_bonus_matched": False,
            "reward_jammer_leakage_bonus_matched": False,
            "reward_teacher_similarity_bonus_matched": False,
            "reward_sinr_loss_bonus_threshold": float("nan"),
            "reward_soi_gain_loss_bonus_threshold": float("nan"),
            "reward_jammer_leakage_bonus_threshold": float("nan"),
            "reward_teacher_similarity_bonus_threshold": float("nan"),
            "reward_sinr_loss_bonus_row_index": -1,
            "reward_soi_gain_loss_bonus_row_index": -1,
            "reward_jammer_leakage_bonus_row_index": -1,
            "reward_teacher_similarity_bonus_row_index": -1,
            "reward_sinr_loss_bonus_num_satisfied_steps": 0,
            "reward_soi_gain_loss_bonus_num_satisfied_steps": 0,
            "reward_jammer_leakage_bonus_num_satisfied_steps": 0,
            "reward_teacher_similarity_bonus_num_satisfied_steps": 0,

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
        fixed_physics_teacher_coefficients: np.ndarray,
        physics_teacher_is_valid: bool,
        action_info: dict,
        num_block_steps: int,
        terminated: bool,
    ) -> dict:
        """Build aggregated information for one Phase 10 control block."""

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
            "physics_teacher_is_valid": bool(physics_teacher_is_valid),
            "fixed_physics_teacher_coefficients": np.asarray(
                fixed_physics_teacher_coefficients,
                dtype=np.complex128,
            ).copy(),
            "current_physical_step": int(self.current_physical_step),
            "episode_length_physical_steps": self.episode_length_physical_steps,
            "terminated": bool(terminated),
            "observation_mode": self.observation_mode,
            "complex_weight_mode": self.complex_weight_mode,
            "action_type": self._get_action_type(),
            "action_dim": self.action_dim,
            "num_elements": self.num_elements,
            "num_independent_weights": self.num_independent_weights,
            "symmetry_mode": self.symmetry_mode,
            "symmetry_primary_flat_indices": (
                self.symmetry_primary_flat_indices.copy()
            ),
            "symmetry_secondary_flat_indices": (
                self.symmetry_secondary_flat_indices.copy()
            ),
            "num_active_jammers": self.num_active_jammers,
            "jammers_powers": self.jammer_powers.copy(),
            "weights": fixed_weights.copy(),
            "raw_action": action_info["raw_action"].copy(),
            "direct_real_action": action_info["direct_real_action"].copy(),
            "direct_imag_action": action_info["direct_imag_action"].copy(),
            "independent_complex_action": action_info[
                "independent_complex_action"
            ].copy(),
            "symmetric_modulation_weights": action_info[
                "symmetric_modulation_weights"
            ].copy(),
            "steering_weights_before_modulation": action_info[
                "steering_weights_before_modulation"
            ].copy(),
            "direct_weights_before_normalization": action_info[
                "direct_weights_before_normalization"
            ].copy(),
            "direct_weight_power_before_normalization": float(
                action_info["direct_weight_power_before_normalization"]
            ),
            "final_magnitude": action_info["final_magnitude"].copy(),
            "final_phase_rad": action_info["final_phase_rad"].copy(),
            "final_phase_norm": action_info["final_phase_norm"].copy(),
            "final_weight_power": float(action_info["final_weight_power"]),
            "invalid_direct_weight_action": bool(
                action_info["invalid_direct_weight_action"]
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
            "reward_mode": self.reward_mode,
            "reward_physics_jammer_weight": (
                self.reward_physics_jammer_weight
            ),
            "reward_physics_soi_weight": (
                self.reward_physics_soi_weight
            ),
            "reward_physics_teacher_direction_weight": (
                self.reward_physics_teacher_direction_weight
            ),
            "reward_physics_jammer_leakage_scale": (
                self.reward_physics_jammer_leakage_scale
            ),
            "reward_physics_soi_gate_db": (
                self.reward_physics_soi_gate_db
            ),
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
            "reward_physics_jammer_loss",
            "reward_physics_soi_loss",
            "reward_physics_teacher_direction_loss",
            "reward_physics_total_loss",
            "physics_teacher_direction_similarity_sq",
            "physics_teacher_direction_loss",
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
            "invalid_direct_weight_action",
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
        info["reward_physics_jammer_loss"] = info[
            "reward_physics_jammer_loss_mean"
        ]
        info["reward_physics_soi_loss"] = info[
            "reward_physics_soi_loss_mean"
        ]
        info["reward_physics_teacher_direction_loss"] = info[
            "reward_physics_teacher_direction_loss_mean"
        ]
        info["reward_physics_total_loss"] = info[
            "reward_physics_total_loss_mean"
        ]
        info["physics_teacher_direction_similarity_sq"] = info[
            "physics_teacher_direction_similarity_sq_mean"
        ]
        info["physics_teacher_direction_loss"] = info[
            "physics_teacher_direction_loss_mean"
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
                "physics_teacher_coefficients": np.asarray(
                    fixed_physics_teacher_coefficients,
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
        """Return the action type used in logs and evaluation files."""

        return "symmetric_steering_modulation_real_imag"

