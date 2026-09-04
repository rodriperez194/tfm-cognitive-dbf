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
from tfm.math.narrow_band.geometry import (
    angles_to_unit_vector,
    unit_vector_to_angles,
)

from tfm.scenario.scenario_generator import ScenarioGenerator
from tfm.targets.aircraft import AircraftTarget
from tfm.targets.drone import DroneTarget
from tfm.targets.dummy import Dummy
from tfm.targets.static import StaticTarget
from tfm.targets.truck import TruckRoadTarget


class BeamformingEnvPhase11(gym.Env):
    """
    Standalone Gymnasium environment for Phase 11 direction-plus-null-width control.

    The policy operates at a high semantic level instead of generating complex
    array weights directly.

    Observation
    -----------
    The SOI is static. Each active jammer contributes its current direction,
    its finite-difference angular motion and a presence mask.

    ``observation_mode="angles"``:
        [SOI theta/phi,
         jammer theta/phi + theta/phi rate + mask] x 3
        -> 17 components.

    ``observation_mode="unit_vector"``:
        [SOI unit vector,
         jammer unit vector + unit-vector rate + mask] x 3
        -> 24 components.

    Action
    ------
    The action predicts the SOI direction and, for every jammer slot, the
    jammer direction plus one normalized null-width parameter.

    ``action_mode="angles"``:
        [SOI theta, SOI phi,
         J1 theta, J1 phi, delta_1,
         J2 theta, J2 phi, delta_2,
         J3 theta, J3 phi, delta_3]
        -> 11 components for ``max_jammers=3``.

    ``action_mode="unit_vector"``:
        [SOI ux, SOI uy, SOI uz,
         J1 ux, J1 uy, J1 uz, delta_1,
         J2 ux, J2 uy, J2 uz, delta_2,
         J3 ux, J3 uy, J3 uz, delta_3]
        -> 15 components for ``max_jammers=3``.

    For each active jammer, the jammer direction predicted by the agent is the
    central zero. Two additional zeros are placed behind/ahead of that predicted
    direction along the current jammer motion contained in the observation.
    The selected ``delta_i`` scales the prediction horizon. These target/zero
    constraints are converted into weights with ``target_or_zero_weights``.

    Temporal control
    ----------------
    One Gymnasium action is converted into one weight vector at the beginning
    of a control block. Those weights remain fixed for ``weight_hold_steps``
    physical samples. The scenario continues evolving during the block and the
    returned reward is the arithmetic mean over those physical samples.

    Reward
    ------
    The dense reward is:

        - beta_sinr_loss * normalized_sinr_loss
        - gamma_soi_gain_loss * normalized_soi_gain_loss
        - gamma_jammer_leakage * normalized_jammer_leakage
        + gamma_hold * exp(-sinr_loss_db / reward_hold_scale_db)
        - gamma_soi_action_error * normalized_soi_action_error
        - gamma_jammer_action_error * normalized_jammer_action_error

    The two angular-action terms explicitly guide the geometric part of the
    policy: the SOI action should reproduce the true SOI direction and each
    active jammer action should reproduce the corresponding true jammer
    direction.

    For K > 1, these geometric guidance errors are evaluated only at the
    beginning of the control block and then kept fixed throughout the K
    physical substeps. This prevents the policy from being penalized simply
    because a jammer moves while the selected weights are intentionally held.
    The physical SINR, SOI-gain and jammer-leakage terms are still evaluated
    at every physical substep and averaged over the block.

    The null-width deltas remain unsupervised and are learned only through the
    physical SINR/leakage trade-off over the complete hold interval.

    Optional stepped bonuses can independently reward low SINR loss, low SOI
    gain loss and low jammer leakage. Within one bonus family only the largest
    satisfied bonus is applied; bonuses from different families accumulate.

    This class is fully standalone and does not inherit from any environment
    from previous phases.
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
        action_mode: str = "unit_vector",
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
        max_null_horizon_steps: float | None = None,
        nulling_diagonal_loading: float = 1e-8,
        nulling_use_pinv: bool = False,
        reward_beta_sinr_loss: float = 1.0,
        reward_gamma_soi_gain_loss: float = 0.25,
        reward_gamma_jammer_leakage: float = 0.50,
        reward_gamma_hold: float = 0.50,
        reward_gamma_soi_action_error: float = 1.0,
        reward_gamma_jammer_action_error: float = 4.0,
        reward_soi_action_error_scale_deg: float = 10.0,
        reward_soi_action_error_clip: float = 18.0,
        reward_jammer_action_error_scale_deg: float = 10.0,
        reward_jammer_action_error_clip: float = 18.0,
        reward_sinr_loss_scale_db: float = 30.0,
        reward_sinr_loss_clip: float = 2.0,
        reward_soi_gain_loss_scale_db: float = 10.0,
        reward_soi_gain_loss_clip: float = 3.0,
        reward_jammer_leakage_scale: float = 0.05,
        reward_jammer_leakage_clip: float = 5.0,
        reward_hold_scale_db: float = 3.0,
        reward_sinr_loss_bonus_steps: (
            list[tuple[float, float]]
            | tuple[tuple[float, float], ...]
            | np.ndarray
            | None
        ) = None,
        reward_soi_gain_loss_bonus_steps: (
            list[tuple[float, float]]
            | tuple[tuple[float, float], ...]
            | np.ndarray
            | None
        ) = None,
        reward_jammer_leakage_bonus_steps: (
            list[tuple[float, float]]
            | tuple[tuple[float, float], ...]
            | np.ndarray
            | None
        ) = None,
        max_sinr_loss_db: float = 60.0,
        max_soi_gain_loss_db: float = 60.0,
        max_jammer_leakage_loss: float = 30.0,
        mvdr_diagonal_loading: float = 1e-4,
        invalid_sinr_db: float = -120.0,
        invalid_value_penalty: float = -1_000.0,
        max_scenario_sampling_attempts: int = 200,
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
            None
            if jammer_powers is None
            else [float(power) for power in jammer_powers]
        )
        self.jammer_powers = self._build_jammer_powers_for_active_count(
            self.num_active_jammers
        )

        self.observation_mode = str(observation_mode)
        self.action_mode = str(action_mode)

        self.weight_hold_steps = int(weight_hold_steps)
        self.episode_length_physical_steps = int(
            episode_length_physical_steps
        )
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

        self.jammer_target_types = [
            str(value) for value in jammer_target_types
        ]

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
        self.enforce_visible_hemisphere = bool(
            enforce_visible_hemisphere
        )

        self.max_null_horizon_steps = (
            float(self.weight_hold_steps)
            if max_null_horizon_steps is None
            else float(max_null_horizon_steps)
        )

        self.nulling_diagonal_loading = float(
            nulling_diagonal_loading
        )
        self.nulling_use_pinv = bool(nulling_use_pinv)

        self.reward_beta_sinr_loss_phase11 = float(
            reward_beta_sinr_loss
        )
        self.reward_gamma_soi_gain_loss_phase11 = float(
            reward_gamma_soi_gain_loss
        )
        self.reward_gamma_jammer_leakage_phase11 = float(
            reward_gamma_jammer_leakage
        )
        self.reward_gamma_hold_phase11 = float(reward_gamma_hold)

        self.reward_gamma_soi_action_error_phase11 = float(
            reward_gamma_soi_action_error
        )
        self.reward_gamma_jammer_action_error_phase11 = float(
            reward_gamma_jammer_action_error
        )

        self.reward_soi_action_error_scale_deg_phase11 = float(
            reward_soi_action_error_scale_deg
        )
        self.reward_soi_action_error_clip_phase11 = float(
            reward_soi_action_error_clip
        )

        self.reward_jammer_action_error_scale_deg_phase11 = float(
            reward_jammer_action_error_scale_deg
        )
        self.reward_jammer_action_error_clip_phase11 = float(
            reward_jammer_action_error_clip
        )

        self.reward_sinr_loss_scale_db_phase11 = float(
            reward_sinr_loss_scale_db
        )
        self.reward_sinr_loss_clip_phase11 = float(
            reward_sinr_loss_clip
        )

        self.reward_soi_gain_loss_scale_db_phase11 = float(
            reward_soi_gain_loss_scale_db
        )
        self.reward_soi_gain_loss_clip_phase11 = float(
            reward_soi_gain_loss_clip
        )

        self.reward_jammer_leakage_scale_phase11 = float(
            reward_jammer_leakage_scale
        )
        self.reward_jammer_leakage_clip_phase11 = float(
            reward_jammer_leakage_clip
        )

        self.reward_hold_scale_db_phase11 = float(
            reward_hold_scale_db
        )

        self.reward_sinr_loss_bonus_steps_phase11 = (
            self._prepare_bonus_steps(
                reward_sinr_loss_bonus_steps,
                parameter_name="reward_sinr_loss_bonus_steps",
            )
        )
        self.reward_soi_gain_loss_bonus_steps_phase11 = (
            self._prepare_bonus_steps(
                reward_soi_gain_loss_bonus_steps,
                parameter_name="reward_soi_gain_loss_bonus_steps",
            )
        )
        self.reward_jammer_leakage_bonus_steps_phase11 = (
            self._prepare_bonus_steps(
                reward_jammer_leakage_bonus_steps,
                parameter_name="reward_jammer_leakage_bonus_steps",
            )
        )

        self.max_sinr_loss_db = float(max_sinr_loss_db)
        self.max_soi_gain_loss_db = float(max_soi_gain_loss_db)
        self.max_jammer_leakage_loss = float(
            max_jammer_leakage_loss
        )

        self.mvdr_diagonal_loading = float(mvdr_diagonal_loading)
        self.invalid_sinr_db = float(invalid_sinr_db)
        self.invalid_value_penalty = float(invalid_value_penalty)
        self.max_scenario_sampling_attempts = int(
            max_scenario_sampling_attempts
        )

        self.num_elements = int(self.array.N * self.array.M)

        self._validate_configuration()
        self._validate_phase11_configuration()

        # ============================================================
        # Observation space
        # ============================================================

        if self.observation_mode == "angles":
            self.observation_dim = 2 + 5 * self.max_jammers
        elif self.observation_mode == "unit_vector":
            self.observation_dim = 3 + 7 * self.max_jammers
        else:
            raise RuntimeError("Invalid observation_mode.")

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
            # SOI (theta, phi) + per jammer (theta, phi, delta).
            self.action_dim = 2 + 3 * self.max_jammers
            action_low = np.zeros(
                self.action_dim,
                dtype=np.float32,
            )
            action_high = np.ones(
                self.action_dim,
                dtype=np.float32,
            )

        elif self.action_mode == "unit_vector":
            # SOI unit vector (3) + per jammer (unit vector 3 + delta 1).
            self.action_dim = 3 + 4 * self.max_jammers

            low_values = [-1.0, -1.0, -1.0]
            high_values = [1.0, 1.0, 1.0]

            for _ in range(self.max_jammers):
                low_values.extend([-1.0, -1.0, -1.0, 0.0])
                high_values.extend([1.0, 1.0, 1.0, 1.0])

            action_low = np.asarray(low_values, dtype=np.float32)
            action_high = np.asarray(high_values, dtype=np.float32)

        else:
            raise RuntimeError("Invalid action_mode.")

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # ============================================================
        # Episode state
        # ============================================================

        self.current_scenario: dict | None = None
        self.current_physical_step: int = 0

        self.current_theta_rad: float | None = None
        self.current_phi_rad: float | None = None
        self.current_jammer_thetas_rad: list[float] = []
        self.current_jammer_phis_rad: list[float] = []

        self.current_state: np.ndarray | None = None
        self.last_final_weights: np.ndarray | None = None

        # ============================================================
        # Scenario kinematics
        # ============================================================

        self._jammer_theta_rates_rad_s: list[np.ndarray] = []
        self._jammer_phi_rates_rad_s: list[np.ndarray] = []
        self._jammer_unit_vectors: list[np.ndarray] = []
        self._jammer_unit_rates_per_s: list[np.ndarray] = []

        self.current_jammer_theta_rates_rad_s: list[float] = []
        self.current_jammer_phi_rates_rad_s: list[float] = []
        self.current_jammer_unit_vectors: list[np.ndarray] = []
        self.current_jammer_unit_rates_per_s: list[np.ndarray] = []

        self.last_phase11_action_info: dict | None = None
        self.last_phase11_weights: np.ndarray | None = None

    def _validate_configuration(self) -> None:
        """Validate common standalone Phase 11 configuration."""

        if self.max_jammers != 3:
            raise ValueError(
                "This environment currently requires max_jammers=3 "
                "to match the fixed Phase 11 state/action format."
            )

        if self.num_active_jammers < 0:
            raise ValueError(
                "num_active_jammers must be non-negative."
            )

        if self.num_active_jammers > self.max_jammers:
            raise ValueError(
                "num_active_jammers cannot exceed max_jammers."
            )

        if self.active_jammers_choices is not None:
            if len(self.active_jammers_choices) == 0:
                raise ValueError(
                    "active_jammers_choices cannot be empty."
                )

            for value in self.active_jammers_choices:
                if value < 0 or value > self.max_jammers:
                    raise ValueError(
                        "All active_jammers_choices values must be "
                        "between 0 and max_jammers."
                    )

        if self.observation_mode not in [
            "angles",
            "unit_vector",
        ]:
            raise ValueError(
                "Unknown observation_mode. Expected 'angles' "
                "or 'unit_vector'."
            )

        if self.action_mode not in [
            "angles",
            "unit_vector",
        ]:
            raise ValueError(
                "Unknown action_mode. Expected 'angles' "
                "or 'unit_vector'."
            )

        if self.weight_hold_steps <= 0:
            raise ValueError(
                "weight_hold_steps must be a positive integer."
            )

        if self.episode_length_physical_steps <= 0:
            raise ValueError(
                "episode_length_physical_steps must be a "
                "positive integer."
            )

        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

        if self.min_source_distance_m < 0.0:
            raise ValueError(
                "min_source_distance_m must be non-negative."
            )

        if self.min_target_jammer_separation_deg < 0.0:
            raise ValueError(
                "min_target_jammer_separation_deg must be "
                "non-negative."
            )

        if len(self.jammer_target_types) == 0:
            raise ValueError(
                "jammer_target_types cannot be empty."
            )

        valid_target_types = {
            "aircraft",
            "drone",
            "dummy",
            "static",
            "truck",
        }

        for target_type in self.jammer_target_types:
            if target_type not in valid_target_types:
                raise ValueError(
                    f"Unknown jammer target type: {target_type}. "
                    f"Expected one of: {sorted(valid_target_types)}."
                )

        if self.max_sinr_loss_db <= 0.0:
            raise ValueError(
                "max_sinr_loss_db must be positive."
            )

        if self.max_soi_gain_loss_db <= 0.0:
            raise ValueError(
                "max_soi_gain_loss_db must be positive."
            )

        if self.max_jammer_leakage_loss <= 0.0:
            raise ValueError(
                "max_jammer_leakage_loss must be positive."
            )

        if self.mvdr_diagonal_loading < 0.0:
            raise ValueError(
                "mvdr_diagonal_loading must be non-negative."
            )

        if self.max_scenario_sampling_attempts <= 0:
            raise ValueError(
                "max_scenario_sampling_attempts must be a "
                "positive integer."
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Generate one dynamic scenario and build the initial kinematic state."""

        # Reset Gymnasium directly and prepare the Phase 11 kinematic state.
        gym.Env.reset(self, seed=seed)

        self._sample_num_active_jammers_for_episode()
        self.jammer_powers = self._build_jammer_powers_for_active_count(
            self.num_active_jammers
        )

        scenario = self._sample_valid_scenario()

        self.current_scenario = scenario
        self.current_physical_step = 0

        self._prepare_scenario_kinematics()
        self._load_current_phase11_kinematics(step_idx=0)

        state = self._build_phase11_state()
        self.current_state = state

        self.last_phase11_action_info = None
        self.last_phase11_weights = None
        self.last_final_weights = None

        info = {
            "phase": 11,
            "num_active_jammers": self.num_active_jammers,
            "observation_mode": self.observation_mode,
            "action_mode": self.action_mode,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "action_type": self._get_phase11_action_type(),
            "weight_hold_steps": self.weight_hold_steps,
            "episode_length_physical_steps": self.episode_length_physical_steps,
            "dt": self.dt,
            "max_null_horizon_steps": self.max_null_horizon_steps,
            "nulling_diagonal_loading": self.nulling_diagonal_loading,
            "nulling_use_pinv": self.nulling_use_pinv,
            "scenario_metadata": scenario.get("metadata", {}),
            "array_normalize_power": bool(self.array.normalize_power),
            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,
            "theta_target_deg": float(np.rad2deg(self.current_theta_rad)),
            "phi_target_deg": float(np.rad2deg(self.current_phi_rad)),
            "jammer_thetas_rad": self.current_jammer_thetas_rad.copy(),
            "jammer_phis_rad": self.current_jammer_phis_rad.copy(),
            "jammer_thetas_deg": [
                float(np.rad2deg(value))
                for value in self.current_jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(value))
                for value in self.current_jammer_phis_rad
            ],
            "jammer_theta_rates_rad_s": (
                self.current_jammer_theta_rates_rad_s.copy()
            ),
            "jammer_phi_rates_rad_s": (
                self.current_jammer_phi_rates_rad_s.copy()
            ),
            "jammer_theta_rates_deg_s": [
                float(np.rad2deg(value))
                for value in self.current_jammer_theta_rates_rad_s
            ],
            "jammer_phi_rates_deg_s": [
                float(np.rad2deg(value))
                for value in self.current_jammer_phi_rates_rad_s
            ],
            "jammers_directions_deg": self._get_current_jammer_directions_deg(),
            "jammers_powers": self.jammer_powers.copy(),
            **self._build_phase11_reward_configuration_info(),
        }

        return state, info

    def step(self, action: np.ndarray):
        """
        Apply one direction-plus-width target-or-zero action for one K-step block.
        """

        if self.current_scenario is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized.")

        remaining_steps = (
            self.episode_length_physical_steps - self.current_physical_step
        )
        num_block_steps = min(self.weight_hold_steps, remaining_steps)

        if num_block_steps <= 0:
            raise RuntimeError(
                "No physical steps remain. Reset the environment before "
                "calling step() again."
            )

        block_start_step_idx = self.current_physical_step
        self._load_current_phase11_kinematics(
            step_idx=block_start_step_idx
        )

        numerical_error = False
        invalid_phase11_action = False

        try:
            (
                proposed_weights,
                action_info,
            ) = self._action_to_target_or_zero_weights(action)

        except Exception:
            proposed_weights = self._build_safe_fallback_weights()
            action_info = self._build_invalid_phase11_action_info()
            invalid_phase11_action = True
            numerical_error = True

        if not np.all(np.isfinite(proposed_weights)):
            proposed_weights = self._build_safe_fallback_weights()
            action_info = self._build_invalid_phase11_action_info()
            invalid_phase11_action = True
            numerical_error = True

        self.array.set_weights(proposed_weights)

        fixed_weights = np.asarray(
            self.array.W,
            dtype=np.complex128,
        ).copy()

        weights_are_finite = bool(np.all(np.isfinite(fixed_weights)))

        if not weights_are_finite:
            self.array.set_weights(self._build_safe_fallback_weights())

            fixed_weights = np.asarray(
                self.array.W,
                dtype=np.complex128,
            ).copy()

            action_info = self._build_invalid_phase11_action_info()
            invalid_phase11_action = True
            numerical_error = True

        action_info = self._finalize_phase11_action_info(
            action_info=action_info,
            normalized_weights=fixed_weights,
            invalid_phase11_action=invalid_phase11_action,
        )

        # ----------------------------------------------------
        # Geometric guidance is defined at block start.
        #
        # For K > 1, the predicted jammer direction is the
        # central zero selected when the action is taken. The
        # jammer is expected to move while these weights are
        # held, so comparing that same central zero against the
        # evolving jammer direction at every substep would
        # incorrectly penalize the intended predictive behavior.
        #
        # Physical performance is still evaluated at every
        # physical substep below.
        # ----------------------------------------------------
        if invalid_phase11_action:
            block_start_soi_action_error_deg = 0.0
            block_start_jammer_action_errors_deg = []
            block_start_jammer_action_error_deg = 0.0
        else:
            block_start_soi_action_error_deg = (
                self._compute_angular_error_deg(
                    theta_a_rad=self.current_theta_rad,
                    phi_a_rad=self.current_phi_rad,
                    theta_b_rad=float(
                        action_info["theta_soi_action_rad"]
                    ),
                    phi_b_rad=float(
                        action_info["phi_soi_action_rad"]
                    ),
                )
            )

            block_start_jammer_action_errors_deg = []

            for jammer_idx in range(self.num_active_jammers):
                (
                    jammer_theta_action_deg,
                    jammer_phi_action_deg,
                ) = action_info[
                    "jammer_action_directions_deg"
                ][jammer_idx]

                block_start_jammer_action_errors_deg.append(
                    self._compute_angular_error_deg(
                        theta_a_rad=(
                            self.current_jammer_thetas_rad[
                                jammer_idx
                            ]
                        ),
                        phi_a_rad=(
                            self.current_jammer_phis_rad[
                                jammer_idx
                            ]
                        ),
                        theta_b_rad=float(
                            np.deg2rad(
                                jammer_theta_action_deg
                            )
                        ),
                        phi_b_rad=float(
                            np.deg2rad(
                                jammer_phi_action_deg
                            )
                        ),
                    )
                )

            block_start_jammer_action_error_deg = (
                float(
                    np.mean(
                        block_start_jammer_action_errors_deg
                    )
                )
                if len(
                    block_start_jammer_action_errors_deg
                ) > 0
                else 0.0
            )

        action_info["block_start_soi_action_error_deg"] = float(
            block_start_soi_action_error_deg
        )
        action_info[
            "block_start_jammer_action_errors_deg"
        ] = list(
            block_start_jammer_action_errors_deg
        )
        action_info[
            "block_start_jammer_action_error_deg"
        ] = float(
            block_start_jammer_action_error_deg
        )

        self.last_phase11_action_info = action_info
        self.last_phase11_weights = fixed_weights.copy()
        self.last_final_weights = fixed_weights.copy()

        block_metrics: list[dict] = []

        for block_offset in range(num_block_steps):
            step_idx = self.current_physical_step + block_offset

            self._load_current_phase11_kinematics(step_idx=step_idx)

            instant_metrics = self._evaluate_phase11_weights_at_current_step(
                weights=fixed_weights,
                action_info=action_info,
                invalid_phase11_action=invalid_phase11_action,
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
            self.current_physical_step
            >= self.episode_length_physical_steps
        )
        truncated = False

        next_step_idx = min(
            self.current_physical_step,
            self.episode_length_physical_steps - 1,
        )

        self._load_current_phase11_kinematics(step_idx=next_step_idx)

        next_state = self._build_phase11_state()
        self.current_state = next_state

        info = self._build_phase11_block_info(
            block_metrics=block_metrics,
            reward=reward,
            numerical_error=numerical_error,
            weights_are_finite=weights_are_finite,
            fixed_weights=fixed_weights,
            action_info=action_info,
            num_block_steps=num_block_steps,
            terminated=terminated,
        )

        return next_state, reward, terminated, truncated, info

    def _validate_phase11_configuration(self) -> None:
        """Validate Phase 11 specific configuration."""

        if self.action_mode not in ["angles", "unit_vector"]:
            raise ValueError(
                "Unknown action_mode. Expected 'angles' or 'unit_vector'."
            )

        if self.max_null_horizon_steps < 0.0:
            raise ValueError(
                "max_null_horizon_steps must be non-negative."
            )

        if self.nulling_diagonal_loading < 0.0:
            raise ValueError(
                "nulling_diagonal_loading must be non-negative."
            )

        non_negative_parameters = {
            "reward_beta_sinr_loss": self.reward_beta_sinr_loss_phase11,
            "reward_gamma_soi_gain_loss": (
                self.reward_gamma_soi_gain_loss_phase11
            ),
            "reward_gamma_jammer_leakage": (
                self.reward_gamma_jammer_leakage_phase11
            ),
            "reward_gamma_hold": self.reward_gamma_hold_phase11,
            "reward_gamma_soi_action_error": (
                self.reward_gamma_soi_action_error_phase11
            ),
            "reward_gamma_jammer_action_error": (
                self.reward_gamma_jammer_action_error_phase11
            ),
            "reward_soi_action_error_clip": (
                self.reward_soi_action_error_clip_phase11
            ),
            "reward_jammer_action_error_clip": (
                self.reward_jammer_action_error_clip_phase11
            ),
            "reward_sinr_loss_clip": self.reward_sinr_loss_clip_phase11,
            "reward_soi_gain_loss_clip": (
                self.reward_soi_gain_loss_clip_phase11
            ),
            "reward_jammer_leakage_clip": (
                self.reward_jammer_leakage_clip_phase11
            ),
        }

        for name, value in non_negative_parameters.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        positive_parameters = {
            "reward_sinr_loss_scale_db": (
                self.reward_sinr_loss_scale_db_phase11
            ),
            "reward_soi_gain_loss_scale_db": (
                self.reward_soi_gain_loss_scale_db_phase11
            ),
            "reward_jammer_leakage_scale": (
                self.reward_jammer_leakage_scale_phase11
            ),
            "reward_hold_scale_db": self.reward_hold_scale_db_phase11,
            "reward_soi_action_error_scale_deg": (
                self.reward_soi_action_error_scale_deg_phase11
            ),
            "reward_jammer_action_error_scale_deg": (
                self.reward_jammer_action_error_scale_deg_phase11
            ),
        }

        for name, value in positive_parameters.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")

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
                "Could not sample a valid Phase 11 scenario."
            ) from last_error

        raise RuntimeError("Could not sample a valid Phase 11 scenario.")

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

    def _prepare_scenario_kinematics(self) -> None:
        """
        Build angular and unit-vector velocity histories from scenario DOAs.

        For sample k > 0:
            velocity[k] = (x[k] - x[k - 1]) / dt

        For sample k = 0:
            velocity[0] = (x[1] - x[0]) / dt

        Phi differences are wrapped to [-pi, pi) before division by dt.
        """

        if self.current_scenario is None:
            raise RuntimeError("No scenario is currently loaded.")

        self._jammer_theta_rates_rad_s = []
        self._jammer_phi_rates_rad_s = []
        self._jammer_unit_vectors = []
        self._jammer_unit_rates_per_s = []

        for jammer in self.current_scenario["jammers"]:
            theta = np.asarray(
                jammer["doa"]["theta"],
                dtype=float,
            ).reshape(-1)

            phi = np.asarray(
                jammer["doa"]["phi"],
                dtype=float,
            ).reshape(-1)

            if theta.size != self.episode_length_physical_steps:
                raise RuntimeError(
                    "Unexpected jammer theta history length."
                )

            if phi.size != self.episode_length_physical_steps:
                raise RuntimeError(
                    "Unexpected jammer phi history length."
                )

            theta_rate = self._causal_scalar_rate(
                values=theta,
                wrap_period=None,
            )

            phi_rate = self._causal_scalar_rate(
                values=phi,
                wrap_period=2.0 * np.pi,
            )

            unit_vectors = np.stack(
                [
                    self._angles_rad_to_unit_vector(
                        theta_rad=float(theta_k),
                        phi_rad=float(phi_k),
                    )
                    for theta_k, phi_k in zip(theta, phi)
                ],
                axis=0,
            )

            unit_rate = self._causal_vector_rate(unit_vectors)

            self._jammer_theta_rates_rad_s.append(theta_rate)
            self._jammer_phi_rates_rad_s.append(phi_rate)
            self._jammer_unit_vectors.append(unit_vectors)
            self._jammer_unit_rates_per_s.append(unit_rate)

    def _load_current_phase11_kinematics(self, step_idx: int) -> None:
        """Load true DOA and the finite-difference velocity estimate."""

        self._load_current_directions_from_scenario(step_idx=step_idx)

        step_idx = int(step_idx)

        self.current_jammer_theta_rates_rad_s = []
        self.current_jammer_phi_rates_rad_s = []
        self.current_jammer_unit_vectors = []
        self.current_jammer_unit_rates_per_s = []

        for jammer_idx in range(self.num_active_jammers):
            self.current_jammer_theta_rates_rad_s.append(
                float(
                    self._jammer_theta_rates_rad_s[
                        jammer_idx
                    ][step_idx]
                )
            )

            self.current_jammer_phi_rates_rad_s.append(
                float(
                    self._jammer_phi_rates_rad_s[
                        jammer_idx
                    ][step_idx]
                )
            )

            self.current_jammer_unit_vectors.append(
                np.asarray(
                    self._jammer_unit_vectors[
                        jammer_idx
                    ][step_idx],
                    dtype=float,
                ).copy()
            )

            self.current_jammer_unit_rates_per_s.append(
                np.asarray(
                    self._jammer_unit_rates_per_s[
                        jammer_idx
                    ][step_idx],
                    dtype=float,
                ).copy()
            )

    def _causal_scalar_rate(
        self,
        values: np.ndarray,
        wrap_period: float | None,
    ) -> np.ndarray:
        """Compute one causal finite-difference scalar-rate history."""

        values = np.asarray(values, dtype=float).reshape(-1)

        rate = np.zeros_like(values, dtype=float)

        if values.size <= 1:
            return rate

        first_difference = values[1] - values[0]

        if wrap_period is not None:
            first_difference = self._wrap_periodic_difference(
                first_difference,
                wrap_period,
            )

        rate[0] = first_difference / self.dt

        for idx in range(1, values.size):
            difference = values[idx] - values[idx - 1]

            if wrap_period is not None:
                difference = self._wrap_periodic_difference(
                    difference,
                    wrap_period,
                )

            rate[idx] = difference / self.dt

        return rate

    def _causal_vector_rate(self, values: np.ndarray) -> np.ndarray:
        """Compute one causal finite-difference vector-rate history."""

        values = np.asarray(values, dtype=float)

        if values.ndim != 2:
            raise ValueError("values must be a 2D array.")

        rate = np.zeros_like(values, dtype=float)

        if values.shape[0] <= 1:
            return rate

        rate[0] = (values[1] - values[0]) / self.dt
        rate[1:] = (values[1:] - values[:-1]) / self.dt

        return rate

    @staticmethod
    def _wrap_periodic_difference(
        difference: float,
        period: float,
    ) -> float:
        """Wrap one periodic coordinate difference to [-period/2, period/2)."""

        half_period = 0.5 * float(period)

        return float(
            (float(difference) + half_period) % float(period) - half_period
        )

    def _build_phase11_state(self) -> np.ndarray:
        """Build the fixed-size Phase 11 position-plus-velocity state."""

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Current SOI direction is not initialized.")

        if self.observation_mode == "angles":
            state = [
                self._normalize_theta(self.current_theta_rad),
                self._normalize_phi(self.current_phi_rad),
            ]

            theta_span = max(
                self.jammer_theta_max - self.jammer_theta_min,
                1e-12,
            )
            phi_span = max(
                self.jammer_phi_max - self.jammer_phi_min,
                1e-12,
            )

            for jammer_idx in range(self.max_jammers):
                if jammer_idx < self.num_active_jammers:
                    theta_norm = self._normalize_jammer_theta(
                        self.current_jammer_thetas_rad[jammer_idx]
                    )
                    phi_norm = self._normalize_jammer_phi(
                        self.current_jammer_phis_rad[jammer_idx]
                    )

                    theta_rate_norm = float(
                        np.clip(
                            self.current_jammer_theta_rates_rad_s[
                                jammer_idx
                            ]
                            * self.dt
                            / theta_span,
                            -1.0,
                            1.0,
                        )
                    )

                    phi_rate_norm = float(
                        np.clip(
                            self.current_jammer_phi_rates_rad_s[
                                jammer_idx
                            ]
                            * self.dt
                            / phi_span,
                            -1.0,
                            1.0,
                        )
                    )

                    state.extend(
                        [
                            float(theta_norm),
                            float(phi_norm),
                            theta_rate_norm,
                            phi_rate_norm,
                            1.0,
                        ]
                    )
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            result = np.asarray(state, dtype=np.float32)

        elif self.observation_mode == "unit_vector":
            theta_target_deg = float(np.rad2deg(self.current_theta_rad))
            phi_target_deg = float(np.rad2deg(self.current_phi_rad))

            u_target = angles_to_unit_vector(
                theta_deg=theta_target_deg,
                phi_deg=phi_target_deg,
                enforce_visible=self.enforce_visible_hemisphere,
            )

            state = [
                float(u_target[0]),
                float(u_target[1]),
                float(u_target[2]),
            ]

            for jammer_idx in range(self.max_jammers):
                if jammer_idx < self.num_active_jammers:
                    u_jammer = np.asarray(
                        self.current_jammer_unit_vectors[jammer_idx],
                        dtype=float,
                    ).reshape(3)

                    # Difference between two unit-vector components is bounded
                    # by 2. Multiplying du/dt by dt gives the per-step change.
                    # Dividing by 2 maps the theoretical coordinate range to
                    # [-1, 1].
                    du_step_norm = np.clip(
                        (
                            self.current_jammer_unit_rates_per_s[
                                jammer_idx
                            ]
                            * self.dt
                            / 2.0
                        ),
                        -1.0,
                        1.0,
                    )

                    state.extend(
                        [
                            float(u_jammer[0]),
                            float(u_jammer[1]),
                            float(u_jammer[2]),
                            float(du_step_norm[0]),
                            float(du_step_norm[1]),
                            float(du_step_norm[2]),
                            1.0,
                        ]
                    )
                else:
                    state.extend(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    )

            result = np.asarray(state, dtype=np.float32)

        else:
            raise RuntimeError("Invalid observation_mode.")

        if result.shape != (self.observation_dim,):
            raise RuntimeError(
                "Invalid Phase 11 state shape: "
                f"expected {(self.observation_dim,)}, got {result.shape}."
            )

        if not np.all(np.isfinite(result)):
            raise RuntimeError("Phase 11 state contains invalid values.")

        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def _action_to_target_or_zero_weights(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Convert one Phase 11 direction-plus-width action into weights."""

        (
            theta_soi_action_rad,
            phi_soi_action_rad,
            jammer_action_unit_vectors,
            jammer_action_directions_deg,
            delta_norm,
            raw_action,
        ) = self._decode_phase11_action(action)

        (
            zero_directions_deg,
            zero_triplets_deg,
            null_horizons_steps,
        ) = self._build_predictive_zero_directions(
            jammer_action_unit_vectors=jammer_action_unit_vectors,
            delta_norm=delta_norm,
        )

        target_direction_deg = (
            float(np.rad2deg(theta_soi_action_rad)),
            float(np.rad2deg(phi_soi_action_rad)),
        )

        zero_directions_unique_deg = self._deduplicate_zero_directions(
            zero_directions_deg
        )

        weights_flat = target_or_zero_weights(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            target_directions=[target_direction_deg],
            zero_directions=zero_directions_unique_deg,
            diagonal_loading=self.nulling_diagonal_loading,
            use_pinv=self.nulling_use_pinv,
        ).astype(np.complex128).reshape(self.num_elements)

        if not np.all(np.isfinite(weights_flat)):
            raise RuntimeError(
                "target_or_zero_weights returned non-finite values."
            )

        action_info = {
            "raw_action": raw_action.copy(),
            "theta_soi_action_rad": float(theta_soi_action_rad),
            "phi_soi_action_rad": float(phi_soi_action_rad),
            "theta_soi_action_deg": float(
                np.rad2deg(theta_soi_action_rad)
            ),
            "phi_soi_action_deg": float(
                np.rad2deg(phi_soi_action_rad)
            ),
            "jammer_action_unit_vectors": [
                np.asarray(vector, dtype=float).copy()
                for vector in jammer_action_unit_vectors
            ],
            "jammer_action_directions_deg": [
                (float(direction[0]), float(direction[1]))
                for direction in jammer_action_directions_deg
            ],
            "delta_norm": delta_norm.copy(),
            "null_horizons_steps": null_horizons_steps.copy(),
            "null_horizons_seconds": (
                null_horizons_steps * self.dt
            ).copy(),
            "zero_triplets_deg": [
                list(triplet) for triplet in zero_triplets_deg
            ],
            "zero_directions_deg": list(zero_directions_deg),
            "zero_directions_unique_deg": list(
                zero_directions_unique_deg
            ),
            "num_zero_directions_requested": int(
                len(zero_directions_deg)
            ),
            "num_zero_directions_unique": int(
                len(zero_directions_unique_deg)
            ),
            "invalid_phase11_action": False,
        }

        return (
            weights_flat.reshape(self.array.N, self.array.M),
            action_info,
        )

    def _decode_phase11_action(
        self,
        action: np.ndarray,
    ) -> tuple[
        float,
        float,
        list[np.ndarray],
        list[tuple[float, float]],
        np.ndarray,
        np.ndarray,
    ]:
        """Decode SOI direction, jammer directions and null widths."""

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(self.action_dim)

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        jammer_action_unit_vectors: list[np.ndarray] = []
        jammer_action_directions_deg: list[tuple[float, float]] = []
        delta_norm = np.zeros(self.max_jammers, dtype=float)

        if self.action_mode == "angles":
            theta_soi_action_rad = self._denormalize_target_theta(
                float(action[0])
            )
            phi_soi_action_rad = self._denormalize_target_phi(
                float(action[1])
            )

            base_idx = 2

            for jammer_idx in range(self.max_jammers):
                theta_norm = float(action[base_idx])
                phi_norm = float(action[base_idx + 1])
                delta_norm[jammer_idx] = float(action[base_idx + 2])

                theta_rad = self._denormalize_jammer_theta(theta_norm)
                phi_rad = self._denormalize_jammer_phi(phi_norm)

                unit_vector = self._angles_rad_to_unit_vector(
                    theta_rad,
                    phi_rad,
                )

                jammer_action_unit_vectors.append(unit_vector)
                jammer_action_directions_deg.append(
                    (
                        float(np.rad2deg(theta_rad)),
                        float(np.rad2deg(phi_rad)),
                    )
                )

                base_idx += 3

        elif self.action_mode == "unit_vector":
            u_soi_action = self._normalize_phase11_action_unit_vector(
                action[:3]
            )

            theta_soi_deg, phi_soi_deg = unit_vector_to_angles(
                u_soi_action,
                enforce_visible=self.enforce_visible_hemisphere,
            )

            theta_soi_action_rad = float(np.deg2rad(theta_soi_deg))
            phi_soi_action_rad = float(np.deg2rad(phi_soi_deg))

            base_idx = 3

            for jammer_idx in range(self.max_jammers):
                jammer_vector = self._normalize_phase11_action_unit_vector(
                    action[base_idx : base_idx + 3]
                )
                delta_norm[jammer_idx] = float(action[base_idx + 3])

                jammer_direction_deg = self._unit_vector_to_direction_deg(
                    jammer_vector
                )

                jammer_action_unit_vectors.append(jammer_vector)
                jammer_action_directions_deg.append(jammer_direction_deg)

                base_idx += 4

        else:
            raise RuntimeError("Invalid action_mode.")

        delta_norm = np.clip(delta_norm, 0.0, 1.0)

        if len(jammer_action_unit_vectors) != self.max_jammers:
            raise RuntimeError(
                "Invalid number of jammer directions decoded from action."
            )

        return (
            theta_soi_action_rad,
            phi_soi_action_rad,
            jammer_action_unit_vectors,
            jammer_action_directions_deg,
            delta_norm,
            action.copy(),
        )

    def _build_predictive_zero_directions(
        self,
        jammer_action_unit_vectors: list[np.ndarray],
        delta_norm: np.ndarray,
    ) -> tuple[
        list[tuple[float, float]],
        list[tuple[tuple[float, float], ...]],
        np.ndarray,
    ]:
        """
        Build Z-, Z0 and Z+ around each jammer direction predicted by the agent.

        For active jammer ``i`` the agent-predicted jammer position is the
        central zero ``Z0``. The two side zeros are displaced along the current
        jammer motion contained in the observation:

            u_minus_raw = u_predicted - h_i * du_per_step
            u_center    = u_predicted
            u_plus_raw  = u_predicted + h_i * du_per_step

        where ``h_i = delta_i * max_null_horizon_steps``. All three vectors are
        projected back to the visible unit hemisphere.
        """

        delta_norm = np.asarray(
            delta_norm,
            dtype=float,
        ).reshape(self.max_jammers)

        if len(jammer_action_unit_vectors) != self.max_jammers:
            raise ValueError(
                "jammer_action_unit_vectors must contain one vector per "
                "jammer slot."
            )

        null_horizons_steps = (
            np.clip(delta_norm, 0.0, 1.0)
            * self.max_null_horizon_steps
        )

        zero_directions_deg: list[tuple[float, float]] = []
        zero_triplets_deg: list[tuple[tuple[float, float], ...]] = []

        for jammer_idx in range(self.num_active_jammers):
            predicted_u = self._project_unit_vector_to_visible_hemisphere(
                jammer_action_unit_vectors[jammer_idx]
            )

            du_per_step = (
                np.asarray(
                    self.current_jammer_unit_rates_per_s[jammer_idx],
                    dtype=float,
                ).reshape(3)
                * self.dt
            )

            horizon_steps = float(
                null_horizons_steps[jammer_idx]
            )

            u_minus = self._project_unit_vector_to_visible_hemisphere(
                predicted_u - horizon_steps * du_per_step
            )
            u_center = predicted_u.copy()
            u_plus = self._project_unit_vector_to_visible_hemisphere(
                predicted_u + horizon_steps * du_per_step
            )

            minus_deg = self._unit_vector_to_direction_deg(u_minus)
            center_deg = self._unit_vector_to_direction_deg(u_center)
            plus_deg = self._unit_vector_to_direction_deg(u_plus)

            triplet = (minus_deg, center_deg, plus_deg)

            zero_triplets_deg.append(triplet)
            zero_directions_deg.extend(triplet)

        return (
            zero_directions_deg,
            zero_triplets_deg,
            null_horizons_steps,
        )

    def _project_unit_vector_to_visible_hemisphere(
        self,
        vector: np.ndarray,
    ) -> np.ndarray:
        """Normalize one vector and apply the existing visible-hemisphere rule."""

        vector = np.asarray(vector, dtype=float).reshape(3)

        norm = float(np.linalg.norm(vector))

        if not np.isfinite(norm) or norm < 1e-12:
            vector = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            vector = vector / norm

        if self.enforce_visible_hemisphere and vector[2] < 0.0:
            vector[2] = abs(vector[2])

            renorm = float(np.linalg.norm(vector))

            if renorm < 1e-12:
                vector = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                vector = vector / renorm

        return vector.astype(float)

    def _unit_vector_to_direction_deg(
        self,
        vector: np.ndarray,
    ) -> tuple[float, float]:
        """Convert one unit vector into the project's angular convention."""

        theta_deg, phi_deg = unit_vector_to_angles(
            np.asarray(vector, dtype=float).reshape(3),
            enforce_visible=self.enforce_visible_hemisphere,
        )

        return float(theta_deg), float(phi_deg)

    def _deduplicate_zero_directions(
        self,
        directions_deg: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Remove only numerically duplicate physical directions.

        This is primarily needed for static jammers, where Z-, Z0 and Z+
        collapse exactly to the same point.
        """

        unique: list[tuple[float, float]] = []

        for candidate in directions_deg:
            candidate_u = self._angles_rad_to_unit_vector(
                theta_rad=float(np.deg2rad(candidate[0])),
                phi_rad=float(np.deg2rad(candidate[1])),
            )

            duplicate = False

            for existing in unique:
                existing_u = self._angles_rad_to_unit_vector(
                    theta_rad=float(np.deg2rad(existing[0])),
                    phi_rad=float(np.deg2rad(existing[1])),
                )

                if np.allclose(
                    candidate_u,
                    existing_u,
                    rtol=0.0,
                    atol=1e-10,
                ):
                    duplicate = True
                    break

            if not duplicate:
                unique.append(
                    (float(candidate[0]), float(candidate[1]))
                )

        return unique

    def _normalize_phase11_action_unit_vector(
        self,
        action_vector: np.ndarray,
    ) -> np.ndarray:
        """Normalize the three SOI unit-vector action components."""

        vector = np.asarray(
            action_vector,
            dtype=float,
        ).reshape(3)

        norm = float(np.linalg.norm(vector))

        if norm < 1e-8:
            vector = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            vector = vector / norm

        if self.enforce_visible_hemisphere and vector[2] < 0.0:
            vector[2] = abs(vector[2])

            vector = vector / (
                float(np.linalg.norm(vector)) + 1e-12
            )

        return vector.astype(float)

    def _denormalize_target_theta(self, value: float) -> float:
        """Convert normalized SOI theta action to radians."""

        return float(
            self.theta_min
            + float(value) * (self.theta_max - self.theta_min)
        )

    def _denormalize_target_phi(self, value: float) -> float:
        """Convert normalized SOI phi action to radians."""

        return float(
            self.phi_min
            + float(value) * (self.phi_max - self.phi_min)
        )

    def _evaluate_phase11_weights_at_current_step(
        self,
        weights: np.ndarray,
        action_info: dict,
        invalid_phase11_action: bool,
    ) -> dict:
        """Evaluate fixed target-or-zero weights at one physical substep."""

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
            reference_sinr_db = self._compute_sinr_for_weights(
                reference_weights
            )
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

        clipped_sinr_loss_db = min(
            sinr_loss_db,
            self.max_sinr_loss_db,
        )

        try:
            soi_gain_metrics = self._compute_soi_gain_metrics(
                weights=weights
            )
        except Exception:
            soi_gain_metrics = self._build_invalid_soi_gain_metrics()
            numerical_error = True

        try:
            jammer_leakage_metrics = self._compute_jammer_leakage_metrics(
                weights=weights,
                soi_gain_linear=soi_gain_metrics["soi_gain_linear"],
            )
        except Exception:
            jammer_leakage_metrics = (
                self._build_invalid_jammer_leakage_metrics()
            )
            numerical_error = True

        soi_action_error_deg = self._compute_angular_error_deg(
            theta_a_rad=self.current_theta_rad,
            phi_a_rad=self.current_phi_rad,
            theta_b_rad=float(action_info["theta_soi_action_rad"]),
            phi_b_rad=float(action_info["phi_soi_action_rad"]),
        )

        jammer_action_errors_deg: list[float] = []

        for jammer_idx in range(self.num_active_jammers):
            jammer_theta_action_deg, jammer_phi_action_deg = (
                action_info["jammer_action_directions_deg"][jammer_idx]
            )

            jammer_action_errors_deg.append(
                self._compute_angular_error_deg(
                    theta_a_rad=self.current_jammer_thetas_rad[jammer_idx],
                    phi_a_rad=self.current_jammer_phis_rad[jammer_idx],
                    theta_b_rad=float(np.deg2rad(jammer_theta_action_deg)),
                    phi_b_rad=float(np.deg2rad(jammer_phi_action_deg)),
                )
            )

        jammer_action_error_deg = (
            float(np.mean(jammer_action_errors_deg))
            if len(jammer_action_errors_deg) > 0
            else 0.0
        )

        reward_soi_action_error_deg = float(
            action_info.get(
                "block_start_soi_action_error_deg",
                soi_action_error_deg,
            )
        )

        reward_jammer_action_error_deg = float(
            action_info.get(
                "block_start_jammer_action_error_deg",
                jammer_action_error_deg,
            )
        )

        reward, reward_info = self._compute_phase11_reward(
            sinr_loss_db=sinr_loss_db,
            soi_gain_loss_db=float(
                soi_gain_metrics["soi_gain_loss_db"]
            ),
            jammer_mean_leakage=float(
                jammer_leakage_metrics["jammer_leakage_loss"]
            ),
            soi_action_error_deg=reward_soi_action_error_deg,
            jammer_action_error_deg=(
                reward_jammer_action_error_deg
            ),
            invalid_phase11_action=invalid_phase11_action,
            numerical_error=numerical_error,
        )

        return {
            "reward": float(reward),
            "sinr_db": float(sinr_db),
            "reference_sinr_db": float(reference_sinr_db),
            "sinr_loss_db": float(sinr_loss_db),
            "clipped_sinr_loss_db": float(clipped_sinr_loss_db),
            # Instantaneous geometry diagnostics compare the
            # block-start action against the current physical
            # substep.
            "soi_action_error_deg": float(soi_action_error_deg),
            "jammer_action_errors_deg": jammer_action_errors_deg.copy(),
            "jammer_action_error_deg": float(jammer_action_error_deg),

            # These fixed block-start values are the ones used
            # by the angular guidance terms in the reward.
            "reward_soi_action_error_deg": float(
                reward_soi_action_error_deg
            ),
            "reward_jammer_action_error_deg": float(
                reward_jammer_action_error_deg
            ),
            "numerical_error": bool(numerical_error),
            "invalid_phase11_action": bool(invalid_phase11_action),
            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,
            "theta_target_deg": float(np.rad2deg(self.current_theta_rad)),
            "phi_target_deg": float(np.rad2deg(self.current_phi_rad)),
            "jammer_thetas_rad": self.current_jammer_thetas_rad.copy(),
            "jammer_phis_rad": self.current_jammer_phis_rad.copy(),
            "jammer_thetas_deg": [
                float(np.rad2deg(value))
                for value in self.current_jammer_thetas_rad
            ],
            "jammer_phis_deg": [
                float(np.rad2deg(value))
                for value in self.current_jammer_phis_rad
            ],
            "jammer_theta_rates_rad_s": (
                self.current_jammer_theta_rates_rad_s.copy()
            ),
            "jammer_phi_rates_rad_s": (
                self.current_jammer_phi_rates_rad_s.copy()
            ),
            "num_active_jammers": self.num_active_jammers,
            "jammers_directions_deg": self._get_current_jammer_directions_deg(),
            "jammers_powers": self.jammer_powers.copy(),
            **soi_gain_metrics,
            **jammer_leakage_metrics,
            **reward_info,
        }

    def _compute_phase11_reward(
        self,
        sinr_loss_db: float,
        soi_gain_loss_db: float,
        jammer_mean_leakage: float,
        soi_action_error_deg: float,
        jammer_action_error_deg: float,
        invalid_phase11_action: bool,
        numerical_error: bool,
    ) -> tuple[float, dict]:
        """
        Compute the dense physical reward plus explicit geometric guidance.

        The SOI and jammer angular terms supervise only the directional
        components of the action. Null-width deltas are not directly rewarded.
        """

        if invalid_phase11_action:
            return float(self.invalid_value_penalty), {
                **self._build_zero_phase11_reward_components(),
                "reward_failure_applied": True,
                "reward_failure_reason": "invalid_phase11_action",
            }

        if numerical_error:
            return float(self.invalid_value_penalty), {
                **self._build_zero_phase11_reward_components(),
                "reward_failure_applied": True,
                "reward_failure_reason": "numerical_error",
            }

        normalized_sinr_loss = float(
            np.clip(
                float(sinr_loss_db) / self.reward_sinr_loss_scale_db_phase11,
                0.0,
                self.reward_sinr_loss_clip_phase11,
            )
        )

        normalized_soi_gain_loss = float(
            np.clip(
                float(soi_gain_loss_db)
                / self.reward_soi_gain_loss_scale_db_phase11,
                0.0,
                self.reward_soi_gain_loss_clip_phase11,
            )
        )

        if self.num_active_jammers == 0:
            normalized_jammer_leakage = 0.0
        else:
            normalized_jammer_leakage = float(
                np.clip(
                    float(jammer_mean_leakage)
                    / self.reward_jammer_leakage_scale_phase11,
                    0.0,
                    self.reward_jammer_leakage_clip_phase11,
                )
            )

        hold_score = float(
            np.exp(
                -max(0.0, float(sinr_loss_db))
                / self.reward_hold_scale_db_phase11
            )
        )

        normalized_soi_action_error = float(
            np.clip(
                float(soi_action_error_deg)
                / self.reward_soi_action_error_scale_deg_phase11,
                0.0,
                self.reward_soi_action_error_clip_phase11,
            )
        )

        if self.num_active_jammers == 0:
            normalized_jammer_action_error = 0.0
        else:
            normalized_jammer_action_error = float(
                np.clip(
                    float(jammer_action_error_deg)
                    / self.reward_jammer_action_error_scale_deg_phase11,
                    0.0,
                    self.reward_jammer_action_error_clip_phase11,
                )
            )

        sinr_loss_component = float(
            -self.reward_beta_sinr_loss_phase11 * normalized_sinr_loss
        )
        soi_gain_loss_component = float(
            -self.reward_gamma_soi_gain_loss_phase11
            * normalized_soi_gain_loss
        )
        jammer_leakage_component = float(
            -self.reward_gamma_jammer_leakage_phase11
            * normalized_jammer_leakage
        )
        hold_component = float(
            self.reward_gamma_hold_phase11 * hold_score
        )
        soi_action_error_component = float(
            -self.reward_gamma_soi_action_error_phase11
            * normalized_soi_action_error
        )
        jammer_action_error_component = float(
            -self.reward_gamma_jammer_action_error_phase11
            * normalized_jammer_action_error
        )

        sinr_bonus_info = self._compute_stepped_bonus(
            metric_value=sinr_loss_db,
            bonus_steps=self.reward_sinr_loss_bonus_steps_phase11,
            lower_is_better=True,
        )
        soi_bonus_info = self._compute_stepped_bonus(
            metric_value=soi_gain_loss_db,
            bonus_steps=self.reward_soi_gain_loss_bonus_steps_phase11,
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
                bonus_steps=self.reward_jammer_leakage_bonus_steps_phase11,
                lower_is_better=True,
            )

        sinr_bonus = float(sinr_bonus_info["bonus"])
        soi_bonus = float(soi_bonus_info["bonus"])
        jammer_bonus = float(jammer_bonus_info["bonus"])
        total_bonus = float(sinr_bonus + soi_bonus + jammer_bonus)

        reward = float(
            sinr_loss_component
            + soi_gain_loss_component
            + jammer_leakage_component
            + hold_component
            + soi_action_error_component
            + jammer_action_error_component
            + total_bonus
        )

        if not np.isfinite(reward):
            reward = self.invalid_value_penalty

        info = {
            "reward_failure_applied": False,
            "reward_failure_reason": "none",
            "reward_normalized_sinr_loss": normalized_sinr_loss,
            "reward_normalized_soi_gain_loss": normalized_soi_gain_loss,
            "reward_normalized_jammer_leakage": normalized_jammer_leakage,
            "reward_hold_score": hold_score,
            "reward_normalized_soi_action_error": (
                normalized_soi_action_error
            ),
            "reward_normalized_jammer_action_error": (
                normalized_jammer_action_error
            ),
            "reward_sinr_loss_component": sinr_loss_component,
            "reward_soi_gain_loss_component": soi_gain_loss_component,
            "reward_jammer_leakage_component": jammer_leakage_component,
            "reward_hold_component": hold_component,
            "reward_soi_action_error_component": (
                soi_action_error_component
            ),
            "reward_jammer_action_error_component": (
                jammer_action_error_component
            ),
            "reward_sinr_loss_bonus": sinr_bonus,
            "reward_soi_gain_loss_bonus": soi_bonus,
            "reward_jammer_leakage_bonus": jammer_bonus,
            "reward_total_stepped_bonus": total_bonus,
            "reward_sinr_loss_bonus_matched": bool(
                sinr_bonus_info["matched"]
            ),
            "reward_soi_gain_loss_bonus_matched": bool(
                soi_bonus_info["matched"]
            ),
            "reward_jammer_leakage_bonus_matched": bool(
                jammer_bonus_info["matched"]
            ),
            "reward_sinr_loss_bonus_threshold": float(
                sinr_bonus_info["matched_threshold"]
            ),
            "reward_soi_gain_loss_bonus_threshold": float(
                soi_bonus_info["matched_threshold"]
            ),
            "reward_jammer_leakage_bonus_threshold": float(
                jammer_bonus_info["matched_threshold"]
            ),
        }

        return reward, info

    @staticmethod
    def _build_zero_phase11_reward_components() -> dict:
        """Return zero-valued reward diagnostics for invalid transitions."""

        return {
            "reward_normalized_sinr_loss": 0.0,
            "reward_normalized_soi_gain_loss": 0.0,
            "reward_normalized_jammer_leakage": 0.0,
            "reward_hold_score": 0.0,
            "reward_normalized_soi_action_error": 0.0,
            "reward_normalized_jammer_action_error": 0.0,
            "reward_sinr_loss_component": 0.0,
            "reward_soi_gain_loss_component": 0.0,
            "reward_jammer_leakage_component": 0.0,
            "reward_hold_component": 0.0,
            "reward_soi_action_error_component": 0.0,
            "reward_jammer_action_error_component": 0.0,
            "reward_sinr_loss_bonus": 0.0,
            "reward_soi_gain_loss_bonus": 0.0,
            "reward_jammer_leakage_bonus": 0.0,
            "reward_total_stepped_bonus": 0.0,
            "reward_sinr_loss_bonus_matched": False,
            "reward_soi_gain_loss_bonus_matched": False,
            "reward_jammer_leakage_bonus_matched": False,
            "reward_sinr_loss_bonus_threshold": float("nan"),
            "reward_soi_gain_loss_bonus_threshold": float("nan"),
            "reward_jammer_leakage_bonus_threshold": float("nan"),
        }

    def _build_phase11_block_info(
        self,
        block_metrics: list[dict],
        reward: float,
        numerical_error: bool,
        weights_are_finite: bool,
        fixed_weights: np.ndarray,
        action_info: dict,
        num_block_steps: int,
        terminated: bool,
    ) -> dict:
        """Build aggregated logging information for one Phase 11 block."""

        last_metrics = (
            block_metrics[-1] if len(block_metrics) > 0 else {}
        )

        info = {
            "phase": 11,
            "reward": float(reward),
            "block_reward_mean": float(reward),
            "num_block_steps": int(num_block_steps),
            "weight_hold_steps": self.weight_hold_steps,
            "current_physical_step": int(self.current_physical_step),
            "episode_length_physical_steps": (
                self.episode_length_physical_steps
            ),
            "terminated": bool(terminated),
            "observation_mode": self.observation_mode,
            "action_mode": self.action_mode,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "action_type": self._get_phase11_action_type(),
            "num_active_jammers": self.num_active_jammers,
            "jammers_powers": self.jammer_powers.copy(),
            "max_null_horizon_steps": self.max_null_horizon_steps,
            "weights": np.asarray(
                fixed_weights,
                dtype=np.complex128,
            ).copy(),
            "weights_are_finite": bool(weights_are_finite),
            "numerical_error": bool(numerical_error),
            "array_normalize_power": bool(self.array.normalize_power),
            "raw_action": action_info["raw_action"].copy(),
            "theta_soi_action_rad": float(
                action_info["theta_soi_action_rad"]
            ),
            "phi_soi_action_rad": float(
                action_info["phi_soi_action_rad"]
            ),
            "theta_soi_action_deg": float(
                action_info["theta_soi_action_deg"]
            ),
            "phi_soi_action_deg": float(
                action_info["phi_soi_action_deg"]
            ),
            "jammer_action_unit_vectors": [
                np.asarray(vector, dtype=float).copy()
                for vector in action_info["jammer_action_unit_vectors"]
            ],
            "jammer_action_directions_deg": list(
                action_info["jammer_action_directions_deg"]
            ),
            "block_start_soi_action_error_deg": float(
                action_info.get(
                    "block_start_soi_action_error_deg",
                    0.0,
                )
            ),
            "block_start_jammer_action_errors_deg": list(
                action_info.get(
                    "block_start_jammer_action_errors_deg",
                    [],
                )
            ),
            "block_start_jammer_action_error_deg": float(
                action_info.get(
                    "block_start_jammer_action_error_deg",
                    0.0,
                )
            ),
            "delta_norm": action_info["delta_norm"].copy(),
            "null_horizons_steps": (
                action_info["null_horizons_steps"].copy()
            ),
            "null_horizons_seconds": (
                action_info["null_horizons_seconds"].copy()
            ),
            "zero_triplets_deg": action_info["zero_triplets_deg"],
            "zero_directions_deg": (
                action_info["zero_directions_deg"]
            ),
            "zero_directions_unique_deg": (
                action_info["zero_directions_unique_deg"]
            ),
            "num_zero_directions_requested": int(
                action_info["num_zero_directions_requested"]
            ),
            "num_zero_directions_unique": int(
                action_info["num_zero_directions_unique"]
            ),
            "invalid_phase11_action": bool(
                action_info["invalid_phase11_action"]
            ),
            "final_weight_power": float(
                action_info["final_weight_power"]
            ),
            "final_magnitude": action_info["final_magnitude"].copy(),
            "final_phase_rad": action_info["final_phase_rad"].copy(),
            "substep_metrics": block_metrics,
            **self._build_phase11_reward_configuration_info(),
        }

        aggregate_keys = [
            "reward",
            "sinr_db",
            "reference_sinr_db",
            "sinr_loss_db",
            "clipped_sinr_loss_db",
            "soi_action_error_deg",
            "jammer_action_error_deg",
            "reward_soi_action_error_deg",
            "reward_jammer_action_error_deg",
            "soi_gain_db",
            "reference_soi_gain_db",
            "soi_gain_loss_db",
            "clipped_soi_gain_loss_db",
            "jammer_leakage_loss",
            "clipped_jammer_leakage_loss",
            "jammer_leakage_max",
            "reward_normalized_sinr_loss",
            "reward_normalized_soi_gain_loss",
            "reward_normalized_jammer_leakage",
            "reward_hold_score",
            "reward_normalized_soi_action_error",
            "reward_normalized_jammer_action_error",
            "reward_sinr_loss_component",
            "reward_soi_gain_loss_component",
            "reward_jammer_leakage_component",
            "reward_hold_component",
            "reward_soi_action_error_component",
            "reward_jammer_action_error_component",
            "reward_sinr_loss_bonus",
            "reward_soi_gain_loss_bonus",
            "reward_jammer_leakage_bonus",
            "reward_total_stepped_bonus",
            "reward_sinr_loss_bonus_matched",
            "reward_soi_gain_loss_bonus_matched",
            "reward_jammer_leakage_bonus_matched",
            "reward_failure_applied",
            "invalid_phase11_action",
        ]

        for key in aggregate_keys:
            info[f"{key}_mean"] = self._safe_mean_metric(
                block_metrics,
                key,
            )
            info[f"{key}_last"] = self._safe_last_metric(
                block_metrics,
                key,
            )

        # Common aliases used by previous evaluation notebooks.
        info["sinr_db"] = info["sinr_db_mean"]
        info["reference_sinr_db"] = info["reference_sinr_db_mean"]
        info["sinr_loss_db"] = info["sinr_loss_db_mean"]
        info["clipped_sinr_loss_db"] = (
            info["clipped_sinr_loss_db_mean"]
        )
        info["soi_gain_loss_db"] = info["soi_gain_loss_db_mean"]
        info["jammer_leakage_loss"] = (
            info["jammer_leakage_loss_mean"]
        )
        info["soi_action_error_deg"] = (
            info["soi_action_error_deg_mean"]
        )
        info["jammer_action_error_deg"] = (
            info["jammer_action_error_deg_mean"]
        )

        info["reward_soi_action_error_deg"] = (
            info["reward_soi_action_error_deg_mean"]
        )
        info["reward_jammer_action_error_deg"] = (
            info["reward_jammer_action_error_deg_mean"]
        )

        info["reward_normalized_soi_action_error"] = (
            info["reward_normalized_soi_action_error_mean"]
        )
        info["reward_normalized_jammer_action_error"] = (
            info["reward_normalized_jammer_action_error_mean"]
        )
        info["reward_soi_action_error_component"] = (
            info["reward_soi_action_error_component_mean"]
        )
        info["reward_jammer_action_error_component"] = (
            info["reward_jammer_action_error_component_mean"]
        )

        info["reward_sinr_loss_bonus"] = (
            info["reward_sinr_loss_bonus_mean"]
        )
        info["reward_soi_gain_loss_bonus"] = (
            info["reward_soi_gain_loss_bonus_mean"]
        )
        info["reward_jammer_leakage_bonus"] = (
            info["reward_jammer_leakage_bonus_mean"]
        )
        info["reward_total_stepped_bonus"] = (
            info["reward_total_stepped_bonus_mean"]
        )

        info.update(
            {
                "theta_target_rad": last_metrics.get(
                    "theta_target_rad"
                ),
                "phi_target_rad": last_metrics.get(
                    "phi_target_rad"
                ),
                "theta_target_deg": last_metrics.get(
                    "theta_target_deg"
                ),
                "phi_target_deg": last_metrics.get(
                    "phi_target_deg"
                ),
                "jammer_thetas_rad": last_metrics.get(
                    "jammer_thetas_rad",
                    [],
                ),
                "jammer_phis_rad": last_metrics.get(
                    "jammer_phis_rad",
                    [],
                ),
                "jammer_thetas_deg": last_metrics.get(
                    "jammer_thetas_deg",
                    [],
                ),
                "jammer_phis_deg": last_metrics.get(
                    "jammer_phis_deg",
                    [],
                ),
                "jammer_theta_rates_rad_s": last_metrics.get(
                    "jammer_theta_rates_rad_s",
                    [],
                ),
                "jammer_phi_rates_rad_s": last_metrics.get(
                    "jammer_phi_rates_rad_s",
                    [],
                ),
                "jammers_directions_deg": last_metrics.get(
                    "jammers_directions_deg",
                    [],
                ),
                "jammer_gains_linear": last_metrics.get(
                    "jammer_gains_linear",
                    [],
                ),
                "jammer_gains_db": last_metrics.get(
                    "jammer_gains_db",
                    [],
                ),
                "jammer_leakage_values": last_metrics.get(
                    "jammer_leakage_values",
                    [],
                ),
            }
        )

        return info

    def _build_phase11_reward_configuration_info(self) -> dict:
        """Return the complete Phase 11 reward configuration."""

        return {
            "reward_type": "phase11_dense_direction_width_with_angular_guidance",
            "reward_beta_sinr_loss": self.reward_beta_sinr_loss_phase11,
            "reward_gamma_soi_gain_loss": (
                self.reward_gamma_soi_gain_loss_phase11
            ),
            "reward_gamma_jammer_leakage": (
                self.reward_gamma_jammer_leakage_phase11
            ),
            "reward_gamma_hold": self.reward_gamma_hold_phase11,
            "reward_gamma_soi_action_error": (
                self.reward_gamma_soi_action_error_phase11
            ),
            "reward_gamma_jammer_action_error": (
                self.reward_gamma_jammer_action_error_phase11
            ),
            "reward_soi_action_error_scale_deg": (
                self.reward_soi_action_error_scale_deg_phase11
            ),
            "reward_soi_action_error_clip": (
                self.reward_soi_action_error_clip_phase11
            ),
            "reward_jammer_action_error_scale_deg": (
                self.reward_jammer_action_error_scale_deg_phase11
            ),
            "reward_jammer_action_error_clip": (
                self.reward_jammer_action_error_clip_phase11
            ),
            "reward_sinr_loss_scale_db": (
                self.reward_sinr_loss_scale_db_phase11
            ),
            "reward_sinr_loss_clip": self.reward_sinr_loss_clip_phase11,
            "reward_soi_gain_loss_scale_db": (
                self.reward_soi_gain_loss_scale_db_phase11
            ),
            "reward_soi_gain_loss_clip": (
                self.reward_soi_gain_loss_clip_phase11
            ),
            "reward_jammer_leakage_scale": (
                self.reward_jammer_leakage_scale_phase11
            ),
            "reward_jammer_leakage_clip": (
                self.reward_jammer_leakage_clip_phase11
            ),
            "reward_hold_scale_db": self.reward_hold_scale_db_phase11,
            "reward_sinr_loss_bonus_steps": (
                self.reward_sinr_loss_bonus_steps_phase11.copy()
            ),
            "reward_soi_gain_loss_bonus_steps": (
                self.reward_soi_gain_loss_bonus_steps_phase11.copy()
            ),
            "reward_jammer_leakage_bonus_steps": (
                self.reward_jammer_leakage_bonus_steps_phase11.copy()
            ),
        }

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

    def _build_invalid_phase11_action_info(self) -> dict:
        """Build safe logging data for a failed Phase 11 action."""

        if self.current_theta_rad is None or self.current_phi_rad is None:
            theta_soi_action_rad = 0.0
            phi_soi_action_rad = 0.0
        else:
            theta_soi_action_rad = float(self.current_theta_rad)
            phi_soi_action_rad = float(self.current_phi_rad)

        zero_triplets = []

        for jammer_idx in range(self.num_active_jammers):
            center = (
                float(
                    np.rad2deg(
                        self.current_jammer_thetas_rad[jammer_idx]
                    )
                ),
                float(
                    np.rad2deg(
                        self.current_jammer_phis_rad[jammer_idx]
                    )
                ),
            )
            zero_triplets.append((center, center, center))

        flat_zeros = [
            direction
            for triplet in zero_triplets
            for direction in triplet
        ]

        return {
            "raw_action": np.zeros(
                self.action_dim,
                dtype=np.float32,
            ),
            "theta_soi_action_rad": theta_soi_action_rad,
            "phi_soi_action_rad": phi_soi_action_rad,
            "theta_soi_action_deg": float(
                np.rad2deg(theta_soi_action_rad)
            ),
            "phi_soi_action_deg": float(
                np.rad2deg(phi_soi_action_rad)
            ),
            "jammer_action_unit_vectors": [
                (
                    np.asarray(
                        self.current_jammer_unit_vectors[idx],
                        dtype=float,
                    ).copy()
                    if idx < self.num_active_jammers
                    else np.array([0.0, 0.0, 1.0], dtype=float)
                )
                for idx in range(self.max_jammers)
            ],
            "jammer_action_directions_deg": [
                (
                    (
                        float(
                            np.rad2deg(
                                self.current_jammer_thetas_rad[idx]
                            )
                        ),
                        float(
                            np.rad2deg(
                                self.current_jammer_phis_rad[idx]
                            )
                        ),
                    )
                    if idx < self.num_active_jammers
                    else (0.0, 0.0)
                )
                for idx in range(self.max_jammers)
            ],
            "delta_norm": np.zeros(
                self.max_jammers,
                dtype=float,
            ),
            "null_horizons_steps": np.zeros(
                self.max_jammers,
                dtype=float,
            ),
            "null_horizons_seconds": np.zeros(
                self.max_jammers,
                dtype=float,
            ),
            "zero_triplets_deg": [
                list(triplet) for triplet in zero_triplets
            ],
            "zero_directions_deg": flat_zeros,
            "zero_directions_unique_deg": (
                self._deduplicate_zero_directions(flat_zeros)
            ),
            "num_zero_directions_requested": int(len(flat_zeros)),
            "num_zero_directions_unique": int(
                len(self._deduplicate_zero_directions(flat_zeros))
            ),
            "invalid_phase11_action": True,
        }

    def _finalize_phase11_action_info(
        self,
        action_info: dict,
        normalized_weights: np.ndarray,
        invalid_phase11_action: bool,
    ) -> dict:
        """Add final normalized weight diagnostics to the action information."""

        result = dict(action_info)

        weights = np.asarray(
            normalized_weights,
            dtype=np.complex128,
        ).reshape(self.array.N, self.array.M)

        weights_flat = weights.reshape(self.num_elements)

        result.update(
            {
                "weights": weights.copy(),
                "final_weight_power": float(
                    np.sum(np.abs(weights_flat) ** 2)
                ),
                "final_magnitude": np.abs(weights_flat).copy(),
                "final_phase_rad": np.angle(weights_flat).copy(),
                "invalid_phase11_action": bool(
                    invalid_phase11_action
                ),
            }
        )

        return result

    def _get_phase11_action_type(self) -> str:
        """Return a stable action identifier for logs and evaluation."""

        return (
            "direction_plus_predictive_null_width_"
            f"{self.action_mode}_soi_plus_{self.max_jammers}_jammers"
        )

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

    def _normalize_theta(self, theta_rad: float) -> float:
        return (theta_rad - self.theta_min) / (self.theta_max - self.theta_min)

    def _normalize_phi(self, phi_rad: float) -> float:
        return (phi_rad - self.phi_min) / (self.phi_max - self.phi_min)

    def _denormalize_jammer_theta(self, value: float) -> float:
        """Convert normalized jammer theta action to radians."""

        return float(
            self.jammer_theta_min
            + float(value)
            * (self.jammer_theta_max - self.jammer_theta_min)
        )

    def _denormalize_jammer_phi(self, value: float) -> float:
        """Convert normalized jammer phi action to radians."""

        return float(
            self.jammer_phi_min
            + float(value)
            * (self.jammer_phi_max - self.jammer_phi_min)
        )

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

