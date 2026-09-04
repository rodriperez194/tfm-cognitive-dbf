from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tfm.physics.narrow_band.phased_array_nb import Phased_Array_NB
from tfm.physics.narrow_band.weights_stochastic_nb import mvdr_weights
from tfm.math.narrow_band.metrics import compute_sinr
from tfm.math.narrow_band.steering_vector import get_steering_vector
from tfm.math.narrow_band.geometry import (
    angles_to_unit_vector,
)

from tfm.scenario.scenario_generator import ScenarioGenerator
from tfm.targets.aircraft import AircraftTarget
from tfm.targets.drone import DroneTarget
from tfm.targets.dummy import Dummy
from tfm.targets.static import StaticTarget
from tfm.targets.truck import TruckRoadTarget


class BeamformingEnvPhase5(gym.Env):
    """
    Gymnasium environment for Phase 5 cognitive beamforming with direct
    complex-weight control under dynamic jammer scenarios.

    This environment is intended for DRL training.

    Main differences with Phase 4
    -----------------------------
    Phase 4:
        - Static one-step angular scenarios.
        - The agent outputs SOI/jammer directions.
        - The environment builds steering, nulling, or MVDR weights.

    Phase 5:
        - Dynamic multi-step scenarios generated with ScenarioGenerator.
        - The agent outputs complex array weights directly.
        - The weights are held constant for K physical timesteps.
        - One env.step(action) corresponds to one control block of K
          physical substeps.

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

    Complex weight modes
    --------------------
    complex_weight_mode = "real_imag"

        action = [
            Re(w_1), ..., Re(w_E),
            Im(w_1), ..., Im(w_E)
        ]

        Action dimension: 2 * E.

    complex_weight_mode = "mag_phase"

        action = [
            m_1, ..., m_E,
            p_1, ..., p_E
        ]

        where:
            magnitude_n = (m_n + 1) / 2  in [0, 1]
            phase_n = pi * p_n           in [-pi, pi]

        Action dimension: 2 * E.

    complex_weight_mode = "phase_only"

        action = [
            p_1, ..., p_E
        ]

        where:
            phase_n = pi * p_n           in [-pi, pi]
            w_n = exp(j * phase_n)

        Action dimension: E.

    where E = array.N * array.M.

    Reward definition
    -----------------
    At every physical substep, the environment computes:

        reward =
            alpha * normalized_sinr
            - beta * normalized_sinr_loss
            - gamma_soi * normalized_soi_gain_loss
            - gamma_jammer * normalized_jammer_leakage_loss
            + milestone_bonus

    The reward returned by env.step(action) is the mean reward over the
    K physical substeps where the same complex weights are held constant.

    Reference
    ---------
    The SINR-loss reference is always instantaneous MVDR computed from the
    true SOI and true active jammer DOAs at the current physical substep.

    Notes
    -----
    - There is no action_mode.
    - There is no beamforming_mode.
    - There is no reference_mode.
    - The agent always controls complex weights directly.
    - Previous weights and jammer velocities are not included in the state.
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
        complex_weight_mode: str = "phase_only",
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
        reward_alpha_sinr: float = 0.0,
        reward_beta_sinr_loss: float = 1.0,
        reward_gamma_soi_gain_loss: float = 0.0,
        reward_gamma_jammer_leakage: float = 0.0,
        normalize_reward_coefficients: bool = True,
        sinr_scale_db: float = 30.0,
        sinr_loss_scale_db: float = 60.0,
        soi_gain_loss_scale_db: float = 30.0,
        jammer_leakage_scale: float = 1.0,
        reward_bonus_good_soi: float = 0.0,
        reward_bonus_good_jammer: float = 0.0,
        reward_bonus_good_sinr_loss: float = 0.0,
        soi_gain_loss_bonus_threshold_db: float = 1.0,
        jammer_leakage_bonus_threshold: float = 0.01,
        sinr_loss_bonus_threshold_db: float = 1.0,
        max_sinr_loss_db: float = 60.0,
        max_soi_gain_loss_db: float = 60.0,
        max_jammer_leakage_loss: float = 10.0,
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

        self.reward_alpha_sinr_raw = float(reward_alpha_sinr)
        self.reward_beta_sinr_loss_raw = float(reward_beta_sinr_loss)
        self.reward_gamma_soi_gain_loss_raw = float(reward_gamma_soi_gain_loss)
        self.reward_gamma_jammer_leakage_raw = float(reward_gamma_jammer_leakage)
        self.normalize_reward_coefficients = bool(normalize_reward_coefficients)

        (
            self.reward_alpha_sinr,
            self.reward_beta_sinr_loss,
            self.reward_gamma_soi_gain_loss,
            self.reward_gamma_jammer_leakage,
        ) = self._build_effective_reward_coefficients(
            alpha=self.reward_alpha_sinr_raw,
            beta=self.reward_beta_sinr_loss_raw,
            gamma_soi=self.reward_gamma_soi_gain_loss_raw,
            gamma_jammer=self.reward_gamma_jammer_leakage_raw,
        )

        self.sinr_scale_db = float(sinr_scale_db)
        self.sinr_loss_scale_db = float(sinr_loss_scale_db)
        self.soi_gain_loss_scale_db = float(soi_gain_loss_scale_db)
        self.jammer_leakage_scale = float(jammer_leakage_scale)

        self.reward_bonus_good_soi = float(reward_bonus_good_soi)
        self.reward_bonus_good_jammer = float(reward_bonus_good_jammer)
        self.reward_bonus_good_sinr_loss = float(reward_bonus_good_sinr_loss)

        self.soi_gain_loss_bonus_threshold_db = float(
            soi_gain_loss_bonus_threshold_db
        )
        self.jammer_leakage_bonus_threshold = float(
            jammer_leakage_bonus_threshold
        )
        self.sinr_loss_bonus_threshold_db = float(sinr_loss_bonus_threshold_db)

        self.max_sinr_loss_db = float(max_sinr_loss_db)
        self.max_soi_gain_loss_db = float(max_soi_gain_loss_db)
        self.max_jammer_leakage_loss = float(max_jammer_leakage_loss)

        self.mvdr_diagonal_loading = float(mvdr_diagonal_loading)
        self.invalid_sinr_db = float(invalid_sinr_db)
        self.invalid_value_penalty = float(invalid_value_penalty)
        self.max_scenario_sampling_attempts = int(max_scenario_sampling_attempts)

        self.num_elements = int(self.array.N * self.array.M)

        self._validate_configuration()

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

        if self.complex_weight_mode == "real_imag":
            self.action_dim = 2 * self.num_elements

        elif self.complex_weight_mode == "mag_phase":
            self.action_dim = 2 * self.num_elements

        elif self.complex_weight_mode == "phase_only":
            self.action_dim = self.num_elements

        else:
            raise RuntimeError("Invalid complex_weight_mode.")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # ============================================================
        # Episode variables
        # ============================================================

        self.current_scenario: dict | None = None
        self.current_physical_step: int = 0

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
        Reset the environment and generate a new dynamic scenario.
        """

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
            "weight_hold_steps": self.weight_hold_steps,
            "episode_length_physical_steps": self.episode_length_physical_steps,
            "dt": self.dt,
            "scenario_metadata": scenario.get("metadata", {}),
            "reward_alpha_sinr": self.reward_alpha_sinr,
            "reward_beta_sinr_loss": self.reward_beta_sinr_loss,
            "reward_gamma_soi_gain_loss": self.reward_gamma_soi_gain_loss,
            "reward_gamma_jammer_leakage": self.reward_gamma_jammer_leakage,
        }

        return state, info

    def step(self, action: np.ndarray):
        """
        Apply one complex-weight action and hold it for weight_hold_steps
        physical substeps.

        One Gymnasium step corresponds to one control block.
        """

        if self.current_scenario is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized.")

        numerical_error = False
        weights_are_finite = True

        try:
            weights = self._action_to_complex_weights(action)
        except Exception:
            weights = self._build_safe_fallback_weights()
            numerical_error = True

        weights_are_finite = bool(np.all(np.isfinite(weights)))

        if not weights_are_finite:
            weights = self._build_safe_fallback_weights()
            numerical_error = True

        self.array.set_weights(weights)
        fixed_weights = self.array.W.copy()

        remaining_steps = (
            self.episode_length_physical_steps - self.current_physical_step
        )
        num_block_steps = min(self.weight_hold_steps, remaining_steps)

        block_metrics: list[dict] = []

        for block_offset in range(num_block_steps):
            step_idx = self.current_physical_step + block_offset

            self._load_current_directions_from_scenario(step_idx=step_idx)

            instant_metrics = self._evaluate_fixed_weights_at_current_step(
                weights=fixed_weights,
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
            num_block_steps=num_block_steps,
            terminated=terminated,
        )

        return next_state, reward, terminated, truncated, info

    # ============================================================
    # Validation
    # ============================================================

    def _validate_configuration(self) -> None:
        """
        Validate constructor configuration.
        """

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
                "Unknown observation_mode. Expected one of: "
                "'angles', 'unit_vector'."
            )

        if self.complex_weight_mode not in ["real_imag", "mag_phase", "phase_only"]:
            raise ValueError(
                "Unknown complex_weight_mode. Expected one of: "
                "'real_imag', 'mag_phase', 'phase_only'."
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

        if self.sinr_scale_db <= 0.0:
            raise ValueError("sinr_scale_db must be positive.")

        if self.sinr_loss_scale_db <= 0.0:
            raise ValueError("sinr_loss_scale_db must be positive.")

        if self.soi_gain_loss_scale_db <= 0.0:
            raise ValueError("soi_gain_loss_scale_db must be positive.")

        if self.jammer_leakage_scale <= 0.0:
            raise ValueError("jammer_leakage_scale must be positive.")

        if self.max_scenario_sampling_attempts <= 0:
            raise ValueError(
                "max_scenario_sampling_attempts must be a positive integer."
            )

    # ============================================================
    # Scenario generation
    # ============================================================

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

        The environment samples a static desired-source position and a set
        of jammer objects. The generated scenario is accepted if its initial
        angular geometry satisfies the minimum separation constraints.
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
                "Could not sample a valid Phase 5 scenario."
            ) from last_error

        raise RuntimeError("Could not sample a valid Phase 5 scenario.")

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

    # ============================================================
    # State and action
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

    def _action_to_complex_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Convert the selected action representation into complex weights.
        """

        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.complex_weight_mode == "real_imag":
            real_part = action[: self.num_elements].astype(float)
            imag_part = action[self.num_elements :].astype(float)

            weights_flat = real_part + 1j * imag_part

        elif self.complex_weight_mode == "mag_phase":
            raw_magnitude = action[: self.num_elements].astype(float)
            raw_phase = action[self.num_elements :].astype(float)

            magnitude = 0.5 * (raw_magnitude + 1.0)
            phase = np.pi * raw_phase

            weights_flat = magnitude * np.exp(1j * phase)

        elif self.complex_weight_mode == "phase_only":
            raw_phase = action.astype(float)
            phase = np.pi * raw_phase

            weights_flat = np.exp(1j * phase)

        else:
            raise RuntimeError("Invalid complex_weight_mode.")

        weights = weights_flat.reshape(self.array.N, self.array.M)

        return weights.astype(np.complex128)

    # ============================================================
    # Beamforming metrics
    # ============================================================

    def _evaluate_fixed_weights_at_current_step(
        self,
        weights: np.ndarray,
    ) -> dict:
        """
        Evaluate a fixed complex-weight matrix at the current physical step.
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
            soi_gain_metrics = self._compute_soi_gain_metrics(
                weights=weights,
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
            jammer_leakage_metrics = self._build_invalid_jammer_leakage_metrics()
            numerical_error = True

        milestone_bonus, milestone_info = self._compute_milestone_bonus(
            soi_gain_loss_db=soi_gain_metrics["soi_gain_loss_db"],
            jammer_leakage_loss=jammer_leakage_metrics["jammer_leakage_loss"],
            clipped_sinr_loss_db=clipped_sinr_loss_db,
        )

        reward = self._compute_reward(
            sinr_db=sinr_db,
            clipped_sinr_loss_db=clipped_sinr_loss_db,
            soi_gain_loss_db=soi_gain_metrics["clipped_soi_gain_loss_db"],
            jammer_leakage_loss=jammer_leakage_metrics[
                "clipped_jammer_leakage_loss"
            ],
            milestone_bonus=milestone_bonus,
        )

        if not np.isfinite(reward):
            reward = self.invalid_value_penalty
            numerical_error = True

        metrics = {
            "reward": float(reward),
            "sinr_db": float(sinr_db),
            "reference_sinr_db": float(reference_sinr_db),
            "sinr_loss_db": float(sinr_loss_db),
            "clipped_sinr_loss_db": float(clipped_sinr_loss_db),
            "numerical_error": bool(numerical_error),
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
            "milestone_bonus": float(milestone_bonus),
            **milestone_info,
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
        Build conventional steering weights for a given direction.
        """

        theta_deg = float(np.rad2deg(theta_rad))
        phi_deg = float(np.rad2deg(phi_rad))

        weights_flat = get_steering_vector(
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            direction=(theta_deg, phi_deg),
        )

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

    # ============================================================
    # Reward
    # ============================================================

    def _build_effective_reward_coefficients(
        self,
        alpha: float,
        beta: float,
        gamma_soi: float,
        gamma_jammer: float,
    ) -> tuple[float, float, float, float]:
        """
        Build effective reward coefficients.

        If normalize_reward_coefficients=True, non-negative coefficients
        are normalized so that their sum equals one.
        """

        coefficients = np.array(
            [
                float(alpha),
                float(beta),
                float(gamma_soi),
                float(gamma_jammer),
            ],
            dtype=float,
        )

        if np.any(coefficients < 0.0):
            raise ValueError("Reward coefficients must be non-negative.")

        if not self.normalize_reward_coefficients:
            return tuple(float(value) for value in coefficients)

        coefficient_sum = float(np.sum(coefficients))

        if coefficient_sum <= 0.0:
            return tuple(float(value) for value in coefficients)

        normalized_coefficients = coefficients / coefficient_sum

        return tuple(float(value) for value in normalized_coefficients)

    def _compute_reward(
        self,
        sinr_db: float,
        clipped_sinr_loss_db: float,
        soi_gain_loss_db: float,
        jammer_leakage_loss: float,
        milestone_bonus: float = 0.0,
    ) -> float:
        """
        Compute normalized weighted reward robustly.
        """

        reward = 0.0
        invalid_term_detected = False

        if self.reward_alpha_sinr != 0.0:
            if np.isfinite(sinr_db):
                normalized_sinr = float(sinr_db) / self.sinr_scale_db
                reward += self.reward_alpha_sinr * normalized_sinr
            else:
                invalid_term_detected = True

        if self.reward_beta_sinr_loss != 0.0:
            if np.isfinite(clipped_sinr_loss_db):
                normalized_sinr_loss = (
                    float(clipped_sinr_loss_db) / self.sinr_loss_scale_db
                )
                reward -= self.reward_beta_sinr_loss * normalized_sinr_loss
            else:
                invalid_term_detected = True

        if self.reward_gamma_soi_gain_loss != 0.0:
            if np.isfinite(soi_gain_loss_db):
                normalized_soi_loss = (
                    float(soi_gain_loss_db) / self.soi_gain_loss_scale_db
                )
                reward -= self.reward_gamma_soi_gain_loss * normalized_soi_loss
            else:
                invalid_term_detected = True

        if self.reward_gamma_jammer_leakage != 0.0:
            if np.isfinite(jammer_leakage_loss):
                normalized_jammer_loss = (
                    float(jammer_leakage_loss) / self.jammer_leakage_scale
                )
                reward -= self.reward_gamma_jammer_leakage * normalized_jammer_loss
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

    def _compute_milestone_bonus(
        self,
        soi_gain_loss_db: float,
        jammer_leakage_loss: float,
        clipped_sinr_loss_db: float,
    ) -> tuple[float, dict]:
        """
        Compute optional bounded milestone bonuses.
        """

        good_soi = bool(
            np.isfinite(soi_gain_loss_db)
            and float(soi_gain_loss_db) <= self.soi_gain_loss_bonus_threshold_db
        )

        if self.num_active_jammers == 0:
            good_jammer = False
        else:
            good_jammer = bool(
                np.isfinite(jammer_leakage_loss)
                and float(jammer_leakage_loss)
                <= self.jammer_leakage_bonus_threshold
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

    # ============================================================
    # Block aggregation
    # ============================================================

    def _build_block_info(
        self,
        block_metrics: list[dict],
        reward: float,
        numerical_error: bool,
        weights_are_finite: bool,
        fixed_weights: np.ndarray,
        num_block_steps: int,
        terminated: bool,
    ) -> dict:
        """
        Build aggregated info for the control block.
        """

        last_metrics = block_metrics[-1] if len(block_metrics) > 0 else {}

        info = {
            "reward": float(reward),
            "block_reward_mean": float(reward),
            "num_block_steps": int(num_block_steps),
            "weight_hold_steps": self.weight_hold_steps,
            "current_physical_step": int(self.current_physical_step),
            "episode_length_physical_steps": self.episode_length_physical_steps,
            "terminated": bool(terminated),
            "observation_mode": self.observation_mode,
            "complex_weight_mode": self.complex_weight_mode,
            "num_active_jammers": self.num_active_jammers,
            "jammers_powers": self.jammer_powers.copy(),
            "weights": fixed_weights.copy(),
            "numerical_error": bool(numerical_error),
            "weights_are_finite": bool(weights_are_finite),
            "reward_alpha_sinr": self.reward_alpha_sinr,
            "reward_beta_sinr_loss": self.reward_beta_sinr_loss,
            "reward_gamma_soi_gain_loss": self.reward_gamma_soi_gain_loss,
            "reward_gamma_jammer_leakage": self.reward_gamma_jammer_leakage,
            "reward_alpha_sinr_raw": self.reward_alpha_sinr_raw,
            "reward_beta_sinr_loss_raw": self.reward_beta_sinr_loss_raw,
            "reward_gamma_soi_gain_loss_raw": self.reward_gamma_soi_gain_loss_raw,
            "reward_gamma_jammer_leakage_raw": self.reward_gamma_jammer_leakage_raw,
            "normalize_reward_coefficients": self.normalize_reward_coefficients,
            "sinr_scale_db": self.sinr_scale_db,
            "sinr_loss_scale_db": self.sinr_loss_scale_db,
            "soi_gain_loss_scale_db": self.soi_gain_loss_scale_db,
            "jammer_leakage_scale": self.jammer_leakage_scale,
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
            "milestone_bonus",
        ]

        for key in aggregate_keys:
            info[f"{key}_mean"] = self._safe_mean_metric(block_metrics, key)
            info[f"{key}_last"] = self._safe_last_metric(block_metrics, key)

        # Backward-friendly aliases for common training/evaluation code.
        info["sinr_db"] = info["sinr_db_mean"]
        info["reference_sinr_db"] = info["reference_sinr_db_mean"]
        info["sinr_loss_db"] = info["sinr_loss_db_mean"]
        info["clipped_sinr_loss_db"] = info["clipped_sinr_loss_db_mean"]
        info["soi_gain_loss_db"] = info["soi_gain_loss_db_mean"]
        info["jammer_leakage_loss"] = info["jammer_leakage_loss_mean"]

        # Current/last geometry.
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
                    "jammers_directions_deg",
                    [],
                ),
                "jammer_gains_linear": last_metrics.get(
                    "jammer_gains_linear",
                    [],
                ),
                "jammer_gains_db": last_metrics.get("jammer_gains_db", []),
                "jammer_leakage_values": last_metrics.get(
                    "jammer_leakage_values",
                    [],
                ),
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

            if isinstance(value, (list, tuple, dict)):
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

        if isinstance(value, (list, tuple, dict)):
            return float("nan")

        if not np.isfinite(value):
            return float("nan")

        return float(value)

    # ============================================================
    # Geometry helpers
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

    def _compute_angular_error_from_positions_deg(
        self,
        position_a: np.ndarray,
        position_b: np.ndarray,
    ) -> float:
        """
        Compute angular separation between two Cartesian positions as seen
        from the array.
        """

        relative_a = np.asarray(position_a, dtype=float).reshape(3) - self.array_position
        relative_b = np.asarray(position_b, dtype=float).reshape(3) - self.array_position

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

    # ============================================================
    # Jammer powers
    # ============================================================

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

    # ============================================================
    # Normalization helpers
    # ============================================================

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

    # ============================================================
    # Numeric helpers
    # ============================================================

    @staticmethod
    def _linear_to_db(value: float, eps: float = 1e-12) -> float:
        """
        Convert a non-negative linear value to dB safely.
        """

        safe_value = max(float(value), eps)

        return float(10.0 * np.log10(safe_value))