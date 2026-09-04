from __future__ import annotations

import numpy as np
from gymnasium import spaces

from tfm.rl.envs.beamforming_env_phase7 import BeamformingEnvPhase7


class BeamformingEnvPhase8(BeamformingEnvPhase7):
    """
    Sequential incremental complex-weight environment for cognitive beamforming.

    Phase 8 keeps one complete scenario during the whole episode. The desired
    source is static, while the active jammers may move according to their
    target models. Every action defines an absolute real/imaginary residual around the
    fixed normalized steering weights of the episode. Actions are not
    accumulated over previous control steps.

    Initial weights
    ---------------
    At reset, the environment initializes the array with conventional steering
    weights towards the fixed SOI.

    Steering-residual action
    ------------------
    For E array elements, the action has 2E real components:

        action = [Re(delta_1), ..., Re(delta_E),
                  Im(delta_1), ..., Im(delta_E)]

    The proposed weights are:

        w_proposed(t) = (w_steering + residual_complex_scale * delta(t))

    ``Phased_Array_NB.set_weights`` then applies the configured total-power
    normalization. Therefore, the final normalized weights become the initial
    weights for the next control decision.

    Observation
    -----------
    The observation combines:

    1. The original Phase 7 geometry state:
       - 11 components for ``observation_mode='angles'``.
       - 15 components for ``observation_mode='unit_vector'``.
    2. The current persistent weights in normalized real/imaginary form: 2E.
    3. Six feedback values:
       - normalized SINR,
       - normalized SINR loss relative to instantaneous MVDR,
       - normalized SOI gain loss,
       - normalized mean jammer leakage,
       - normalized maximum jammer leakage,
       - normalized episode progress.

    For a 6x6 array and unit-vector geometry, the observation dimension is:

        15 + 72 + 6 = 93.

    Reward
    ------
    The Phase 8 reward is dense and is evaluated at every physical substep:

        reward =
            - beta_sinr_loss * normalized_sinr_loss
            - gamma_soi_loss * normalized_soi_gain_loss
            - gamma_jammer * normalized_jammer_leakage
            - gamma_action * normalized_increment_size
            + gamma_hold * exp(-sinr_loss_db / hold_scale_db)
            + gamma_improvement * normalized_sinr_loss_improvement
            + gamma_teacher * teacher_similarity

    The main objective is not only to reach a good pattern, but to maintain a
    low SINR loss while the jammers evolve. The improvement and teacher terms
    are optional and disabled by default.

    Notes
    -----
    - The SOI is required to remain fixed for the complete scenario.
    - Only ``complex_weight_mode='real_imag'`` is supported.
    - One Gymnasium step corresponds to one control block. The updated weights
      are held for ``weight_hold_steps`` physical samples.
    """

    metadata = {"render_modes": []}

    _NUM_FEEDBACK_FEATURES = 6

    def __init__(
        self,
        array,
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
        episode_length_physical_steps: int = 50,
        dt: float = 0.1,
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
        residual_complex_scale: float = 0.25,
        residual_weight_min_power: float = 1e-12,
        reward_beta_sinr_loss: float = 1.0,
        reward_gamma_soi_gain_loss: float = 0.25,
        reward_gamma_jammer_leakage: float = 0.50,
        reward_gamma_action: float = 0.01,
        reward_gamma_hold: float = 0.50,
        reward_gamma_improvement: float = 0.0,
        reward_gamma_teacher_similarity: float = 0.0,
        reward_sinr_loss_scale_db: float = 30.0,
        reward_sinr_loss_clip: float = 2.0,
        reward_soi_gain_loss_scale_db: float = 10.0,
        reward_soi_gain_loss_clip: float = 3.0,
        reward_jammer_leakage_scale: float = 0.05,
        reward_jammer_leakage_clip: float = 5.0,
        reward_action_scale: float = 1.0,
        reward_action_clip: float = 1.0,
        reward_hold_scale_db: float = 3.0,
        reward_improvement_scale_db: float = 10.0,
        reward_improvement_clip: float = 2.0,
        observation_sinr_scale_db: float = 30.0,
        observation_jammer_leakage_scale: float = 0.10,
        teacher_diagonal_loading: float = 1e-8,
        teacher_use_pinv: bool = False,
        teacher_similarity_epsilon: float = 1e-12,
        max_sinr_loss_db: float = 60.0,
        max_soi_gain_loss_db: float = 60.0,
        max_jammer_leakage_loss: float = 30.0,
        mvdr_diagonal_loading: float = 1e-4,
        invalid_sinr_db: float = -120.0,
        invalid_value_penalty: float = -1_000.0,
        max_scenario_sampling_attempts: int = 200,
    ) -> None:
        # Phase 7 is used only as the common implementation of scenario
        # generation, array metrics, teacher weights and geometry helpers.
        # Its threshold reward is not used by Phase 8.
        super().__init__(
            array=array,
            array_position=array_position,
            desired_power=desired_power,
            noise_power=noise_power,
            max_jammers=max_jammers,
            num_active_jammers=num_active_jammers,
            active_jammers_choices=active_jammers_choices,
            jammer_powers=jammer_powers,
            observation_mode=observation_mode,
            complex_weight_mode=complex_weight_mode,
            weight_hold_steps=weight_hold_steps,
            episode_length_physical_steps=episode_length_physical_steps,
            dt=dt,
            target_position_x_limits_m=target_position_x_limits_m,
            target_position_y_limits_m=target_position_y_limits_m,
            target_position_z_limits_m=target_position_z_limits_m,
            jammer_target_types=jammer_target_types,
            theta_limits_rad=theta_limits_rad,
            phi_limits_rad=phi_limits_rad,
            jammer_theta_limits_rad=jammer_theta_limits_rad,
            jammer_phi_limits_rad=jammer_phi_limits_rad,
            min_source_distance_m=min_source_distance_m,
            min_target_jammer_separation_deg=min_target_jammer_separation_deg,
            enforce_visible_hemisphere=enforce_visible_hemisphere,
            reward_failure_penalty=-1.0,
            reward_soi_max_gain_loss_db=max_soi_gain_loss_db,
            reward_jammer_max_mean_leakage=max_jammer_leakage_loss,
            reward_sinr_scale_db=max(observation_sinr_scale_db, 1e-12),
            reward_valid_min=-10.0,
            reward_valid_max=10.0,
            reward_teacher_similarity_weight=0.0,
            reward_jammer_leakage_penalty_weight=0.0,
            reward_jammer_leakage_penalty_scale=max(
                observation_jammer_leakage_scale,
                1e-12,
            ),
            reward_jammer_leakage_penalty_clip=0.0,
            reward_sinr_loss_bonus_steps=None,
            reward_soi_gain_loss_bonus_steps=None,
            reward_jammer_leakage_bonus_steps=None,
            reward_teacher_similarity_bonus_steps=None,
            teacher_diagonal_loading=teacher_diagonal_loading,
            teacher_use_pinv=teacher_use_pinv,
            teacher_similarity_epsilon=teacher_similarity_epsilon,
            direct_weight_min_power=residual_weight_min_power,
            max_sinr_loss_db=max_sinr_loss_db,
            max_soi_gain_loss_db=max_soi_gain_loss_db,
            max_jammer_leakage_loss=max_jammer_leakage_loss,
            mvdr_diagonal_loading=mvdr_diagonal_loading,
            invalid_sinr_db=invalid_sinr_db,
            invalid_value_penalty=invalid_value_penalty,
            max_scenario_sampling_attempts=max_scenario_sampling_attempts,
        )

        self.residual_complex_scale = float(residual_complex_scale)
        self.residual_weight_min_power = float(residual_weight_min_power)

        self.reward_beta_sinr_loss = float(reward_beta_sinr_loss)
        self.reward_gamma_soi_gain_loss = float(reward_gamma_soi_gain_loss)
        self.reward_gamma_jammer_leakage = float(
            reward_gamma_jammer_leakage
        )
        self.reward_gamma_action = float(reward_gamma_action)
        self.reward_gamma_hold = float(reward_gamma_hold)
        self.reward_gamma_improvement = float(reward_gamma_improvement)
        self.reward_gamma_teacher_similarity = float(
            reward_gamma_teacher_similarity
        )

        self.reward_sinr_loss_scale_db = float(reward_sinr_loss_scale_db)
        self.reward_sinr_loss_clip = float(reward_sinr_loss_clip)
        self.reward_soi_gain_loss_scale_db = float(
            reward_soi_gain_loss_scale_db
        )
        self.reward_soi_gain_loss_clip = float(reward_soi_gain_loss_clip)
        self.reward_jammer_leakage_scale = float(
            reward_jammer_leakage_scale
        )
        self.reward_jammer_leakage_clip = float(
            reward_jammer_leakage_clip
        )
        self.reward_action_scale = float(reward_action_scale)
        self.reward_action_clip = float(reward_action_clip)
        self.reward_hold_scale_db = float(reward_hold_scale_db)
        self.reward_improvement_scale_db = float(
            reward_improvement_scale_db
        )
        self.reward_improvement_clip = float(reward_improvement_clip)

        self.observation_sinr_scale_db = float(observation_sinr_scale_db)
        self.observation_jammer_leakage_scale = float(
            observation_jammer_leakage_scale
        )

        self._validate_phase8_configuration()

        # The inherited action space already has shape (2E,), but it is
        # redefined explicitly to document the Phase 8 interpretation.
        self.action_dim = 2 * self.num_elements
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.geometry_observation_dim = int(self.observation_dim)
        self.weight_observation_dim = 2 * self.num_elements
        self.feedback_observation_dim = self._NUM_FEEDBACK_FEATURES
        self.observation_dim = (
            self.geometry_observation_dim
            + self.weight_observation_dim
            + self.feedback_observation_dim
        )

        # A single symmetric space is used because geometry, normalized
        # weights and feedback values all lie inside [-1, 1].
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        self.current_weights: np.ndarray | None = None
        self.current_feedback_metrics: dict | None = None

        # Fixed normalized steering reference for the complete episode.
        self.steering_reference_weights: np.ndarray | None = None

        self.last_residual_action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

    # ============================================================
    # Gymnasium API
    # ============================================================
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Generate one scenario and initialize the fixed steering reference."""
    
        _, base_info = super().reset(seed=seed, options=options)
    
        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("SOI direction was not initialized correctly.")
    
        steering_weights = self._build_steering_weights(
            theta_rad=self.current_theta_rad,
            phi_rad=self.current_phi_rad,
        )
        self.array.set_weights(steering_weights)
    
        # Store the normalized steering solution as an immutable episode
        # reference. Every subsequent action is built from these weights.
        self.steering_reference_weights = np.asarray(
            self.array.W,
            dtype=np.complex128,
        ).copy()
    
        self.current_weights = self.steering_reference_weights.copy()
        self.last_final_weights = self.current_weights.copy()
        self.last_residual_action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )
    
        feedback_metrics = self._compute_feedback_metrics(
            weights=self.current_weights,
        )
        self.current_feedback_metrics = feedback_metrics
    
        state = self._build_phase8_observation(
            feedback_metrics=feedback_metrics,
        )
        self.current_state = state
    
        info = {
            key: value
            for key, value in base_info.items()
            if not (
                key.startswith("reward_")
                or key.startswith("teacher_")
                or key == "direct_weight_min_power"
            )
        }
    
        info.update(
            {
                "phase": 8,
                "action_type": self._get_action_type(),
                "weight_update_mode": "steering_residual_real_imag",
                "soi_is_static": True,
                "geometry_observation_dim": self.geometry_observation_dim,
                "weight_observation_dim": self.weight_observation_dim,
                "feedback_observation_dim": self.feedback_observation_dim,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "residual_complex_scale": self.residual_complex_scale,
                "residual_weight_min_power": self.residual_weight_min_power,
                "weights": self.current_weights.copy(),
                "initial_weights": self.current_weights.copy(),
                "steering_reference_weights": (
                    self.steering_reference_weights.copy()
                ),
                **self._build_phase8_reward_configuration_info(),
                **feedback_metrics,
            }
        )
    
        return state, info

    def step(self, action: np.ndarray):
        """Apply one absolute complex residual around fixed steering weights."""

        if self.current_scenario is None:
            raise RuntimeError("Environment must be reset before step().")

        if self.current_weights is None:
            raise RuntimeError("Current weights are not initialized.")

        if self.steering_reference_weights is None:
            raise RuntimeError(
                "Steering reference weights are not initialized."
            )

        if self.current_feedback_metrics is None:
            raise RuntimeError("Feedback metrics are not initialized.")

        previous_weights = self.current_weights.copy()
        previous_feedback = dict(self.current_feedback_metrics)

        numerical_error = False
        invalid_residual_action = False

        try:
            proposed_weights, action_info = self._action_to_residual_weights(
                action=action,
                previous_weights=previous_weights,
            )

            invalid_residual_action = bool(
                action_info["invalid_residual_weight_action"]
            )

        except Exception:
            proposed_weights = self._build_safe_fallback_weights()

            action_info = self._build_invalid_residual_action_info(
                previous_weights=previous_weights,
            )

            invalid_residual_action = True
            numerical_error = True

        if not np.all(np.isfinite(proposed_weights)):
            proposed_weights = self._build_safe_fallback_weights()

            action_info = self._build_invalid_residual_action_info(
                previous_weights=previous_weights,
            )

            invalid_residual_action = True
            numerical_error = True

        self.array.set_weights(proposed_weights)

        updated_weights = np.asarray(
            self.array.W,
            dtype=np.complex128,
        ).copy()

        weights_are_finite = bool(
            np.all(np.isfinite(updated_weights))
        )

        if not weights_are_finite:
            self.array.set_weights(
                self._build_safe_fallback_weights()
            )

            updated_weights = np.asarray(
                self.array.W,
                dtype=np.complex128,
            ).copy()

            action_info = self._build_invalid_residual_action_info(
                previous_weights=previous_weights,
            )

            invalid_residual_action = True
            numerical_error = True

        action_info = self._finalize_residual_action_info(
            action_info=action_info,
            previous_weights=previous_weights,
            normalized_weights=updated_weights,
            invalid_residual_action=invalid_residual_action,
        )

        self.current_weights = updated_weights.copy()
        self.last_final_weights = updated_weights.copy()
        self.last_residual_action = action_info["raw_action"].copy()

        remaining_steps = (
            self.episode_length_physical_steps
            - self.current_physical_step
        )

        num_block_steps = min(
            self.weight_hold_steps,
            remaining_steps,
        )

        if num_block_steps <= 0:
            raise RuntimeError(
                "No physical steps remain. Reset the environment before "
                "calling step() again."
            )

        block_metrics: list[dict] = []

        for block_offset in range(num_block_steps):
            step_idx = self.current_physical_step + block_offset

            self._load_current_directions_from_scenario(
                step_idx=step_idx
            )

            instant_metrics = (
                self._evaluate_residual_weights_at_current_step(
                    weights=updated_weights,
                    action_info=action_info,
                    previous_sinr_loss_db=float(
                        previous_feedback["sinr_loss_db"]
                    ),
                    invalid_residual_action=invalid_residual_action,
                )
            )

            if bool(instant_metrics["numerical_error"]):
                numerical_error = True

            block_metrics.append(instant_metrics)

        reward = self._safe_mean_metric(
            block_metrics,
            "reward",
        )

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

        self._load_current_directions_from_scenario(
            step_idx=next_step_idx
        )

        next_feedback = self._compute_feedback_metrics(
            weights=self.current_weights,
        )
        self.current_feedback_metrics = next_feedback

        next_state = self._build_phase8_observation(
            feedback_metrics=next_feedback,
        )
        self.current_state = next_state

        info = self._build_phase8_block_info(
            block_metrics=block_metrics,
            reward=reward,
            numerical_error=numerical_error,
            weights_are_finite=weights_are_finite,
            action_info=action_info,
            num_block_steps=num_block_steps,
            terminated=terminated,
            next_feedback=next_feedback,
        )

        return next_state, reward, terminated, truncated, info

    # ============================================================
    # Scenario constraints
    # ============================================================

    def _scenario_has_valid_initial_geometry(self, scenario: dict) -> bool:
        """Validate geometry and explicitly require a static SOI."""

        if not super()._scenario_has_valid_initial_geometry(scenario):
            return False

        desired_positions = np.asarray(
            scenario["desired"]["position"],
            dtype=float,
        )
        desired_theta = np.asarray(
            scenario["desired"]["doa"]["theta"],
            dtype=float,
        )
        desired_phi = np.asarray(
            scenario["desired"]["doa"]["phi"],
            dtype=float,
        )

        if desired_positions.ndim != 2 or desired_positions.shape[1] != 3:
            return False

        if not np.allclose(
            desired_positions,
            desired_positions[0],
            rtol=0.0,
            atol=1e-10,
        ):
            return False

        if not np.allclose(
            desired_theta,
            desired_theta[0],
            rtol=0.0,
            atol=1e-12,
        ):
            return False

        if not np.allclose(
            desired_phi,
            desired_phi[0],
            rtol=0.0,
            atol=1e-12,
        ):
            return False

        return True

    # ============================================================
    # Observation
    # ============================================================

    def _build_phase8_observation(
        self,
        feedback_metrics: dict,
    ) -> np.ndarray:
        """Build geometry, persistent-weight and feedback observation."""

        if self.current_weights is None:
            raise RuntimeError("Current weights are required for observation.")

        geometry_state = super()._build_state(
            theta_target_rad=self.current_theta_rad,
            phi_target_rad=self.current_phi_rad,
            jammer_thetas_rad=self.current_jammer_thetas_rad,
            jammer_phis_rad=self.current_jammer_phis_rad,
        ).astype(np.float32)

        weights_flat = np.asarray(
            self.current_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        # With array power normalization, ||w||_2 = sqrt(E). Dividing by
        # sqrt(E) guarantees every real and imaginary component lies in [-1, 1].
        weight_scale = max(np.sqrt(float(self.num_elements)), 1e-12)
        normalized_weights = weights_flat / weight_scale

        weight_state = np.concatenate(
            [
                np.real(normalized_weights),
                np.imag(normalized_weights),
            ]
        ).astype(np.float32)

        sinr_norm = float(
            np.clip(
                float(feedback_metrics["sinr_db"])
                / self.observation_sinr_scale_db,
                -1.0,
                1.0,
            )
        )
        sinr_loss_norm = float(
            np.clip(
                float(feedback_metrics["sinr_loss_db"])
                / self.max_sinr_loss_db,
                0.0,
                1.0,
            )
        )
        soi_loss_norm = float(
            np.clip(
                float(feedback_metrics["soi_gain_loss_db"])
                / self.max_soi_gain_loss_db,
                0.0,
                1.0,
            )
        )
        jammer_mean_norm = float(
            np.clip(
                float(feedback_metrics["jammer_leakage_loss"])
                / self.observation_jammer_leakage_scale,
                0.0,
                1.0,
            )
        )
        jammer_max_norm = float(
            np.clip(
                float(feedback_metrics["jammer_leakage_max"])
                / self.observation_jammer_leakage_scale,
                0.0,
                1.0,
            )
        )

        progress_denominator = max(
            self.episode_length_physical_steps - 1,
            1,
        )
        progress = float(
            np.clip(
                self.current_physical_step / progress_denominator,
                0.0,
                1.0,
            )
        )

        feedback_state = np.array(
            [
                sinr_norm,
                sinr_loss_norm,
                soi_loss_norm,
                jammer_mean_norm,
                jammer_max_norm,
                progress,
            ],
            dtype=np.float32,
        )

        state = np.concatenate(
            [geometry_state, weight_state, feedback_state]
        ).astype(np.float32)

        if state.shape != (self.observation_dim,):
            raise RuntimeError(
                "Invalid Phase 8 observation shape: "
                f"expected {(self.observation_dim,)}, got {state.shape}."
            )

        if not np.all(np.isfinite(state)):
            raise RuntimeError("Phase 8 observation contains invalid values.")

        return np.clip(state, -1.0, 1.0).astype(np.float32)

    # ============================================================
    # Steering-residual action
    # ============================================================

    def _action_to_residual_weights(
        self,
        action: np.ndarray,
        previous_weights: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Convert a 2E action into an absolute residual around steering."""

        if self.steering_reference_weights is None:
            raise RuntimeError(
                "Steering reference weights are not initialized."
            )

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(self.action_dim)

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        real_action = action[: self.num_elements].astype(float)
        imag_action = action[self.num_elements :].astype(float)

        normalized_residual = real_action + 1j * imag_action

        complex_residual = (
            self.residual_complex_scale
            * normalized_residual
        )

        steering_flat = np.asarray(
            self.steering_reference_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        previous_flat = np.asarray(
            previous_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        # Critical Phase 8 residual equation:
        # the previous weights are NOT used to construct the proposal.
        proposed_flat = steering_flat + complex_residual

        proposed_power = float(
            np.sum(np.abs(proposed_flat) ** 2)
        )

        invalid_residual_action = bool(
            not np.isfinite(proposed_power)
            or proposed_power <= self.residual_weight_min_power
            or not np.all(np.isfinite(proposed_flat))
        )

        if invalid_residual_action:
            proposed_flat = steering_flat.copy()

            proposed_power = float(
                np.sum(np.abs(proposed_flat) ** 2)
            )

        normalized_action_energy = float(
            np.mean(
                np.square(real_action)
                + np.square(imag_action)
            )
        )

        residual_abs_mean = float(
            np.mean(np.abs(complex_residual))
        )
        residual_abs_max = float(
            np.max(np.abs(complex_residual))
        )
        residual_rms = float(
            np.sqrt(
                np.mean(np.abs(complex_residual) ** 2)
            )
        )

        action_info = {
            "raw_action": action.copy(),

            "residual_real_action": real_action.copy(),
            "residual_imag_action": imag_action.copy(),
            "normalized_complex_residual": (
                normalized_residual.copy()
            ),
            "complex_residual": complex_residual.copy(),

            "residual_complex_scale": (
                self.residual_complex_scale
            ),
            "residual_abs_mean": residual_abs_mean,
            "residual_abs_max": residual_abs_max,
            "residual_rms": residual_rms,

            "normalized_action_energy": (
                normalized_action_energy
            ),

            "previous_weights": previous_flat.reshape(
                self.array.N,
                self.array.M,
            ).copy(),

            "steering_reference_weights": (
                steering_flat.reshape(
                    self.array.N,
                    self.array.M,
                ).copy()
            ),

            "proposed_weights_before_normalization": (
                proposed_flat.reshape(
                    self.array.N,
                    self.array.M,
                ).copy()
            ),

            "proposed_weight_power_before_normalization": (
                proposed_power
            ),

            "invalid_residual_weight_action": (
                invalid_residual_action
            ),

            # Backward-compatible aliases.
            "incremental_real_action": real_action.copy(),
            "incremental_imag_action": imag_action.copy(),
            "normalized_complex_increment": (
                normalized_residual.copy()
            ),
            "complex_increment": complex_residual.copy(),
            "increment_abs_mean": residual_abs_mean,
            "increment_abs_max": residual_abs_max,
            "increment_rms": residual_rms,
            "invalid_incremental_weight_action": (
                invalid_residual_action
            ),
        }

        return (
            proposed_flat.reshape(
                self.array.N,
                self.array.M,
            ),
            action_info,
        )


    def _build_invalid_residual_action_info(
        self,
        previous_weights: np.ndarray,
    ) -> dict:
        """Build diagnostics for an invalid steering-residual action."""

        if self.steering_reference_weights is None:
            raise RuntimeError(
                "Steering reference weights are not initialized."
            )

        previous_weights = np.asarray(
            previous_weights,
            dtype=np.complex128,
        ).reshape(
            self.array.N,
            self.array.M,
        )

        steering_weights = np.asarray(
            self.steering_reference_weights,
            dtype=np.complex128,
        ).reshape(
            self.array.N,
            self.array.M,
        )

        zeros = np.zeros(
            self.num_elements,
            dtype=float,
        )

        complex_zeros = np.zeros(
            self.num_elements,
            dtype=np.complex128,
        )

        return {
            "raw_action": np.zeros(
                self.action_dim,
                dtype=np.float32,
            ),

            "residual_real_action": zeros.copy(),
            "residual_imag_action": zeros.copy(),
            "normalized_complex_residual": complex_zeros.copy(),
            "complex_residual": complex_zeros.copy(),

            "residual_complex_scale": (
                self.residual_complex_scale
            ),

            "residual_abs_mean": 0.0,
            "residual_abs_max": 0.0,
            "residual_rms": 0.0,

            "normalized_action_energy": 0.0,

            "previous_weights": previous_weights.copy(),

            "steering_reference_weights": (
                steering_weights.copy()
            ),

            "proposed_weights_before_normalization": (
                steering_weights.copy()
            ),

            "proposed_weight_power_before_normalization": float(
                np.sum(np.abs(steering_weights) ** 2)
            ),

            "invalid_residual_weight_action": True,

            # Backward-compatible aliases.
            "incremental_real_action": zeros.copy(),
            "incremental_imag_action": zeros.copy(),
            "normalized_complex_increment": complex_zeros.copy(),
            "complex_increment": complex_zeros.copy(),
            "increment_abs_mean": 0.0,
            "increment_abs_max": 0.0,
            "increment_rms": 0.0,
            "invalid_incremental_weight_action": True,
        }


    def _finalize_residual_action_info(
        self,
        action_info: dict,
        previous_weights: np.ndarray,
        normalized_weights: np.ndarray,
        invalid_residual_action: bool,
    ) -> dict:
        """Add normalized residual and inter-step diagnostics."""

        if self.steering_reference_weights is None:
            raise RuntimeError(
                "Steering reference weights are not initialized."
            )

        steering_flat = np.asarray(
            self.steering_reference_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        previous_flat = np.asarray(
            previous_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        final_flat = np.asarray(
            normalized_weights,
            dtype=np.complex128,
        ).reshape(self.num_elements)

        # Absolute residual relative to the fixed steering solution.
        realized_residual = final_flat - steering_flat

        # Actual change relative to the previous control step.
        realized_step_change = final_flat - previous_flat

        def complex_similarity(
            first: np.ndarray,
            second: np.ndarray,
        ) -> float:
            denominator = float(
                np.linalg.norm(first)
                * np.linalg.norm(second)
            )

            if (
                denominator <= 1e-12
                or not np.isfinite(denominator)
            ):
                return 0.0

            return float(
                np.clip(
                    np.abs(np.vdot(first, second))
                    / denominator,
                    0.0,
                    1.0,
                )
            )

        steering_similarity = complex_similarity(
            steering_flat,
            final_flat,
        )

        previous_similarity = complex_similarity(
            previous_flat,
            final_flat,
        )

        realized_residual_abs_mean = float(
            np.mean(np.abs(realized_residual))
        )
        realized_residual_abs_max = float(
            np.max(np.abs(realized_residual))
        )
        realized_residual_rms = float(
            np.sqrt(
                np.mean(np.abs(realized_residual) ** 2)
            )
        )

        result = dict(action_info)

        result.update(
            {
                "weights": final_flat.reshape(
                    self.array.N,
                    self.array.M,
                ).copy(),

                "final_magnitude": np.abs(final_flat).copy(),
                "final_phase_rad": np.angle(final_flat).copy(),
                "final_phase_norm": (
                    np.angle(final_flat).copy() / np.pi
                ),

                "final_weight_power": float(
                    np.sum(np.abs(final_flat) ** 2)
                ),

                "realized_residual": (
                    realized_residual.copy()
                ),
                "realized_residual_abs_mean": (
                    realized_residual_abs_mean
                ),
                "realized_residual_abs_max": (
                    realized_residual_abs_max
                ),
                "realized_residual_rms": (
                    realized_residual_rms
                ),

                "realized_step_change": (
                    realized_step_change.copy()
                ),
                "realized_step_change_abs_mean": float(
                    np.mean(np.abs(realized_step_change))
                ),
                "realized_step_change_abs_max": float(
                    np.max(np.abs(realized_step_change))
                ),
                "realized_step_change_rms": float(
                    np.sqrt(
                        np.mean(
                            np.abs(realized_step_change) ** 2
                        )
                    )
                ),

                "steering_final_weight_similarity": (
                    steering_similarity
                ),
                "previous_final_weight_similarity": (
                    previous_similarity
                ),

                "invalid_residual_weight_action": bool(
                    invalid_residual_action
                ),

                # Backward-compatible aliases.
                "realized_increment": (
                    realized_residual.copy()
                ),
                "realized_increment_abs_mean": (
                    realized_residual_abs_mean
                ),
                "realized_increment_abs_max": (
                    realized_residual_abs_max
                ),
                "realized_increment_rms": (
                    realized_residual_rms
                ),
                "invalid_incremental_weight_action": bool(
                    invalid_residual_action
                ),
            }
        )

        return result

    def _build_safe_fallback_weights(self) -> np.ndarray:
        """Return the fixed steering reference as the safe fallback."""
    
        if self.steering_reference_weights is not None:
            return self.steering_reference_weights.copy()
    
        return super()._build_safe_fallback_weights()

    # ============================================================
    # Metrics and reward
    # ============================================================

    def _compute_feedback_metrics(self, weights: np.ndarray) -> dict:
        """Compute current physical metrics without changing the weights."""

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

        sinr_loss_db = float(reference_sinr_db - sinr_db)

        if not np.isfinite(sinr_loss_db):
            sinr_loss_db = self.max_sinr_loss_db
            numerical_error = True

        sinr_loss_db = max(0.0, sinr_loss_db)
        clipped_sinr_loss_db = min(
            sinr_loss_db,
            self.max_sinr_loss_db,
        )

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
            jammer_leakage_metrics = (
                self._build_invalid_jammer_leakage_metrics()
            )
            numerical_error = True

        return {
            "sinr_db": float(sinr_db),
            "reference_sinr_db": float(reference_sinr_db),
            "sinr_loss_db": float(sinr_loss_db),
            "clipped_sinr_loss_db": float(clipped_sinr_loss_db),
            "numerical_error": bool(numerical_error),
            **soi_gain_metrics,
            **jammer_leakage_metrics,
        }

    def _evaluate_residual_weights_at_current_step(
        self,
        weights: np.ndarray,
        action_info: dict,
        previous_sinr_loss_db: float,
        invalid_residual_action: bool,
    ) -> dict:
        """Evaluate one steering-residual action at the current substep."""

        feedback_metrics = self._compute_feedback_metrics(
            weights=weights
        )

        numerical_error = bool(
            feedback_metrics["numerical_error"]
        )

        teacher_weights = np.zeros(
            (self.array.N, self.array.M),
            dtype=np.complex128,
        )

        teacher_metrics = self._build_invalid_teacher_metrics()

        try:
            teacher_weights = self._build_teacher_weights()

            teacher_metrics = (
                self._compute_teacher_weight_similarity(
                    agent_weights=weights,
                    teacher_weights=teacher_weights,
                )
            )

        except Exception:
            if self.reward_gamma_teacher_similarity != 0.0:
                numerical_error = True

        reward, reward_info = self._compute_phase8_reward(
            sinr_loss_db=float(
                feedback_metrics["sinr_loss_db"]
            ),
            soi_gain_loss_db=float(
                feedback_metrics["soi_gain_loss_db"]
            ),
            jammer_mean_leakage=float(
                feedback_metrics["jammer_leakage_loss"]
            ),
            normalized_action_energy=float(
                action_info["normalized_action_energy"]
            ),
            previous_sinr_loss_db=float(
                previous_sinr_loss_db
            ),
            teacher_weight_similarity=float(
                teacher_metrics["teacher_weight_similarity"]
            ),
            invalid_residual_action=invalid_residual_action,
            numerical_error=numerical_error,
        )

        return {
            "reward": float(reward),
            "numerical_error": bool(numerical_error),

            "invalid_residual_weight_action": bool(
                invalid_residual_action
            ),

            # Backward-compatible aliases.
            "invalid_incremental_weight_action": bool(
                invalid_residual_action
            ),
            "invalid_direct_weight_action": bool(
                invalid_residual_action
            ),

            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,

            "theta_target_deg": float(
                np.rad2deg(self.current_theta_rad)
            ),
            "phi_target_deg": float(
                np.rad2deg(self.current_phi_rad)
            ),

            "jammer_thetas_rad": (
                self.current_jammer_thetas_rad.copy()
            ),
            "jammer_phis_rad": (
                self.current_jammer_phis_rad.copy()
            ),

            "jammer_thetas_deg": [
                float(np.rad2deg(theta))
                for theta in self.current_jammer_thetas_rad
            ],

            "jammer_phis_deg": [
                float(np.rad2deg(phi))
                for phi in self.current_jammer_phis_rad
            ],

            "jammers_directions_deg": (
                self._get_current_jammer_directions_deg()
            ),

            "num_active_jammers": self.num_active_jammers,
            "jammers_powers": self.jammer_powers.copy(),
            "teacher_weights": teacher_weights.copy(),

            "residual_abs_mean": float(
                action_info["residual_abs_mean"]
            ),
            "residual_abs_max": float(
                action_info["residual_abs_max"]
            ),
            "residual_rms": float(
                action_info["residual_rms"]
            ),

            "realized_residual_abs_mean": float(
                action_info["realized_residual_abs_mean"]
            ),
            "realized_residual_abs_max": float(
                action_info["realized_residual_abs_max"]
            ),
            "realized_residual_rms": float(
                action_info["realized_residual_rms"]
            ),

            "realized_step_change_abs_mean": float(
                action_info["realized_step_change_abs_mean"]
            ),
            "realized_step_change_abs_max": float(
                action_info["realized_step_change_abs_max"]
            ),
            "realized_step_change_rms": float(
                action_info["realized_step_change_rms"]
            ),

            "steering_final_weight_similarity": float(
                action_info[
                    "steering_final_weight_similarity"
                ]
            ),

            "previous_final_weight_similarity": float(
                action_info[
                    "previous_final_weight_similarity"
                ]
            ),

            "normalized_action_energy": float(
                action_info["normalized_action_energy"]
            ),

            # Backward-compatible metric aliases.
            "increment_abs_mean": float(
                action_info["residual_abs_mean"]
            ),
            "increment_abs_max": float(
                action_info["residual_abs_max"]
            ),
            "increment_rms": float(
                action_info["residual_rms"]
            ),
            "realized_increment_abs_mean": float(
                action_info["realized_residual_abs_mean"]
            ),
            "realized_increment_abs_max": float(
                action_info["realized_residual_abs_max"]
            ),
            "realized_increment_rms": float(
                action_info["realized_residual_rms"]
            ),

            **feedback_metrics,
            **teacher_metrics,
            **reward_info,
        }    
    
    def _compute_phase8_reward(
        self,
        sinr_loss_db: float,
        soi_gain_loss_db: float,
        jammer_mean_leakage: float,
        normalized_action_energy: float,
        previous_sinr_loss_db: float,
        teacher_weight_similarity: float,
        invalid_residual_action: bool,
        numerical_error: bool,
    ) -> tuple[float, dict]:
        """Compute the dense sequential Phase 8 reward."""

        if invalid_residual_action:
            return float(self.invalid_value_penalty), {
                **self._build_zero_reward_components(),
                "reward_failure_applied": True,
                "reward_failure_reason": "invalid_residual_action",
            }

        if numerical_error:
            return float(self.invalid_value_penalty), {
                **self._build_zero_reward_components(),
                "reward_failure_applied": True,
                "reward_failure_reason": "numerical_error",
            }

        normalized_sinr_loss = float(
            np.clip(
                sinr_loss_db / self.reward_sinr_loss_scale_db,
                0.0,
                self.reward_sinr_loss_clip,
            )
        )
        normalized_soi_gain_loss = float(
            np.clip(
                soi_gain_loss_db / self.reward_soi_gain_loss_scale_db,
                0.0,
                self.reward_soi_gain_loss_clip,
            )
        )

        if self.num_active_jammers == 0:
            normalized_jammer_leakage = 0.0
        else:
            normalized_jammer_leakage = float(
                np.clip(
                    jammer_mean_leakage
                    / self.reward_jammer_leakage_scale,
                    0.0,
                    self.reward_jammer_leakage_clip,
                )
            )

        normalized_action_cost = float(
            np.clip(
                normalized_action_energy / self.reward_action_scale,
                0.0,
                self.reward_action_clip,
            )
        )

        hold_score = float(
            np.exp(
                -max(0.0, sinr_loss_db)
                / self.reward_hold_scale_db
            )
        )

        sinr_loss_improvement_db = float(
            previous_sinr_loss_db - sinr_loss_db
        )
        normalized_improvement = float(
            np.clip(
                sinr_loss_improvement_db
                / self.reward_improvement_scale_db,
                -self.reward_improvement_clip,
                self.reward_improvement_clip,
            )
        )

        teacher_similarity = float(
            np.clip(teacher_weight_similarity, 0.0, 1.0)
        )

        sinr_loss_component = float(
            -self.reward_beta_sinr_loss * normalized_sinr_loss
        )
        soi_gain_loss_component = float(
            -self.reward_gamma_soi_gain_loss
            * normalized_soi_gain_loss
        )
        jammer_leakage_component = float(
            -self.reward_gamma_jammer_leakage
            * normalized_jammer_leakage
        )
        action_component = float(
            -self.reward_gamma_action * normalized_action_cost
        )
        hold_component = float(
            self.reward_gamma_hold * hold_score
        )
        improvement_component = float(
            self.reward_gamma_improvement * normalized_improvement
        )
        teacher_component = float(
            self.reward_gamma_teacher_similarity * teacher_similarity
        )

        reward = float(
            sinr_loss_component
            + soi_gain_loss_component
            + jammer_leakage_component
            + action_component
            + hold_component
            + improvement_component
            + teacher_component
        )

        if not np.isfinite(reward):
            reward = self.invalid_value_penalty

        info = {
            "reward_failure_applied": False,
            "reward_failure_reason": "none",
            "reward_normalized_sinr_loss": normalized_sinr_loss,
            "reward_normalized_soi_gain_loss": normalized_soi_gain_loss,
            "reward_normalized_jammer_leakage": (
                normalized_jammer_leakage
            ),
            "reward_normalized_action_cost": normalized_action_cost,
            "reward_hold_score": hold_score,
            "sinr_loss_improvement_db": sinr_loss_improvement_db,
            "reward_normalized_improvement": normalized_improvement,
            "reward_teacher_similarity": teacher_similarity,
            "reward_sinr_loss_component": sinr_loss_component,
            "reward_soi_gain_loss_component": soi_gain_loss_component,
            "reward_jammer_leakage_component": jammer_leakage_component,
            "reward_action_component": action_component,
            "reward_hold_component": hold_component,
            "reward_improvement_component": improvement_component,
            "reward_teacher_component": teacher_component,
        }

        return reward, info

    @staticmethod
    def _build_zero_reward_components() -> dict:
        """Return zero-valued reward diagnostics for invalid transitions."""

        return {
            "reward_normalized_sinr_loss": 0.0,
            "reward_normalized_soi_gain_loss": 0.0,
            "reward_normalized_jammer_leakage": 0.0,
            "reward_normalized_action_cost": 0.0,
            "reward_hold_score": 0.0,
            "sinr_loss_improvement_db": 0.0,
            "reward_normalized_improvement": 0.0,
            "reward_teacher_similarity": 0.0,
            "reward_sinr_loss_component": 0.0,
            "reward_soi_gain_loss_component": 0.0,
            "reward_jammer_leakage_component": 0.0,
            "reward_action_component": 0.0,
            "reward_hold_component": 0.0,
            "reward_improvement_component": 0.0,
            "reward_teacher_component": 0.0,
        }

    # ============================================================
    # Logging
    # ============================================================

    def _build_phase8_block_info(
        self,
        block_metrics: list[dict],
        reward: float,
        numerical_error: bool,
        weights_are_finite: bool,
        action_info: dict,
        num_block_steps: int,
        terminated: bool,
        next_feedback: dict,
    ) -> dict:
        """Build aggregated information for one Phase 8 control block."""

        last_metrics = block_metrics[-1] if block_metrics else {}

        info = {
            "phase": 8,
            "reward": float(reward),
            "block_reward_mean": float(reward),
            "num_block_steps": int(num_block_steps),
            "weight_hold_steps": self.weight_hold_steps,
            "current_physical_step": int(self.current_physical_step),
            "episode_length_physical_steps": (
                self.episode_length_physical_steps
            ),
            "terminated": bool(terminated),
            "soi_is_static": True,
            "observation_mode": self.observation_mode,
            "complex_weight_mode": self.complex_weight_mode,
            "action_type": self._get_action_type(),
            "weight_update_mode": "steering_residual_real_imag",
            "geometry_observation_dim": self.geometry_observation_dim,
            "weight_observation_dim": self.weight_observation_dim,
            "feedback_observation_dim": self.feedback_observation_dim,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "num_active_jammers": self.num_active_jammers,
            "jammers_powers": self.jammer_powers.copy(),
            "weights": self.current_weights.copy(),
            "previous_weights": action_info["previous_weights"].copy(),
            "raw_action": action_info["raw_action"].copy(),
            "steering_reference_weights": action_info[
                "steering_reference_weights"
            ].copy(),
            
            "residual_real_action": action_info[
                "residual_real_action"
            ].copy(),
            
            "residual_imag_action": action_info[
                "residual_imag_action"
            ].copy(),
            
            "complex_residual": action_info[
                "complex_residual"
            ].copy(),
            "incremental_real_action": action_info[
                "incremental_real_action"
            ].copy(),
            "incremental_imag_action": action_info[
                "incremental_imag_action"
            ].copy(),
            "complex_increment": action_info["complex_increment"].copy(),
            "proposed_weights_before_normalization": action_info[
                "proposed_weights_before_normalization"
            ].copy(),
            "proposed_weight_power_before_normalization": float(
                action_info["proposed_weight_power_before_normalization"]
            ),
            "final_magnitude": action_info["final_magnitude"].copy(),
            "final_phase_rad": action_info["final_phase_rad"].copy(),
            "final_phase_norm": action_info["final_phase_norm"].copy(),
            "final_weight_power": float(action_info["final_weight_power"]),
            "invalid_residual_weight_action": bool(
                action_info["invalid_residual_weight_action"]
            ),

            # Backward-compatible aliases.
            "invalid_incremental_weight_action": bool(
                action_info["invalid_residual_weight_action"]
            ),

            "invalid_direct_weight_action": bool(
                action_info["invalid_residual_weight_action"]
            ),
            "numerical_error": bool(numerical_error),
            "weights_are_finite": bool(weights_are_finite),
            "array_normalize_power": bool(self.array.normalize_power),
            "next_feedback_metrics": dict(next_feedback),
            "substep_metrics": block_metrics,
            **self._build_phase8_reward_configuration_info(),
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
            "teacher_weight_similarity",
            "teacher_weight_loss",
            "residual_abs_mean",
            "residual_abs_max",
            "residual_rms",
            "realized_residual_abs_mean",
            "realized_residual_abs_max",
            "realized_residual_rms",
            "realized_step_change_abs_mean",
            "realized_step_change_abs_max",
            "realized_step_change_rms",
            "steering_final_weight_similarity",            
            "increment_abs_mean",
            "increment_abs_max",
            "increment_rms",
            "realized_increment_abs_mean",
            "realized_increment_abs_max",
            "realized_increment_rms",
            "normalized_action_energy",
            "previous_final_weight_similarity",
            "reward_normalized_sinr_loss",
            "reward_normalized_soi_gain_loss",
            "reward_normalized_jammer_leakage",
            "reward_normalized_action_cost",
            "reward_hold_score",
            "sinr_loss_improvement_db",
            "reward_normalized_improvement",
            "reward_teacher_similarity",
            "reward_sinr_loss_component",
            "reward_soi_gain_loss_component",
            "reward_jammer_leakage_component",
            "reward_action_component",
            "reward_hold_component",
            "reward_improvement_component",
            "reward_teacher_component",
            "reward_failure_applied",
            "invalid_incremental_weight_action",
            "invalid_residual_weight_action",
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

        # Main aliases expected by Monitor and evaluation notebooks.
        info["sinr_db"] = info["sinr_db_mean"]
        info["reference_sinr_db"] = info["reference_sinr_db_mean"]
        info["sinr_loss_db"] = info["sinr_loss_db_mean"]
        info["clipped_sinr_loss_db"] = info[
            "clipped_sinr_loss_db_mean"
        ]
        info["soi_gain_loss_db"] = info["soi_gain_loss_db_mean"]
        info["jammer_leakage_loss"] = info[
            "jammer_leakage_loss_mean"
        ]
        info["teacher_weight_similarity"] = info[
            "teacher_weight_similarity_mean"
        ]
        info["teacher_weight_loss"] = info["teacher_weight_loss_mean"]
        info["reward_sinr_loss_component"] = info[
            "reward_sinr_loss_component_mean"
        ]
        info["reward_soi_gain_loss_component"] = info[
            "reward_soi_gain_loss_component_mean"
        ]
        info["reward_jammer_leakage_component"] = info[
            "reward_jammer_leakage_component_mean"
        ]
        info["reward_action_component"] = info[
            "reward_action_component_mean"
        ]
        info["reward_hold_component"] = info[
            "reward_hold_component_mean"
        ]
        info["reward_improvement_component"] = info[
            "reward_improvement_component_mean"
        ]
        info["reward_teacher_component"] = info[
            "reward_teacher_component_mean"
        ]
        info["failure_penalty_fraction"] = info[
            "reward_failure_applied_mean"
        ]

        failure_reasons = [
            item.get("reward_failure_reason", "unknown")
            for item in block_metrics
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
                "teacher_weights": last_metrics.get(
                    "teacher_weights",
                    np.zeros(
                        (self.array.N, self.array.M),
                        dtype=np.complex128,
                    ),
                ),
            }
        )

        return info

    def _build_phase8_reward_configuration_info(self) -> dict:
        """Return Phase 8 configuration values for reset and step info."""

        return {
            "residual_complex_scale": self.residual_complex_scale,
            "residual_weight_min_power": (
                self.residual_weight_min_power
            ),
            "reward_beta_sinr_loss": self.reward_beta_sinr_loss,
            "reward_gamma_soi_gain_loss": (
                self.reward_gamma_soi_gain_loss
            ),
            "reward_gamma_jammer_leakage": (
                self.reward_gamma_jammer_leakage
            ),
            "reward_gamma_action": self.reward_gamma_action,
            "reward_gamma_hold": self.reward_gamma_hold,
            "reward_gamma_improvement": self.reward_gamma_improvement,
            "reward_gamma_teacher_similarity": (
                self.reward_gamma_teacher_similarity
            ),
            "reward_sinr_loss_scale_db": (
                self.reward_sinr_loss_scale_db
            ),
            "reward_sinr_loss_clip": self.reward_sinr_loss_clip,
            "reward_soi_gain_loss_scale_db": (
                self.reward_soi_gain_loss_scale_db
            ),
            "reward_soi_gain_loss_clip": (
                self.reward_soi_gain_loss_clip
            ),
            "reward_jammer_leakage_scale": (
                self.reward_jammer_leakage_scale
            ),
            "reward_jammer_leakage_clip": (
                self.reward_jammer_leakage_clip
            ),
            "reward_action_scale": self.reward_action_scale,
            "reward_action_clip": self.reward_action_clip,
            "reward_hold_scale_db": self.reward_hold_scale_db,
            "reward_improvement_scale_db": (
                self.reward_improvement_scale_db
            ),
            "reward_improvement_clip": self.reward_improvement_clip,
            "observation_sinr_scale_db": (
                self.observation_sinr_scale_db
            ),
            "observation_jammer_leakage_scale": (
                self.observation_jammer_leakage_scale
            ),
        }

    # ============================================================
    # Validation and labels
    # ============================================================

    def _validate_phase8_configuration(self) -> None:
        """Validate parameters introduced by Phase 8."""

        positive_parameters = {
            "residual_complex_scale": self.residual_complex_scale,
            "reward_sinr_loss_scale_db": self.reward_sinr_loss_scale_db,
            "reward_soi_gain_loss_scale_db": (
                self.reward_soi_gain_loss_scale_db
            ),
            "reward_jammer_leakage_scale": (
                self.reward_jammer_leakage_scale
            ),
            "reward_action_scale": self.reward_action_scale,
            "reward_hold_scale_db": self.reward_hold_scale_db,
            "reward_improvement_scale_db": (
                self.reward_improvement_scale_db
            ),
            "observation_sinr_scale_db": (
                self.observation_sinr_scale_db
            ),
            "observation_jammer_leakage_scale": (
                self.observation_jammer_leakage_scale
            ),
        }

        for parameter_name, value in positive_parameters.items():
            if value <= 0.0:
                raise ValueError(f"{parameter_name} must be positive.")

        non_negative_parameters = {
            "residual_weight_min_power": (
                self.residual_weight_min_power
            ),
            "reward_beta_sinr_loss": self.reward_beta_sinr_loss,
            "reward_gamma_soi_gain_loss": (
                self.reward_gamma_soi_gain_loss
            ),
            "reward_gamma_jammer_leakage": (
                self.reward_gamma_jammer_leakage
            ),
            "reward_gamma_action": self.reward_gamma_action,
            "reward_gamma_hold": self.reward_gamma_hold,
            "reward_gamma_improvement": self.reward_gamma_improvement,
            "reward_gamma_teacher_similarity": (
                self.reward_gamma_teacher_similarity
            ),
            "reward_sinr_loss_clip": self.reward_sinr_loss_clip,
            "reward_soi_gain_loss_clip": (
                self.reward_soi_gain_loss_clip
            ),
            "reward_jammer_leakage_clip": (
                self.reward_jammer_leakage_clip
            ),
            "reward_action_clip": self.reward_action_clip,
            "reward_improvement_clip": self.reward_improvement_clip,
        }

        for parameter_name, value in non_negative_parameters.items():
            if value < 0.0:
                raise ValueError(
                    f"{parameter_name} must be non-negative."
                )

    def _get_action_type(self) -> str:
        """Return the Phase 8 action label used by logs and notebooks."""
    
        return "residual_real_imag"
