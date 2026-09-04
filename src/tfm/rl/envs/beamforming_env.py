from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tfm.physics.narrow_band.phased_array_nb import Phased_Array_NB
from tfm.math.narrow_band.metrics import compute_sinr
from tfm.math.narrow_band.steering_vector import get_steering_vector
from tfm.math.narrow_band.geometry import (
    angles_to_unit_vector,
    unit_vector_to_angles,
)


class BeamformingEnv(gym.Env):
    """
    Gymnasium environment for Phase 2 beam steering.

    This environment is designed for training a DRL agent to steer the
    main beam towards a desired signal direction without jammers.

    Key design choice
    -----------------
    The environment does not use ScenarioGenerator internally.

    The target DOA is generated directly in angular coordinates at reset().
    This keeps the training environment simple, controlled and independent
    from the external ground-truth simulation pipeline.

    Observation modes
    -----------------
    observation_mode = "angles"
        state = [
            theta_target_norm,
            phi_target_norm,
            jammer_slot_1,
            jammer_slot_2,
            jammer_slot_3
        ]

        Since Phase 2 has no jammers, the jammer slots are filled with zeros:

        state = [
            theta_target_norm,
            phi_target_norm,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0
        ]

        Observation dimension: 11.

    observation_mode = "unit_vector"
        state = [
            u_target_x,
            u_target_y,
            u_target_z,
            jammer_slot_1,
            jammer_slot_2,
            jammer_slot_3
        ]

        Since Phase 2 has no jammers, the jammer slots are filled with zeros:

        state = [
            u_target_x,
            u_target_y,
            u_target_z,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        ]

        Observation dimension: 15.

    Action modes
    ------------
    action_mode = "angles"
        action = [theta_steer_norm, phi_steer_norm]

        Both values are normalized to [0, 1].

    action_mode = "unit_vector"
        action = [ux, uy, uz]

        The vector is clipped to [-1, 1], normalized internally,
        optionally projected to the visible hemisphere, converted to angular
        coordinates and then used to build conventional steering weights.

    Target modes
    ------------
    target_mode = "random_static"
        A random target DOA is sampled at every reset().

    target_mode = "fixed_static"
        A fixed target DOA is used. This is useful for debugging and controlled
        evaluation, but not recommended for the main training runs.

    Reward definition
    -----------------
    reward = alpha * sinr_db
             - beta * clipped_sinr_loss_db
             - gamma * angle_loss

    where:

        angle_loss = (angle_error_deg / 180)^2

    The reference SINR is computed by steering exactly towards the true target
    DOA. Therefore:

        sinr_loss_db = reference_sinr_db - sinr_db

    By default:

        alpha = 1.0
        beta  = 0.0
        gamma = 0.0

    Therefore, the default behavior is the original SINR-based reward.

    Notes
    -----
    This is a one-step environment:

        reset() -> generate target DOA
        step()  -> evaluate one steering action -> terminated = True

    This makes the environment equivalent to a contextual bandit / one-step MDP,
    which is appropriate for Phase 2 static beam steering.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        array: Phased_Array_NB,
        array_position: np.ndarray,
        desired_power: float = 1.0,
        noise_power: float = 1e-3,
        max_jammers: int = 3,
        target_mode: str = "random_static",
        fixed_target_doa_rad: tuple[float, float] | None = None,
        theta_limits_rad: tuple[float, float] = (0.0, np.pi / 2.0),
        phi_limits_rad: tuple[float, float] = (0.0, 2.0 * np.pi),
        observation_mode: str = "angles",
        action_mode: str = "angles",
        enforce_visible_hemisphere: bool = True,
        reward_alpha_sinr: float = 1.0,
        reward_beta_sinr_loss: float = 0.0,
        reward_gamma_angle: float = 0.0,
        max_sinr_loss_db: float = 60.0,
    ) -> None:
        super().__init__()

        self.array = array
        self.array_position = np.asarray(array_position, dtype=float).reshape(3)

        self.desired_power = float(desired_power)
        self.noise_power = float(noise_power)

        self.max_jammers = int(max_jammers)

        self.target_mode = str(target_mode)
        self.fixed_target_doa_rad = fixed_target_doa_rad

        self.theta_min = float(theta_limits_rad[0])
        self.theta_max = float(theta_limits_rad[1])
        self.phi_min = float(phi_limits_rad[0])
        self.phi_max = float(phi_limits_rad[1])

        self.observation_mode = str(observation_mode)
        self.action_mode = str(action_mode)
        self.enforce_visible_hemisphere = bool(enforce_visible_hemisphere)

        self.reward_alpha_sinr = float(reward_alpha_sinr)
        self.reward_beta_sinr_loss = float(reward_beta_sinr_loss)
        self.reward_gamma_angle = float(reward_gamma_angle)
        self.max_sinr_loss_db = float(max_sinr_loss_db)

        if self.max_jammers != 3:
            raise ValueError(
                "This environment currently requires max_jammers=3 "
                "to match the fixed roadmap state format."
            )

        valid_target_modes = {"random_static", "fixed_static"}

        if self.target_mode not in valid_target_modes:
            raise ValueError(
                f"Unknown target_mode: {self.target_mode}. "
                f"Valid modes are: {valid_target_modes}"
            )

        if self.target_mode == "fixed_static":
            if self.fixed_target_doa_rad is None:
                raise ValueError(
                    "fixed_target_doa_rad must be provided when "
                    "target_mode='fixed_static'."
                )

            if len(self.fixed_target_doa_rad) != 2:
                raise ValueError(
                    "fixed_target_doa_rad must be a tuple "
                    "(theta_rad, phi_rad)."
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

        else:
            raise ValueError(f"Unknown observation_mode: {self.observation_mode}")

        # ============================================================
        # Action space
        # ============================================================

        if self.action_mode == "angles":
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )

        elif self.action_mode == "unit_vector":
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            )

        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        # ============================================================
        # Episode variables
        # ============================================================

        self.current_theta_rad: float | None = None
        self.current_phi_rad: float | None = None
        self.current_state: np.ndarray | None = None

    # ============================================================
    # Gym API
    # ============================================================

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment and generate the target DOA for the episode.
        """

        super().reset(seed=seed)

        theta_rad, phi_rad = self._generate_target_doa_for_episode()

        self.current_theta_rad = theta_rad
        self.current_phi_rad = phi_rad

        state = self._build_state(
            theta_target_rad=theta_rad,
            phi_target_rad=phi_rad,
        )

        self.current_state = state

        info = {
            "theta_target_rad": theta_rad,
            "phi_target_rad": phi_rad,
            "theta_target_deg": float(np.rad2deg(theta_rad)),
            "phi_target_deg": float(np.rad2deg(phi_rad)),
            "target_mode": self.target_mode,
        }

        return state, info

    def step(self, action: np.ndarray):
        """
        Evaluate one steering action and terminate the episode.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized.")

        theta_steer_rad, phi_steer_rad = self._action_to_angles(action)

        weights = self._build_steering_weights(
            theta_rad=theta_steer_rad,
            phi_rad=phi_steer_rad,
        )

        self.array.set_weights(weights)

        target_direction_deg = (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.rad2deg(self.current_phi_rad)),
        )

        sinr_db = compute_sinr(
            weights=self.array.W,
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            target_direction=target_direction_deg,
            target_power=self.desired_power,
            jammers_directions=[],
            jammers_powers=[],
            noise_power=self.noise_power,
        )

        sinr_db = float(sinr_db)

        angle_error_deg = self._compute_angular_error_deg(
            theta_a_rad=self.current_theta_rad,
            phi_a_rad=self.current_phi_rad,
            theta_b_rad=theta_steer_rad,
            phi_b_rad=phi_steer_rad,
        )

        angle_loss = self._compute_angle_loss(angle_error_deg)

        reference_sinr_db = self._compute_reference_steering_sinr_db()

        sinr_loss_db = reference_sinr_db - sinr_db
        sinr_loss_db = max(0.0, float(sinr_loss_db))

        clipped_sinr_loss_db = min(sinr_loss_db, self.max_sinr_loss_db)

        reward = self._compute_reward(
            sinr_db=sinr_db,
            clipped_sinr_loss_db=clipped_sinr_loss_db,
            angle_loss=angle_loss,
        )

        reward = float(reward)

        terminated = True
        truncated = False

        next_state = self.current_state.copy()

        info = {
            "reward": reward,
            "reward_alpha_sinr": self.reward_alpha_sinr,
            "reward_beta_sinr_loss": self.reward_beta_sinr_loss,
            "reward_gamma_angle": self.reward_gamma_angle,
            "sinr_db": sinr_db,
            "reference_sinr_db": reference_sinr_db,
            "sinr_loss_db": sinr_loss_db,
            "clipped_sinr_loss_db": clipped_sinr_loss_db,
            "angle_error_deg": angle_error_deg,
            "angle_loss": angle_loss,
            "theta_target_rad": self.current_theta_rad,
            "phi_target_rad": self.current_phi_rad,
            "theta_target_deg": float(np.rad2deg(self.current_theta_rad)),
            "phi_target_deg": float(np.rad2deg(self.current_phi_rad)),
            "theta_steer_rad": theta_steer_rad,
            "phi_steer_rad": phi_steer_rad,
            "theta_steer_deg": float(np.rad2deg(theta_steer_rad)),
            "phi_steer_deg": float(np.rad2deg(phi_steer_rad)),
            "weights": self.array.W.copy(),
        }

        return next_state, reward, terminated, truncated, info

    # ============================================================
    # Internal helpers: target generation
    # ============================================================

    def _generate_target_doa_for_episode(self) -> tuple[float, float]:
        """
        Generate or retrieve the target DOA for the current episode.
        """

        if self.target_mode == "fixed_static":
            assert self.fixed_target_doa_rad is not None

            theta_rad = float(self.fixed_target_doa_rad[0])
            phi_rad = float(self.fixed_target_doa_rad[1])

            return theta_rad, phi_rad

        if self.target_mode == "random_static":
            theta_rad = float(self.np_random.uniform(self.theta_min, self.theta_max))
            phi_rad = float(self.np_random.uniform(self.phi_min, self.phi_max))

            return theta_rad, phi_rad

        raise RuntimeError("Invalid target mode.")

    # ============================================================
    # Internal helpers: state and action
    # ============================================================

    def _build_state(
        self,
        theta_target_rad: float,
        phi_target_rad: float,
    ) -> np.ndarray:
        """
        Build the fixed-size Phase 2 state.

        The roadmap format is preserved:
        - target information is active
        - jammer slots are padded with zeros
        """

        if self.observation_mode == "angles":
            theta_norm = self._normalize_theta(theta_target_rad)
            phi_norm = self._normalize_phi(phi_target_rad)

            state = [theta_norm, phi_norm]

            for _ in range(self.max_jammers):
                state.extend([0.0, 0.0, 0.0])

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

            for _ in range(self.max_jammers):
                state.extend([0.0, 0.0, 0.0, 0.0])

            return np.array(state, dtype=np.float32)

        raise RuntimeError("Invalid observation mode.")

    def _action_to_angles(self, action: np.ndarray) -> tuple[float, float]:
        """
        Convert the selected action representation into steering angles in radians.
        """

        if self.action_mode == "angles":
            action = np.asarray(action, dtype=np.float32).reshape(2)
            action = np.clip(action, self.action_space.low, self.action_space.high)

            theta_steer_rad = self._denormalize_theta(float(action[0]))
            phi_steer_rad = self._denormalize_phi(float(action[1]))

            return theta_steer_rad, phi_steer_rad

        if self.action_mode == "unit_vector":
            action = np.asarray(action, dtype=np.float32).reshape(3)
            action = np.clip(action, self.action_space.low, self.action_space.high)

            norm = np.linalg.norm(action)

            if norm < 1e-8:
                u_steer = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                u_steer = action / norm

            if self.enforce_visible_hemisphere and u_steer[2] < 0.0:
                u_steer[2] = abs(u_steer[2])
                u_steer = u_steer / (np.linalg.norm(u_steer) + 1e-8)

            theta_deg, phi_deg = unit_vector_to_angles(
                u_steer,
                enforce_visible=self.enforce_visible_hemisphere,
            )

            theta_steer_rad = np.deg2rad(theta_deg)
            phi_steer_rad = np.deg2rad(phi_deg)

            return float(theta_steer_rad), float(phi_steer_rad)

        raise RuntimeError("Invalid action mode.")

    # ============================================================
    # Internal helpers: beamforming and reward
    # ============================================================

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

    def _compute_reference_steering_sinr_db(self) -> float:
        """
        Compute the reference SINR obtained by steering exactly towards the target DOA.
        """

        if self.current_theta_rad is None or self.current_phi_rad is None:
            raise RuntimeError(
                "Environment must be reset before computing reference SINR."
            )

        reference_weights = self._build_steering_weights(
            theta_rad=self.current_theta_rad,
            phi_rad=self.current_phi_rad,
        )

        target_direction_deg = (
            float(np.rad2deg(self.current_theta_rad)),
            float(np.rad2deg(self.current_phi_rad)),
        )

        reference_sinr_db = compute_sinr(
            weights=reference_weights,
            element_positions=self.array.element_positions,
            wavenumber_k=self.array.k_num,
            target_direction=target_direction_deg,
            target_power=self.desired_power,
            jammers_directions=[],
            jammers_powers=[],
            noise_power=self.noise_power,
        )

        return float(reference_sinr_db)

    def _compute_reward(
        self,
        sinr_db: float,
        clipped_sinr_loss_db: float,
        angle_loss: float,
    ) -> float:
        """
        Compute the weighted reward.
        """

        reward = (
            self.reward_alpha_sinr * float(sinr_db)
            - self.reward_beta_sinr_loss * float(clipped_sinr_loss_db)
            - self.reward_gamma_angle * float(angle_loss)
        )

        return float(reward)

    # ============================================================
    # Internal helpers: geometry
    # ============================================================

    def _compute_angular_error_deg(
        self,
        theta_a_rad: float,
        phi_a_rad: float,
        theta_b_rad: float,
        phi_b_rad: float,
    ) -> float:
        """
        Compute the 3D angular separation between two directions.

        Angle convention:
        theta: polar angle from +z.
        phi: azimuth angle in the xy-plane from +x to +y.
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