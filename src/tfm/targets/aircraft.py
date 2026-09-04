import numpy as np


class AircraftTarget:
    """
    Aircraft target model for realistic 3D flight-motion simulation.

    The aircraft moves through random flight segments while respecting
    realistic kinematic constraints:
        - bounded speed
        - bounded longitudinal acceleration
        - bounded jerk
        - bounded bank angle
        - bounded bank rate
        - bounded climb / descent angle
        - bounded climb-angle rate
        - altitude recovery tendency if initial altitude is outside limits

    Outputs:
        - position history      (N, 3)
        - velocity history      (N, 3)
        - acceleration history  (N, 3)

    Parameters
    ----------
    x0 : float
        Initial x position.
    y0 : float
        Initial y position.
    z0 : float
        Initial z position.
    dt : float
        Simulation time step.
    num_steps : int
        Number of simulation steps.
    seed : int | None, optional
        Random seed for reproducible trajectory generation.
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        z0: float,
        dt: float,
        num_steps: int,
        seed: int | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")

        self.x0 = float(x0)
        self.y0 = float(y0)
        self.z0 = float(z0)
        self.dt = float(dt)
        self.num_steps = int(num_steps)
        self.seed = seed

        # Random generator
        self.rng = np.random.default_rng(seed)

        # ============================================================
        # Internal aircraft parameters
        # ============================================================

        # Gravity
        self.g = 9.81  # [m/s^2]

        # Speed limits (generic tactical / patrol-like aircraft)
        self.v_min = 70.0    # [m/s]
        self.v_max = 220.0   # [m/s]
        self.v0 = 140.0      # [m/s]

        # Longitudinal acceleration
        self.a_t_min = -8.0   # [m/s^2]
        self.a_t_max = 10.0   # [m/s^2]
        self.jerk_limit = 4.0 # [m/s^3]

        # Bank angle dynamics
        self.bank_max = np.deg2rad(35.0)        # [rad]
        self.bank_rate_max = np.deg2rad(8.0)    # [rad/s]

        # Flight-path angle dynamics
        self.gamma_max_climb = np.deg2rad(18.0)   # [rad]
        self.gamma_max_dive = np.deg2rad(-20.0)   # [rad]
        self.gamma_rate_max = np.deg2rad(3.0)     # [rad/s]

        # Altitude limits
        self.z_min = 300.0    # [m]
        self.z_max = 4000.0   # [m]

        # Segment duration
        self.segment_duration_min = 5.0   # [s]
        self.segment_duration_max = 18.0  # [s]

        # Small perturbations
        self.bank_noise_std = np.deg2rad(0.08)
        self.gamma_noise_std = np.deg2rad(0.03)
        self.acc_noise_std = 0.15

        # Initial motion state
        self.psi0 = self._sample_initial_heading()
        self.gamma0 = self._sample_initial_gamma(self.z0)

        self._position: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._acceleration: np.ndarray | None = None

    def generate(self) -> None:
        """
        Generate trajectory histories (position, velocity, acceleration).
        """
        position = np.zeros((self.num_steps, 3), dtype=float)
        velocity = np.zeros((self.num_steps, 3), dtype=float)
        acceleration = np.zeros((self.num_steps, 3), dtype=float)

        # ============================================================
        # Dynamic states
        # ============================================================
        x = self.x0
        y = self.y0
        z = self.z0

        v = self.v0
        psi = self.psi0
        gamma = self.gamma0
        bank = 0.0
        a_t = 0.0

        # Segment targets
        target_a_t = 0.0
        target_bank = 0.0
        target_gamma = gamma
        remaining_segment_steps = 0

        prev_vx = None
        prev_vy = None
        prev_vz = None

        for k in range(self.num_steps):
            # ========================================================
            # 1. New random flight segment if needed
            # ========================================================
            if remaining_segment_steps <= 0:
                (
                    target_a_t,
                    target_bank,
                    target_gamma,
                    remaining_segment_steps,
                ) = self._sample_new_segment(
                    current_speed=v,
                    current_altitude=z,
                    current_gamma=gamma,
                )

            remaining_segment_steps -= 1

            # ========================================================
            # 2. Longitudinal acceleration update with jerk limit
            # ========================================================
            a_error = target_a_t - a_t
            max_da = self.jerk_limit * self.dt
            da = np.clip(a_error, -max_da, max_da)
            a_t += da

            a_t += self.rng.normal(0.0, self.acc_noise_std)
            a_t = np.clip(a_t, self.a_t_min, self.a_t_max)

            # ========================================================
            # 3. Bank update with bank-rate limit
            # ========================================================
            bank_error = target_bank - bank
            max_dbank = self.bank_rate_max * self.dt
            dbank = np.clip(bank_error, -max_dbank, max_dbank)
            bank += dbank

            bank += self.rng.normal(0.0, self.bank_noise_std)
            bank = np.clip(bank, -self.bank_max, self.bank_max)

            # ========================================================
            # 4. Gamma update with gamma-rate limit
            # ========================================================
            gamma_error = target_gamma - gamma
            max_dgamma = self.gamma_rate_max * self.dt
            dgamma = np.clip(gamma_error, -max_dgamma, max_dgamma)
            gamma += dgamma

            gamma += self.rng.normal(0.0, self.gamma_noise_std)

            # Altitude recovery tendency if outside admissible range
            gamma += self._altitude_recovery_bias(z) * self.dt

            gamma = np.clip(gamma, self.gamma_max_dive, self.gamma_max_climb)

            # ========================================================
            # 5. Speed update
            # ========================================================
            v = v + a_t * self.dt
            v = np.clip(v, self.v_min, self.v_max)

            # ========================================================
            # 6. Heading update from coordinated-turn approximation
            # ========================================================
            cos_gamma = max(np.cos(gamma), 1e-6)
            yaw_rate = self.g * np.tan(bank) / max(v * cos_gamma, 1e-6)
            psi = self._wrap_angle(psi + yaw_rate * self.dt)

            # ========================================================
            # 7. Velocity components in 3D
            # ========================================================
            v_xy = v * np.cos(gamma)
            vx = v_xy * np.cos(psi)
            vy = v_xy * np.sin(psi)
            vz = v * np.sin(gamma)

            # ========================================================
            # 8. Position update
            # ========================================================
            x += vx * self.dt
            y += vy * self.dt
            z += vz * self.dt

            # ========================================================
            # 9. Save position and velocity
            # ========================================================
            position[k, 0] = x
            position[k, 1] = y
            position[k, 2] = z

            velocity[k, 0] = vx
            velocity[k, 1] = vy
            velocity[k, 2] = vz

            # ========================================================
            # 10. Save acceleration
            # ========================================================
            if k == 0:
                acceleration[k, 0] = a_t * np.cos(gamma) * np.cos(psi)
                acceleration[k, 1] = a_t * np.cos(gamma) * np.sin(psi)
                acceleration[k, 2] = a_t * np.sin(gamma)
            else:
                acceleration[k, 0] = (vx - prev_vx) / self.dt
                acceleration[k, 1] = (vy - prev_vy) / self.dt
                acceleration[k, 2] = (vz - prev_vz) / self.dt

            prev_vx = vx
            prev_vy = vy
            prev_vz = vz

        self._position = position
        self._velocity = velocity
        self._acceleration = acceleration

    def get_position(self) -> np.ndarray:
        """
        Return position history of shape (N, 3).
        """
        self._check_generated()
        return self._position.copy()

    def get_velocity(self) -> np.ndarray:
        """
        Return velocity history of shape (N, 3).
        """
        self._check_generated()
        return self._velocity.copy()

    def get_acceleration(self) -> np.ndarray:
        """
        Return acceleration history of shape (N, 3).
        """
        self._check_generated()
        return self._acceleration.copy()

    def get_trajectory_dict(self) -> dict[str, np.ndarray]:
        """
        Return all trajectory data in a dictionary.
        """
        self._check_generated()
        return {
            "position": self._position.copy(),
            "velocity": self._velocity.copy(),
            "acceleration": self._acceleration.copy(),
        }

    def _sample_new_segment(
        self,
        current_speed: float,
        current_altitude: float,
        current_gamma: float,
    ) -> tuple[float, float, float, int]:
        """
        Sample a new random flight segment.
        """
        duration = self.rng.uniform(
            self.segment_duration_min,
            self.segment_duration_max,
        )
        num_segment_steps = max(1, int(np.round(duration / self.dt)))

        # ============================================================
        # 1. Target longitudinal acceleration
        # ============================================================
        cruise_speed = self.rng.uniform(100.0, 180.0)
        speed_error = cruise_speed - current_speed
        target_a_t = 0.12 * speed_error + self.rng.normal(0.0, 0.5)
        target_a_t = np.clip(target_a_t, self.a_t_min, self.a_t_max)

        # ============================================================
        # 2. Target bank angle
        # Mostly straight flight, sometimes mild turn, occasionally stronger
        # ============================================================
        bank_mode = self.rng.choice([0, 1, 2], p=[0.55, 0.30, 0.15])

        if bank_mode == 0:
            target_bank = self.rng.normal(0.0, np.deg2rad(2.0))
        elif bank_mode == 1:
            target_bank = self.rng.uniform(
                np.deg2rad(-12.0),
                np.deg2rad(12.0),
            )
        else:
            target_bank = self.rng.uniform(
                np.deg2rad(-25.0),
                np.deg2rad(25.0),
            )

        target_bank = np.clip(target_bank, -self.bank_max, self.bank_max)

        # ============================================================
        # 3. Target gamma
        # Altitude-aware: if outside range, bias toward re-entry
        # ============================================================
        if current_altitude < self.z_min:
            target_gamma = self.rng.uniform(
                np.deg2rad(4.0),
                np.deg2rad(12.0),
            )
        elif current_altitude > self.z_max:
            target_gamma = self.rng.uniform(
                np.deg2rad(-12.0),
                np.deg2rad(-4.0),
            )
        else:
            gamma_mode = self.rng.choice([0, 1, 2], p=[0.55, 0.25, 0.20])

            if gamma_mode == 0:
                target_gamma = self.rng.normal(0.0, np.deg2rad(1.0))
            elif gamma_mode == 1:
                target_gamma = self.rng.uniform(
                    np.deg2rad(2.0),
                    np.deg2rad(8.0),
                )
            else:
                target_gamma = self.rng.uniform(
                    np.deg2rad(-8.0),
                    np.deg2rad(-2.0),
                )

            # Small persistence around current gamma
            target_gamma = 0.65 * target_gamma + 0.35 * current_gamma

        target_gamma = np.clip(
            target_gamma,
            self.gamma_max_dive,
            self.gamma_max_climb,
        )

        return target_a_t, target_bank, target_gamma, num_segment_steps

    def _sample_initial_heading(self) -> float:
        """
        Sample initial heading.
        """
        return self.rng.uniform(
            np.deg2rad(-25.0),
            np.deg2rad(25.0),
        )

    def _sample_initial_gamma(self, z0: float) -> float:
        """
        Sample initial flight-path angle.

        If the initial altitude is outside the admissible altitude interval,
        the initial gamma is biased to drive the aircraft back into range.
        """
        if z0 < self.z_min:
            return self.rng.uniform(
                np.deg2rad(3.0),
                np.deg2rad(10.0),
            )

        if z0 > self.z_max:
            return self.rng.uniform(
                np.deg2rad(-10.0),
                np.deg2rad(-3.0),
            )

        return self.rng.uniform(
            np.deg2rad(-2.0),
            np.deg2rad(2.0),
        )

    def _altitude_recovery_bias(self, z: float) -> float:
        """
        Return an additional gamma-rate bias to bring altitude back into range.

        Output units: [rad/s]
        """
        if z < self.z_min:
            error = self.z_min - z
            bias = 0.0008 * error
            return min(bias, np.deg2rad(1.2))

        if z > self.z_max:
            error = z - self.z_max
            bias = -0.0008 * error
            return max(bias, -np.deg2rad(1.2))

        # Mild centering tendency near boundaries
        mid = 0.5 * (self.z_min + self.z_max)
        half_span = 0.5 * (self.z_max - self.z_min)
        normalized = (z - mid) / max(half_span, 1e-6)

        if abs(normalized) < 0.7:
            return 0.0

        return -0.15 * normalized * np.deg2rad(1.0)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """
        Wrap angle to [-pi, pi].
        """
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _check_generated(self) -> None:
        """
        Ensure trajectory has been generated.
        """
        if (
            self._position is None
            or self._velocity is None
            or self._acceleration is None
        ):
            raise RuntimeError(
                "Trajectory not generated yet. Call generate() first."
            )