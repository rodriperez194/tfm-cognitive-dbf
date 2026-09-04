import numpy as np


class DroneTarget:
    """
    Drone target model for agile 3D flight-motion simulation.

    The drone moves through random flight segments while respecting
    realistic kinematic constraints for a small aerial vehicle:
        - bounded speed
        - bounded longitudinal acceleration
        - bounded jerk
        - bounded heading-rate
        - bounded heading-acceleration
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
        # Internal drone parameters
        # ============================================================

        # Speed limits
        self.v_min = 3.0     # [m/s]
        self.v_max = 28.0    # [m/s]
        self.v0 = 12.0       # [m/s]

        # Longitudinal acceleration
        self.a_t_min = -5.0   # [m/s^2]
        self.a_t_max = 6.0    # [m/s^2]
        self.jerk_limit = 8.0 # [m/s^3]

        # Heading dynamics
        self.heading_rate_max = np.deg2rad(45.0)      # [rad/s]
        self.heading_accel_max = np.deg2rad(120.0)    # [rad/s^2]

        # Flight-path angle dynamics
        self.gamma_max_climb = np.deg2rad(35.0)   # [rad]
        self.gamma_max_dive = np.deg2rad(-40.0)   # [rad]
        self.gamma_rate_max = np.deg2rad(20.0)    # [rad/s]

        # Altitude limits
        self.z_min = 20.0     # [m]
        self.z_max = 500.0    # [m]

        # Segment duration
        self.segment_duration_min = 1.5   # [s]
        self.segment_duration_max = 6.0   # [s]

        # Small perturbations
        self.heading_rate_noise_std = np.deg2rad(1.2)
        self.gamma_noise_std = np.deg2rad(0.6)
        self.acc_noise_std = 0.25

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
        a_t = 0.0
        heading_rate = 0.0

        # Segment targets
        target_a_t = 0.0
        target_heading_rate = 0.0
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
                    target_heading_rate,
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
            # 3. Heading-rate update with bounded angular acceleration
            # ========================================================
            heading_rate_error = target_heading_rate - heading_rate
            max_dheading_rate = self.heading_accel_max * self.dt
            dheading_rate = np.clip(
                heading_rate_error,
                -max_dheading_rate,
                max_dheading_rate,
            )
            heading_rate += dheading_rate

            heading_rate += self.rng.normal(0.0, self.heading_rate_noise_std)
            heading_rate = np.clip(
                heading_rate,
                -self.heading_rate_max,
                self.heading_rate_max,
            )

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
            # 6. Heading update
            # ========================================================
            psi = self._wrap_angle(psi + heading_rate * self.dt)

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
        Sample a new random drone-flight segment.
        """
        duration = self.rng.uniform(
            self.segment_duration_min,
            self.segment_duration_max,
        )
        num_segment_steps = max(1, int(np.round(duration / self.dt)))

        # ============================================================
        # 1. Target longitudinal acceleration
        # ============================================================
        cruise_speed = self.rng.uniform(6.0, 20.0)
        speed_error = cruise_speed - current_speed
        target_a_t = 0.35 * speed_error + self.rng.normal(0.0, 0.5)
        target_a_t = np.clip(target_a_t, self.a_t_min, self.a_t_max)

        # ============================================================
        # 2. Target heading rate
        # Drones can keep straight segments, but also turn sharply
        # ============================================================
        heading_mode = self.rng.choice([0, 1, 2, 3], p=[0.30, 0.30, 0.25, 0.15])

        if heading_mode == 0:
            target_heading_rate = self.rng.normal(0.0, np.deg2rad(4.0))
        elif heading_mode == 1:
            target_heading_rate = self.rng.uniform(
                np.deg2rad(-15.0),
                np.deg2rad(15.0),
            )
        elif heading_mode == 2:
            target_heading_rate = self.rng.uniform(
                np.deg2rad(-30.0),
                np.deg2rad(30.0),
            )
        else:
            target_heading_rate = self.rng.uniform(
                np.deg2rad(-45.0),
                np.deg2rad(45.0),
            )

        target_heading_rate = np.clip(
            target_heading_rate,
            -self.heading_rate_max,
            self.heading_rate_max,
        )

        # ============================================================
        # 3. Target gamma
        # Altitude-aware: if outside range, bias toward re-entry
        # ============================================================
        if current_altitude < self.z_min:
            target_gamma = self.rng.uniform(
                np.deg2rad(8.0),
                np.deg2rad(20.0),
            )
        elif current_altitude > self.z_max:
            target_gamma = self.rng.uniform(
                np.deg2rad(-20.0),
                np.deg2rad(-8.0),
            )
        else:
            gamma_mode = self.rng.choice([0, 1, 2, 3], p=[0.30, 0.25, 0.25, 0.20])

            if gamma_mode == 0:
                target_gamma = self.rng.normal(0.0, np.deg2rad(2.0))
            elif gamma_mode == 1:
                target_gamma = self.rng.uniform(
                    np.deg2rad(4.0),
                    np.deg2rad(16.0),
                )
            elif gamma_mode == 2:
                target_gamma = self.rng.uniform(
                    np.deg2rad(-16.0),
                    np.deg2rad(-4.0),
                )
            else:
                target_gamma = self.rng.uniform(
                    np.deg2rad(-25.0),
                    np.deg2rad(25.0),
                )

            target_gamma = 0.55 * target_gamma + 0.45 * current_gamma

        target_gamma = np.clip(
            target_gamma,
            self.gamma_max_dive,
            self.gamma_max_climb,
        )

        return target_a_t, target_heading_rate, target_gamma, num_segment_steps

    def _sample_initial_heading(self) -> float:
        """
        Sample initial heading.
        """
        return self.rng.uniform(-np.pi, np.pi)

    def _sample_initial_gamma(self, z0: float) -> float:
        """
        Sample initial flight-path angle.

        If the initial altitude is outside the admissible altitude interval,
        the initial gamma is biased to drive the drone back into range.
        """
        if z0 < self.z_min:
            return self.rng.uniform(
                np.deg2rad(6.0),
                np.deg2rad(18.0),
            )

        if z0 > self.z_max:
            return self.rng.uniform(
                np.deg2rad(-18.0),
                np.deg2rad(-6.0),
            )

        return self.rng.uniform(
            np.deg2rad(-5.0),
            np.deg2rad(5.0),
        )

    def _altitude_recovery_bias(self, z: float) -> float:
        """
        Return an additional gamma-rate bias to bring altitude back into range.

        Output units: [rad/s]
        """
        if z < self.z_min:
            error = self.z_min - z
            bias = 0.01 * error
            return min(bias, np.deg2rad(10.0))

        if z > self.z_max:
            error = z - self.z_max
            bias = -0.01 * error
            return max(bias, -np.deg2rad(10.0))

        # Mild centering tendency near boundaries
        mid = 0.5 * (self.z_min + self.z_max)
        half_span = 0.5 * (self.z_max - self.z_min)
        normalized = (z - mid) / max(half_span, 1e-6)

        if abs(normalized) < 0.6:
            return 0.0

        return -0.5 * normalized * np.deg2rad(3.0)

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