import numpy as np


class TruckRoadTarget:
    """
    Truck target model for realistic road-motion simulation.

    The truck moves along a road-like trajectory generated through random
    motion segments, while respecting physical constraints typical of a
    heavy ground vehicle:
        - bounded speed
        - bounded longitudinal acceleration
        - bounded jerk
        - bounded steering angle
        - bounded steering rate
        - smooth road slope changes

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
        # Internal truck / road parameters
        # ============================================================

        # Vehicle geometry
        self.wheelbase = 6.0  # [m]

        # Speed limits
        self.v_min = 8.0      # [m/s]
        self.v_max = 28.0     # [m/s]

        # Initial motion state
        self.v0 = 18.0        # [m/s]
        self.psi0 = self._sample_initial_heading()  # [rad]

        # Longitudinal dynamics
        self.a_min = -2.5      # [m/s^2]
        self.a_max = 1.8       # [m/s^2]
        self.jerk_limit = 0.6  # [m/s^3]

        # Steering dynamics
        self.delta_max = np.deg2rad(8.0)         # [rad]
        self.delta_rate_max = np.deg2rad(1.5)    # [rad/s]

        # Slope / grade dynamics
        self.grade_max = np.deg2rad(4.0)         # [rad]
        self.grade_rate_max = np.deg2rad(0.35)   # [rad/s]

        # Random segment duration
        self.segment_duration_min = 4.0   # [s]
        self.segment_duration_max = 14.0  # [s]

        # Small perturbations
        self.steering_noise_std = np.deg2rad(0.03)
        self.acc_noise_std = 0.03
        self.grade_noise_std = np.deg2rad(0.01)

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

        psi = self.psi0
        v = self.v0
        a_long = 0.0
        delta = 0.0
        grade = 0.0

        target_a_long = 0.0
        target_delta = 0.0
        target_grade = 0.0
        remaining_segment_steps = 0

        prev_vx = None
        prev_vy = None
        prev_vz = None

        for k in range(self.num_steps):
            # ========================================================
            # 1. New random road segment if needed
            # ========================================================
            if remaining_segment_steps <= 0:
                (
                    target_a_long,
                    target_delta,
                    target_grade,
                    remaining_segment_steps,
                ) = self._sample_new_segment(
                    current_speed=v,
                    current_grade=grade,
                )

            remaining_segment_steps -= 1

            # ========================================================
            # 2. Longitudinal acceleration update with jerk limit
            # ========================================================
            a_error = target_a_long - a_long
            max_da = self.jerk_limit * self.dt
            da = np.clip(a_error, -max_da, max_da)
            a_long += da

            a_long += self.rng.normal(0.0, self.acc_noise_std)
            a_long = np.clip(a_long, self.a_min, self.a_max)

            # ========================================================
            # 3. Steering update with steering-rate limit
            # ========================================================
            delta_error = target_delta - delta
            max_ddelta = self.delta_rate_max * self.dt
            ddelta = np.clip(delta_error, -max_ddelta, max_ddelta)
            delta += ddelta

            delta += self.rng.normal(0.0, self.steering_noise_std)
            delta = np.clip(delta, -self.delta_max, self.delta_max)

            # ========================================================
            # 4. Grade update with bounded slope rate
            # ========================================================
            grade_error = target_grade - grade
            max_dgrade = self.grade_rate_max * self.dt
            dgrade = np.clip(grade_error, -max_dgrade, max_dgrade)
            grade += dgrade

            grade += self.rng.normal(0.0, self.grade_noise_std)
            grade = np.clip(grade, -self.grade_max, self.grade_max)

            # ========================================================
            # 5. Speed update
            # ========================================================
            v = v + a_long * self.dt
            v = np.clip(v, self.v_min, self.v_max)

            # ========================================================
            # 6. Heading update using bicycle model
            # ========================================================
            yaw_rate = (v / self.wheelbase) * np.tan(delta)
            psi = self._wrap_angle(psi + yaw_rate * self.dt)

            # ========================================================
            # 7. Velocity components in 3D
            # ========================================================
            v_xy = v * np.cos(grade)
            vx = v_xy * np.cos(psi)
            vy = v_xy * np.sin(psi)
            vz = v * np.sin(grade)

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
                acceleration[k, 0] = a_long * np.cos(psi) * np.cos(grade)
                acceleration[k, 1] = a_long * np.sin(psi) * np.cos(grade)
                acceleration[k, 2] = a_long * np.sin(grade)
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
        current_grade: float,
    ) -> tuple[float, float, float, int]:
        """
        Sample a new random road segment.
        """
        duration = self.rng.uniform(
            self.segment_duration_min,
            self.segment_duration_max,
        )
        num_segment_steps = max(1, int(np.round(duration / self.dt)))

        # Cruise-speed attraction
        cruise_speed = self.rng.uniform(14.0, 24.0)
        speed_error = cruise_speed - current_speed
        target_a_long = 0.18 * speed_error + self.rng.normal(0.0, 0.15)
        target_a_long = np.clip(target_a_long, self.a_min, self.a_max)

        # Mostly straight road, sometimes gentle turn, rarely stronger turn
        steering_mode = self.rng.choice([0, 1, 2], p=[0.65, 0.28, 0.07])

        if steering_mode == 0:
            target_delta = self.rng.normal(0.0, np.deg2rad(0.5))
        elif steering_mode == 1:
            target_delta = self.rng.uniform(
                np.deg2rad(-3.0),
                np.deg2rad(3.0),
            )
        else:
            target_delta = self.rng.uniform(
                np.deg2rad(-6.0),
                np.deg2rad(6.0),
            )

        target_delta = np.clip(target_delta, -self.delta_max, self.delta_max)

        # Small slope changes
        target_grade = current_grade + self.rng.uniform(
            np.deg2rad(-1.0),
            np.deg2rad(1.0),
        )
        target_grade = np.clip(target_grade, -self.grade_max, self.grade_max)

        return target_a_long, target_delta, target_grade, num_segment_steps

    def _sample_initial_heading(self) -> float:
        """
        Sample initial heading.
        """
        return self.rng.uniform(
            np.deg2rad(-20.0),
            np.deg2rad(20.0),
        )

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