import numpy as np


class Dummy:
    """
    Dummy target model for interface testing.

    The target moves with constant velocity in the horizontal plane z = z0.

    Outputs:
        - position history  (N, 3)
        - velocity history  (N, 3)
        - acceleration history (N, 3)

    Parameters
    ----------
    x0 : float
        Initial x position.
    y0 : float
        Initial y position.
    z0 : float
        Constant height.
    dt : float
        Simulation time step.
    num_steps : int
        Number of simulation steps.
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        z0: float,
        dt: float,
        num_steps: int,
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

        # Internal motion parameters (deterministic for testing)
        self.vx = 20.0
        self.vy = 10.0
        self.vz = 0.0

        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0

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

        for k in range(self.num_steps):
            t = k * self.dt

            # Position (MRU in XY, constant Z)
            position[k, 0] = self.x0 + self.vx * t
            position[k, 1] = self.y0 + self.vy * t
            position[k, 2] = self.z0

            # Velocity (constant)
            velocity[k, 0] = self.vx
            velocity[k, 1] = self.vy
            velocity[k, 2] = self.vz

            # Acceleration (zero)
            acceleration[k, 0] = self.ax
            acceleration[k, 1] = self.ay
            acceleration[k, 2] = self.az

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