import numpy as np


class StaticTarget:
    """
    Static target model.

    The target remains fixed in 3D space.

    Outputs:
        - position history  (N, 3)
        - velocity history  (N, 3)
        - acceleration history (N, 3)

    Parameters
    ----------
    x0 : float
        Fixed x position.
    y0 : float
        Fixed y position.
    z0 : float
        Fixed z position.
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
            # Constant position
            position[k, 0] = self.x0
            position[k, 1] = self.y0
            position[k, 2] = self.z0

            # Zero velocity
            velocity[k, :] = 0.0

            # Zero acceleration
            acceleration[k, :] = 0.0

        self._position = position
        self._velocity = velocity
        self._acceleration = acceleration

    def get_position(self) -> np.ndarray:
        self._check_generated()
        return self._position.copy()

    def get_velocity(self) -> np.ndarray:
        self._check_generated()
        return self._velocity.copy()

    def get_acceleration(self) -> np.ndarray:
        self._check_generated()
        return self._acceleration.copy()

    def get_trajectory_dict(self) -> dict[str, np.ndarray]:
        self._check_generated()
        return {
            "position": self._position.copy(),
            "velocity": self._velocity.copy(),
            "acceleration": self._acceleration.copy(),
        }

    def _check_generated(self) -> None:
        if (
            self._position is None
            or self._velocity is None
            or self._acceleration is None
        ):
            raise RuntimeError(
                "Trajectory not generated yet. Call generate() first."
            )