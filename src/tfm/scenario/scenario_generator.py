from __future__ import annotations

import numpy as np

from tfm.selection.scenario_signature import build_target_signature


class ScenarioGenerator:
    """
    Scenario generator for cognitive beamforming simulations.

    This class builds a full electromagnetic scenario including:
        - one fixed desired source
        - multiple dynamic jammers
        - one fixed array position

    It generates:
        - desired-source positions over time
        - jammer trajectories
        - DOAs (theta, phi) for desired source and jammers
        - signal powers
        - environment parameters

    Angular convention:
        - theta: polar angle from +z
        - phi: azimuth from +x towards +y, wrapped to [0, 2*pi)

    Notes
    -----
    - The desired source is static for now.
    - Jammers are motion-model objects such as AircraftTarget, DroneTarget,
      Dummy, or TruckRoadTarget.
    - Each jammer object must implement:
          generate()
          get_position()
    """

    def __init__(
        self,
        desired_source_position: np.ndarray,
        jammers: list,
        array_position: np.ndarray,
        num_steps: int,
        dt: float,
        desired_power: float,
        jammer_powers: list[float],
        noise_power: float,
    ) -> None:
        self.desired_source_position = np.asarray(
            desired_source_position, dtype=float
        ).reshape(3)

        self.jammers = list(jammers)

        self.array_position = np.asarray(
            array_position, dtype=float
        ).reshape(3)

        self.num_steps = int(num_steps)
        self.dt = float(dt)

        self.desired_power = float(desired_power)
        self.jammer_powers = [float(p) for p in jammer_powers]
        self.noise_power = float(noise_power)

        self._validate_inputs()

    # ============================================================
    # Public API
    # ============================================================

    def generate(self) -> dict:
        """
        Generate the full scenario.

        Returns
        -------
        dict
            Structure:
            {
                "desired": {
                    "position": (N, 3),
                    "doa": {
                        "theta": (N,),
                        "phi": (N,),
                    },
                    "power": float,
                },
                "jammers": [
                    {
                        "position": (N, 3),
                        "doa": {
                            "theta": (N,),
                            "phi": (N,),
                        },
                        "power": float,
                        "class_name": str,
                    },
                    ...
                ],
                "environment": {
                    "noise_power": float,
                },
                "metadata": {
                    "num_steps": int,
                    "dt": float,
                    "array_position": (3,),
                    "num_jammers": int,
                    "target_signature": str,
                }
            }
        """

        # --------------------------------------------------------
        # 1. Desired source data (static over time)
        # --------------------------------------------------------
        desired_positions = np.tile(
            self.desired_source_position, (self.num_steps, 1)
        )

        desired_doa = self._compute_doa(desired_positions)

        desired_data = {
            "position": desired_positions,
            "doa": desired_doa,
            "power": self.desired_power,
        }

        # --------------------------------------------------------
        # 2. Jammer data
        # --------------------------------------------------------
        jammer_data = []

        for jammer, jammer_power in zip(self.jammers, self.jammer_powers):
            jammer.generate()

            jammer_positions = jammer.get_position()
            jammer_positions = np.asarray(jammer_positions, dtype=float)

            self._validate_jammer_positions(jammer_positions, jammer)

            jammer_doa = self._compute_doa(jammer_positions)

            jammer_data.append(
                {
                    "position": jammer_positions,
                    "doa": jammer_doa,
                    "power": jammer_power,
                    "class_name": jammer.__class__.__name__,
                }
            )

        target_signature = build_target_signature(self.jammers)

        # --------------------------------------------------------
        # 3. Full scenario
        # --------------------------------------------------------
        scenario = {
            "desired": desired_data,
            "jammers": jammer_data,
            "environment": {
                "noise_power": self.noise_power,
            },
            "metadata": {
                "num_steps": self.num_steps,
                "dt": self.dt,
                "array_position": self.array_position.copy(),
                "num_jammers": len(self.jammers),
                "target_signature": target_signature,
            },
        }

        return scenario

    # ============================================================
    # Internal methods
    # ============================================================

    def _compute_doa(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute DOA (theta, phi) from Cartesian positions.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (N, 3).

        Returns
        -------
        dict[str, np.ndarray]
            {
                "theta": np.ndarray of shape (N,),
                "phi": np.ndarray of shape (N,),
            }
        """
        relative_positions = positions - self.array_position

        x = relative_positions[:, 0]
        y = relative_positions[:, 1]
        z = relative_positions[:, 2]

        r = np.linalg.norm(relative_positions, axis=1)

        eps = 1e-12
        r_safe = np.maximum(r, eps)

        cos_theta = np.clip(z / r_safe, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        phi = np.arctan2(y, x)
        phi = np.mod(phi, 2.0 * np.pi)

        return {
            "theta": theta,
            "phi": phi,
        }

    def _validate_inputs(self) -> None:
        """
        Validate constructor inputs.
        """
        if self.num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")

        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

        if len(self.jammers) != len(self.jammer_powers):
            raise ValueError(
                "Number of jammer objects and jammer_powers must match."
            )

        if self.desired_power < 0.0:
            raise ValueError("desired_power must be non-negative.")

        if self.noise_power < 0.0:
            raise ValueError("noise_power must be non-negative.")

        if any(p < 0.0 for p in self.jammer_powers):
            raise ValueError("All jammer powers must be non-negative.")

        for jammer in self.jammers:
            if not hasattr(jammer, "generate"):
                raise ValueError(
                    f"{jammer.__class__.__name__} must implement generate()."
                )
            if not hasattr(jammer, "get_position"):
                raise ValueError(
                    f"{jammer.__class__.__name__} must implement get_position()."
                )

    def _validate_jammer_positions(
        self,
        jammer_positions: np.ndarray,
        jammer: object,
    ) -> None:
        """
        Validate jammer trajectory output.
        """
        if jammer_positions.ndim != 2 or jammer_positions.shape[1] != 3:
            raise ValueError(
                f"{jammer.__class__.__name__}.get_position() must return "
                "an array of shape (N, 3)."
            )

        if jammer_positions.shape[0] != self.num_steps:
            raise ValueError(
                f"{jammer.__class__.__name__}.get_position() returned "
                f"{jammer_positions.shape[0]} steps, but ScenarioGenerator "
                f"expects {self.num_steps}."
            )