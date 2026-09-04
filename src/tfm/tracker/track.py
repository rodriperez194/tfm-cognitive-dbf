from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


TrackStatus = Literal["tentative", "confirmed", "coasting", "deleted"]


@dataclass
class Track:
    """
    Multi-target track wrapper around a single-target tracker.

    A Track is a persistent hypothesis that one physical target exists and is
    being followed over time. The internal estimation is delegated to the
    wrapped tracker (CV / CA / IMM), while this class handles:
        - track identity
        - lifecycle management
        - hit/miss counters
        - history storage
        - predicted measurement exposure

    Expected tracker interface:
        - predict() -> None
        - update(theta_meas_deg: float, phi_meas_deg: float) -> None
        - get_unit_vector() -> np.ndarray shape (3, 1)
        - get_angles() -> tuple[float, float]
        - get_covariance() -> np.ndarray
        - get_full_state() -> np.ndarray

    Notes:
        - This class does NOT perform data association.
        - This class does NOT create or delete other tracks.
        - This class assumes angular I/O compatibility with the current MUSIC
          pipeline, while the wrapped tracker internally works in unit-vector
          space.
    """

    track_id: int
    tracker: Any
    confirm_hits: int = 3
    max_misses: int = 5
    tentative_max_misses: int = 2

    status: TrackStatus = "tentative"
    age: int = 0
    hits: int = 0
    misses: int = 0
    total_updates: int = 0
    total_predictions: int = 0

    last_prediction_u: np.ndarray | None = None
    last_update_u: np.ndarray | None = None
    last_measurement_theta_deg: float | None = None
    last_measurement_phi_deg: float | None = None
    was_updated_in_current_step: bool = False

    history_u: list[np.ndarray] = field(default_factory=list)
    history_theta_deg: list[float] = field(default_factory=list)
    history_phi_deg: list[float] = field(default_factory=list)
    history_status: list[str] = field(default_factory=list)

    # ============================================================
    # INITIALIZATION
    # ============================================================

    @classmethod
    def from_first_measurement(
        cls,
        track_id: int,
        tracker: Any,
        theta_meas_deg: float,
        phi_meas_deg: float,
        confirm_hits: int = 3,
        max_misses: int = 5,
        tentative_max_misses: int = 2,
    ) -> "Track":
        """
        Creates and initializes a Track from its first assigned measurement.

        This is the standard birth mechanism for a new target hypothesis.
        """
        track = cls(
            track_id=track_id,
            tracker=tracker,
            confirm_hits=confirm_hits,
            max_misses=max_misses,
            tentative_max_misses=tentative_max_misses,
        )

        track.update_assigned(theta_meas_deg, phi_meas_deg, is_birth=True)
        return track

    # ============================================================
    # CORE LIFECYCLE METHODS
    # ============================================================

    def predict(self) -> None:
        """
        Runs the prediction step of the wrapped tracker.

        This should be called once per scan before data association.
        """
        if self.status == "deleted":
            return

        self.tracker.predict()
        self.total_predictions += 1
        self.age += 1
        self.was_updated_in_current_step = False

        self.last_prediction_u = self._safe_copy_vector(self.tracker.get_unit_vector())

        # Keep current state in history if desired only after update/miss.
        # Here we do not append to history yet to avoid double-counting one scan.

    def update_assigned(
        self,
        theta_meas_deg: float,
        phi_meas_deg: float,
        is_birth: bool = False,
    ) -> None:
        """
        Updates the wrapped tracker with an assigned angular measurement.

        Args:
            theta_meas_deg: Measured polar angle in degrees.
            phi_meas_deg: Measured azimuth angle in degrees.
            is_birth: True when this is the first measurement of a new track.
        """
        if self.status == "deleted":
            return

        self.tracker.update(theta_meas_deg, phi_meas_deg)

        self.hits += 1
        self.misses = 0
        self.total_updates += 1
        self.was_updated_in_current_step = True

        if is_birth and self.age == 0:
            self.age = 1

        self.last_measurement_theta_deg = float(theta_meas_deg)
        self.last_measurement_phi_deg = float(phi_meas_deg)
        self.last_update_u = self._safe_copy_vector(self.tracker.get_unit_vector())

        self._update_status_after_hit()
        self._append_history()

    def mark_missed(self) -> None:
        """
        Marks the track as unassigned in the current scan.

        The internal tracker is not corrected with a measurement. The track
        survives only by prediction.
        """
        if self.status == "deleted":
            return

        self.misses += 1
        self.was_updated_in_current_step = False

        self._update_status_after_miss()
        self._append_history()

    # ============================================================
    # STATUS MANAGEMENT
    # ============================================================

    def _update_status_after_hit(self) -> None:
        """
        Updates lifecycle status after a successful measurement association.
        """
        if self.status == "deleted":
            return

        if self.hits >= self.confirm_hits:
            self.status = "confirmed"
        else:
            self.status = "tentative"

    def _update_status_after_miss(self) -> None:
        """
        Updates lifecycle status after a missed detection.
        """
        if self.status == "deleted":
            return

        if self.status == "tentative":
            if self.misses > self.tentative_max_misses:
                self.status = "deleted"
            else:
                self.status = "tentative"
            return

        if self.status in ("confirmed", "coasting"):
            if self.misses > self.max_misses:
                self.status = "deleted"
            else:
                self.status = "coasting"

    # ============================================================
    # HISTORY
    # ============================================================

    def _append_history(self) -> None:
        """
        Appends the current filtered estimate and status to the track history.
        """
        if self.status == "deleted":
            self.history_status.append(self.status)
            return

        u = self._safe_copy_vector(self.tracker.get_unit_vector())
        theta_deg, phi_deg = self.tracker.get_angles()

        self.history_u.append(u)
        self.history_theta_deg.append(float(theta_deg))
        self.history_phi_deg.append(float(phi_deg))
        self.history_status.append(self.status)

    # ============================================================
    # GETTERS FOR ASSOCIATION / MANAGEMENT
    # ============================================================

    def get_predicted_measurement_u(self) -> np.ndarray:
        """
        Returns the predicted measurement in unit-vector space.

        In the current architecture, the measurement model directly observes
        the direction vector, so the predicted measurement is the current
        tracker estimate of u.
        """
        return self._safe_copy_vector(self.tracker.get_unit_vector())

    def get_angles(self) -> tuple[float, float]:
        """
        Returns the current filtered angular estimate.

        The wrapped tracker is responsible for applying the angular output
        constraints already defined in your pipeline.
        """
        theta_deg, phi_deg = self.tracker.get_angles()
        return float(theta_deg), float(phi_deg)

    def get_unit_vector(self) -> np.ndarray:
        """
        Returns the current filtered unit vector estimate.
        """
        return self._safe_copy_vector(self.tracker.get_unit_vector())

    def get_covariance(self) -> np.ndarray:
        """
        Returns the current covariance matrix of the wrapped tracker.
        """
        return np.asarray(self.tracker.get_covariance(), dtype=float).copy()

    def get_full_state(self) -> np.ndarray:
        """
        Returns the current full state of the wrapped tracker.
        """
        return np.asarray(self.tracker.get_full_state(), dtype=float).copy()

    def get_predicted_measurement_cov_u(self) -> np.ndarray:
        """
        Returns the predicted measurement covariance in unit-vector space.
    
        Assumes that the wrapped tracker state is ordered with the observable
        unit-vector components in the first three positions, e.g.:
    
            x = [u_x, u_y, u_z, ...]
    
        Therefore, the covariance of the predicted measurement u is taken as
        the upper-left 3x3 block of the full state covariance.
        """
        P = np.asarray(self.tracker.get_covariance(), dtype=float).copy()
    
        if P.shape[0] < 3 or P.shape[1] < 3:
            raise ValueError(
                f"Tracker covariance has shape {P.shape}, but at least a 3x3 "
                "matrix is required to extract unit-vector covariance."
            )
    
        return P[:3, :3]

    # ============================================================
    # BOOLEAN HELPERS
    # ============================================================

    def is_tentative(self) -> bool:
        return self.status == "tentative"

    def is_confirmed(self) -> bool:
        return self.status == "confirmed"

    def is_coasting(self) -> bool:
        return self.status == "coasting"

    def is_deleted(self) -> bool:
        return self.status == "deleted"

    def is_alive(self) -> bool:
        return self.status != "deleted"

    # ============================================================
    # SUMMARY / DEBUG
    # ============================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a compact serializable summary of the track.
        """
        theta_deg, phi_deg = self.get_angles()
        u = self.get_unit_vector().reshape(-1)

        return {
            "track_id": self.track_id,
            "status": self.status,
            "age": self.age,
            "hits": self.hits,
            "misses": self.misses,
            "total_predictions": self.total_predictions,
            "total_updates": self.total_updates,
            "theta_deg": float(theta_deg),
            "phi_deg": float(phi_deg),
            "u_x": float(u[0]),
            "u_y": float(u[1]),
            "u_z": float(u[2]),
            "was_updated_in_current_step": self.was_updated_in_current_step,
        }

    def __repr__(self) -> str:
        theta_deg, phi_deg = self.get_angles()
        return (
            f"Track(track_id={self.track_id}, status='{self.status}', "
            f"age={self.age}, hits={self.hits}, misses={self.misses}, "
            f"theta_deg={theta_deg:.3f}, phi_deg={phi_deg:.3f})"
        )

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    @staticmethod
    def _safe_copy_vector(u: np.ndarray) -> np.ndarray:
        """
        Returns a safe copy of a 3x1 vector.
        """
        return np.asarray(u, dtype=float).reshape(3, 1).copy()