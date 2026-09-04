import numpy as np
from .track import Track


class MultiTargetTracker:
    """
    Multi-target tracker managing multiple Track objects.

    Responsibilities:
        - Predict all tracks
        - Delegate measurement-to-track association to an external policy
        - Update assigned tracks
        - Mark missed tracks
        - Create new tracks from unassigned measurements
        - Delete dead tracks

    The association logic is intentionally externalized so that different
    association policies can be plugged in without modifying this class.
    """

    def __init__(
        self,
        tracker_factory,
        association_policy,
        distance_threshold: float = 0.3,
        confirm_hits: int = 3,
        max_misses: int = 5,
        tentative_max_misses: int = 2,
    ):
        """
        Args:
            tracker_factory: callable with no arguments that returns a new
                             tracker instance (CV / CA / IMM).
            association_policy: object implementing an `associate(...)`
                                method for measurement-to-track assignment.
            distance_threshold: gating threshold in unit-vector Euclidean
                                space. Passed to the association policy.
            confirm_hits: number of consecutive hits required to transition
                          a track from tentative to confirmed.
            max_misses: maximum consecutive misses before a confirmed or
                        coasting track is deleted.
            tentative_max_misses: maximum consecutive misses before a
                                  tentative track is deleted.
        """
        self.tracker_factory = tracker_factory
        self.association_policy = association_policy

        self.distance_threshold = distance_threshold
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.tentative_max_misses = tentative_max_misses

        self.tracks = []
        self.next_track_id = 0

    # ============================================================
    # MAIN STEP
    # ============================================================

    def step(self, measurements):
        """
        One full multi-target tracking step.

        Args:
            measurements: list of dicts with keys "theta" and "phi" (degrees):
                [{"theta": float, "phi": float}, ...]
        """

        # --------------------------------------------------------
        # 1. Predict all tracks
        # --------------------------------------------------------
        for track in self.tracks:
            track.predict()

        # --------------------------------------------------------
        # 2. Delegate association to the selected policy
        # --------------------------------------------------------
        assignments, unassigned_tracks, unassigned_meas = (
            self.association_policy.associate(
                tracks=self.tracks,
                measurements=measurements,
                angles_to_unit_vector_fn=self._angles_to_unit_vector,
                distance_threshold=self.distance_threshold,
            )
        )

        # --------------------------------------------------------
        # 3. Update assigned tracks
        # --------------------------------------------------------
        for track_idx, meas_idx in assignments:
            m = measurements[meas_idx]
            self.tracks[track_idx].update_assigned(m["theta"], m["phi"])

        # --------------------------------------------------------
        # 4. Mark missed tracks
        # --------------------------------------------------------
        for track_idx in unassigned_tracks:
            self.tracks[track_idx].mark_missed()

        # --------------------------------------------------------
        # 5. Create new tracks from unassigned measurements
        # --------------------------------------------------------
        for meas_idx in unassigned_meas:
            m = measurements[meas_idx]
            self._create_track(m["theta"], m["phi"])

        # --------------------------------------------------------
        # 6. Remove deleted tracks
        # --------------------------------------------------------
        self.tracks = [t for t in self.tracks if t.is_alive()]

    # ============================================================
    # TRACK MANAGEMENT
    # ============================================================

    def _create_track(self, theta_deg, phi_deg):
        """
        Instantiates a new tentative track from a birth measurement.

        Args:
            theta_deg: polar angle of the birth measurement in degrees.
            phi_deg: azimuth angle of the birth measurement in degrees.
        """
        tracker = self.tracker_factory()

        track = Track.from_first_measurement(
            track_id=self.next_track_id,
            tracker=tracker,
            theta_meas_deg=theta_deg,
            phi_meas_deg=phi_deg,
            confirm_hits=self.confirm_hits,
            max_misses=self.max_misses,
            tentative_max_misses=self.tentative_max_misses,
        )

        self.tracks.append(track)
        self.next_track_id += 1

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _angles_to_unit_vector(theta_deg, phi_deg):
        """
        Converts (theta, phi) in degrees to a 3x1 unit vector.

        Angular convention:
            theta: polar angle from +z axis.
            phi: azimuth in xy-plane from +x towards +y.
        """
        theta = np.deg2rad(theta_deg)
        phi = np.deg2rad(phi_deg)

        return np.array([
            [np.sin(theta) * np.cos(phi)],
            [np.sin(theta) * np.sin(phi)],
            [np.cos(theta)]
        ])

    # ============================================================
    # DEBUG / OUTPUT
    # ============================================================

    def get_active_tracks(self):
        return [t for t in self.tracks if t.is_alive()]

    def get_confirmed_tracks(self):
        return [t for t in self.tracks if t.is_confirmed()]