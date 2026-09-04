import numpy as np
from .track import Track


class MultiTargetTracker:
    """
    Multi-target tracker managing multiple Track objects.

    Responsibilities:
        - Predict all tracks
        - Associate measurements to tracks
        - Update tracks
        - Create new tracks
        - Delete dead tracks

    Association strategy:
        Confirmed and coasting tracks have priority over tentative tracks.
        Within each priority group, a greedy nearest-neighbour algorithm
        resolves assignments. This prevents tentative tracks from stealing
        measurements from already-confirmed targets.
    """

    def __init__(
        self,
        tracker_factory,
        distance_threshold: float = 0.3,
        confirm_hits: int = 3,
        max_misses: int = 5,
        tentative_max_misses: int = 2,
    ):
        """
        Args:
            tracker_factory: callable with no arguments that returns a new
                             tracker instance (CV / CA / IMM).
            distance_threshold: gating threshold in unit-vector Euclidean
                                 space. Pairs whose distance exceeds this
                                 value are never associated.
            confirm_hits: number of consecutive hits required to transition
                          a track from tentative to confirmed.
            max_misses: maximum consecutive misses before a confirmed or
                        coasting track is deleted.
            tentative_max_misses: maximum consecutive misses before a
                                  tentative track is deleted.
        """
        self.tracker_factory = tracker_factory
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
        # 2. Convert measurements to unit vectors
        # --------------------------------------------------------
        meas_u = [
            self._angles_to_unit_vector(m["theta"], m["phi"])
            for m in measurements
        ]

        # --------------------------------------------------------
        # 3. Associate with confirmed/coasting priority
        # --------------------------------------------------------
        assignments, unassigned_tracks, unassigned_meas = \
            self._associate_with_priority(meas_u)

        # --------------------------------------------------------
        # 4. Update assigned tracks
        # --------------------------------------------------------
        for track_idx, meas_idx in assignments:
            m = measurements[meas_idx]
            self.tracks[track_idx].update_assigned(m["theta"], m["phi"])

        # --------------------------------------------------------
        # 5. Mark missed tracks
        # --------------------------------------------------------
        for track_idx in unassigned_tracks:
            self.tracks[track_idx].mark_missed()

        # --------------------------------------------------------
        # 6. Create new tracks from unassigned measurements
        # --------------------------------------------------------
        for meas_idx in unassigned_meas:
            m = measurements[meas_idx]
            self._create_track(m["theta"], m["phi"])

        # --------------------------------------------------------
        # 7. Remove deleted tracks
        # --------------------------------------------------------
        self.tracks = [t for t in self.tracks if t.is_alive()]

    # ============================================================
    # ASSOCIATION
    # ============================================================

    def _associate_with_priority(self, meas_u):
        """
        Two-pass greedy nearest-neighbour association with status priority.

        Pass 1 — confirmed and coasting tracks compete for measurements first.
        Pass 2 — tentative tracks compete for the remaining unassigned
                 measurements only.

        This guarantees that a tentative track can never steal a measurement
        from a confirmed or coasting track, which was the root cause of
        premature track death in the original single-pass implementation.

        Args:
            meas_u: list of unit-vector np.ndarray of shape (3, 1),
                    one per measurement.

        Returns:
            assignments: list of (track_idx, meas_idx) pairs.
            unassigned_tracks: list of track indices with no assignment.
            unassigned_meas: list of measurement indices with no assignment.
        """
        num_tracks = len(self.tracks)
        num_meas = len(meas_u)

        # Split track indices into priority groups
        priority_indices = [
            i for i, t in enumerate(self.tracks)
            if t.status in ("confirmed", "coasting")
        ]
        tentative_indices = [
            i for i, t in enumerate(self.tracks)
            if t.status == "tentative"
        ]

        assignments = []
        assigned_tracks = set()
        assigned_meas = set()

        # Run greedy NN on each priority group in order
        for group in (priority_indices, tentative_indices):
            if not group:
                continue

            # Build cost sub-matrix for this group vs all unassigned measurements
            unassigned_meas_list = [
                j for j in range(num_meas) if j not in assigned_meas
            ]

            if not unassigned_meas_list:
                break

            # cost[local_i, local_j] = distance(track_group[i], meas[j])
            cost = np.full((len(group), len(unassigned_meas_list)), np.inf)

            for local_i, track_idx in enumerate(group):
                u_pred = self.tracks[track_idx].get_predicted_measurement_u()
                for local_j, meas_idx in enumerate(unassigned_meas_list):
                    cost[local_i, local_j] = np.linalg.norm(
                        u_pred - meas_u[meas_idx]
                    )

            # Greedy NN on the sub-matrix
            local_assigned_tracks = set()
            local_assigned_meas = set()

            while True:
                if cost.size == 0:
                    break

                flat_idx = np.argmin(cost)
                local_i, local_j = np.unravel_index(flat_idx, cost.shape)
                min_cost = cost[local_i, local_j]

                if min_cost > self.distance_threshold:
                    break

                if local_i in local_assigned_tracks or local_j in local_assigned_meas:
                    cost[local_i, local_j] = np.inf
                    continue

                track_idx = group[local_i]
                meas_idx = unassigned_meas_list[local_j]

                assignments.append((track_idx, meas_idx))
                assigned_tracks.add(track_idx)
                assigned_meas.add(meas_idx)
                local_assigned_tracks.add(local_i)
                local_assigned_meas.add(local_j)

                cost[local_i, :] = np.inf
                cost[:, local_j] = np.inf

        unassigned_tracks = [
            i for i in range(num_tracks) if i not in assigned_tracks
        ]
        unassigned_meas = [
            j for j in range(num_meas) if j not in assigned_meas
        ]

        return assignments, unassigned_tracks, unassigned_meas

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
        ], dtype=float)

    # ============================================================
    # PUBLIC GETTERS
    # ============================================================

    def get_active_tracks(self):
        """Returns all tracks that are not deleted."""
        return [t for t in self.tracks if t.is_alive()]

    def get_confirmed_tracks(self):
        """Returns only confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed()]