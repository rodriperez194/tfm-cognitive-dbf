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
            tracker_factory: function -> returns a new tracker instance
            distance_threshold: gating threshold in unit-vector space
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
            measurements: list of dicts:
                [
                    {"theta": float, "phi": float},
                    ...
                ]
        """

        # --------------------------------------------------------
        # 1. Predict all tracks
        # --------------------------------------------------------
        for track in self.tracks:
            track.predict()

        # --------------------------------------------------------
        # 2. Convert measurements to unit vectors
        # --------------------------------------------------------
        meas_u = [self._angles_to_unit_vector(m["theta"], m["phi"])
                  for m in measurements]

        # --------------------------------------------------------
        # 3. Compute cost matrix (Euclidean in u)
        # --------------------------------------------------------
        cost_matrix = self._compute_cost_matrix(meas_u)

        # --------------------------------------------------------
        # 4. Associate (nearest neighbor + gating)
        # --------------------------------------------------------
        assignments, unassigned_tracks, unassigned_meas = \
            self._associate(cost_matrix)

        # --------------------------------------------------------
        # 5. Update assigned tracks
        # --------------------------------------------------------
        for track_idx, meas_idx in assignments:
            m = measurements[meas_idx]
            self.tracks[track_idx].update_assigned(m["theta"], m["phi"])

        # --------------------------------------------------------
        # 6. Mark missed tracks
        # --------------------------------------------------------
        for track_idx in unassigned_tracks:
            self.tracks[track_idx].mark_missed()

        # --------------------------------------------------------
        # 7. Create new tracks
        # --------------------------------------------------------
        for meas_idx in unassigned_meas:
            m = measurements[meas_idx]
            self._create_track(m["theta"], m["phi"])

        # --------------------------------------------------------
        # 8. Remove deleted tracks
        # --------------------------------------------------------
        self.tracks = [t for t in self.tracks if t.is_alive()]

    # ============================================================
    # ASSOCIATION
    # ============================================================

    def _compute_cost_matrix(self, meas_u):
        """
        Cost = ||u_meas - u_track||
        """
        if len(self.tracks) == 0 or len(meas_u) == 0:
            return np.empty((len(self.tracks), len(meas_u)))

        cost = np.zeros((len(self.tracks), len(meas_u)))

        for i, track in enumerate(self.tracks):
            u_pred = track.get_predicted_measurement_u()

            for j, u_m in enumerate(meas_u):
                cost[i, j] = np.linalg.norm(u_pred - u_m)

        return cost

    def _associate(self, cost_matrix):
        """
        Nearest Neighbor association with gating.
        """

        num_tracks, num_meas = cost_matrix.shape

        assignments = []
        assigned_tracks = set()
        assigned_meas = set()

        # Greedy NN
        while True:
            if cost_matrix.size == 0:
                break

            i, j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            min_cost = cost_matrix[i, j]

            if min_cost > self.distance_threshold:
                break

            if i in assigned_tracks or j in assigned_meas:
                cost_matrix[i, j] = np.inf
                continue

            assignments.append((i, j))
            assigned_tracks.add(i)
            assigned_meas.add(j)

            cost_matrix[i, :] = np.inf
            cost_matrix[:, j] = np.inf

        unassigned_tracks = [i for i in range(num_tracks) if i not in assigned_tracks]
        unassigned_meas = [j for j in range(num_meas) if j not in assigned_meas]

        return assignments, unassigned_tracks, unassigned_meas

    # ============================================================
    # TRACK MANAGEMENT
    # ============================================================

    def _create_track(self, theta, phi):
        tracker = self.tracker_factory()

        track = Track.from_first_measurement(
            track_id=self.next_track_id,
            tracker=tracker,
            theta_meas_deg=theta,
            phi_meas_deg=phi,
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