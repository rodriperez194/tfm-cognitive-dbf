from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment


class AssociationPolicy(ABC):
    """
    Base interface for measurement-to-track association policies.

    Any concrete association policy must implement `associate(...)` and return:
        - assignments: list of (track_idx, meas_idx)
        - unassigned_tracks: list of track indices
        - unassigned_meas: list of measurement indices
    """

    @abstractmethod
    def associate(
        self,
        tracks: list,
        measurements: list[dict],
        angles_to_unit_vector_fn: Callable[[float, float], np.ndarray],
        distance_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Associates measurements to tracks.

        Args:
            tracks: list of Track objects.
            measurements: list of dicts with keys "theta" and "phi" (degrees).
            angles_to_unit_vector_fn: helper function converting angles to a
                                      3x1 unit vector.
            distance_threshold: gating threshold in unit-vector Euclidean space.

        Returns:
            assignments: list of (track_idx, meas_idx) pairs.
            unassigned_tracks: list of track indices without assigned measurement.
            unassigned_meas: list of measurement indices not assigned to any track.
        """
        raise NotImplementedError


class GreedyNearestNeighbor(AssociationPolicy):
    """
    Greedy Nearest Neighbor (NN) association with gating.

    All tracks compete equally for all measurements.
    Associations are resolved iteratively by selecting the
    minimum-distance pair (greedy), without global optimization.
    """

    def associate(
        self,
        tracks: list,
        measurements: list[dict],
        angles_to_unit_vector_fn,
        distance_threshold: float,
    ):
        meas_u = [
            angles_to_unit_vector_fn(m["theta"], m["phi"])
            for m in measurements
        ]

        num_tracks = len(tracks)
        num_meas = len(meas_u)

        if num_tracks == 0 or num_meas == 0:
            return [], list(range(num_tracks)), list(range(num_meas))

        cost = np.zeros((num_tracks, num_meas))

        for i, track in enumerate(tracks):
            u_pred = track.get_predicted_measurement_u()
            for j, u_m in enumerate(meas_u):
                cost[i, j] = np.linalg.norm(u_pred - u_m)

        assignments = []
        assigned_tracks = set()
        assigned_meas = set()

        while True:
            if cost.size == 0:
                break

            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            min_cost = cost[i, j]

            if min_cost > distance_threshold:
                break

            if i in assigned_tracks or j in assigned_meas:
                cost[i, j] = np.inf
                continue

            assignments.append((i, j))
            assigned_tracks.add(i)
            assigned_meas.add(j)

            cost[i, :] = np.inf
            cost[:, j] = np.inf

        unassigned_tracks = [
            i for i in range(num_tracks)
            if i not in assigned_tracks
        ]
        unassigned_meas = [
            j for j in range(num_meas)
            if j not in assigned_meas
        ]

        return assignments, unassigned_tracks, unassigned_meas


class PriorityNearestNeighbor(AssociationPolicy):
    """
    Priority-based Greedy Nearest Neighbor (NN) association.

    Two-pass strategy:
        1) Confirmed + coasting tracks
        2) Tentative tracks

    Within each group, association is solved via greedy NN with gating.
    """

    def associate(
        self,
        tracks: list,
        measurements: list[dict],
        angles_to_unit_vector_fn,
        distance_threshold: float,
    ):
        meas_u = [
            angles_to_unit_vector_fn(m["theta"], m["phi"])
            for m in measurements
        ]

        num_tracks = len(tracks)
        num_meas = len(meas_u)

        priority_indices = [
            i for i, t in enumerate(tracks)
            if t.status in ("confirmed", "coasting")
        ]
        tentative_indices = [
            i for i, t in enumerate(tracks)
            if t.status == "tentative"
        ]

        assignments = []
        assigned_tracks = set()
        assigned_meas = set()

        for group in (priority_indices, tentative_indices):
            if not group:
                continue

            unassigned_meas_list = [
                j for j in range(num_meas)
                if j not in assigned_meas
            ]

            if not unassigned_meas_list:
                break

            cost = np.full((len(group), len(unassigned_meas_list)), np.inf)

            for local_i, track_idx in enumerate(group):
                u_pred = tracks[track_idx].get_predicted_measurement_u()
                for local_j, meas_idx in enumerate(unassigned_meas_list):
                    cost[local_i, local_j] = np.linalg.norm(
                        u_pred - meas_u[meas_idx]
                    )

            local_assigned_tracks = set()
            local_assigned_meas = set()

            while True:
                if cost.size == 0:
                    break

                flat_idx = np.argmin(cost)
                local_i, local_j = np.unravel_index(flat_idx, cost.shape)
                min_cost = cost[local_i, local_j]

                if min_cost > distance_threshold:
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
            i for i in range(num_tracks)
            if i not in assigned_tracks
        ]
        unassigned_meas = [
            j for j in range(num_meas)
            if j not in assigned_meas
        ]

        return assignments, unassigned_tracks, unassigned_meas
    
class HungarianNearestNeighbor(AssociationPolicy):
    """
    Global Nearest Neighbor (GNN) association using the Hungarian algorithm.

    This policy computes the full track-to-measurement cost matrix in
    unit-vector Euclidean space and finds the globally optimal assignment
    minimizing the total association cost.

    Gating is enforced by assigning a very large cost to invalid pairs
    (those whose distance exceeds `distance_threshold`). After the Hungarian
    solution is obtained, invalid assignments are discarded.
    """

    def associate(
        self,
        tracks: list,
        measurements: list[dict],
        angles_to_unit_vector_fn: Callable[[float, float], np.ndarray],
        distance_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        meas_u = [
            angles_to_unit_vector_fn(m["theta"], m["phi"])
            for m in measurements
        ]

        num_tracks = len(tracks)
        num_meas = len(meas_u)

        if num_tracks == 0 or num_meas == 0:
            return [], list(range(num_tracks)), list(range(num_meas))

        cost = np.zeros((num_tracks, num_meas), dtype=float)

        for i, track in enumerate(tracks):
            u_pred = track.get_predicted_measurement_u()
            for j, u_m in enumerate(meas_u):
                cost[i, j] = np.linalg.norm(u_pred - u_m)

        gated_cost = cost.copy()
        large_cost = 1e9
        gated_cost[gated_cost > distance_threshold] = large_cost

        row_ind, col_ind = linear_sum_assignment(gated_cost)

        assignments = []
        assigned_tracks = set()
        assigned_meas = set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] <= distance_threshold:
                assignments.append((i, j))
                assigned_tracks.add(i)
                assigned_meas.add(j)

        unassigned_tracks = [
            i for i in range(num_tracks)
            if i not in assigned_tracks
        ]
        unassigned_meas = [
            j for j in range(num_meas)
            if j not in assigned_meas
        ]

        return assignments, unassigned_tracks, unassigned_meas
    
class MahalanobisHungarian(AssociationPolicy):
    """
    Global Nearest Neighbor (GNN) association using the Hungarian algorithm
    with Mahalanobis distance in unit-vector space.

    The association cost between track i and measurement j is:

        d_M(i, j) = sqrt((u_meas - u_pred)^T S^{-1} (u_meas - u_pred))

    where:
        - u_pred is the predicted unit-vector measurement of the track
        - u_meas is the measured unit-vector
        - S is the predicted measurement covariance in unit-vector space

    Gating is enforced by assigning a very large cost to invalid pairs
    (those whose Mahalanobis distance exceeds `distance_threshold`).
    After the Hungarian solution is obtained, invalid assignments are discarded.
    """

    def associate(
        self,
        tracks: list,
        measurements: list[dict],
        angles_to_unit_vector_fn: Callable[[float, float], np.ndarray],
        distance_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        meas_u = [
            angles_to_unit_vector_fn(m["theta"], m["phi"])
            for m in measurements
        ]

        num_tracks = len(tracks)
        num_meas = len(meas_u)

        if num_tracks == 0 or num_meas == 0:
            return [], list(range(num_tracks)), list(range(num_meas))

        cost = np.full((num_tracks, num_meas), np.inf, dtype=float)

        for i, track in enumerate(tracks):
            u_pred = np.asarray(
                track.get_predicted_measurement_u(),
                dtype=float,
            ).reshape(3, 1)

            S = np.asarray(
                track.get_predicted_measurement_cov_u(),
                dtype=float,
            )

            if S.shape != (3, 3):
                raise ValueError(
                    f"Track {i} returned covariance with shape {S.shape}, "
                    "but MahalanobisHungarian expects a 3x3 covariance in "
                    "unit-vector measurement space."
                )

            S_inv = np.linalg.pinv(S)

            for j, u_m in enumerate(meas_u):
                u_m = np.asarray(u_m, dtype=float).reshape(3, 1)
                delta = u_m - u_pred

                d2 = (delta.T @ S_inv @ delta).item()
                d2 = max(d2, 0.0)  # numerical safety
                cost[i, j] = np.sqrt(d2)

        gated_cost = cost.copy()
        large_cost = 1e9
        gated_cost[gated_cost > distance_threshold] = large_cost

        row_ind, col_ind = linear_sum_assignment(gated_cost)

        assignments = []
        assigned_tracks = set()
        assigned_meas = set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] <= distance_threshold:
                assignments.append((i, j))
                assigned_tracks.add(i)
                assigned_meas.add(j)

        unassigned_tracks = [
            i for i in range(num_tracks)
            if i not in assigned_tracks
        ]
        unassigned_meas = [
            j for j in range(num_meas)
            if j not in assigned_meas
        ]

        return assignments, unassigned_tracks, unassigned_meas
    
class ScoreBasedHungarian(AssociationPolicy):
    """
    Global Nearest Neighbor (GNN) association using the Hungarian algorithm
    with a custom score-based cost.

    The association cost between track i and measurement j is:

        cost(i, j) =
            distance_weight * d_u(i, j)
            + miss_weight * misses_i
            + tentative_penalty * I(track_i is tentative)

    where:
        - d_u(i, j) is the Euclidean distance in unit-vector space
        - misses_i is the current miss count of the track
        - I(.) is the indicator function

    This allows the association policy to combine geometric proximity with
    simple track-management heuristics.

    Gating is first applied on geometric distance only. Pairs whose distance
    exceeds `distance_threshold` are marked as invalid with a very large cost.
    After the Hungarian solution is obtained, invalid assignments are discarded.
    """

    def __init__(
        self,
        distance_weight: float = 1.0,
        miss_weight: float = 0.0,
        tentative_penalty: float = 0.0,
    ) -> None:
        self.distance_weight = float(distance_weight)
        self.miss_weight = float(miss_weight)
        self.tentative_penalty = float(tentative_penalty)

    def associate(
        self,
        tracks: list,
        measurements: list[dict],
        angles_to_unit_vector_fn: Callable[[float, float], np.ndarray],
        distance_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        meas_u = [
            angles_to_unit_vector_fn(m["theta"], m["phi"])
            for m in measurements
        ]

        num_tracks = len(tracks)
        num_meas = len(meas_u)

        if num_tracks == 0 or num_meas == 0:
            return [], list(range(num_tracks)), list(range(num_meas))

        geom_dist = np.zeros((num_tracks, num_meas), dtype=float)
        cost = np.zeros((num_tracks, num_meas), dtype=float)

        for i, track in enumerate(tracks):
            u_pred = track.get_predicted_measurement_u()

            miss_penalty_i = self.miss_weight * track.misses
            tentative_penalty_i = (
                self.tentative_penalty
                if track.status == "tentative"
                else 0.0
            )

            track_penalty_i = miss_penalty_i + tentative_penalty_i

            for j, u_m in enumerate(meas_u):
                d = np.linalg.norm(u_pred - u_m)
                geom_dist[i, j] = d

                cost[i, j] = (
                    self.distance_weight * d
                    + track_penalty_i
                )

        gated_cost = cost.copy()
        large_cost = 1e9
        gated_cost[geom_dist > distance_threshold] = large_cost

        row_ind, col_ind = linear_sum_assignment(gated_cost)

        assignments = []
        assigned_tracks = set()
        assigned_meas = set()

        for i, j in zip(row_ind, col_ind):
            if geom_dist[i, j] <= distance_threshold:
                assignments.append((i, j))
                assigned_tracks.add(i)
                assigned_meas.add(j)

        unassigned_tracks = [
            i for i in range(num_tracks)
            if i not in assigned_tracks
        ]
        unassigned_meas = [
            j for j in range(num_meas)
            if j not in assigned_meas
        ]

        return assignments, unassigned_tracks, unassigned_meas