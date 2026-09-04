import numpy as np


class UnitVectorKalmanTracker:
    """
    Kalman Filter based Direction of Arrival (DOA) Tracker using a unit direction vector.

    This class implements the Step 1 tracker, where the internal state is not formed
    by the angles themselves, but by the 3D unit vector associated with the incoming
    direction of arrival.

    Internal State Vector (x): [u_x, u_y, u_z]^T
    Internal Measurement Vector (z): [u_x_meas, u_y_meas, u_z_meas]^T

    Angular convention used in this tracker:
    - theta: polar angle measured from the +z axis
    - phi: azimuth angle measured in the xy-plane from +x towards +y

    Therefore:

        u_x = sin(theta) * cos(phi)
        u_y = sin(theta) * sin(phi)
        u_z = cos(theta)

    Notes:
    - The tracker receives angular measurements in degrees.
    - The tracker internally filters the unit vector in Cartesian space.
    - The tracker returns filtered angles in degrees for compatibility with the
      rest of the pipeline.
    - A visibility constraint is internally enforced so that the returned polar
      angle always satisfies: 0 <= theta <= 90 deg.
    """

    def __init__(self, q_noise_std: float = 0.02, r_noise_std: float = 0.05):
        """
        Initializes the Unit Vector Kalman Tracker.

        Step 1 uses a random walk model over the unit direction vector:

            x_{k+1} = x_k + w_k

        Args:
            q_noise_std (float): Standard deviation for the process noise (Q).
                                 Controls how much the true direction is allowed
                                 to change between time steps.
            r_noise_std (float): Standard deviation for the measurement noise (R)
                                 in unit-vector space.
        """
        # 1. Initialization flag
        # The tracker is initialized only after the first angular measurement arrives.
        self.is_initialized = False

        # 2. State Vector estimation (3x1)
        # NOTE: This is \hat{x}, the ESTIMATED state, not the true physical state.
        # [u_x_hat, u_y_hat, u_z_hat]^T
        self.x = np.zeros((3, 1))

        # 3. State Uncertainty Covariance Matrix (3x3)
        # Initialized with high uncertainty until the first valid measurement arrives.
        self.P = np.eye(3) * 500.0

        # 4. Dynamic Model Transition Matrix F (3x3)
        # Step 1 assumes a random walk model.
        self.F = np.eye(3)

        # 5. Process Noise Covariance Matrix Q (3x3)
        # This controls how much the direction is allowed to drift between updates.
        q_scalar = q_noise_std ** 2
        self.Q = np.eye(3) * q_scalar

        # 6. Measurement Matrix H (3x3)
        # The measurement is also the unit direction vector.
        self.H = np.eye(3)

        # 7. Measurement Noise Covariance Matrix R (3x3)
        # Simplified isotropic noise model in unit-vector space for Step 1.
        # This can later be replaced by a Jacobian-based angular-to-Cartesian mapping.
        r_scalar = r_noise_std ** 2
        self.R = np.eye(3) * r_scalar

        # Identity matrix cached for efficient update steps
        self.I = np.eye(3)

    # ============================================================
    # GEOMETRY HELPERS
    # ============================================================

    @staticmethod
    def _wrap_360(angle_deg: float) -> float:
        """Wrap angle to [0, 360)."""
        return angle_deg % 360.0

    @staticmethod
    def _angles_to_unit_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
        """
        Converts angular coordinates (theta, phi) in degrees into a 3D unit vector.

        Angular convention:
        - theta: polar angle from +z
        - phi: azimuth in xy-plane from +x towards +y

        Args:
            theta_deg (float): Polar angle in degrees.
            phi_deg (float): Azimuth angle in degrees.

        Returns:
            np.ndarray: 3x1 unit vector [u_x, u_y, u_z]^T.
        """
        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(phi_deg)

        ux = np.sin(theta_rad) * np.cos(phi_rad)
        uy = np.sin(theta_rad) * np.sin(phi_rad)
        uz = np.cos(theta_rad)

        return np.array([[ux], [uy], [uz]], dtype=float)

    @classmethod
    def _unit_vector_to_angles(cls, u: np.ndarray) -> tuple[float, float]:
        """
        Converts a 3D unit vector into angular coordinates (theta, phi) in degrees.

        Angular convention:
        - theta: polar angle from +z
        - phi: azimuth in xy-plane from +x towards +y

        Args:
            u (np.ndarray): 3x1 or length-3 direction vector.

        Returns:
            tuple: (theta_deg, phi_deg)
        """
        u = np.asarray(u, dtype=float).reshape(3, 1)

        ux = float(u[0, 0])
        uy = float(u[1, 0])
        uz = float(u[2, 0])

        # Numerical safety clamp for arccos
        uz = np.clip(uz, -1.0, 1.0)

        theta_deg = float(np.rad2deg(np.arccos(uz)))
        phi_deg = float(np.rad2deg(np.arctan2(uy, ux)))
        phi_deg = cls._wrap_360(phi_deg)

        return theta_deg, phi_deg

    def _normalize_unit_vector(self) -> None:
        """
        Enforces the unit-norm constraint on the internal state vector.

        Since the Kalman Filter operates through linear combinations, the updated
        state may slightly drift away from the unit sphere. This method projects
        the state back onto the sphere.

        If the vector norm is too small, normalization is skipped to avoid division
        by zero. This should only happen before proper initialization.
        """
        norm = float(np.linalg.norm(self.x))

        if norm > 1e-12:
            self.x = self.x / norm

    def _project_to_visible_hemisphere(self) -> None:
        """
        Enforces the visibility constraint for the polar angle:

            0 <= theta <= 90 deg

        Since:
            u_z = cos(theta)

        this is equivalent to enforcing:
            u_z >= 0

        If the estimated state falls outside the visible hemisphere (u_z < 0),
        it is projected onto the visibility boundary theta = 90 deg, i.e. u_z = 0,
        and then re-normalized.

        This implements an internal saturation of the polar angle at 90 degrees.
        """
        # If already in visible hemisphere, nothing to do
        if self.x[2, 0] >= 0.0:
            return

        # Project to boundary theta = 90 deg <=> u_z = 0
        self.x[2, 0] = 0.0

        # Re-normalize only the horizontal part
        horiz_norm = float(np.linalg.norm(self.x[:2, 0]))

        if horiz_norm > 1e-12:
            self.x[0, 0] /= horiz_norm
            self.x[1, 0] /= horiz_norm
            self.x[2, 0] = 0.0
        else:
            # Degenerate case: if horizontal part is numerically zero,
            # choose a default visible boundary direction.
            self.x[0, 0] = 1.0
            self.x[1, 0] = 0.0
            self.x[2, 0] = 0.0

    def _initialize_from_measurement(self, measurement: tuple[float, float]) -> None:
        """
        Initializes the tracker from the first angular measurement.

        Args:
            measurement (tuple): (theta_meas, phi_meas) in degrees.
        """
        z = self._angles_to_unit_vector(measurement[0], measurement[1])

        self.x = z.copy()
        self._normalize_unit_vector()
        self._project_to_visible_hemisphere()

        # High initial uncertainty, but now centered at the first measurement
        self.P = np.eye(3) * 10.0

        self.is_initialized = True

    # ============================================================
    # KALMAN CORE
    # ============================================================

    def predict(self) -> tuple[float, float]:
        """
        Prediction Phase (Time Update).

        Projects the current state and error covariance matrix forward in time
        using the random walk model, BEFORE a new measurement is taken.

        Returns:
            tuple: The predicted (theta, phi) angles in degrees.
                   If the tracker is not initialized yet, returns (nan, nan).
        """
        if not self.is_initialized:
            return float("nan"), float("nan")

        # 1. Project the state ahead: x_{k|k-1} = F * x_{k-1|k-1}
        self.x = np.dot(self.F, self.x)

        # 2. Project the error covariance ahead: P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # 3. Re-normalize and enforce visible hemisphere
        self._normalize_unit_vector()
        self._project_to_visible_hemisphere()

        # Return the predicted angular representation
        return self._unit_vector_to_angles(self.x)

    def update(self, measurement: tuple[float, float]) -> tuple[float, float]:
        """
        Correction Phase (Measurement Update).

        Fuses the predicted state with the new noisy angular DOA measurement to
        obtain the optimal a posteriori estimate in unit-vector space.

        Args:
            measurement (tuple): A tuple containing (theta_meas, phi_meas) in degrees
                                 from the DOA estimator (e.g., MUSIC/Capon).

        Returns:
            tuple: The updated/smoothed (theta, phi) angles in degrees.
        """
        # Deferred initialization using the first measurement
        if not self.is_initialized:
            self._initialize_from_measurement(measurement)
            return self._unit_vector_to_angles(self.x)

        # Format the incoming angular measurement as a 3x1 unit vector (z_k)
        z = self._angles_to_unit_vector(measurement[0], measurement[1])

        # 1. Compute predicted measurement: z_hat_k = H * x_{k|k-1}
        z_pred = np.dot(self.H, self.x)

        # 2. Compute Innovation (Residual): y_k = z_k - z_hat_k
        y = z - z_pred

        # 3. Compute Innovation Covariance: S_k = H * P_{k|k-1} * H^T + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # 4. Compute Kalman Gain: K_k = P_{k|k-1} * H^T * S_k^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 5. Update the State Estimate: x_{k|k} = x_{k|k-1} + K_k * y_k
        self.x = self.x + np.dot(K, y)

        # 6. Update the Error Covariance: P_{k|k} = (I - K_k * H) * P_{k|k-1}
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)

        # 7. Re-normalize and enforce visible hemisphere
        self._normalize_unit_vector()
        self._project_to_visible_hemisphere()

        # Return the final, smoothed angular representation
        return self._unit_vector_to_angles(self.x)

    # ============================================================
    # ACCESSORS
    # ============================================================

    def get_unit_vector(self) -> tuple[float, float, float]:
        """
        Returns the current estimated unit direction vector.

        Returns:
            tuple: (u_x, u_y, u_z)
                   If the tracker is not initialized yet, returns (nan, nan, nan).
        """
        if not self.is_initialized:
            return float("nan"), float("nan"), float("nan")

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

    def get_full_state(self) -> np.ndarray:
        """
        Returns the full 3D estimated state vector [u_x, u_y, u_z]^T.

        Useful for debugging, logging, or building a future observation tensor.

        Returns:
            np.ndarray: The 3x1 state vector.
        """
        return self.x.copy()