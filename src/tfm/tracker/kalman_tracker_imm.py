import numpy as np


class IMMKalmanTracker:
    """
    Interacting Multiple Model (IMM) Kalman tracker for DOA tracking using
    a 3D unit-direction vector representation.

    This tracker combines two linear dynamic models:

        Model 1 (CV):
            x_cv = [u_x, u_y, u_z, du_x, du_y, du_z]^T

        Model 2 (CA):
            x_ca = [u_x, u_y, u_z, du_x, du_y, du_z,
                    ddu_x, ddu_y, ddu_z]^T

    Measurement model for both:
        z = [u_x_meas, u_y_meas, u_z_meas]^T

    Angular convention:
    - theta: polar angle measured from the +z axis
    - phi: azimuth angle measured in the xy-plane from +x towards +y

    Therefore:

        u_x = sin(theta) * cos(phi)
        u_y = sin(theta) * sin(phi)
        u_z = cos(theta)

    Notes:
    - The tracker receives angular measurements in degrees.
    - The tracker internally filters the unit vector in Cartesian space.
    - The tracker returns filtered angles in degrees for compatibility with
      the rest of the pipeline.
    - Output angular constraints are applied only when returning angles:
        * phi is circular in [0, 360)
        * if theta < 0:
              theta = -theta
              phi = phi + 180 deg
        * if theta > 90:
              theta = 90 deg
    - The global IMM output is represented in the CA state space (9x1).
    """

    def __init__(
        self,
        dt: float,
        cv_q_noise_std: float = 0.02,
        ca_q_noise_std: float = 0.02,
        r_noise_std: float = 0.05,
        transition_matrix: np.ndarray | None = None,
        initial_mode_probabilities: np.ndarray | None = None,
        cv_to_ca_accel_var: float = 1.0,
    ):
        """
        Initializes the IMM Kalman tracker.

        Args:
            dt (float): Sampling period.
            cv_q_noise_std (float): Process noise std for the CV model.
                                    Interpreted as acceleration uncertainty in
                                    Cartesian unit-vector space.
            ca_q_noise_std (float): Process noise std for the CA model.
                                    Interpreted as jerk uncertainty in
                                    Cartesian unit-vector space.
            r_noise_std (float): Measurement noise std in unit-vector space.
            transition_matrix (np.ndarray | None): 2x2 mode transition matrix.
                                                   If None, a default matrix is used.
            initial_mode_probabilities (np.ndarray | None): Initial model probabilities.
                                                            If None, [0.5, 0.5] is used.
            cv_to_ca_accel_var (float): Extra variance assigned to the acceleration
                                        block when converting CV -> CA covariance.
        """
        self.dt = float(dt)

        # ============================================================
        # IMM STATE
        # ============================================================

        self.is_initialized = False

        if transition_matrix is None:
            self.PI = np.array([
                [0.95, 0.05],
                [0.05, 0.95]
            ], dtype=float)
        else:
            self.PI = np.asarray(transition_matrix, dtype=float).reshape(2, 2)

        if initial_mode_probabilities is None:
            self.mu = np.array([[0.5], [0.5]], dtype=float)
        else:
            self.mu = np.asarray(initial_mode_probabilities, dtype=float).reshape(2, 1)
            self.mu = self.mu / np.sum(self.mu)

        self.cv_to_ca_accel_var = float(cv_to_ca_accel_var)

        # These are useful for IMM internals
        self.cbar = np.zeros((2, 1), dtype=float)
        self.mixing_probabilities = np.zeros((2, 2), dtype=float)

        # Prediction bookkeeping
        self._prediction_done = False

        # Optional diagnostics buffers (updated externally if needed)
        self.last_y_cv = None
        self.last_y_ca = None
        self.last_S_cv = None
        self.last_S_ca = None
        self.last_lambda_cv = None
        self.last_lambda_ca = None
        self.last_cbar_cv = None
        self.last_cbar_ca = None
        self.last_mahal_cv = None
        self.last_mahal_ca = None

        # ============================================================
        # MODEL 1: CV
        # ============================================================

        self.x_cv = np.zeros((6, 1))
        self.P_cv = np.eye(6) * 500.0

        self.F_cv = np.array([
            [1.0, 0.0, 0.0, self.dt, 0.0,    0.0],
            [0.0, 1.0, 0.0, 0.0,    self.dt, 0.0],
            [0.0, 0.0, 1.0, 0.0,    0.0,    self.dt],
            [0.0, 0.0, 0.0, 1.0,    0.0,    0.0],
            [0.0, 0.0, 0.0, 0.0,    1.0,    0.0],
            [0.0, 0.0, 0.0, 0.0,    0.0,    1.0]
        ], dtype=float)

        q_cv = cv_q_noise_std ** 2
        q_pos = (self.dt ** 4) / 4.0
        q_cross = (self.dt ** 3) / 2.0
        q_vel = self.dt ** 2

        self.Q_cv = q_cv * np.block([
            [q_pos * np.eye(3),   q_cross * np.eye(3)],
            [q_cross * np.eye(3), q_vel * np.eye(3)]
        ])

        self.H_cv = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ], dtype=float)

        self.I_cv = np.eye(6)

        # ============================================================
        # MODEL 2: CA
        # ============================================================

        self.x_ca = np.zeros((9, 1))
        self.P_ca = np.eye(9) * 500.0

        dt = self.dt
        dt2_half = 0.5 * (dt ** 2)

        self.F_ca = np.array([
            [1.0, 0.0, 0.0, dt,  0.0, 0.0, dt2_half, 0.0,      0.0],
            [0.0, 1.0, 0.0, 0.0, dt,  0.0, 0.0,      dt2_half, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt,  0.0,      0.0,      dt2_half],

            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, dt,       0.0,      0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,      dt,       0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,      0.0,      dt],

            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,      0.0,      0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      1.0,      0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      0.0,      1.0]
        ], dtype=float)

        q_ca = ca_q_noise_std ** 2
        q11 = (dt ** 6) / 36.0
        q12 = (dt ** 5) / 12.0
        q13 = (dt ** 4) / 6.0
        q22 = (dt ** 4) / 4.0
        q23 = (dt ** 3) / 2.0
        q33 = (dt ** 2)

        q_axis = np.array([
            [q11, q12, q13],
            [q12, q22, q23],
            [q13, q23, q33]
        ], dtype=float)

        self.Q_ca = q_ca * np.block([
            [q_axis[0, 0] * np.eye(3), q_axis[0, 1] * np.eye(3), q_axis[0, 2] * np.eye(3)],
            [q_axis[1, 0] * np.eye(3), q_axis[1, 1] * np.eye(3), q_axis[1, 2] * np.eye(3)],
            [q_axis[2, 0] * np.eye(3), q_axis[2, 1] * np.eye(3), q_axis[2, 2] * np.eye(3)]
        ])

        self.H_ca = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=float)

        self.I_ca = np.eye(9)

        # ============================================================
        # COMMON MEASUREMENT NOISE
        # ============================================================

        r_scalar = r_noise_std ** 2
        self.R = np.eye(3) * r_scalar

        # ============================================================
        # GLOBAL IMM OUTPUT (CA SPACE)
        # ============================================================

        self.x = np.zeros((9, 1))
        self.P = np.eye(9) * 500.0

    # ============================================================
    # ANGLE HELPERS
    # ============================================================

    @staticmethod
    def _wrap_360(angle_deg: float) -> float:
        """Wrap angle to [0, 360)."""
        return angle_deg % 360.0

    @classmethod
    def _apply_output_angle_constraints(cls, theta_deg: float, phi_deg: float) -> tuple[float, float]:
        """
        Applies the angular output constraints of the tracker.

        Rules:
        - Azimuth phi is circular in [0, 360).
        - If theta < 0:
              reflect theta -> -theta
              shift phi by +180 deg
        - If theta > 90:
              saturate theta at 90 deg
        """
        theta = float(theta_deg)
        phi = float(phi_deg)

        if theta < 0.0:
            theta = -theta
            phi = phi + 180.0

        if theta > 90.0:
            theta = 90.0

        phi = cls._wrap_360(phi)

        return theta, phi

    @staticmethod
    def _angles_to_unit_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
        """
        Converts angular coordinates (theta, phi) in degrees into a 3D unit vector.
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
        """
        u = np.asarray(u, dtype=float).reshape(3, 1)

        norm_u = np.linalg.norm(u)
        if norm_u < 1e-12:
            return 0.0, 0.0

        u = u / norm_u

        ux = float(u[0, 0])
        uy = float(u[1, 0])
        uz = float(u[2, 0])

        uz = np.clip(uz, -1.0, 1.0)

        theta_deg = float(np.rad2deg(np.arccos(uz)))
        phi_deg = float(np.rad2deg(np.arctan2(uy, ux)))

        return theta_deg, phi_deg

    # ============================================================
    # GEOMETRIC CONSTRAINTS: CV
    # ============================================================

    @staticmethod
    def _normalize_direction_in_state(x: np.ndarray) -> np.ndarray:
        """
        Enforces unit norm on the direction component x[0:3].
        """
        x = x.copy()
        u = x[0:3, :]
        norm_u = np.linalg.norm(u)

        if norm_u < 1e-12:
            x[0:3, 0] = np.array([0.0, 0.0, 1.0], dtype=float)
            return x

        x[0:3, :] = u / norm_u
        return x

    @staticmethod
    def _project_velocity_to_tangent_in_state(x: np.ndarray) -> np.ndarray:
        """
        Projects the velocity component x[3:6] onto the tangent plane of the sphere.
        Constraint:
            u^T du = 0
        """
        x = x.copy()

        u = x[0:3, :]
        du = x[3:6, :]

        norm_u = np.linalg.norm(u)
        if norm_u < 1e-12:
            return x

        u = u / norm_u
        radial_component = np.vdot(u.ravel(), du.ravel())
        du_tangent = du - radial_component * u

        x[3:6, :] = du_tangent
        return x

    @classmethod
    def _enforce_cv_constraints(cls, x: np.ndarray) -> np.ndarray:
        """
        Enforces geometric constraints for the CV state.
        """
        x = cls._normalize_direction_in_state(x)
        x = cls._project_velocity_to_tangent_in_state(x)
        return x

    # ============================================================
    # GEOMETRIC CONSTRAINTS: CA
    # ============================================================

    @staticmethod
    def _enforce_acceleration_sphere_constraint_in_state(x: np.ndarray) -> np.ndarray:
        """
        Enforces the second-order spherical geometry constraint:
            u^T ddu = -||du||^2
        """
        x = x.copy()

        u = x[0:3, :]
        du = x[3:6, :]
        ddu = x[6:9, :]

        norm_u = np.linalg.norm(u)
        if norm_u < 1e-12:
            return x

        u = u / norm_u

        radial_component_current = (u.T @ ddu).item()
        radial_component_required = -(du.T @ du).item()

        ddu_corrected = ddu + (radial_component_required - radial_component_current) * u
        x[6:9, :] = ddu_corrected

        return x

    @classmethod
    def _enforce_ca_constraints(cls, x: np.ndarray) -> np.ndarray:
        """
        Enforces geometric constraints for the CA state.
        """
        x = cls._normalize_direction_in_state(x)
        x = cls._project_velocity_to_tangent_in_state(x)
        x = cls._enforce_acceleration_sphere_constraint_in_state(x)
        return x

    # ============================================================
    # MODEL CONVERSIONS
    # ============================================================

    @staticmethod
    def _cv_to_ca_state(x_cv: np.ndarray) -> np.ndarray:
        """
        Converts a CV state (6x1) to CA state (9x1) by appending zero acceleration.
        """
        x_cv = np.asarray(x_cv, dtype=float).reshape(6, 1)
        x_ca = np.zeros((9, 1), dtype=float)
        x_ca[0:6, :] = x_cv
        x_ca[6:9, :] = 0.0
        return x_ca

    @staticmethod
    def _ca_to_cv_state(x_ca: np.ndarray) -> np.ndarray:
        """
        Converts a CA state (9x1) to CV state (6x1) by truncating acceleration.
        """
        x_ca = np.asarray(x_ca, dtype=float).reshape(9, 1)
        return x_ca[0:6, :].copy()

    def _cv_to_ca_covariance(self, P_cv: np.ndarray) -> np.ndarray:
        """
        Converts a CV covariance (6x6) into a CA covariance (9x9), assigning
        additional uncertainty to the acceleration block.
        """
        P_cv = np.asarray(P_cv, dtype=float).reshape(6, 6)
        P_ca = np.zeros((9, 9), dtype=float)

        P_ca[0:6, 0:6] = P_cv
        P_ca[6:9, 6:9] = self.cv_to_ca_accel_var * np.eye(3)

        return P_ca

    @staticmethod
    def _ca_to_cv_covariance(P_ca: np.ndarray) -> np.ndarray:
        """
        Converts a CA covariance (9x9) into a CV covariance (6x6) by truncation.
        """
        P_ca = np.asarray(P_ca, dtype=float).reshape(9, 9)
        return P_ca[0:6, 0:6].copy()

    # ============================================================
    # IMM INTERNAL HELPERS
    # ============================================================

    @staticmethod
    def _gaussian_likelihood(y: np.ndarray, S: np.ndarray) -> float:
        """
        Computes a practical IMM likelihood based only on the Mahalanobis distance.

        This avoids excessive penalization of higher-uncertainty models through
        the det(S) normalization term of the full Gaussian likelihood.
        """
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        S = np.asarray(S, dtype=float)

        dim = y.shape[0]
        S_reg = S + 1e-12 * np.eye(dim)

        try:
            mahal = (y.T @ np.linalg.inv(S_reg) @ y).item()
        except np.linalg.LinAlgError:
            return 1e-300

        log_likelihood = -0.5 * mahal
        log_likelihood = max(log_likelihood, -700.0)

        return float(np.exp(log_likelihood))

    @staticmethod
    def _symmetrize(P: np.ndarray) -> np.ndarray:
        """
        Forces symmetry of a covariance matrix.
        """
        return 0.5 * (P + P.T)

    def _update_global_output_from_models(self) -> None:
        """
        Builds the fused IMM global estimate in CA space.
        """
        x_cv_as_ca = self._cv_to_ca_state(self.x_cv)
        P_cv_as_ca = self._cv_to_ca_covariance(self.P_cv)

        x_ca = self.x_ca.copy()
        P_ca = self.P_ca.copy()

        mu_cv = float(self.mu[0, 0])
        mu_ca = float(self.mu[1, 0])

        self.x = mu_cv * x_cv_as_ca + mu_ca * x_ca

        dx_cv = x_cv_as_ca - self.x
        dx_ca = x_ca - self.x

        self.P = (
            mu_cv * (P_cv_as_ca + dx_cv @ dx_cv.T) +
            mu_ca * (P_ca + dx_ca @ dx_ca.T)
        )

        self.P = self._symmetrize(self.P)
        self.x = self._enforce_ca_constraints(self.x)

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize_from_measurement(self, theta_meas_deg: float, phi_meas_deg: float) -> None:
        """
        Delayed initialization from the first angular measurement.
        """
        z = self._angles_to_unit_vector(theta_meas_deg, phi_meas_deg)

        # Initialize CV
        self.x_cv[0:3, :] = z
        self.x_cv[3:6, :] = 0.0
        self.x_cv = self._enforce_cv_constraints(self.x_cv)

        # Initialize CA
        self.x_ca[0:3, :] = z
        self.x_ca[3:6, :] = 0.0
        self.x_ca[6:9, :] = 0.0
        self.x_ca = self._enforce_ca_constraints(self.x_ca)

        # Keep the original large uncertainty style
        self.P_cv = np.eye(6) * 500.0
        self.P_ca = np.eye(9) * 500.0

        self.mu = self.mu / np.sum(self.mu)
        self._update_global_output_from_models()

        self.is_initialized = True
        self._prediction_done = False

    # ============================================================
    # IMM PREDICT
    # ============================================================

    def predict(self) -> None:
        """
        IMM prediction step.

        If the tracker is not initialized yet, this method does nothing.

        Steps:
        1) Compute mixing probabilities
        2) Mix states/covariances for each destination model
        3) Run model-matched prediction
        """
        if not self.is_initialized:
            return

        # --------------------------------------------------------
        # 1) Mixing probabilities
        # --------------------------------------------------------
        # cbar_j = sum_i p_ij * mu_i
        self.cbar[0, 0] = self.PI[0, 0] * self.mu[0, 0] + self.PI[1, 0] * self.mu[1, 0]
        self.cbar[1, 0] = self.PI[0, 1] * self.mu[0, 0] + self.PI[1, 1] * self.mu[1, 0]

        # mu_{i|j}
        for j in range(2):
            denom = float(self.cbar[j, 0])
            if denom < 1e-15:
                self.mixing_probabilities[:, j] = 0.5
            else:
                for i in range(2):
                    self.mixing_probabilities[i, j] = (self.PI[i, j] * self.mu[i, 0]) / denom

        # --------------------------------------------------------
        # 2) Mixed initial condition for CV destination
        # --------------------------------------------------------
        # Sources converted to CV space
        x0_cv_from_cv = self.x_cv.copy()
        P0_cv_from_cv = self.P_cv.copy()

        x0_cv_from_ca = self._ca_to_cv_state(self.x_ca)
        P0_cv_from_ca = self._ca_to_cv_covariance(self.P_ca)

        mu_cv_cv = self.mixing_probabilities[0, 0]
        mu_ca_cv = self.mixing_probabilities[1, 0]

        x0_cv = mu_cv_cv * x0_cv_from_cv + mu_ca_cv * x0_cv_from_ca

        dx_cv_from_cv = x0_cv_from_cv - x0_cv
        dx_cv_from_ca = x0_cv_from_ca - x0_cv

        P0_cv = (
            mu_cv_cv * (P0_cv_from_cv + dx_cv_from_cv @ dx_cv_from_cv.T) +
            mu_ca_cv * (P0_cv_from_ca + dx_cv_from_ca @ dx_cv_from_ca.T)
        )

        P0_cv = self._symmetrize(P0_cv)
        x0_cv = self._enforce_cv_constraints(x0_cv)

        # --------------------------------------------------------
        # 3) Mixed initial condition for CA destination
        # --------------------------------------------------------
        # Sources converted to CA space
        x0_ca_from_cv = self._cv_to_ca_state(self.x_cv)
        P0_ca_from_cv = self._cv_to_ca_covariance(self.P_cv)

        x0_ca_from_ca = self.x_ca.copy()
        P0_ca_from_ca = self.P_ca.copy()

        mu_cv_ca = self.mixing_probabilities[0, 1]
        mu_ca_ca = self.mixing_probabilities[1, 1]

        x0_ca = mu_cv_ca * x0_ca_from_cv + mu_ca_ca * x0_ca_from_ca

        dx_ca_from_cv = x0_ca_from_cv - x0_ca
        dx_ca_from_ca = x0_ca_from_ca - x0_ca

        P0_ca = (
            mu_cv_ca * (P0_ca_from_cv + dx_ca_from_cv @ dx_ca_from_cv.T) +
            mu_ca_ca * (P0_ca_from_ca + dx_ca_from_ca @ dx_ca_from_ca.T)
        )

        P0_ca = self._symmetrize(P0_ca)
        x0_ca = self._enforce_ca_constraints(x0_ca)

        # --------------------------------------------------------
        # 4) Model-matched prediction
        # --------------------------------------------------------
        self.x_cv = self.F_cv @ x0_cv
        self.P_cv = self.F_cv @ P0_cv @ self.F_cv.T + self.Q_cv
        self.P_cv = self._symmetrize(self.P_cv)
        self.x_cv = self._enforce_cv_constraints(self.x_cv)

        self.x_ca = self.F_ca @ x0_ca
        self.P_ca = self.F_ca @ P0_ca @ self.F_ca.T + self.Q_ca
        self.P_ca = self._symmetrize(self.P_ca)
        self.x_ca = self._enforce_ca_constraints(self.x_ca)

        self._update_global_output_from_models()
        self._prediction_done = True

    # ============================================================
    # IMM UPDATE
    # ============================================================

    def update(self, theta_meas_deg: float, phi_meas_deg: float) -> None:
        """
        IMM measurement update step.

        On the first valid measurement, the tracker is initialized using:
        - direction from the measurement
        - zero initial velocity in both models
        - zero initial acceleration in CA

        If update() is called before predict() on a later step, this method
        still works, but it will behave as a pure measurement update over the
        current model states. The standard intended usage remains:

            predict()
            update(theta_meas_deg, phi_meas_deg)
        """
        z = self._angles_to_unit_vector(theta_meas_deg, phi_meas_deg)

        # Delayed initialization
        if not self.is_initialized:
            self._initialize_from_measurement(theta_meas_deg, phi_meas_deg)
            return

        # --------------------------------------------------------
        # 1) CV model update
        # --------------------------------------------------------
        y_cv = z - (self.H_cv @ self.x_cv)
        S_cv = self.H_cv @ self.P_cv @ self.H_cv.T + self.R
        K_cv = self.P_cv @ self.H_cv.T @ np.linalg.inv(S_cv)

        self.x_cv = self.x_cv + K_cv @ y_cv

        I_KH_cv = self.I_cv - K_cv @ self.H_cv
        self.P_cv = I_KH_cv @ self.P_cv @ I_KH_cv.T + K_cv @ self.R @ K_cv.T
        self.P_cv = self._symmetrize(self.P_cv)

        self.x_cv = self._enforce_cv_constraints(self.x_cv)

        # --------------------------------------------------------
        # 2) CA model update
        # --------------------------------------------------------
        y_ca = z - (self.H_ca @ self.x_ca)
        S_ca = self.H_ca @ self.P_ca @ self.H_ca.T + self.R
        K_ca = self.P_ca @ self.H_ca.T @ np.linalg.inv(S_ca)

        self.x_ca = self.x_ca + K_ca @ y_ca

        I_KH_ca = self.I_ca - K_ca @ self.H_ca
        self.P_ca = I_KH_ca @ self.P_ca @ I_KH_ca.T + K_ca @ self.R @ K_ca.T
        self.P_ca = self._symmetrize(self.P_ca)

        self.x_ca = self._enforce_ca_constraints(self.x_ca)

        # --------------------------------------------------------
        # 3) Model likelihoods
        # --------------------------------------------------------
        lambda_cv = self._gaussian_likelihood(y_cv, S_cv)
        lambda_ca = self._gaussian_likelihood(y_ca, S_ca)

        # Optional diagnostics storage
        self.last_y_cv = y_cv.copy()
        self.last_y_ca = y_ca.copy()
        self.last_S_cv = S_cv.copy()
        self.last_S_ca = S_ca.copy()
        self.last_lambda_cv = float(lambda_cv)
        self.last_lambda_ca = float(lambda_ca)
        self.last_mahal_cv = (y_cv.T @ np.linalg.inv(S_cv + 1e-12 * np.eye(3)) @ y_cv).item()
        self.last_mahal_ca = (y_ca.T @ np.linalg.inv(S_ca + 1e-12 * np.eye(3)) @ y_ca).item()

        # --------------------------------------------------------
        # 4) Mode probability update
        #     mu_j = lambda_j * cbar_j / sum_l(...)
        # --------------------------------------------------------
        if self._prediction_done:
            cbar_cv = float(self.cbar[0, 0])
            cbar_ca = float(self.cbar[1, 0])
        else:
            # Fallback if predict() was not called
            cbar_cv = float(self.mu[0, 0])
            cbar_ca = float(self.mu[1, 0])

        self.last_cbar_cv = cbar_cv
        self.last_cbar_ca = cbar_ca

        numer_cv = lambda_cv * cbar_cv
        numer_ca = lambda_ca * cbar_ca
        denom = numer_cv + numer_ca

        if denom < 1e-300:
            self.mu[0, 0] = 0.5
            self.mu[1, 0] = 0.5
        else:
            self.mu[0, 0] = numer_cv / denom
            self.mu[1, 0] = numer_ca / denom

        self.mu = self.mu / np.sum(self.mu)

        # --------------------------------------------------------
        # 5) Global fusion in CA space
        # --------------------------------------------------------
        self._update_global_output_from_models()
        self._prediction_done = False

    # ============================================================
    # PUBLIC GETTERS
    # ============================================================

    def get_unit_vector(self) -> np.ndarray:
        """
        Returns the global IMM estimated unit direction vector.
        """
        return self.x[0:3, :].copy()

    def get_velocity_vector(self) -> np.ndarray:
        """
        Returns the global IMM estimated Cartesian velocity vector.
        """
        return self.x[3:6, :].copy()

    def get_acceleration_vector(self) -> np.ndarray:
        """
        Returns the global IMM estimated Cartesian acceleration vector
        in the CA global state space.
        """
        return self.x[6:9, :].copy()

    def get_angles(self) -> tuple[float, float]:
        """
        Returns the current filtered angular estimate in degrees.

        Output angular constraints are applied here.
        """
        theta_deg, phi_deg = self._unit_vector_to_angles(self.x[0:3, :])
        theta_deg, phi_deg = self._apply_output_angle_constraints(theta_deg, phi_deg)
        return theta_deg, phi_deg

    def get_full_state(self) -> np.ndarray:
        """
        Returns the global fused IMM state in CA space (9x1).
        """
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """
        Returns the global fused IMM covariance in CA space (9x9).
        """
        return self.P.copy()

    def get_model_probabilities(self) -> np.ndarray:
        """
        Returns the posterior model probabilities:

            [mu_cv, mu_ca]^T
        """
        return self.mu.copy()

    def get_cv_state(self) -> np.ndarray:
        """
        Returns the posterior CV model state (6x1).
        """
        return self.x_cv.copy()

    def get_ca_state(self) -> np.ndarray:
        """
        Returns the posterior CA model state (9x1).
        """
        return self.x_ca.copy()

    def get_cv_covariance(self) -> np.ndarray:
        """
        Returns the posterior CV covariance (6x6).
        """
        return self.P_cv.copy()

    def get_ca_covariance(self) -> np.ndarray:
        """
        Returns the posterior CA covariance (9x9).
        """
        return self.P_ca.copy()