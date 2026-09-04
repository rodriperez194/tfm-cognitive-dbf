import numpy as np


class UnitVectorCAKalmanTracker:
    """
    Kalman Filter based Direction of Arrival (DOA) Tracker using a unit direction
    vector and a Constant Acceleration (CA) dynamic model.

    This tracker extends the unit-vector constant-velocity tracker by augmenting
    the state with the Cartesian acceleration of the direction vector.

    Internal State Vector (x):
        [u_x, u_y, u_z, du_x, du_y, du_z, ddu_x, ddu_y, ddu_z]^T

    Internal Measurement Vector (z):
        [u_x_meas, u_y_meas, u_z_meas]^T

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
    - Output angular constraints are applied only when returning angles:
        * phi is circular in [0, 360)
        * if theta < 0:
              theta = -theta
              phi = phi + 180 deg
        * if theta > 90:
              theta = 90 deg
    """

    def __init__(self, dt: float, q_noise_std: float = 0.02, r_noise_std: float = 0.05):
        """
        Initializes the Unit Vector Constant Acceleration Kalman Tracker.

        The dynamic model is:

            x_{k+1} = F x_k + w_k

        with state:

            x = [u_x, u_y, u_z, du_x, du_y, du_z, ddu_x, ddu_y, ddu_z]^T

        and measurement:

            z = [u_x_meas, u_y_meas, u_z_meas]^T

        Args:
            dt (float): Sampling period.
            q_noise_std (float): Standard deviation of the process noise.
                                 Interpreted as jerk uncertainty in Cartesian
                                 unit-vector space.
            r_noise_std (float): Standard deviation of the measurement noise
                                 in unit-vector space.
        """
        self.dt = float(dt)

        # 1. Initialization flag
        self.is_initialized = False

        # 2. State Vector estimation (9x1)
        # NOTE: This is x_hat, the ESTIMATED state, not the true physical state.
        # [u_x_hat, u_y_hat, u_z_hat, du_x_hat, du_y_hat, du_z_hat,
        #  ddu_x_hat, ddu_y_hat, ddu_z_hat]^T
        self.x = np.zeros((9, 1))

        # 3. State Uncertainty Covariance Matrix (9x9)
        # Initialized with high uncertainty until the first valid measurement arrives.
        self.P = np.eye(9) * 500.0

        # 4. Dynamic Model Transition Matrix F (9x9)
        # Constant Acceleration model in Cartesian unit-vector space.
        dt = self.dt
        dt2_half = 0.5 * (dt ** 2)

        self.F = np.array([
            [1.0, 0.0, 0.0, dt,  0.0, 0.0, dt2_half, 0.0,      0.0],
            [0.0, 1.0, 0.0, 0.0, dt,  0.0, 0.0,      dt2_half, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt,  0.0,      0.0,      dt2_half],

            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, dt,       0.0,      0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,      dt,       0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,      0.0,      dt],

            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,      0.0,      0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      1.0,      0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      0.0,      1.0]
        ])

        # 5. Process Noise Covariance Matrix Q (9x9)
        # White-jerk model replicated independently over x, y, z.
        q_scalar = q_noise_std ** 2

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

        self.Q = q_scalar * np.block([
            [q_axis[0, 0] * np.eye(3), q_axis[0, 1] * np.eye(3), q_axis[0, 2] * np.eye(3)],
            [q_axis[1, 0] * np.eye(3), q_axis[1, 1] * np.eye(3), q_axis[1, 2] * np.eye(3)],
            [q_axis[2, 0] * np.eye(3), q_axis[2, 1] * np.eye(3), q_axis[2, 2] * np.eye(3)]
        ])

        # 6. Measurement Matrix H (3x9)
        # The measurement only observes the direction vector, not its derivatives.
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

        # 7. Measurement Noise Covariance Matrix R (3x3)
        # Simplified isotropic noise model in unit-vector space.
        r_scalar = r_noise_std ** 2
        self.R = np.eye(3) * r_scalar

        # Identity matrix cached for efficient update steps
        self.I = np.eye(9)

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

    # ============================================================
    # GEOMETRY HELPERS
    # ============================================================

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

    def _normalize_direction_state(self) -> None:
        """
        Enforces unit norm on the direction component of the internal state.
        """
        u = self.x[0:3, :]
        norm_u = np.linalg.norm(u)

        if norm_u < 1e-12:
            # Safe fallback: +z direction
            self.x[0:3, 0] = np.array([0.0, 0.0, 1.0], dtype=float)
            return

        self.x[0:3, :] = u / norm_u

    def _project_velocity_to_tangent_plane(self) -> None:
        """
        Projects the velocity component of the state onto the tangent plane of the
        unit sphere at the current direction vector.

        Geometric constraint:
            u^T * du = 0
        """
        u = self.x[0:3, :]
        du = self.x[3:6, :]

        norm_u = np.linalg.norm(u)
        if norm_u < 1e-12:
            return

        u = u / norm_u
        radial_component = np.vdot(u.ravel(), du.ravel())
        du_tangent = du - radial_component * u

        self.x[3:6, :] = du_tangent

    def _enforce_acceleration_sphere_constraint(self) -> None:
        """
        Enforces the second-order geometric constraint of motion on the unit sphere.

        For a unit direction vector u(t), the following must hold:

            u^T * ddu = -||du||^2

        Therefore, the radial component of the acceleration is not zero, but the
        centripetal term required to keep u on the unit sphere.

        This method preserves the tangential component estimated by the Kalman
        filter and corrects only the radial component.
        """
        u = self.x[0:3, :]
        du = self.x[3:6, :]
        ddu = self.x[6:9, :]

        norm_u = np.linalg.norm(u)
        if norm_u < 1e-12:
            return

        u = u / norm_u

        # Current radial component of the estimated acceleration
        radial_component_current = (u.T @ ddu).item()

        # Required radial component to satisfy spherical geometry
        radial_component_required = -(du.T @ du).item()

        # Correct only the radial part
        ddu_corrected = ddu + (radial_component_required - radial_component_current) * u

        self.x[6:9, :] = ddu_corrected

    def _enforce_geometric_constraints(self) -> None:
        """
        Applies all geometric consistency corrections to the internal state.
        """
        self._normalize_direction_state()
        self._project_velocity_to_tangent_plane()
        self._enforce_acceleration_sphere_constraint()

    # ============================================================
    # KALMAN FILTER CORE
    # ============================================================

    def predict(self) -> None:
        """
        Kalman prediction step.

        If the tracker is not initialized yet, this method does nothing.
        """
        if not self.is_initialized:
            return

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self._enforce_geometric_constraints()

    def update(self, theta_meas_deg: float, phi_meas_deg: float) -> None:
        """
        Kalman update step using an angular measurement.

        On the first valid measurement, the tracker is initialized using:
        - direction from the measurement
        - zero initial velocity
        - zero initial acceleration

        Args:
            theta_meas_deg (float): Measured polar angle in degrees.
            phi_meas_deg (float): Measured azimuth angle in degrees.
        """
        z = self._angles_to_unit_vector(theta_meas_deg, phi_meas_deg)

        # Delayed initialization
        if not self.is_initialized:
            self.x[0:3, :] = z
            self.x[3:6, :] = 0.0
            self.x[6:9, :] = 0.0
            self._enforce_geometric_constraints()
            self.is_initialized = True
            return

        # Innovation
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update using Joseph form for better numerical stability
        I_KH = self.I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self._enforce_geometric_constraints()

    # ============================================================
    # PUBLIC GETTERS
    # ============================================================

    def get_unit_vector(self) -> np.ndarray:
        """
        Returns the current estimated unit direction vector.

        Returns:
            np.ndarray: 3x1 unit vector [u_x, u_y, u_z]^T.
        """
        return self.x[0:3, :].copy()

    def get_velocity_vector(self) -> np.ndarray:
        """
        Returns the current estimated Cartesian velocity vector.

        Returns:
            np.ndarray: 3x1 velocity vector [du_x, du_y, du_z]^T.
        """
        return self.x[3:6, :].copy()

    def get_acceleration_vector(self) -> np.ndarray:
        """
        Returns the current estimated Cartesian acceleration vector.

        Returns:
            np.ndarray: 3x1 acceleration vector [ddu_x, ddu_y, ddu_z]^T.
        """
        return self.x[6:9, :].copy()

    def get_angles(self) -> tuple[float, float]:
        """
        Returns the current filtered angular estimate in degrees.

        Output angular constraints are applied here.

        Returns:
            tuple: (theta_deg, phi_deg)
        """
        theta_deg, phi_deg = self._unit_vector_to_angles(self.x[0:3, :])
        theta_deg, phi_deg = self._apply_output_angle_constraints(theta_deg, phi_deg)
        return theta_deg, phi_deg

    def get_full_state(self) -> np.ndarray:
        """
        Returns the full internal state vector.

        Returns:
            np.ndarray: 9x1 state vector.
        """
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """
        Returns the current covariance matrix.

        Returns:
            np.ndarray: 9x9 covariance matrix.
        """
        return self.P.copy()