import numpy as np

class KalmanTracker:
    """
    Kalman Filter based Direction of Arrival (DOA) Tracker.
    
    This class implements a dynamic tracker to estimate the temporal evolution 
    of arrival angles (theta, phi) of a signal, smoothing fluctuations caused 
    by noise and providing predictive capabilities to solve POMDP environments.
    
    State Vector (x_hat): [theta, phi, theta_velocity, phi_velocity]^T
    Measurement Vector (z): [theta_meas, phi_meas]^T
    """

    def __init__(self, dt: float, q_noise_std: float = 0.1, r_noise_std: float = 1.0):
        """
        Initializes the Kalman Tracker with the dynamic and measurement models.
        
        Args:
            dt (float): Update period of the system (Delta t).
            q_noise_std (float): Standard deviation for the process noise (Q).
                                 Models unpredicted maneuvers of the jammer.
            r_noise_std (float): Standard deviation for the measurement noise (R).
                                 Models the uncertainty of the DOA estimator.
        """
        self.dt = float(dt)
        
        # 1. State Vector estimation (4x1)
        # NOTE: This is \hat{x}, the ESTIMATED state, not the true physical state.
        # [theta_hat, phi_hat, d_theta_hat, d_phi_hat]^T
        self.x = np.zeros((4, 1))
        
        # 2. State Uncertainty Covariance Matrix (4x4)
        # Initialized with high uncertainty (500) since we don't know where 
        # the jammer is when the simulation starts.
        self.P = np.eye(4) * 500.0
        
        # 3. Dynamic Model Transition Matrix F (4x4)
        # Assumes constant angular velocity between sampling instances.
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0,       self.dt],
            [0, 0, 1,       0],
            [0, 0, 0,       1]
        ])
        
        # 4. Process Noise Covariance Matrix Q (4x4)
        # Simplified continuous white noise acceleration model.
        # This prevents the filter from overly trusting the "constant velocity" 
        # assumption when the jammer performs sudden evasive maneuvers.
        q_scalar = q_noise_std ** 2
        self.Q = np.array([
            [(self.dt**4)/4, 0,              (self.dt**3)/2, 0],
            [0,              (self.dt**4)/4, 0,              (self.dt**3)/2],
            [(self.dt**3)/2, 0,              self.dt**2,     0],
            [0,              (self.dt**3)/2, 0,              self.dt**2]
        ]) * q_scalar
        
        # 5. Measurement Matrix H (2x4)
        # The DOA estimator only observes the angles (theta, phi), not the velocities.
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 6. Measurement Noise Covariance Matrix R (2x2)
        # Diagonal matrix representing the variance/error of the DOA algorithm.
        r_scalar = r_noise_std ** 2
        self.R = np.eye(2) * r_scalar
        
        # Identity matrix cached for efficient update steps
        self.I = np.eye(4)

    # ============================================================
    # ANGLE HELPERS
    # ============================================================

    @staticmethod
    def _wrap_360(angle_deg: float) -> float:
        """Wrap angle to [0,360)."""
        return angle_deg % 360.0
    
    @staticmethod
    def _wrap_180(angle_deg: float) -> float:
        """Wrap angle difference to [-180,180)."""
        return ((angle_deg + 180.0) % 360.0) - 180.0
    
    def _normalize_state(self) -> None:
        """
        Enforces the tracker's angular convention on the internal state.

        Rules:
        - Azimuth phi is circular in [0, 360).
        - If theta < 0:
              reflect theta -> -theta
              shift phi by +180 deg
        - If theta > 90:
              saturate theta at 90 deg
              keep theta velocity unchanged
        """
        theta = float(self.x[0, 0])
        phi = float(self.x[1, 0])
        theta_dot = float(self.x[2, 0])

        # Case 1: theta below lower bound -> reflect and rotate azimuth
        if theta < 0.0:
            theta = -theta
            phi = phi + 180.0
            theta_dot = -theta_dot

        # Case 2: theta above upper bound -> saturate at 90 deg
        if theta > 90.0:
            theta = 90.0
            # theta_dot is intentionally preserved, as requested

        # Always wrap azimuth
        phi = self._wrap_360(phi)

        # Write back
        self.x[0, 0] = theta
        self.x[1, 0] = phi
        self.x[2, 0] = theta_dot
    
    # ============================================================
    # KALMAN CORE
    # ============================================================

    def predict(self) -> tuple[float, float]:
        """
        Prediction Phase (Time Update).
        Projects the current state and error covariance matrix forward in time 
        using the dynamic model, BEFORE a new measurement is taken.
        
        Returns:
            tuple: The predicted (theta, phi) angles in degrees.
        """
        # 1. Project the state ahead: x_{k|k-1} = F * x_{k-1|k-1}
        #
        self.x = np.dot(self.F, self.x)
        
        # 2. Project the error covariance ahead: P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        #
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # 3. Normalize the predicted state to enforce angular conventions
        self._normalize_state()
        
        # Return the predicted positions (first two elements of the state vector)
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, measurement: tuple[float, float]) -> tuple[float, float]:
        """
        Correction Phase (Measurement Update).
        Fuses the predicted state with the new noisy DOA measurement to obtain 
        the optimal a posteriori estimate.
        
        Args:
            measurement (tuple): A tuple containing the noisy (theta_meas, phi_meas) 
                                 from the DOA estimator (e.g., MUSIC/Capon).
            
        Returns:
            tuple: The updated/smoothed (theta, phi) angles in degrees.
        """
        # Format the incoming measurement as a 2x1 column vector (z_k)
        z = np.array([[measurement[0]], [measurement[1]]])

        # 1. Compute predicted measurement: z_hat_k = H * x_{k|k-1}
        #
        z_pred = np.dot(self.H, self.x)

        # 2. Compute Innovation (Residual)
        # Theta uses standard subtraction
        # Phi uses wrapped angular difference in [-180, 180)
        #
        y_theta = z[0, 0] - z_pred[0, 0]
        y_phi = self._wrap_180(z[1, 0] - z_pred[1, 0])

        y = np.array([[y_theta], [y_phi]])
        
        # 2. Compute Innovation Covariance: S_k = H * P_{k|k-1} * H^T + R
        #
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        # 3. Compute Kalman Gain: K_k = P_{k|k-1} * H^T * S_k^-1
        #
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 4. Update the State Estimate: x_{k|k} = x_{k|k-1} + K_k * y_k
        #
        self.x = self.x + np.dot(K, y)
        
        # 5. Update the Error Covariance: P_{k|k} = (I - K_k * H) * P_{k|k-1}
        #
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)

        # 6. Normalize the updated state to enforce angular conventions
        self._normalize_state()
        
        # Return the final, smoothed positions
        return float(self.x[0, 0]), float(self.x[1, 0])
    
    # ============================================================
    # ACCESSORS
    # ============================================================
        
    def get_velocities(self) -> tuple[float, float]:
        """
        Extracts the hidden angular velocities inferred by the Kalman Filter.
        This is a critical feature to break the POMDP ambiguity for the DRL Agent.
        
        Returns:
            tuple: The estimated (theta_velocity, phi_velocity) in degrees/step.
        """
        return float(self.x[2, 0]), float(self.x[3, 0])
        
    def get_full_state(self) -> np.ndarray:
        """
        Returns the full 4D estimated state vector [theta, phi, d_theta, d_phi]^T.
        Useful for building the observation tensor (O_t) in the Gym environment.
        
        Returns:
            np.ndarray: The 4x1 state vector.
        """
        return self.x.copy()
    
