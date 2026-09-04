import numpy as np 
import scipy.constants as constants

class Phased_Array_NB:
    """
    Simulation class for a Uniform Rectangular Array (URA) in Narrowband regime.
    
    Implements the physics of the array, managing geometry, wavelength, 
    and beamforming weights state.

    Coordinate System (ISO 80000-2 / Physics Convention):
        - Array lies in the XY plane at Z=0.
        - Rows align with X-axis.
        - Columns align with Y-axis.
        - Theta (Polar): Angle from Z-axis (Normal). 0=Broadside.
        - Phi (Azimuth): Angle from X-axis in XY plane.
    """

    def __init__(self, num_rows, num_cols, carrier_freq, normalize_power=True, d_x=None, d_y=None):
        """
        Initializes the Narrowband Phased Array object.

        Args:
            num_rows (int): Number of elements along the X-axis (N).
            num_cols (int): Number of elements along the Y-axis (M).
            carrier_freq (float): Operation frequency in Hz.
            normalize_power (bool, optional): If True, enforces a total power constraint.
                                              Total power (||w||^2) will equal N*M.
                                              Default is True.
            d_x (float, optional): Spacing along X-axis in meters. 
                                   Default is lambda/2.
            d_y (float, optional): Spacing along Y-axis in meters. 
                                   Default is lambda/2.
        """
        # 1. Geometric and Physical Parameters
        self.N = int(num_rows) # Number of elements along X-axis
        self.M = int(num_cols) # Number of elements along Y-axis
        self.fc = float(carrier_freq)
        self.normalize_power = bool(normalize_power)

        # 2. Derived Parameters
        self.lambda_w = constants.c / self.fc  # Wavelength (m)
        self.k_num = 2 * np.pi / self.lambda_w  # Wavenumber (rad/m)

        # 3. Element Spacing
        self.d_x = d_x if d_x is not None else self.lambda_w / 2  # Spacing along X-axis (m)
        self.d_y = d_y if d_y is not None else self.lambda_w / 2  # Spacing along Y-axis (m)

        # 4. Beamforming Weights Initialization
        # Initialize with 'Broadside' pattern (pointing to normal, Z-axis, theta=0)
        # Shape is (N, M) corresponding to the physical grid
        # Initialized to 1.0 + 0j (Max gain, zero phase shift)
        self.W = np.ones((self.N, self.M), dtype=complex)  # Complex weights (N x M)

        # 5. Element Positions (Array Manifold Pre-calculation)
        # Coordinate system: Array lies in XY plane at Z=0
        # pos[n, m] = [x, y, z]
        self.element_positions = np.zeros((self.N, self.M, 3))  # (N x M x 3)

        # Geometry definition:
        # - Rows align with X-axis: x = n * d_x, n=0,...,N-1
        # - Columns align with Y-axis: y = m * d_y, m=0,...,M-1
        # - Z coordinate is always 0 since array is in XY plane
        for n in range(self.N):
            for m in range(self.M):
                self.element_positions[n, m, 0] = n * self.d_x  # x-coordinate
                self.element_positions[n, m, 1] = m * self.d_y  # y-coordinate
                self.element_positions[n, m, 2] = 0.0           # z-coordinate (array plane)

    def set_weights(self, new_weights: np.ndarray):
        """
        Updates the beamforming weights, enforcing unit gain if configured.
        
        Args:
            new_weights (np.ndarray): Complex weights. 
                                      Can be shape (N, M) or flattened (N*M,).
        """
        # Ensure correct shape (N, M) for internal storage
        # Note: N corresponds to X-axis, M to Y-axis based on init
        w = new_weights.reshape((self.N, self.M)).astype(complex)  # Ensure complex type

        if self.normalize_power:
            # Power Normalization: Ensure total power equals number of elements
            # This prevents the DRL agent from artificially inflating the SINR
            # by scaling up the amplitudes infinitely.
            total_power = np.sum(np.abs(w)**2)
            
            if total_power > 1e-12:  # Avoid division by zero
                # Scale weights so that new total power is (N * M)
                scaling_factor = np.sqrt((self.N * self.M) / total_power)
                w = w * scaling_factor
            else:
                # Fallback if the agent outputs an all-zero vector
                # Reset to uniform broadside array
                w = np.ones((self.N, self.M), dtype=complex)

        self.W = w  # Update internal weights state
