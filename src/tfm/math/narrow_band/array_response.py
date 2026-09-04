import numpy as np
from tfm.math.narrow_band.steering_vector import get_steering_vector

def compute_array_factor(weights: np.ndarray, element_positions: np.ndarray, wavenumber_k: float, direction: tuple[float, float]) -> float:
    """
    Computes the Array Factor (Voltage Gain) for a specific direction.
    Used to measure the array response in a single target or jammer direction.
    
    AF = | w^H * a(theta, phi) |
    
    Args:
        weights (np.ndarray): Complex weights applied to the array elements.
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        direction (tuple): A tuple (theta_deg, phi_deg) representing the DOA in degrees.
            
    Returns:
        float: Magnitude of the array response (Linear scale).
    """
    # 1. Get the steering vector for the specific direction
    a_vec = get_steering_vector(element_positions, wavenumber_k, direction)
    
    # 2. Flatten current weights to match vector dimensions
    w_vec = weights.flatten()
    
    # 3. Beamforming calculation: y = w^H * a
    # np.vdot(w, a) handles complex conjugation of 'w': w^H * a
    response = np.vdot(w_vec, a_vec)
    
    return np.abs(response)

def compute_beampattern(weights: np.ndarray, element_positions: np.ndarray, wavenumber_k: float, theta_res: float = 1.0, phi_res: float = 1.0, normalize: bool = True):
    """
    Computes the full 3D radiation pattern (Array Factor) efficiently using matrix operations.
    Avoids slow Python loops by calculating all directions simultaneously using tensors.

    Args:
        weights (np.ndarray): Complex weights applied to the array elements.
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        theta_res (float): Resolution for Theta (elevation) in degrees.
        phi_res (float): Resolution for Phi (azimuth) in degrees.
        normalize (bool): If True, normalizes the maximum magnitude to 1.0 (0 dB). 
                          Set to False for exact power/directivity calculations.

    Returns:
        tuple:
            - AF_magnitude (np.ndarray): 2D array of magnitude values [shape: (n_theta, n_phi)].
            - THETA (np.ndarray): Meshgrid of Theta angles in radians. Useful for plotting.
            - PHI (np.ndarray): Meshgrid of Phi angles in radians. Useful for plotting.
    """
    # 1. Define the angular grid
    theta_vals = np.deg2rad(np.arange(0, 181, theta_res))
    phi_vals = np.deg2rad(np.arange(0, 360, phi_res))

    # Create 2D coordinate matrices (Meshgrid)
    THETA, PHI = np.meshgrid(theta_vals, phi_vals, indexing='ij')

    # 2. Calculate Direction Vectors (u, v, w) for the entire grid
    U = np.sin(THETA) * np.cos(PHI)
    V = np.sin(THETA) * np.sin(PHI)
    W = np.cos(THETA)

    directions = np.array([U, V, W])  # Shape: (3, n_theta, n_phi)

    # 3. Calculate Phase Delays (Spatial Projections)
    pos_flat = element_positions.reshape(-1, 3)
    # Tensor dot product over the spatial dimension
    phases = np.tensordot(pos_flat, directions, axes=([1], [0]))

    # 4. Compute the Steering Matrix (The Manifold)
    # Using negative exponents for our convention
    manifold_matrix = np.exp(-1j * wavenumber_k * phases)

    # 5. Beamforming (Weight Application)
    w_vec = weights.flatten()

    # Perform the dot product: sum(w_conjugate * manifold_element)
    array_factor_complex = np.tensordot(np.conj(w_vec), manifold_matrix, axes=([0], [0]))

    # 6. Get Magnitude (Voltage Gain)
    AF_magnitude = np.abs(array_factor_complex)

    # 7. Normalize if required
    if normalize:
        max_val = np.max(AF_magnitude)
        if max_val > 0:
            AF_magnitude = AF_magnitude / max_val

    return AF_magnitude, THETA, PHI