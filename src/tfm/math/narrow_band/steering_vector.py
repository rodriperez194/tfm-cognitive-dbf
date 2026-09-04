import numpy as np

def get_steering_vector(element_positions: np.ndarray, wavenumber_k: float, direction: tuple[float, float]) -> np.ndarray:
    """
    Computes the spatial steering vector for a given Direction of Arrival (DOA).
    Calculates phase delays based on exact array geometry and the wave vector.
    
    Args:
        element_positions (np.ndarray): Array of shape (N, M, 3) or (N*M, 3) containing 
                                        the (x, y, z) coordinates of the antenna elements.
        wavenumber_k (float): The wavenumber (2 * pi / lambda) of the carrier frequency.
        direction (tuple[float, float]): A tuple (theta_deg, phi_deg) representing the target DOA.
                                         - theta_deg (float): Polar angle from Z-axis [0, 180].
                                         - phi_deg (float): Azimuth angle from X-axis [0, 360).
        
    Returns:
        np.ndarray: A 1D complex array of shape (N*M,) containing the steering vector 
                    (type complex128).
    """
    theta_deg, phi_deg = direction
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    
    # 1. Direction Unit Vector (Spherical to Cartesian coordinates)
    u = np.sin(theta) * np.cos(phi) 
    v = np.sin(theta) * np.sin(phi) 
    w_z = np.cos(theta)             
    direction_vector = np.array([u, v, w_z])

    # 2. Spatial Phase Delays (Projection of positions onto the DOA vector)
    pos_flat = element_positions.reshape(-1, 3)
    spatial_projection = pos_flat @ direction_vector
    
    # 3. Complex Steering Vector (Using the negative exponent convention)
    steering_vec = np.exp(-1j * wavenumber_k * spatial_projection)
    
    return steering_vec.astype(np.complex128)