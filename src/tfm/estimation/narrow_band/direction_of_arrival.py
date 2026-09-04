import numpy as np

def doa_periodogram(covariance_matrix, steering_vectors, scan_angles):
    """
    Estimates the Direction of Arrival (DOA) for a Uniform Rectangular Array (URA)
    using the conventional Periodogram method, fully vectorized for max performance.
    
    Args:
        covariance_matrix (np.ndarray): The spatial sample covariance matrix (N, N).
        steering_vectors (np.ndarray): Precomputed steering vectors for the 2D scan grid (N, K).
        scan_angles (np.ndarray): Array of shape (K, 2) containing the (theta, phi) pairs.
        
    Returns:
        tuple: The estimated DOA as a tuple (theta_estimate, phi_estimate).
        np.ndarray: The spatial power spectrum array of size (K,).
    """
    
    # 1. Multiply R_xx by all steering vectors simultaneously
    # covariance_matrix is (N, N), steering_vectors is (N, K)
    # Resulting temp_matrix is (N, K)
    temp_matrix = np.dot(covariance_matrix, steering_vectors)
    
    # 2. Element-wise multiply W* (conjugate) with temp_matrix and sum over the N elements (axis=0)
    # This efficiently computes the diagonal of W^H * R_xx * W without building the massive K x K matrix
    power_array = np.sum(np.conj(steering_vectors) * temp_matrix, axis=0)
    
    # 3. Discard negligible numerical imaginary parts to get real power
    spatial_spectrum = np.abs(power_array)
    
    # 4. Find the global peak
    max_power_idx = np.argmax(spatial_spectrum)
    doa_estimate = tuple(scan_angles[max_power_idx])
    
    return doa_estimate, spatial_spectrum

def doa_capon(covariance_matrix, steering_vectors, scan_angles):
    """
    Estimates the Direction of Arrival (DOA) for a Uniform Rectangular Array (URA)
    using the Capon (MVDR / Maximum Likelihood) high-resolution method.
    Fully vectorized for maximum performance.
    
    Args:
        covariance_matrix (np.ndarray): The spatial sample covariance matrix (N, N).
        steering_vectors (np.ndarray): Precomputed steering vectors for the 2D scan grid (N, K).
                                       Assumes the e^-j(n*psi_x + m*psi_y) phase convention.
        scan_angles (np.ndarray): Array of shape (K, 2) containing the (theta, phi) pairs.
        
    Returns:
        tuple: The estimated DOA as a tuple (theta_estimate, phi_estimate).
        np.ndarray: The spatial power spectrum array of size (K,).
    """
    
    # 1. Calculate the pseudo-inverse of the covariance matrix.
    # Using pinv ensures numerical stability if the matrix is ill-conditioned 
    # (e.g., highly correlated signals or very low noise environments in simulation).
    inv_cov_matrix = np.linalg.pinv(covariance_matrix)
    
    # 2. Multiply R_xx^-1 by all steering vectors simultaneously
    # inv_cov_matrix is (N, N), steering_vectors is (N, K) -> temp_matrix is (N, K)
    temp_matrix = np.dot(inv_cov_matrix, steering_vectors)
    
    # 3. Element-wise multiply W* (conjugate) with temp_matrix and sum over N elements (axis=0)
    # This efficiently computes the denominator diagonal: W^H * R_xx^-1 * W
    denominator_array = np.sum(np.conj(steering_vectors) * temp_matrix, axis=0)
    
    # 4. Discard negligible imaginary parts and calculate the Capon spectrum
    # P(theta) = 1 / abs(w^H * R_xx^-1 * w)
    spatial_spectrum = 1.0 / np.abs(denominator_array)
    
    # 5. Find the global peak
    max_power_idx = np.argmax(spatial_spectrum)
    doa_estimate = tuple(scan_angles[max_power_idx])
    
    return doa_estimate, spatial_spectrum

def doa_music(covariance_matrix, steering_vectors, scan_angles, num_signals=1):
    """
    Estimates the Direction of Arrival (DOA) for a Uniform Rectangular Array (URA)
    using the MUSIC (MUltiple SIgnal Classification) subspace method.
    Fully vectorized for maximum performance in DRL loops.
    
    Args:
        covariance_matrix (np.ndarray): The spatial sample covariance matrix (N, N).
        steering_vectors (np.ndarray): Precomputed steering vectors for the 2D scan grid (N, K).
                                       Assumes the e^-j(n*psi_x + m*psi_y) phase convention.
        scan_angles (np.ndarray): Array of shape (K, 2) containing the (theta, phi) pairs.
        num_signals (int): The assumed number of incident signals (signal subspace dimension).
        
    Returns:
        tuple: The estimated DOA as a tuple (theta_estimate, phi_estimate).
        np.ndarray: The spatial power spectrum array of size (K,).
    """
    
    # 1. Eigenvalue decomposition
    # eigh is highly optimized for Hermitian matrices and guarantees real eigenvalues.
    # It returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # 2. Extract the Noise Subspace (V_n)
    # The noise subspace corresponds to the smallest (N - num_signals) eigenvalues.
    noise_eigenvectors = eigenvectors[:, :-num_signals]
    
    # 3. Project the scanning steering vectors onto the noise subspace
    # noise_eigenvectors is (N, N - num_signals)
    # steering_vectors is (N, K)
    # Projection V_n^H * W results in a shape of (N - num_signals, K)
    projection = np.dot(np.conj(noise_eigenvectors).T, steering_vectors)
    
    # 4. Calculate the denominator for the MUSIC spectrum
    # Denominator = || V_n^H * w ||^2 = sum of squared magnitudes along axis 0
    denominator = np.sum(np.abs(projection)**2, axis=0)
    
    # Prevent division by zero in perfectly ideal simulated environments
    epsilon = 1e-12
    denominator = np.maximum(denominator, epsilon)
    
    # 5. Calculate the MUSIC spatial spectrum
    spatial_spectrum = 1.0 / denominator
    
    # 6. Find the global peak
    max_power_idx = np.argmax(spatial_spectrum)
    doa_estimate = tuple(scan_angles[max_power_idx])
    
    return doa_estimate, spatial_spectrum