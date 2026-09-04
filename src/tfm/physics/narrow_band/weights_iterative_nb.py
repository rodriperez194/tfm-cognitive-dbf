import numpy as np
from tfm.math.narrow_band.steering_vector import get_steering_vector

def lms_weights(x, d, mu, initial_weights=None):
    """
    Least Mean Squares (LMS) algorithm for adaptive beamforming.
    Uses the w^H * x convention matching the array factor computation.
    
    Args:
        x (np.ndarray): Signal matrix of shape (N, K).
        d (np.ndarray): Desired reference signal of shape (K,).
        mu (float): Step size or learning rate.
        initial_weights (np.ndarray, optional): Initial weights of shape (N,).
        
    Returns:
        np.ndarray: Final adapted weights of shape (N,).
        np.ndarray: Weight history of shape (K, N).
        np.ndarray: Error history of shape (K,).
    """
    num_elements, num_samples = x.shape
    if initial_weights is None:
        w = np.zeros(num_elements, dtype=complex)
    else:
        w = initial_weights.copy()
        
    w_history = np.zeros((num_samples, num_elements), dtype=complex)
    error_history = np.zeros(num_samples, dtype=complex)
    
    for k in range(num_samples):
        # Extract signal snapshot at time k
        x_k = x[:, k]
        
        # Calculate array output using Hermitian transpose: y(k) = w^H * x(k)
        y_k = np.dot(np.conj(w), x_k)
        
        # Calculate instantaneous error: e(k) = d(k) - y(k)
        e_k = d[k] - y_k
        
        # LMS weight update rule for complex signals: w(k+1) = w(k) + 2*mu * e(k)^* * x(k)
        w = w + 2 * mu * np.conj(e_k) * x_k
        
        # Store metrics
        w_history[k, :] = w
        error_history[k] = e_k
        
    return w, w_history, error_history


def howells_applebaum_weights(x, element_positions, wavenumber_k, direction, signal_power, mu, initial_weights=None):
    """
    Howells-Applebaum algorithm for adaptive beamforming.
    Dynamically computes the steering vector based on the target's DOA.
    Uses the w^H * x convention matching the array factor computation.
    
    Args:
        x (np.ndarray): Signal matrix of shape (N, K).
        element_positions (np.ndarray): Array of shape (N, 3) containing antenna coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        direction (tuple[float, float]): Target DOA as (theta_deg, phi_deg).
        signal_power (float): Estimated power of the target signal (sigma_s^2).
        mu (float): Step size or learning rate.
        initial_weights (np.ndarray, optional): Initial weights of shape (N,).
        
    Returns:
        np.ndarray: Final adapted weights of shape (N,).
        np.ndarray: Weight history of shape (K, N).
    """
    num_elements, num_samples = x.shape
    if initial_weights is None:
        w = np.zeros(num_elements, dtype=complex)
    else:
        w = initial_weights.copy()
        
    w_history = np.zeros((num_samples, num_elements), dtype=complex)
    
    # 1. Compute the steering vector dynamically
    steering_vector = get_steering_vector(element_positions, wavenumber_k, direction)
    
    # 2. Pre-calculate the steering target component.
    # Since we now use the w^H * x convention, the cross-correlation target 
    # aligns directly with the steering vector without needing conjugation.
    steering_target = signal_power * steering_vector
    
    for k in range(num_samples):
        # Extract signal snapshot at time k
        x_k = x[:, k]
        
        # Calculate array output: y(k) = w^H * x(k)
        y_k = np.dot(np.conj(w), x_k)
        
        # Howells-Applebaum weight update rule:
        # Stationary point is Rxx * w = steering_target.
        w = w + 2 * mu * (steering_target - np.conj(y_k) * x_k)
        
        # Store metrics
        w_history[k, :] = w
        
    return w, w_history