import numpy as np
from tfm.math.narrow_band.steering_vector import get_steering_vector

def mse_weights(R_xx: np.ndarray, element_positions: np.ndarray, wavenumber_k: float, target_direction: tuple[float, float], target_power: float) -> np.ndarray:
    """
    Computes the Minimum Mean Square Error (MSE) / Wiener beamforming weights.
    Minimizes the error between the array output and the desired signal.
    
    Args:
        R_xx (np.ndarray): Total received signal covariance matrix (N*M, N*M).
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        target_direction (tuple): Target DOA (theta_deg, phi_deg).
        target_power (float): Linear power of the desired signal (S).
        
    Returns:
        np.ndarray: A 1D array containing complex weights.
    """
    # 1. Generate the spatial signature WITH negative exponents (our convention)
    target_sv = get_steering_vector(element_positions, wavenumber_k, target_direction)
    
    # 2. Solve the Wiener-Hopf equation: R_xx * w = S * target_sv
    # NOTE: No conjugation needed here because target_sv already has the correct phase sign.
    w_mse = np.linalg.solve(R_xx, target_power * target_sv)
    
    return w_mse


def max_snr_weights(R_rr: np.ndarray, element_positions: np.ndarray, wavenumber_k: float, target_direction: tuple[float, float]) -> np.ndarray:
    """
    Computes the Maximum SNR beamforming weights.
    Maximizes the signal-to-interference-plus-noise ratio directly.
    
    Args:
        R_rr (np.ndarray): Interference + Noise covariance matrix (N*M, N*M).
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        target_direction (tuple): Target DOA (theta_deg, phi_deg).
        
    Returns:
        np.ndarray: A 1D array containing complex weights (normalized).
    """
    # 1. Generate the spatial signature WITH negative exponents (our convention)
    target_sv = get_steering_vector(element_positions, wavenumber_k, target_direction)
    
    # 2. Solve the linear system: R_rr * w = target_sv
    # (No conjugation needed because target_sv already has the correct phase sign)
    w_snr = np.linalg.solve(R_rr, target_sv)
    
    # 3. Normalize the resulting vector to maintain unit norm
    norm_factor = np.linalg.norm(w_snr)
    if norm_factor > 1e-12:
        w_snr = w_snr / norm_factor
        
    return w_snr


def max_likelihood_weights(R_rr: np.ndarray, element_positions: np.ndarray, wavenumber_k: float, target_direction: tuple[float, float]) -> np.ndarray:
    """
    Computes the Maximum Likelihood (ML) beamforming weights.
    Estimates the target signal assuming a Gaussian noise/interference process.
    
    Args:
        R_rr (np.ndarray): Interference + Noise covariance matrix (N*M, N*M).
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        target_direction (tuple): Target DOA (theta_deg, phi_deg).
        
    Returns:
        np.ndarray: A 1D array containing complex weights.
    """
    # 1. Generate spatial signature WITH negative exponents (our convention)
    target_sv = get_steering_vector(element_positions, wavenumber_k, target_direction)
    
    # 2. Numerator: inv(R_rr) * target_sv
    # (No conjugation needed because target_sv already has the correct phase sign)
    numerator = np.linalg.solve(R_rr, target_sv)
    
    # 3. Denominator: target_sv^H * inv(R_rr) * target_sv
    # np.vdot(a, b) computes a^H * b, matching the mathematical formulation
    denominator = np.vdot(target_sv, numerator) 
    
    # 4. Scale weights. Denominator is theoretically real, but we ensure 
    # we don't carry floating-point imaginary artifacts.
    w_ml = numerator / np.real(denominator)
    
    return w_ml


def mvdr_weights(R_xx: np.ndarray, element_positions: np.ndarray, wavenumber_k: float, target_direction: tuple[float, float]) -> np.ndarray:
    """
    Computes the Minimum Variance Distortionless Response (MVDR) / Minimum Power (MP) weights.
    Forces unit gain towards the target while minimizing total output power.
    
    Args:
        R_xx (np.ndarray): Total received signal covariance matrix (N*M, N*M).
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        target_direction (tuple): Target DOA (theta_deg, phi_deg).
        
    Returns:
        np.ndarray: A 1D array containing complex weights.
    """
    # Spatial signature with negative exponents (our convention)
    target_sv = get_steering_vector(element_positions, wavenumber_k, target_direction)
    
    # Numerator: inv(R_xx) * v
    # Solved optimally via linear system: R_xx * x = v
    numerator = np.linalg.solve(R_xx, target_sv)
    
    # Denominator: v_H * inv(R_xx) * v
    # np.vdot applies the conjugate transpose to target_sv automatically
    denominator = np.vdot(target_sv, numerator)
    
    # Scale by the real part to prevent floating-point imaginary artifacts
    w_mvdr = numerator / np.real(denominator)
    
    return w_mvdr