import numpy as np

from tfm.math.narrow_band.steering_vector import get_steering_vector

def compute_covariance_matrices(array, target_direction, target_power, jammers_directions, jammers_powers, noise_power):
    """
    Computes the theoretical spatial covariance matrices of the received signals.
    Essential for optimal stochastic beamforming algorithms.
    
    Mathematical Formulation:
        R_rr = Sum(P_j * a_j * a_j^H) + noise_power * I
        R_ss = P_s * a_s * a_s^H
        R_xx = R_ss + R_rr
        
    Args:
        array (Phased_Array_NB): Instance of the phased array to compute steering vectors.
        target_direction (tuple): (theta_deg, phi_deg) of the desired signal.
        target_power (float): Linear power of the desired signal.
        jammers_directions (list of tuples): DOAs of the active jammers.
        jammers_powers (list of floats): Linear powers of the active jammers.
        noise_power (float): Thermal noise floor variance (linear).
        
    Returns:
        tuple: (R_xx, R_rr) both as complex np.ndarray of shape (num_elements, num_elements).
    """
    num_elements = array.N * array.M
    
    # 1. Initialize R_rr (Interference + Noise Autocorrelation Matrix)
    # Modeled as uncorrelated thermal noise (scaled Identity matrix)
    R_rr = noise_power * np.eye(num_elements, dtype=complex)
    
    # Add spatial correlation contribution of each independent jammer
    for jammer_dir, jammer_pow in zip(jammers_directions, jammers_powers):
        a_j = get_steering_vector(
            element_positions=array.element_positions,
            wavenumber_k=array.k_num,
            direction=jammer_dir
            )
        R_rr += jammer_pow * np.outer(a_j, np.conj(a_j))
        
    # 2. Calculate the Target Signal Covariance Matrix (R_ss)
    a_s = get_steering_vector(
        element_positions=array.element_positions,
        wavenumber_k=array.k_num,
        direction=target_direction
        )
    R_ss = target_power * np.outer(a_s, np.conj(a_s))
    
    # 3. Calculate Total Received Signal Covariance Matrix (R_xx)
    R_xx = R_ss + R_rr
    
    return R_xx, R_rr