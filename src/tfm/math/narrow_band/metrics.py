import numpy as np
from tfm.math.narrow_band.array_response import compute_array_factor, compute_beampattern

def compute_directivity(weights: np.ndarray, element_positions: np.ndarray, wavenumber_k: float, target_direction: tuple[float, float], theta_res: float = 1.0, phi_res: float = 1.0) -> float:
    """
    Computes the Directivity of the array in a specific target direction.
    
    Directivity (D) = 4 * pi * U(target) / P_rad
    where U is the radiation intensity and P_rad is the total radiated power.
    
    Args:
        weights (np.ndarray): Complex weights applied to the array elements.
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        target_direction (tuple): A tuple (theta_deg, phi_deg) representing the DOA in degrees.
        theta_res (float): Resolution for Theta integration in degrees.
        phi_res (float): Resolution for Phi integration in degrees.
        
    Returns:
        float: Directivity in dBi (decibels relative to an isotropic radiator).
    """
    # 1. Get the UNNORMALIZED radiation pattern for accurate power calculations
    af_mag, THETA, PHI = compute_beampattern(
        weights, element_positions, wavenumber_k, theta_res, phi_res, normalize=False
    )
    
    # 2. Calculate the Power Pattern U(theta, phi) = |AF|^2
    power_pattern = af_mag ** 2
    
    # 3. Calculate Total Radiated Power (P_rad) via numerical integration
    d_theta = np.deg2rad(theta_res)
    d_phi = np.deg2rad(phi_res)
    
    # Surface element in spherical coordinates: dS = sin(theta) * d_theta * d_phi
    radiated_power = np.sum(power_pattern * np.sin(THETA) * d_theta * d_phi)
    
    # 4. Calculate Radiation Intensity in the specific target direction
    target_voltage = compute_array_factor(weights, element_positions, wavenumber_k, target_direction)
    target_power = target_voltage ** 2
    
    # 5. Compute Directivity
    if radiated_power == 0:
        return 0.0
        
    directivity_linear = (4 * np.pi * target_power) / radiated_power
    
    if directivity_linear <= 1e-12:
        return -np.inf
        
    return 10 * np.log10(directivity_linear)

def compute_sinr(weights: np.ndarray, element_positions: np.ndarray, wavenumber_k: float, target_direction: tuple[float, float], target_power: float, jammers_directions: list[tuple[float, float]], jammers_powers: list[float], noise_power: float) -> float:
    """
    Computes the Signal-to-Interference-plus-Noise Ratio (SINR) at the output of the array.
    
    Args:
        weights (np.ndarray): Complex weights applied to the array elements.
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        target_direction (tuple): A tuple (theta_deg, phi_deg) representing the DOA of the legitimate signal.
        target_power (float): Received power of the legitimate signal (linear scale).
        jammers_directions (list of tuples): A list of (theta_deg, phi_deg) tuples for each active jammer.
        jammers_powers (list of floats): A list of received powers for each jammer (linear scale).
        noise_power (float): Thermal noise power at the receiver (linear scale).
        
    Returns:
        float: The computed SINR in decibels (dB).
    """
    # 1. Calculate received signal power
    signal_voltage_gain = compute_array_factor(weights, element_positions, wavenumber_k, target_direction)
    received_signal_power = target_power * (signal_voltage_gain ** 2)
    
    # 2. Calculate total interference power from all jammers
    total_interference_power = 0.0
    for jammer_dir, jammer_pow in zip(jammers_directions, jammers_powers):
        jammer_voltage_gain = compute_array_factor(weights, element_positions, wavenumber_k, jammer_dir)
        total_interference_power += jammer_pow * (jammer_voltage_gain ** 2)

    # 3. Calculate noise power at the array output
    w_vec = weights.flatten()
    w_norm = np.sum(np.abs(w_vec)**2)
    received_noise_power = noise_power * w_norm

    # 4. Calculate SINR in linear scale
    interference_plus_noise = total_interference_power + received_noise_power
    
    if interference_plus_noise == 0:
        return np.inf
        
    sinr_linear = received_signal_power / interference_plus_noise
    
    # 5. Convert to decibels (dB)
    if sinr_linear <= 1e-12:
        return -np.inf
        
    return 10 * np.log10(sinr_linear)

def evaluate_null_depth(weights: np.ndarray, element_positions: np.ndarray, wavenumber_k: float, target_direction: tuple[float, float], null_direction: tuple[float, float]) -> float:
    """
    Evaluates the depth of a null in a specific direction relative to the target direction.

    Args:
        weights (np.ndarray): Complex weights applied to the array elements.
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda).
        target_direction (tuple): A tuple (theta_deg, phi_deg) representing the main lobe DOA.
        null_direction (tuple): A tuple (theta_deg, phi_deg) representing the direction of the jammer.

    Returns:
        float: The relative null depth in decibels (dB). A positive value indicates 
               how many dBs the null direction is attenuated compared to the target direction.
               Returns np.inf if the null is practically perfect.
    """
    # 1. Get the voltage gain for both the desired direction and the jammer direction
    target_voltage = compute_array_factor(weights, element_positions, wavenumber_k, target_direction)
    null_voltage = compute_array_factor(weights, element_positions, wavenumber_k, null_direction)
    
    # 2. Convert voltage gain to power gain
    target_power = target_voltage ** 2
    null_power = null_voltage ** 2
    
    # 3. Robustness checks for edge cases
    if null_power <= 1e-12:
        return np.inf
        
    if target_power <= 1e-12:
        return -np.inf
        
    # 4. Calculate relative depth (Ratio of Target Power to Null Power)
    depth_linear = target_power / null_power
    
    # 5. Convert to decibels (dB)
    return 10 * np.log10(depth_linear)