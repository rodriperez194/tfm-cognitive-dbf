import numpy as np
from scipy.signal.windows import chebwin
from scipy.signal import remez
from tfm.math.narrow_band.steering_vector import get_steering_vector

def random_weights(num_elements: int, seed: int = None) -> np.ndarray:
    """
    Generates a complex weight vector with random amplitudes and phases.
    Useful for DRL environment exploration and random initialization.
    
    Args:
        num_elements (int): Total number of antenna elements (N * M).
        seed (int, optional): Random seed for reproducibility across episodes.
        
    Returns:
        np.ndarray: A 1D array of shape (num_elements,) containing complex weights
                    (type complex128).
    """
    if seed is not None:
        np.random.seed(seed)
        
    amplitudes = np.random.uniform(low=0.0, high=1.0, size=num_elements)
    phases = np.random.uniform(low=0.0, high=2 * np.pi, size=num_elements)
    weights = amplitudes * np.exp(1j * phases)
    
    return weights

def broadside_weights(num_elements: int) -> np.ndarray:
    """
    Generates a uniform weight vector to steer the main beam to broadside (theta = 0).
    All elements are assigned a weight of 1.0 + 0j (uniform amplitude, zero phase).
    
    Args:
        num_elements (int): Total number of antenna elements (N * M).
        
    Returns:
        np.ndarray: A 1D array of shape (num_elements,) containing complex weights 
                    (type complex128).
    """
    weights = np.ones(num_elements, dtype=np.complex128)
    return weights

def steering_weights(element_positions: np.ndarray, wavenumber_k: float, direction: tuple[float, float]) -> np.ndarray:
    """
    Generates complex phase-only weights to steer the array's main beam 
    towards a specific direction (DOA).
    
    Args:
        element_positions (np.ndarray): Array of shape (N, M, 3) or (N*M, 3) 
                                        containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda) of the carrier.
        direction (tuple): A tuple (theta_deg, phi_deg) representing the target DOA.
                           - theta_deg (float): Polar angle from Z-axis [0, 180].
                           - phi_deg (float): Azimuth angle from X-axis [0, 360).
        
    Returns:
        np.ndarray: A 1D array of shape (N*M,) containing complex weights 
                    (type complex128).
    """
    return get_steering_vector(element_positions, wavenumber_k, direction)

def hamming_weights(element_positions: np.ndarray, wavenumber_k: float, num_rows: int, num_cols: int, direction: tuple[float, float] | None = None) -> np.ndarray:
    """
    Generates a 2D Hamming window for amplitude tapering to reduce sidelobes.
    Optionally steers the beam if a target direction is provided.

    Args:
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda) of the carrier.
        num_rows (int): Number of antenna elements along the X-axis (N).
        num_cols (int): Number of antenna elements along the Y-axis (M).
        direction (tuple[float, float] | None): Optional target DOA (theta_deg, phi_deg).

    Returns:
        np.ndarray: A 1D flattened array of shape (N*M,) containing complex weights.
    """
    window_x = np.hamming(num_rows)
    window_y = np.hamming(num_cols)
    window_2d = np.outer(window_x, window_y)
    weights = window_2d.flatten().astype(np.complex128)

    if direction is not None:
        steering_vec = get_steering_vector(element_positions, wavenumber_k, direction)
        weights *= steering_vec

    return weights

def dolph_chebyshev_weights(element_positions: np.ndarray, wavenumber_k: float, num_rows: int, num_cols: int, sidelobe_level_db: float, direction: tuple[float, float] | None = None) -> np.ndarray:
    """
    Generates a 2D Dolph-Chebyshev window for optimal amplitude tapering.
    Optionally steers the beam if a target direction is provided.

    Args:
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda) of the carrier.
        num_rows (int): Number of antenna elements along the X-axis (N).
        num_cols (int): Number of antenna elements along the Y-axis (M).
        sidelobe_level_db (float): Required sidelobe attenuation in dB (positive).
        direction (tuple[float, float] | None): Optional target DOA (theta_deg, phi_deg).

    Returns:
        np.ndarray: A 1D flattened array of shape (N*M,) containing complex weights.
    """
    window_x = chebwin(num_rows, at=sidelobe_level_db)
    window_y = chebwin(num_cols, at=sidelobe_level_db)
    window_2d = np.outer(window_x, window_y)
    weights = window_2d.flatten().astype(np.complex128)

    if direction is not None:
        steering_vec = get_steering_vector(element_positions, wavenumber_k, direction)
        weights *= steering_vec

    return weights

def minimax_weights(element_positions: np.ndarray, wavenumber_k: float, num_rows: int, num_cols: int, passband_edge: float, stopband_edge: float, stopband_weight: float = 10.0, direction: tuple[float, float] | None = None) -> np.ndarray:
    """
    Generates a 2D Minimax (Parks-McClellan/Remez) window for amplitude tapering.
    Optimizes weights to minimize the maximum error in the specified bands.
    Optionally steers the beam if a target direction is provided.

    Args:
        element_positions (np.ndarray): Array containing the (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda) of the carrier.
        num_rows (int): Elements in X-axis (N).
        num_cols (int): Elements in Y-axis (M).
        passband_edge (float): End of the main lobe (normalized spatial freq, 0 to 0.5).
        stopband_edge (float): Start of the sidelobe region (must be > passband_edge).
        stopband_weight (float): Penalty for stopband error relative to passband.
        direction (tuple[float, float] | None): Optional target DOA (theta_deg, phi_deg).

    Returns:
        np.ndarray: A 1D flattened array of shape (N*M,) containing complex weights.
    """
    if not (0.0 <= passband_edge < stopband_edge <= 0.5):
        raise ValueError("passband_edge and stopband_edge must satisfy 0 <= passband_edge < stopband_edge <= 0.5")

    bands = [0.0, passband_edge, stopband_edge, 0.5]
    desired_gains = [1.0, 0.0]
    band_weights = [1.0, float(stopband_weight)]

    window_x = remez(numtaps=num_rows, bands=bands, desired=desired_gains, weight=band_weights, fs=1.0)
    window_y = remez(numtaps=num_cols, bands=bands, desired=desired_gains, weight=band_weights, fs=1.0)

    window_2d = np.outer(window_x, window_y)
    weights = window_2d.flatten().astype(np.complex128)

    if direction is not None:
        steering_vec = get_steering_vector(element_positions, wavenumber_k, direction)
        weights *= steering_vec

    return weights

def sampled_pattern_weights(num_rows: int, num_cols: int, desired_response: np.ndarray) -> np.ndarray:
    """
    Synthesizes complex weights by sampling a desired 2D spatial response 
    and applying a 2D Inverse Fast Fourier Transform (IFFT2).
    
    Args:
        num_rows (int): Number of antenna elements along the X-axis (N).
        num_cols (int): Number of antenna elements along the Y-axis (M).
        desired_response (np.ndarray): 2D array of shape (N, M) containing the target gain grid.
        
    Returns:
        np.ndarray: A 1D flattened array of shape (N*M,) containing complex weights.
    """
    if desired_response.shape != (num_rows, num_cols):
        raise ValueError(f"Shape of desired_response {desired_response.shape} must match ({num_rows}, {num_cols}).")
    
    shifted_response = np.fft.ifftshift(desired_response)
    weights_2d = np.fft.ifft2(shifted_response)
    weights_2d_centered = np.fft.fftshift(weights_2d)
    weights = weights_2d_centered.flatten().astype(np.complex128)
    
    return weights

def interference_suppression_weights(element_positions: np.ndarray, wavenumber_k: float, target_direction: tuple[float, float], jammer_direction: tuple[float, float]) -> np.ndarray:
    """
    Places a deterministic null in the jammer direction while steering towards the target 
    using spatial subtraction (linear combination), based on exact DOA angles.
    
    Args:
        element_positions (np.ndarray): Array containing (x, y, z) coordinates.
        wavenumber_k (float): The wavenumber (2 * pi / lambda) of the carrier.
        target_direction (tuple): (theta_deg, phi_deg) for the desired signal.
        jammer_direction (tuple): (theta_deg, phi_deg) for the interference to be nulled.
                                             
    Returns:
        np.ndarray: A 1D array of shape (N*M,) containing complex weights with a forced null.
    """
    # 1. Generate the spatial signatures using the common helper function
    v_target = get_steering_vector(element_positions, wavenumber_k, target_direction)
    v_jammer = get_steering_vector(element_positions, wavenumber_k, jammer_direction)
    
    num_elements = len(v_target)
    
    # 2. Base combiner pointing to the target (normalized by N)
    w_1 = v_target / num_elements
    
    # 3. Combiner pointing to the jammer (normalized by N)
    w_2 = v_jammer / num_elements
    
    # 4. Complex gain of the target weights in the direction of the jammer
    B1_theta0 = np.sum(w_1 * np.conj(v_jammer))
    
    # 5. Combine both to achieve the exact mathematical null
    w_null = w_1 - (B1_theta0 * w_2)

    weights = w_null.astype(np.complex128)
    
    return weights

def multi_interference_suppression_weights(
    element_positions: np.ndarray,
    wavenumber_k: float,
    target_direction: tuple[float, float],
    jammer_directions: list[tuple[float, float]],
    diagonal_loading: float = 1e-8,
    use_pinv: bool = False,
) -> np.ndarray:
    """
    Compute deterministic beamforming weights with unity response towards the
    target direction and spatial nulls towards multiple jammer directions.

    The method solves the linearly constrained minimum-norm problem:

        C^H w = f

    where C contains the steering vectors of the target and the jammers:

        C = [a_target, a_jammer_1, ..., a_jammer_K]

    and f imposes:

        w^H a_target = 1
        w^H a_jammer_k = 0

    Args:
        element_positions (np.ndarray): Array containing (x, y, z) coordinates.
        wavenumber_k (float): Wavenumber of the carrier.
        target_direction (tuple[float, float]): Target direction as (theta_deg, phi_deg).
        jammer_directions (list[tuple[float, float]]): Jammer directions as
            [(theta_deg, phi_deg), ...].
        diagonal_loading (float): Small regularization term for numerical stability.
        use_pinv (bool): If True, use pseudo-inverse instead of solve.

    Returns:
        np.ndarray: A 1D array of shape (num_elements,) containing complex weights.
    """

    # Target steering vector
    a_target = get_steering_vector(
        element_positions=element_positions,
        wavenumber_k=wavenumber_k,
        direction=target_direction,
    ).astype(np.complex128).reshape(-1)

    steering_vectors = [a_target]

    # Jammer steering vectors
    for jammer_direction in jammer_directions:
        a_jammer = get_steering_vector(
            element_positions=element_positions,
            wavenumber_k=wavenumber_k,
            direction=jammer_direction,
        ).astype(np.complex128).reshape(-1)

        steering_vectors.append(a_jammer)

    # Constraint matrix: C = [a_target, a_j1, ..., a_jK]
    C = np.column_stack(steering_vectors)

    num_constraints = C.shape[1]

    # Desired responses: unity gain for SOI, zero gain for jammers
    f = np.zeros(num_constraints, dtype=np.complex128)
    f[0] = 1.0 + 0.0j

    # Gram matrix
    gram = C.conj().T @ C

    if diagonal_loading > 0.0:
        gram = gram + diagonal_loading * np.eye(
            num_constraints,
            dtype=np.complex128,
        )

    if use_pinv:
        weights = C @ (np.linalg.pinv(gram) @ f)
    else:
        try:
            weights = C @ np.linalg.solve(gram, f)
        except np.linalg.LinAlgError:
            weights = C @ (np.linalg.pinv(gram) @ f)

    return weights.astype(np.complex128)

def target_or_zero_weights(
    element_positions: np.ndarray,
    wavenumber_k: float,
    target_directions: list[tuple[float, float]],
    zero_directions: list[tuple[float, float]],
    diagonal_loading: float = 1e-8,
    use_pinv: bool = False,
) -> np.ndarray:
    """
    Compute deterministic minimum-norm beamforming weights that impose
    unity response towards multiple target directions and spatial nulls
    towards multiple zero directions.

    The method solves the linearly constrained minimum-norm problem:

        C^H w = f

    where:

        C = [
            a_target_1,
            ...,
            a_target_Ns,
            a_zero_1,
            ...,
            a_zero_N0,
        ]

    and:

        f = [1, ..., 1, 0, ..., 0]^T

    Therefore, the resulting weights satisfy approximately:

        w^H a(target_i) = 1
        w^H a(zero_j) = 0

    Args:
        element_positions (np.ndarray): Array containing the antenna element
            coordinates with shape (N, M, 3) or (N*M, 3).
        wavenumber_k (float): Wavenumber of the carrier
            defined as 2 * pi / wavelength.
        target_directions (list[tuple[float, float]]): Directions where
            unity response is required, expressed as
            [(theta_deg, phi_deg), ...].
        zero_directions (list[tuple[float, float]]): Directions where
            spatial nulls are required, expressed as
            [(theta_deg, phi_deg), ...].
        diagonal_loading (float): Non-negative regularization term added
            to C^H C for numerical stability.
        use_pinv (bool): If True, use the Moore-Penrose pseudo-inverse
            instead of solving the linear system directly.

    Returns:
        np.ndarray: A 1D array of shape (num_elements,) containing the
            complex minimum-norm beamforming weights.

    Raises:
        ValueError: If no target direction is provided, if diagonal loading
            is negative, or if the number of constraints exceeds the number
            of antenna elements.
    """
    if len(target_directions) == 0:
        raise ValueError(
            "At least one target direction must be provided."
        )

    if diagonal_loading < 0.0:
        raise ValueError(
            "diagonal_loading must be greater than or equal to zero."
        )

    constraint_directions = [
        *target_directions,
        *zero_directions,
    ]

    steering_vectors = []

    for direction in constraint_directions:
        steering_vector = get_steering_vector(
            element_positions=element_positions,
            wavenumber_k=wavenumber_k,
            direction=direction,
        ).astype(np.complex128).reshape(-1)

        steering_vectors.append(steering_vector)

    # Constraint matrix:
    # C = [a_target_1, ..., a_target_Ns, a_zero_1, ..., a_zero_N0]
    C = np.column_stack(steering_vectors)

    num_elements = C.shape[0]
    num_targets = len(target_directions)
    num_constraints = C.shape[1]

    if num_constraints > num_elements:
        raise ValueError(
            f"The number of constraints ({num_constraints}) cannot exceed "
            f"the number of antenna elements ({num_elements})."
        )

    # Desired responses:
    # f = [1, ..., 1, 0, ..., 0]^T
    desired_response = np.zeros(
        num_constraints,
        dtype=np.complex128,
    )
    desired_response[:num_targets] = 1.0 + 0.0j

    # Gram matrix: C^H C
    gram = C.conj().T @ C

    if diagonal_loading > 0.0:
        gram = gram + diagonal_loading * np.eye(
            num_constraints,
            dtype=np.complex128,
        )

    if use_pinv:
        weights = C @ (
            np.linalg.pinv(gram) @ desired_response
        )
    else:
        try:
            weights = C @ np.linalg.solve(
                gram,
                desired_response,
            )
        except np.linalg.LinAlgError:
            weights = C @ (
                np.linalg.pinv(gram) @ desired_response
            )

    return weights.astype(np.complex128)