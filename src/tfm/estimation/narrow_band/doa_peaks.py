import numpy as np


def extract_top_doa_peaks(
    spectrum_flat,
    theta_grid_deg,
    phi_grid_deg,
    num_peaks=2,
    guard_theta_bins=2,
    guard_phi_bins=5,
    min_relative_power=0.0,
):
    """
    Extract top 'num_peaks' DOA peaks from a flat spectrum on a theta-phi grid using 2D NMS.

    Assumes the spectrum is ordered as:
        theta_mesh, phi_mesh = np.meshgrid(theta_grid_deg, phi_grid_deg, indexing="ij")
        scan_angles = np.stack([theta_mesh.ravel(), phi_mesh.ravel()], axis=1)
        spectrum_flat aligns with scan_angles order

    Args:
        spectrum_flat (np.ndarray): Flat spatial spectrum of shape (K,).
        theta_grid_deg (np.ndarray): Theta scan grid in degrees.
        phi_grid_deg (np.ndarray): Phi scan grid in degrees.
        num_peaks (int): Maximum number of peaks to extract.
        guard_theta_bins (int): NMS suppression half-width in theta bins.
        guard_phi_bins (int): NMS suppression half-width in phi bins (with wrap-around).
        min_relative_power (float): Minimum peak power relative to the global spectrum
            maximum, in the range [0.0, 1.0]. Peaks below this threshold are discarded.
            Default is 0.0 (no filtering). Example: 0.1 discards any peak whose absolute
            value is less than 10% of the strongest peak in the spectrum.

    Returns:
        list of tuple: Each element is (theta_deg, phi_deg, value), sorted by descending
            power. May contain fewer than num_peaks entries if candidates fall below
            min_relative_power.
    """
    n_theta = len(theta_grid_deg)
    n_phi = len(phi_grid_deg)

    spec_2d = np.asarray(spectrum_flat).reshape(n_theta, n_phi).copy()

    # Global maximum used as reference for relative power threshold.
    # Computed once before NMS starts so that suppression does not affect the reference.
    global_max = float(np.max(spec_2d))

    # Absolute power threshold derived from the relative one.
    # If global_max is zero or negative (degenerate spectrum), threshold is set to
    # -inf so that no peak is rejected on power grounds.
    if global_max > 0.0 and min_relative_power > 0.0:
        abs_threshold = min_relative_power * global_max
    else:
        abs_threshold = -np.inf

    peaks = []

    for _ in range(num_peaks):

        idx = np.argmax(spec_2d)
        i_theta, j_phi = np.unravel_index(idx, spec_2d.shape)
        peak_val = float(spec_2d[i_theta, j_phi])

        # Stop if the best remaining candidate is below the absolute threshold.
        # This also catches the case where all remaining values are -inf
        # after NMS suppression.
        if peak_val < abs_threshold:
            break

        theta_est = float(theta_grid_deg[i_theta])
        phi_est = float(phi_grid_deg[j_phi])
        peaks.append((theta_est, phi_est, peak_val))

        # Non-maximum suppression neighborhood in theta (no wrap-around).
        i0 = max(0, i_theta - guard_theta_bins)
        i1 = min(n_theta, i_theta + guard_theta_bins + 1)

        # Phi suppression with wrap-around.
        for it in range(i0, i1):
            for jp in range(j_phi - guard_phi_bins, j_phi + guard_phi_bins + 1):
                spec_2d[it, jp % n_phi] = -np.inf

    return peaks