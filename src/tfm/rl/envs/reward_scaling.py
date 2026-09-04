# ============================================================
# Reward coefficient scaling utilities
# ============================================================

def build_scaled_reward_coefficients(
    alpha: float,
    beta: float,
    gamma_soi: float,
    gamma_jammer: float,
    sinr_scale_db: float = 30.0,
    sinr_loss_scale_db: float = 1.0,
    soi_angle_error_scale_deg: float = 10.0,
    jammer_angle_error_scale_deg: float = 10.0,
) -> dict:
    """
    Convert interpretable reward weights into BeamformingEnv coefficients.

    BeamformingEnvPhase5 combines reward components that may have very
    different numerical scales:

        r_sinr          = sinr_db
        r_sinr_loss     = - clipped_sinr_loss_db
        r_soi_angle     = - (soi_angle_error_deg / 180)^2
        r_jammer_angle  = - mean_j [(jammer_angle_error_j_deg / 180)^2]

    Therefore, alpha, beta, gamma_soi and gamma_jammer should not be passed
    directly to the environment if they are intended to represent relative
    interpretable weights.

    This helper rescales the interpretable weights according to characteristic
    component scales:

        reward_alpha_sinr =
            alpha / sinr_scale_db

        reward_beta_sinr_loss =
            beta / sinr_loss_scale_db

        reward_gamma_soi_angle =
            gamma_soi / (soi_angle_error_scale_deg / 180)^2

        reward_gamma_jammer_angle =
            gamma_jammer / (jammer_angle_error_scale_deg / 180)^2

    Parameters
    ----------
    alpha : float
        Interpretable weight associated with the absolute SINR reward term.

    beta : float
        Interpretable weight associated with the SINR-loss penalty term.

    gamma_soi : float
        Interpretable weight associated with the SOI angular-error penalty.

    gamma_jammer : float
        Interpretable weight associated with the jammer angular-error penalty.

    sinr_scale_db : float, optional
        Reference SINR scale in dB. For desired_power=1 and noise_power=1e-3,
        the nominal SNR is around 30 dB.

    sinr_loss_scale_db : float, optional
        Reference SINR-loss scale in dB. A value of 1 dB is suitable when
        losses above 1 dB are considered clearly undesirable.

    soi_angle_error_scale_deg : float, optional
        Reference SOI angular error in degrees. A value of 10 degrees makes
        the internal SOI angular term comparable to one unit of normalized
        penalty at 10 degrees.

    jammer_angle_error_scale_deg : float, optional
        Reference jammer angular error in degrees. A value of 10 degrees makes
        the internal jammer angular term comparable to one unit of normalized
        penalty at 10 degrees.

    Returns
    -------
    dict
        Dictionary containing the effective BeamformingEnv coefficients,
        the original interpretable weights, and the scaling constants used.
    """

    alpha = float(alpha)
    beta = float(beta)
    gamma_soi = float(gamma_soi)
    gamma_jammer = float(gamma_jammer)

    sinr_scale_db = float(sinr_scale_db)
    sinr_loss_scale_db = float(sinr_loss_scale_db)
    soi_angle_error_scale_deg = float(soi_angle_error_scale_deg)
    jammer_angle_error_scale_deg = float(jammer_angle_error_scale_deg)

    if sinr_scale_db <= 0.0:
        raise ValueError("sinr_scale_db must be greater than zero.")

    if sinr_loss_scale_db <= 0.0:
        raise ValueError("sinr_loss_scale_db must be greater than zero.")

    if soi_angle_error_scale_deg <= 0.0:
        raise ValueError("soi_angle_error_scale_deg must be greater than zero.")

    if jammer_angle_error_scale_deg <= 0.0:
        raise ValueError("jammer_angle_error_scale_deg must be greater than zero.")

    soi_angle_component_scale = (soi_angle_error_scale_deg / 180.0) ** 2
    jammer_angle_component_scale = (jammer_angle_error_scale_deg / 180.0) ** 2

    reward_alpha_sinr = alpha / sinr_scale_db
    reward_beta_sinr_loss = beta / sinr_loss_scale_db
    reward_gamma_soi_angle = gamma_soi / soi_angle_component_scale
    reward_gamma_jammer_angle = gamma_jammer / jammer_angle_component_scale

    return {
        # Effective coefficients passed to BeamformingEnvPhase5
        "reward_alpha_sinr": float(reward_alpha_sinr),
        "reward_beta_sinr_loss": float(reward_beta_sinr_loss),
        "reward_gamma_soi_angle": float(reward_gamma_soi_angle),
        "reward_gamma_jammer_angle": float(reward_gamma_jammer_angle),

        # Interpretable user-defined weights
        "interpretable_alpha_sinr": float(alpha),
        "interpretable_beta_sinr_loss": float(beta),
        "interpretable_gamma_soi_angle": float(gamma_soi),
        "interpretable_gamma_jammer_angle": float(gamma_jammer),

        # Scaling constants
        "sinr_component_scale_db": float(sinr_scale_db),
        "sinr_loss_component_scale_db": float(sinr_loss_scale_db),
        "soi_angle_component_scale": float(soi_angle_component_scale),
        "jammer_angle_component_scale": float(jammer_angle_component_scale),
        "soi_angle_error_scale_deg": float(soi_angle_error_scale_deg),
        "jammer_angle_error_scale_deg": float(jammer_angle_error_scale_deg),

        # Human-readable definition
        "reward_definition": (
            "Scaled hybrid reward coefficients for BeamformingEnvPhase5. "
            "Effective coefficients are computed as: "
            "alpha_eff = alpha / sinr_scale_db, "
            "beta_eff = beta / sinr_loss_scale_db, "
            "gamma_soi_eff = gamma_soi / ((soi_angle_error_scale_deg / 180)^2), "
            "gamma_jammer_eff = gamma_jammer / "
            "((jammer_angle_error_scale_deg / 180)^2)."
        ),
    }

def build_phase5_scaled_reward_config(
    alpha_sinr: float = 0.0,
    beta_sinr_loss: float = 1.0,
    gamma_soi_gain_loss: float = 1.0,
    gamma_jammer_leakage: float = 1.0,
    normalize_reward_coefficients: bool = False,
    sinr_scale_db: float = 30.0,
    sinr_loss_scale_db: float = 60.0,
    soi_gain_loss_scale_db: float = 30.0,
    jammer_leakage_scale: float = 1.0,
    reward_bonus_good_soi: float = 0.0,
    reward_bonus_good_jammer: float = 0.0,
    reward_bonus_good_sinr_loss: float = 0.0,
    soi_gain_loss_bonus_threshold_db: float = 1.0,
    jammer_leakage_bonus_threshold: float = 0.01,
    sinr_loss_bonus_threshold_db: float = 1.0,
    max_sinr_loss_db: float = 60.0,
    max_soi_gain_loss_db: float = 60.0,
    max_jammer_leakage_loss: float = 10.0,
) -> dict:
    """
    Build a complete reward configuration for BeamformingEnvPhase5.

    This helper is the Phase 5 equivalent of the older reward-scaling
    utilities used in previous phases.

    BeamformingEnvPhase5 computes, at each physical substep:

        reward =
            alpha * normalized_sinr
            - beta * normalized_sinr_loss
            - gamma_soi * normalized_soi_gain_loss
            - gamma_jammer * normalized_jammer_leakage_loss
            + milestone_bonus

    where:

        normalized_sinr =
            sinr_db / sinr_scale_db

        normalized_sinr_loss =
            clipped_sinr_loss_db / sinr_loss_scale_db

        normalized_soi_gain_loss =
            clipped_soi_gain_loss_db / soi_gain_loss_scale_db

        normalized_jammer_leakage_loss =
            clipped_jammer_leakage_loss / jammer_leakage_scale

    This function does not assume a specific number of jammers, temporal
    window K, action representation, or training curriculum. It only builds
    the reward-related keyword arguments expected by BeamformingEnvPhase5.

    Parameters
    ----------
    alpha_sinr : float
        Weight of the absolute SINR reward term.

    beta_sinr_loss : float
        Weight of the SINR-loss penalty term. Higher values penalize being
        far from the instantaneous MVDR reference.

    gamma_soi_gain_loss : float
        Weight of the SOI gain-loss penalty term. Higher values preserve the
        main lobe toward the desired source.

    gamma_jammer_leakage : float
        Weight of the jammer leakage penalty term. Higher values encourage
        lower gain toward active jammer directions relative to the SOI gain.

    normalize_reward_coefficients : bool
        If True, BeamformingEnvPhase5 normalizes the four coefficients so
        that their sum equals one. If False, the values are used directly.

    sinr_scale_db : float
        Scale used by BeamformingEnvPhase5 to normalize sinr_db.

    sinr_loss_scale_db : float
        Scale used by BeamformingEnvPhase5 to normalize clipped SINR loss.

    soi_gain_loss_scale_db : float
        Scale used by BeamformingEnvPhase5 to normalize clipped SOI gain loss.

    jammer_leakage_scale : float
        Scale used by BeamformingEnvPhase5 to normalize clipped jammer leakage.

    reward_bonus_good_soi : float
        Bonus added when SOI gain loss is below
        soi_gain_loss_bonus_threshold_db.

    reward_bonus_good_jammer : float
        Bonus added when jammer leakage is below
        jammer_leakage_bonus_threshold.

    reward_bonus_good_sinr_loss : float
        Bonus added when clipped SINR loss is below
        sinr_loss_bonus_threshold_db.

    soi_gain_loss_bonus_threshold_db : float
        SOI gain-loss threshold in dB used for the SOI milestone.

    jammer_leakage_bonus_threshold : float
        Linear jammer leakage threshold used for the jammer milestone.

    sinr_loss_bonus_threshold_db : float
        SINR-loss threshold in dB used for the SINR-loss milestone.

    max_sinr_loss_db : float
        Maximum SINR loss used for clipping inside BeamformingEnvPhase5.

    max_soi_gain_loss_db : float
        Maximum SOI gain loss used for clipping inside BeamformingEnvPhase5.

    max_jammer_leakage_loss : float
        Maximum jammer leakage used for clipping inside BeamformingEnvPhase5.

    Returns
    -------
    dict
        Dictionary directly compatible with BeamformingEnvPhase5 constructor.
    """

    alpha_sinr = float(alpha_sinr)
    beta_sinr_loss = float(beta_sinr_loss)
    gamma_soi_gain_loss = float(gamma_soi_gain_loss)
    gamma_jammer_leakage = float(gamma_jammer_leakage)

    normalize_reward_coefficients = bool(normalize_reward_coefficients)

    sinr_scale_db = float(sinr_scale_db)
    sinr_loss_scale_db = float(sinr_loss_scale_db)
    soi_gain_loss_scale_db = float(soi_gain_loss_scale_db)
    jammer_leakage_scale = float(jammer_leakage_scale)

    reward_bonus_good_soi = float(reward_bonus_good_soi)
    reward_bonus_good_jammer = float(reward_bonus_good_jammer)
    reward_bonus_good_sinr_loss = float(reward_bonus_good_sinr_loss)

    soi_gain_loss_bonus_threshold_db = float(
        soi_gain_loss_bonus_threshold_db
    )
    jammer_leakage_bonus_threshold = float(
        jammer_leakage_bonus_threshold
    )
    sinr_loss_bonus_threshold_db = float(
        sinr_loss_bonus_threshold_db
    )

    max_sinr_loss_db = float(max_sinr_loss_db)
    max_soi_gain_loss_db = float(max_soi_gain_loss_db)
    max_jammer_leakage_loss = float(max_jammer_leakage_loss)

    reward_coefficients = {
        "alpha_sinr": alpha_sinr,
        "beta_sinr_loss": beta_sinr_loss,
        "gamma_soi_gain_loss": gamma_soi_gain_loss,
        "gamma_jammer_leakage": gamma_jammer_leakage,
    }

    for name, value in reward_coefficients.items():
        if value < 0.0:
            raise ValueError(
                f"{name} must be non-negative. Received {value}."
            )

    reward_scales = {
        "sinr_scale_db": sinr_scale_db,
        "sinr_loss_scale_db": sinr_loss_scale_db,
        "soi_gain_loss_scale_db": soi_gain_loss_scale_db,
        "jammer_leakage_scale": jammer_leakage_scale,
    }

    for name, value in reward_scales.items():
        if value <= 0.0:
            raise ValueError(
                f"{name} must be positive. Received {value}."
            )

    clipping_values = {
        "max_sinr_loss_db": max_sinr_loss_db,
        "max_soi_gain_loss_db": max_soi_gain_loss_db,
        "max_jammer_leakage_loss": max_jammer_leakage_loss,
    }

    for name, value in clipping_values.items():
        if value <= 0.0:
            raise ValueError(
                f"{name} must be positive. Received {value}."
            )

    if soi_gain_loss_bonus_threshold_db < 0.0:
        raise ValueError(
            "soi_gain_loss_bonus_threshold_db must be non-negative."
        )

    if jammer_leakage_bonus_threshold < 0.0:
        raise ValueError(
            "jammer_leakage_bonus_threshold must be non-negative."
        )

    if sinr_loss_bonus_threshold_db < 0.0:
        raise ValueError(
            "sinr_loss_bonus_threshold_db must be non-negative."
        )

    config = {
        "reward_alpha_sinr": alpha_sinr,
        "reward_beta_sinr_loss": beta_sinr_loss,
        "reward_gamma_soi_gain_loss": gamma_soi_gain_loss,
        "reward_gamma_jammer_leakage": gamma_jammer_leakage,

        "normalize_reward_coefficients": normalize_reward_coefficients,

        "sinr_scale_db": sinr_scale_db,
        "sinr_loss_scale_db": sinr_loss_scale_db,
        "soi_gain_loss_scale_db": soi_gain_loss_scale_db,
        "jammer_leakage_scale": jammer_leakage_scale,

        "reward_bonus_good_soi": reward_bonus_good_soi,
        "reward_bonus_good_jammer": reward_bonus_good_jammer,
        "reward_bonus_good_sinr_loss": reward_bonus_good_sinr_loss,

        "soi_gain_loss_bonus_threshold_db": (
            soi_gain_loss_bonus_threshold_db
        ),
        "jammer_leakage_bonus_threshold": (
            jammer_leakage_bonus_threshold
        ),
        "sinr_loss_bonus_threshold_db": (
            sinr_loss_bonus_threshold_db
        ),

        "max_sinr_loss_db": max_sinr_loss_db,
        "max_soi_gain_loss_db": max_soi_gain_loss_db,
        "max_jammer_leakage_loss": max_jammer_leakage_loss,

        "phase": "phase_5",
        "reward_type": "physical_beamforming_reward",
        "reward_components": {
            "sinr": {
                "coefficient": alpha_sinr,
                "scale_db": sinr_scale_db,
                "sign": "+",
                "description": "Absolute SINR reward term.",
            },
            "sinr_loss": {
                "coefficient": beta_sinr_loss,
                "scale_db": sinr_loss_scale_db,
                "sign": "-",
                "description": (
                    "Penalty with respect to the instantaneous MVDR "
                    "reference SINR."
                ),
            },
            "soi_gain_loss": {
                "coefficient": gamma_soi_gain_loss,
                "scale_db": soi_gain_loss_scale_db,
                "sign": "-",
                "description": (
                    "Penalty for losing array gain toward the SOI compared "
                    "with conventional steering."
                ),
            },
            "jammer_leakage": {
                "coefficient": gamma_jammer_leakage,
                "scale": jammer_leakage_scale,
                "sign": "-",
                "description": (
                    "Penalty for jammer-direction gain relative to SOI gain."
                ),
            },
            "milestone_bonus": {
                "reward_bonus_good_soi": reward_bonus_good_soi,
                "reward_bonus_good_jammer": reward_bonus_good_jammer,
                "reward_bonus_good_sinr_loss": (
                    reward_bonus_good_sinr_loss
                ),
                "soi_gain_loss_bonus_threshold_db": (
                    soi_gain_loss_bonus_threshold_db
                ),
                "jammer_leakage_bonus_threshold": (
                    jammer_leakage_bonus_threshold
                ),
                "sinr_loss_bonus_threshold_db": (
                    sinr_loss_bonus_threshold_db
                ),
                "description": (
                    "Optional bounded milestone bonuses for reaching "
                    "physically meaningful performance regions."
                ),
            },
        },
        "reward_definition": (
            "BeamformingEnvPhase5 reward: "
            "reward_alpha_sinr * (sinr_db / sinr_scale_db) "
            "- reward_beta_sinr_loss * "
            "(clipped_sinr_loss_db / sinr_loss_scale_db) "
            "- reward_gamma_soi_gain_loss * "
            "(clipped_soi_gain_loss_db / soi_gain_loss_scale_db) "
            "- reward_gamma_jammer_leakage * "
            "(clipped_jammer_leakage_loss / jammer_leakage_scale) "
            "+ milestone_bonus."
        ),
    }

    return config

def build_phase5_residual_phase_scaled_reward_config(
    alpha_sinr: float = 0.0,
    beta_sinr_loss: float = 1.0,
    gamma_soi_gain_loss: float = 0.5,
    gamma_jammer_leakage: float = 0.5,
    gamma_residual_phase: float = 0.05,
    gamma_base_improvement: float = 0.0,
    normalize_reward_coefficients: bool = True,
    sinr_scale_db: float = 30.0,
    sinr_loss_scale_db: float = 60.0,
    soi_gain_loss_scale_db: float = 30.0,
    jammer_leakage_scale: float = 10.0,
    residual_phase_loss_scale: float = 1.0,
    base_improvement_scale_db: float = 30.0,
    base_improvement_clip_db: float = 60.0,
    reward_bonus_good_soi: float = 0.0,
    reward_bonus_good_jammer: float = 0.0,
    reward_bonus_good_sinr_loss: float = 0.0,
    soi_gain_loss_bonus_threshold_db: float = 1.0,
    jammer_leakage_bonus_threshold: float = 0.01,
    sinr_loss_bonus_threshold_db: float = 1.0,
    max_sinr_loss_db: float = 60.0,
    max_soi_gain_loss_db: float = 60.0,
    max_jammer_leakage_loss: float = 30.0,
    max_residual_phase_loss: float = 1.0,
) -> dict:
    """
    Build a complete reward configuration for the residual BeamformingEnvPhase5
    variant.

    This helper supports residual phase-only, residual magnitude-phase and
    residual real-imaginary action modes, depending on the environment
    complex_weight_mode.

    The environment computes, at each physical substep:

        reward =
            alpha * normalized_sinr
            - beta * normalized_sinr_loss
            - gamma_soi * normalized_soi_gain_loss
            - gamma_jammer * normalized_jammer_leakage_loss
            - gamma_residual * normalized_residual_control_loss
            + gamma_base * normalized_base_improvement
            + milestone_bonus

    where:

        base_improvement_db =
            base_sinr_loss_db - sinr_loss_db

    Therefore:

        base_improvement_db > 0
            The agent improves the deterministic steering base.

        base_improvement_db = 0
            The agent matches the deterministic steering base.

        base_improvement_db < 0
            The agent is worse than the deterministic steering base.
    """

    alpha_sinr = float(alpha_sinr)
    beta_sinr_loss = float(beta_sinr_loss)
    gamma_soi_gain_loss = float(gamma_soi_gain_loss)
    gamma_jammer_leakage = float(gamma_jammer_leakage)
    gamma_residual_phase = float(gamma_residual_phase)
    gamma_base_improvement = float(gamma_base_improvement)

    normalize_reward_coefficients = bool(normalize_reward_coefficients)

    sinr_scale_db = float(sinr_scale_db)
    sinr_loss_scale_db = float(sinr_loss_scale_db)
    soi_gain_loss_scale_db = float(soi_gain_loss_scale_db)
    jammer_leakage_scale = float(jammer_leakage_scale)
    residual_phase_loss_scale = float(residual_phase_loss_scale)
    base_improvement_scale_db = float(base_improvement_scale_db)
    base_improvement_clip_db = float(base_improvement_clip_db)

    reward_bonus_good_soi = float(reward_bonus_good_soi)
    reward_bonus_good_jammer = float(reward_bonus_good_jammer)
    reward_bonus_good_sinr_loss = float(reward_bonus_good_sinr_loss)

    soi_gain_loss_bonus_threshold_db = float(
        soi_gain_loss_bonus_threshold_db
    )
    jammer_leakage_bonus_threshold = float(
        jammer_leakage_bonus_threshold
    )
    sinr_loss_bonus_threshold_db = float(
        sinr_loss_bonus_threshold_db
    )

    max_sinr_loss_db = float(max_sinr_loss_db)
    max_soi_gain_loss_db = float(max_soi_gain_loss_db)
    max_jammer_leakage_loss = float(max_jammer_leakage_loss)
    max_residual_phase_loss = float(max_residual_phase_loss)

    reward_coefficients = {
        "alpha_sinr": alpha_sinr,
        "beta_sinr_loss": beta_sinr_loss,
        "gamma_soi_gain_loss": gamma_soi_gain_loss,
        "gamma_jammer_leakage": gamma_jammer_leakage,
        "gamma_residual_phase": gamma_residual_phase,
        "gamma_base_improvement": gamma_base_improvement,
    }

    for name, value in reward_coefficients.items():
        if value < 0.0:
            raise ValueError(
                f"{name} must be non-negative. Received {value}."
            )

    reward_scales = {
        "sinr_scale_db": sinr_scale_db,
        "sinr_loss_scale_db": sinr_loss_scale_db,
        "soi_gain_loss_scale_db": soi_gain_loss_scale_db,
        "jammer_leakage_scale": jammer_leakage_scale,
        "residual_phase_loss_scale": residual_phase_loss_scale,
        "base_improvement_scale_db": base_improvement_scale_db,
    }

    for name, value in reward_scales.items():
        if value <= 0.0:
            raise ValueError(
                f"{name} must be positive. Received {value}."
            )

    clipping_values = {
        "max_sinr_loss_db": max_sinr_loss_db,
        "max_soi_gain_loss_db": max_soi_gain_loss_db,
        "max_jammer_leakage_loss": max_jammer_leakage_loss,
        "max_residual_phase_loss": max_residual_phase_loss,
        "base_improvement_clip_db": base_improvement_clip_db,
    }

    for name, value in clipping_values.items():
        if value <= 0.0:
            raise ValueError(
                f"{name} must be positive. Received {value}."
            )

    if soi_gain_loss_bonus_threshold_db < 0.0:
        raise ValueError(
            "soi_gain_loss_bonus_threshold_db must be non-negative."
        )

    if jammer_leakage_bonus_threshold < 0.0:
        raise ValueError(
            "jammer_leakage_bonus_threshold must be non-negative."
        )

    if sinr_loss_bonus_threshold_db < 0.0:
        raise ValueError(
            "sinr_loss_bonus_threshold_db must be non-negative."
        )

    config = {
        "reward_alpha_sinr": alpha_sinr,
        "reward_beta_sinr_loss": beta_sinr_loss,
        "reward_gamma_soi_gain_loss": gamma_soi_gain_loss,
        "reward_gamma_jammer_leakage": gamma_jammer_leakage,
        "reward_gamma_residual_phase": gamma_residual_phase,
        "reward_gamma_base_improvement": gamma_base_improvement,

        "normalize_reward_coefficients": normalize_reward_coefficients,

        "sinr_scale_db": sinr_scale_db,
        "sinr_loss_scale_db": sinr_loss_scale_db,
        "soi_gain_loss_scale_db": soi_gain_loss_scale_db,
        "jammer_leakage_scale": jammer_leakage_scale,
        "residual_phase_loss_scale": residual_phase_loss_scale,
        "base_improvement_scale_db": base_improvement_scale_db,
        "base_improvement_clip_db": base_improvement_clip_db,

        "reward_bonus_good_soi": reward_bonus_good_soi,
        "reward_bonus_good_jammer": reward_bonus_good_jammer,
        "reward_bonus_good_sinr_loss": reward_bonus_good_sinr_loss,

        "soi_gain_loss_bonus_threshold_db": (
            soi_gain_loss_bonus_threshold_db
        ),
        "jammer_leakage_bonus_threshold": (
            jammer_leakage_bonus_threshold
        ),
        "sinr_loss_bonus_threshold_db": (
            sinr_loss_bonus_threshold_db
        ),

        "max_sinr_loss_db": max_sinr_loss_db,
        "max_soi_gain_loss_db": max_soi_gain_loss_db,
        "max_jammer_leakage_loss": max_jammer_leakage_loss,
        "max_residual_phase_loss": max_residual_phase_loss,

        "phase": "phase_5",
        "experimental_folder": "phase_6",
        "reward_type": (
            "residual_complex_physical_beamforming_reward_with_base_improvement"
        ),
        "reward_components": {
            "sinr": {
                "coefficient": alpha_sinr,
                "scale_db": sinr_scale_db,
                "sign": "+",
                "description": "Absolute SINR reward term.",
            },
            "sinr_loss": {
                "coefficient": beta_sinr_loss,
                "scale_db": sinr_loss_scale_db,
                "sign": "-",
                "description": (
                    "Penalty with respect to the instantaneous MVDR "
                    "reference SINR."
                ),
            },
            "soi_gain_loss": {
                "coefficient": gamma_soi_gain_loss,
                "scale_db": soi_gain_loss_scale_db,
                "sign": "-",
                "description": (
                    "Penalty for losing array gain toward the SOI compared "
                    "with conventional phase-only steering."
                ),
            },
            "jammer_leakage": {
                "coefficient": gamma_jammer_leakage,
                "scale": jammer_leakage_scale,
                "sign": "-",
                "description": (
                    "Penalty for jammer-direction gain relative to SOI gain."
                ),
            },
            "residual_control_loss": {
                "coefficient": gamma_residual_phase,
                "scale": residual_phase_loss_scale,
                "max_value": max_residual_phase_loss,
                "sign": "-",
                "description": (
                    "Penalty for large residual corrections with respect to "
                    "the internal base SOI-steering weights. In phase_only "
                    "mode this is the mean squared normalized residual phase. "
                    "In mag_phase and real_imag modes this is the mean "
                    "squared normalized residual action vector."
                ),
            },
            "base_improvement": {
                "coefficient": gamma_base_improvement,
                "scale_db": base_improvement_scale_db,
                "clip_db": base_improvement_clip_db,
                "sign": "+/-",
                "description": (
                    "Reward component based on improvement over the "
                    "deterministic steering base. Positive values reward "
                    "SINR-loss reduction relative to the base weights, while "
                    "negative values penalize policies that are worse than "
                    "the base steering solution."
                ),
            },
            "milestone_bonus": {
                "reward_bonus_good_soi": reward_bonus_good_soi,
                "reward_bonus_good_jammer": reward_bonus_good_jammer,
                "reward_bonus_good_sinr_loss": (
                    reward_bonus_good_sinr_loss
                ),
                "soi_gain_loss_bonus_threshold_db": (
                    soi_gain_loss_bonus_threshold_db
                ),
                "jammer_leakage_bonus_threshold": (
                    jammer_leakage_bonus_threshold
                ),
                "sinr_loss_bonus_threshold_db": (
                    sinr_loss_bonus_threshold_db
                ),
                "description": (
                    "Optional bounded milestone bonuses for reaching "
                    "physically meaningful performance regions."
                ),
            },
        },
        "reward_definition": (
            "Residual BeamformingEnvPhase5 reward: "
            "reward_alpha_sinr * (sinr_db / sinr_scale_db) "
            "- reward_beta_sinr_loss * "
            "(clipped_sinr_loss_db / sinr_loss_scale_db) "
            "- reward_gamma_soi_gain_loss * "
            "(clipped_soi_gain_loss_db / soi_gain_loss_scale_db) "
            "- reward_gamma_jammer_leakage * "
            "(clipped_jammer_leakage_loss / jammer_leakage_scale) "
            "- reward_gamma_residual_phase * "
            "(clipped_residual_control_loss / residual_phase_loss_scale) "
            "+ reward_gamma_base_improvement * "
            "(clipped_base_improvement_db / base_improvement_scale_db) "
            "+ milestone_bonus."
        ),
    }

    return config