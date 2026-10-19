
"""Basic residual-based GNSS pseudorange fault detection."""

import numpy as np

from gnss_resilience.position import estimate_position


def detect_pseudorange_fault(
    satellite_positions_m,
    pseudoranges_m,
    initial_position_m,
    residual_threshold_m=10.0,
    initial_clock_bias_m=0.0,
):
    """Flag the measurement with the largest post-fit residual if it exceeds a threshold.

    Returns a dictionary containing the position estimate, residuals,
    detection status, and suspected satellite index.

    This is a basic residual check, not a complete RAIM implementation.
    """
    satellites = np.asarray(satellite_positions_m, dtype=float)
    ranges = np.asarray(pseudoranges_m, dtype=float)

    if satellites.ndim != 2 or satellites.shape[1] != 3:
        raise ValueError("Satellite positions must have shape (n, 3).")

    if ranges.shape != (satellites.shape[0],):
        raise ValueError("Pseudoranges must have shape (n,).")

    if not np.isfinite(residual_threshold_m) or residual_threshold_m <= 0:
        raise ValueError("Residual threshold must be finite and positive.")

    position, clock_bias = estimate_position(
        satellites,
        ranges,
        initial_position_m=initial_position_m,
        initial_clock_bias_m=initial_clock_bias_m,
    )

    predicted_ranges = (
        np.linalg.norm(satellites - position, axis=1) + clock_bias
    )
    residuals = ranges - predicted_ranges

    suspected_index = int(np.argmax(np.abs(residuals)))
    detected = bool(
        abs(residuals[suspected_index]) > residual_threshold_m
    )

    return {
        "detected": detected,
        "suspected_satellite_index": suspected_index if detected else None,
        "residuals_m": residuals,
        "max_abs_residual_m": float(np.max(np.abs(residuals))),
        "estimated_position_m": position,
        "estimated_clock_bias_m": float(clock_bias),
    }
