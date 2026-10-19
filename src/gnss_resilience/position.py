
"""Least-squares GNSS receiver position estimation."""

import numpy as np


def estimate_position(
    satellite_positions_m,
    pseudoranges_m,
    initial_position_m,
    initial_clock_bias_m=0.0,
    max_iterations=20,
    tolerance_m=1e-4,
):
    """Estimate receiver ECEF position and clock bias using least squares.

    Returns:
        (position_m, clock_bias_m): estimated ECEF position and clock bias,
        both expressed in metres.

    Raises:
        ValueError: if inputs are invalid or the geometry is unsolvable.
        RuntimeError: if the estimator fails to converge.
    """
    satellites = np.asarray(satellite_positions_m, dtype=float)
    ranges = np.asarray(pseudoranges_m, dtype=float)
    position = np.asarray(initial_position_m, dtype=float).copy()
    clock_bias = float(initial_clock_bias_m)

    if satellites.ndim != 2 or satellites.shape[1] != 3:
        raise ValueError("Satellite positions must have shape (n, 3).")

    count = satellites.shape[0]

    if count < 4:
        raise ValueError("At least four satellites are required.")

    if ranges.shape != (count,):
        raise ValueError("Pseudoranges must have shape (n,).")

    if position.shape != (3,):
        raise ValueError("Initial position must have shape (3,).")

    if not np.all(np.isfinite(satellites)):
        raise ValueError("Satellite coordinates must be finite.")

    if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
        raise ValueError("Pseudoranges must be finite and positive.")

    if not np.all(np.isfinite(position)) or not np.isfinite(clock_bias):
        raise ValueError("Initial estimates must be finite.")

    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")

    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance_m must be finite and positive.")

    for _ in range(max_iterations):
        differences = position - satellites
        geometric_ranges = np.linalg.norm(differences, axis=1)

        if np.any(geometric_ranges == 0):
            raise ValueError("Receiver estimate coincides with a satellite.")

        predicted = geometric_ranges + clock_bias
        residuals = ranges - predicted

        # Jacobian of predicted pseudoranges with respect to x, y, z, bias.
        jacobian = np.column_stack((
            differences / geometric_ranges[:, None],
            np.ones(count),
        ))

        if np.linalg.matrix_rank(jacobian) < 4:
            raise ValueError("Satellite geometry is insufficient for a solution.")

        update, _, _, _ = np.linalg.lstsq(
            jacobian, residuals, rcond=None
        )

        position += update[:3]
        clock_bias += update[3]

        if np.linalg.norm(update) < tolerance_m:
            return position, clock_bias

    raise RuntimeError("Position estimator did not converge.")
