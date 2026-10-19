
"""Reproducible GNSS measurement-noise experiments."""

import numpy as np

from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


def run_noise_experiment(
    receiver_position_m,
    satellite_positions_m,
    noise_std_m,
    trials=100,
    seed=42,
):
    """Evaluate positioning error under simulated Gaussian noise.

    Args:
        receiver_position_m: Known receiver ECEF position, shape (3,).
        satellite_positions_m: Satellite ECEF positions, shape (n, 3).
        noise_std_m: Standard deviation of pseudorange noise, in metres.
        trials: Number of independent simulation trials.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing position-error statistics in metres.
    """
    receiver = np.asarray(receiver_position_m, dtype=float)
    satellites = np.asarray(satellite_positions_m, dtype=float)

    if receiver.shape != (3,) or not np.all(np.isfinite(receiver)):
        raise ValueError("Receiver position must contain three finite values.")

    if (
        satellites.ndim != 2
        or satellites.shape[1] != 3
        or not np.all(np.isfinite(satellites))
    ):
        raise ValueError("Satellite positions must have shape (n, 3) and be finite.")

    if satellites.shape[0] < 4:
        raise ValueError("At least four satellites are required.")

    if not np.isfinite(noise_std_m) or noise_std_m < 0:
        raise ValueError("Noise standard deviation must be finite and non-negative.")

    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise ValueError("trials must be a positive integer.")

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")

    rng = np.random.default_rng(seed)
    position_errors = []
    failed_trials = 0

    for _ in range(trials):
        noise = rng.normal(
            loc=0.0,
            scale=noise_std_m,
            size=satellites.shape[0],
        )

        pseudoranges = calculate_pseudoranges(
            receiver,
            satellites,
            receiver_clock_bias_m=125.0,
            noise_m=noise,
        )

        try:
            estimated_position, _ = estimate_position(
                satellites,
                pseudoranges,
                initial_position_m=receiver + np.array([100.0, -100.0, 50.0]),
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failed_trials += 1
            continue

        position_errors.append(
            np.linalg.norm(estimated_position - receiver)
        )

    errors = np.asarray(position_errors, dtype=float)

    if errors.size == 0:
        raise RuntimeError("All simulation trials failed to produce a solution.")

    return {
        "trials_requested": trials,
        "trials_succeeded": int(errors.size),
        "trials_failed": failed_trials,
        "noise_std_m": float(noise_std_m),
        "mean_error_m": float(np.mean(errors)),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "median_error_m": float(np.median(errors)),
        "max_error_m": float(np.max(errors)),
    }
