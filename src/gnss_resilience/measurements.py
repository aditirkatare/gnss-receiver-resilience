
"""GNSS pseudorange measurement models."""

import numpy as np


def calculate_pseudoranges(
    receiver_position_m,
    satellite_positions_m,
    receiver_clock_bias_m=0.0,
    noise_m=0.0,
):
    """Calculate simulated GNSS pseudorange measurements in metres.

    Args:
        receiver_position_m: Receiver ECEF position, shape (3,).
        satellite_positions_m: Satellite ECEF positions, shape (n, 3).
        receiver_clock_bias_m: Receiver clock bias expressed in metres.
        noise_m: Measurement noise in metres, scalar or shape (n,).

    Returns:
        NumPy array of pseudorange measurements, shape (n,).
    """
    receiver = np.asarray(receiver_position_m, dtype=float)
    satellites = np.asarray(satellite_positions_m, dtype=float)

    if receiver.shape != (3,):
        raise ValueError("receiver_position_m must have shape (3,).")

    if satellites.ndim != 2 or satellites.shape[1] != 3:
        raise ValueError("satellite_positions_m must have shape (n, 3).")

    if satellites.shape[0] == 0:
        raise ValueError("At least one satellite position is required.")

    if not np.all(np.isfinite(receiver)):
        raise ValueError("Receiver coordinates must be finite.")

    if not np.all(np.isfinite(satellites)):
        raise ValueError("Satellite coordinates must be finite.")

    if not np.isfinite(receiver_clock_bias_m):
        raise ValueError("Receiver clock bias must be finite.")

    noise = np.asarray(noise_m, dtype=float)

    if not np.all(np.isfinite(noise)):
        raise ValueError("Measurement noise must be finite.")

    try:
        noise = np.broadcast_to(noise, (satellites.shape[0],))
    except ValueError as exc:
        raise ValueError("Noise must be a scalar or have shape (n,).") from exc

    geometric_ranges = np.linalg.norm(satellites - receiver, axis=1)

    return geometric_ranges + receiver_clock_bias_m + noise
