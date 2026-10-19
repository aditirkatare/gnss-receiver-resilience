
"""GNSS satellite geometry and dilution-of-precision metrics."""

import numpy as np


def calculate_dop(receiver_position_m, satellite_positions_m):
    """Calculate PDOP, TDOP, and GDOP from satellite geometry.

    Coordinates must be in the same ECEF coordinate frame and in metres.

    Returns:
        Dictionary containing PDOP, TDOP, and GDOP.

    Raises:
        ValueError: if inputs are invalid or geometry is insufficient.
    """
    receiver = np.asarray(receiver_position_m, dtype=float)
    satellites = np.asarray(satellite_positions_m, dtype=float)

    if receiver.shape != (3,):
        raise ValueError("Receiver position must have shape (3,).")

    if satellites.ndim != 2 or satellites.shape[1] != 3:
        raise ValueError("Satellite positions must have shape (n, 3).")

    if satellites.shape[0] < 4:
        raise ValueError("At least four satellites are required.")

    if not np.all(np.isfinite(receiver)):
        raise ValueError("Receiver coordinates must be finite.")

    if not np.all(np.isfinite(satellites)):
        raise ValueError("Satellite coordinates must be finite.")

    differences = satellites - receiver
    distances = np.linalg.norm(differences, axis=1)

    if np.any(distances == 0):
        raise ValueError("Receiver position coincides with a satellite.")

    # Each row contains the line-of-sight direction and clock-bias term.
    unit_vectors = differences / distances[:, None]
    geometry_matrix = np.column_stack((unit_vectors, np.ones(len(satellites))))

    if np.linalg.matrix_rank(geometry_matrix) < 4:
        raise ValueError("Satellite geometry is insufficient for DOP.")

    covariance = np.linalg.inv(geometry_matrix.T @ geometry_matrix)

    pdop = np.sqrt(np.trace(covariance[:3, :3]))
    tdop = np.sqrt(covariance[3, 3])
    gdop = np.sqrt(np.trace(covariance))

    return {
        "PDOP": float(pdop),
        "TDOP": float(tdop),
        "GDOP": float(gdop),
    }
