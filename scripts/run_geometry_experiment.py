
import json

import numpy as np

from gnss_resilience.geometry import calculate_dop
from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


def make_satellite_positions(receiver_position_m, directions):
    """Create synthetic satellite positions from line-of-sight directions."""
    directions = np.asarray(directions, dtype=float)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return receiver_position_m + 20_000_000.0 * directions


def run_geometry_comparison(
    trials=500,
    noise_std_m=2.0,
    seed=42,
):
    """Compare positioning accuracy under good and poor satellite geometry."""
    receiver_position_m = np.array([1e6, 2e6, 3e6], dtype=float)
    receiver_clock_bias_m = 125.0

    # Satellites distributed across a wider range of directions.
    good_directions = [
        [0.5, 0.0, 1.0],
        [-0.5, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, -0.5, 1.0],
        [0.5, 0.5, 1.0],
        [-0.5, -0.5, 1.0],
    ]

    # Satellites clustered in similar directions, producing weaker geometry.
    poor_directions = [
        [0.15, 0.00, 1.0],
        [-0.15, 0.05, 1.0],
        [0.05, -0.15, 1.0],
        [0.10, 0.15, 1.0],
        [-0.10, -0.10, 1.0],
        [0.15, -0.15, 1.0],
    ]

    geometries = {
        "good_geometry": make_satellite_positions(
            receiver_position_m, good_directions
        ),
        "poor_geometry": make_satellite_positions(
            receiver_position_m, poor_directions
        ),
    }

    rng = np.random.default_rng(seed)
    noise_samples_by_trial = rng.normal(
    0.0,
    noise_std_m,
    size=(trials, len(good_directions)),
)
results = []

    for geometry_name, satellite_positions_m in geometries.items():
        dop = calculate_dop(receiver_position_m, satellite_positions_m)
        position_errors_m = []
        failures = 0

        for _ in range(trials):
            # Use the same noise samples for both geometries for a fair comparison.
           noise = noise_samples_by_trial[_]

            pseudoranges_m = calculate_pseudoranges(
                receiver_position_m,
                satellite_positions_m,
                receiver_clock_bias_m=receiver_clock_bias_m,
                noise_m=noise,
            )

            try:
                estimated_position_m, _ = estimate_position(
                    satellite_positions_m,
                    pseudoranges_m,
                    initial_position_m=receiver_position_m
                    + np.array([100.0, -100.0, 50.0]),
                )
                error_m = float(
                    np.linalg.norm(estimated_position_m - receiver_position_m)
                )
                position_errors_m.append(error_m)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                failures += 1

        errors = np.asarray(position_errors_m, dtype=float)

        results.append(
            {
                "geometry": geometry_name,
                "pdop": float(dop["PDOP"]),
                "tdop": float(dop["TDOP"]),
                "gdop": float(dop["GDOP"]),
                "noise_std_m": noise_std_m,
                "trials_requested": trials,
                "trials_succeeded": int(len(errors)),
                "trials_failed": failures,
                "mean_position_error_m": (
                    float(np.mean(errors)) if len(errors) else None
                ),
                "position_rmse_m": (
                    float(np.sqrt(np.mean(errors**2))) if len(errors) else None
                ),
                "median_position_error_m": (
                    float(np.median(errors)) if len(errors) else None
                ),
                "max_position_error_m": (
                    float(np.max(errors)) if len(errors) else None
                ),
            }
        )

    return {
        "experiment": "Satellite geometry comparison",
        "receiver_position_m": receiver_position_m.tolist(),
        "receiver_clock_bias_m": receiver_clock_bias_m,
        "noise_std_m": noise_std_m,
        "seed": seed,
        "results": results,
        "limitations": [
            "Synthetic satellite geometry, not real orbital data.",
            "Both configurations use the same number of satellites.",
            "Only independent pseudorange noise is modelled.",
            "This experiment does not model multipath, atmospheric errors, "
            "satellite clock errors, or orbital errors.",
        ],
    }


if __name__ == "__main__":
    report = run_geometry_comparison()
    print(json.dumps(report, indent=2))
