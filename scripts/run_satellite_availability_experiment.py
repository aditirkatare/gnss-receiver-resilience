"""Compare GNSS positioning as satellite availability decreases."""

import json

import numpy as np

from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


def main():
    receiver_position_m = np.array([1e6, 2e6, 3e6], dtype=float)
    receiver_clock_bias_m = 125.0
    noise_std_m = 2.0
    trials = 200
    seed = 42

    directions = np.array([
        [0.5, 0.0, 1.0],
        [-0.5, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, -0.5, 1.0],
        [0.5, 0.5, 1.0],
        [-0.5, -0.5, 1.0],
    ], dtype=float)

    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    all_satellites_m = (
        receiver_position_m + 20_000_000.0 * directions
    )

    rng = np.random.default_rng(seed)
    results = []

    for available_count in [6, 5, 4, 3]:
        satellites_m = all_satellites_m[:available_count]
        errors_m = []
        failures = 0

        for _ in range(trials):
            noise_m = rng.normal(
                0.0, noise_std_m, size=available_count
            )

            pseudoranges_m = calculate_pseudoranges(
                receiver_position_m,
                satellites_m,
                receiver_clock_bias_m=receiver_clock_bias_m,
                noise_m=noise_m,
            )

            try:
                estimated_position_m, _ = estimate_position(
                    satellites_m,
                    pseudoranges_m,
                    initial_position_m=receiver_position_m
                    + np.array([100.0, -100.0, 50.0]),
                )

                errors_m.append(
                    float(np.linalg.norm(
                        estimated_position_m - receiver_position_m
                    ))
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                failures += 1

        results.append({
            "available_satellites": available_count,
            "trials_requested": trials,
            "trials_succeeded": len(errors_m),
            "trials_failed": failures,
            "position_rmse_m": (
                float(np.sqrt(np.mean(np.square(errors_m))))
                if errors_m else None
            ),
            "noise_std_m": noise_std_m,
        })

    print("GNSS SATELLITE AVAILABILITY EXPERIMENT")
    print("Synthetic satellite positions and pseudorange measurements")
    print(json.dumps({
        "seed": seed,
        "trials_per_satellite_count": trials,
        "results": results,
        "limitations": [
            "Synthetic satellite geometry, not real orbital data.",
            "Independent Gaussian pseudorange noise only.",
            "No multipath, atmospheric, clock, or orbital error modelling.",
            "Results apply only to these satellite subsets and this estimator.",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
