
"""Run reproducible GNSS positioning-noise experiments."""

import json

import numpy as np

from gnss_resilience.simulation import run_noise_experiment


def main():
    receiver = np.array([
        1_000_000.0,
        2_000_000.0,
        3_000_000.0,
    ])

    satellites = np.array([
        [20_000_000.0, 0.0, 0.0],
        [0.0, 21_000_000.0, 0.0],
        [0.0, 0.0, 22_000_000.0],
        [-15_000_000.0, -16_000_000.0, -17_000_000.0],
        [18_000_000.0, -19_000_000.0, 12_000_000.0],
    ])

    noise_levels_m = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []

    for noise_std_m in noise_levels_m:
        result = run_noise_experiment(
            receiver_position_m=receiver,
            satellite_positions_m=satellites,
            noise_std_m=noise_std_m,
            trials=500,
            seed=42,
        )
        results.append(result)

    print("GNSS POSITIONING NOISE EXPERIMENT")
    print("Synthetic measurements; fixed satellite geometry")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
