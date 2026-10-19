
"""Run a reproducible GNSS pseudorange fault-detection experiment."""

import json
from pathlib import Path

import numpy as np

from gnss_resilience.fault_detection import detect_pseudorange_fault
from gnss_resilience.measurements import calculate_pseudoranges


def main():
    receiver = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])

    satellites = np.array([
        [20_000_000.0, 0.0, 0.0],
        [0.0, 21_000_000.0, 0.0],
        [0.0, 0.0, 22_000_000.0],
        [-15_000_000.0, -16_000_000.0, -17_000_000.0],
        [18_000_000.0, -19_000_000.0, 12_000_000.0],
    ])

    rng = np.random.default_rng(42)
    clean_ranges = calculate_pseudoranges(
        receiver,
        satellites,
        receiver_clock_bias_m=125.0,
    )
    clean_ranges = clean_ranges + rng.normal(
        0.0, 1.0, size=len(clean_ranges)
    )

    scenarios = [
        ("clean_measurements", clean_ranges.copy()),
        ("corrupted_measurement", clean_ranges.copy()),
    ]

    corrupted_index = 0
    scenarios[1][1][corrupted_index] += 1_000.0

    results = []

    for scenario_name, pseudoranges in scenarios:
        result = detect_pseudorange_fault(
            satellites,
            pseudoranges,
            initial_position_m=receiver + np.array([100.0, -80.0, 50.0]),
            residual_threshold_m=10.0,
            initial_clock_bias_m=125.0,
        )

        results.append({
            "scenario": scenario_name,
            "fault_detected": result["detected"],
            "suspected_satellite_index": result["suspected_satellite_index"],
            "max_abs_residual_m": result["max_abs_residual_m"],
            "residual_threshold_m": 10.0,
        })

    output = {
        "experiment": "GNSS pseudorange fault detection",
        "seed": 42,
        "noise_std_m": 1.0,
        "injected_fault_m": 1000.0,
        "injected_satellite_index": corrupted_index,
        "results": results,
        "limitations": [
            "Synthetic measurements only.",
            "Residual thresholding is a basic fault check, not full RAIM.",
            "The largest residual does not guarantee correct fault isolation.",
        ],
    }

    output_path = Path("results/fault_detection_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
