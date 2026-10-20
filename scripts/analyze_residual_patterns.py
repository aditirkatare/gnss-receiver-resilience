
"""Analyze post-fit residual patterns for injected GNSS faults."""

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

    trials = 200
    noise_std_m = 1.0
    fault_size_m = 1000.0
    threshold_m = 10.0
    clock_bias_m = 125.0
    rng = np.random.default_rng(42)

    initial_position = receiver + np.array([100.0, -80.0, 50.0])

    # Track which satellite receives the largest absolute residual.
    results = {
        str(index): {
            "trials": 0,
            "errors": 0,
            "correct_largest_residual": 0,
            "largest_residual_counts": {
                str(predicted): 0 for predicted in range(len(satellites))
            },
            "mean_residual_by_satellite_m": [
                0.0 for _ in range(len(satellites))
            ],
            "mean_absolute_residual_by_satellite_m": [
                0.0 for _ in range(len(satellites))
            ],
        }
        for index in range(len(satellites))
    }

    for _ in range(trials):
        noise = rng.normal(0.0, noise_std_m, len(satellites))

        clean_ranges = calculate_pseudoranges(
            receiver,
            satellites,
            receiver_clock_bias_m=clock_bias_m,
            noise_m=noise,
        )

        for faulty_index in range(len(satellites)):
            ranges = clean_ranges.copy()
            ranges[faulty_index] += fault_size_m

            bucket = results[str(faulty_index)]
            bucket["trials"] += 1

            try:
                detection = detect_pseudorange_fault(
                    satellites,
                    ranges,
                    initial_position_m=initial_position,
                    initial_clock_bias_m=clock_bias_m,
                    residual_threshold_m=threshold_m,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                bucket["errors"] += 1
                continue

            residuals = np.asarray(detection["residuals_m"])
            largest_index = int(np.argmax(np.abs(residuals)))

            bucket["largest_residual_counts"][str(largest_index)] += 1
            if largest_index == faulty_index:
                bucket["correct_largest_residual"] += 1

            for index, residual in enumerate(residuals):
                bucket["mean_residual_by_satellite_m"][index] += float(
                    residual
                )
                bucket["mean_absolute_residual_by_satellite_m"][index] += (
                    abs(float(residual))
                )

    for bucket in results.values():
        valid_trials = bucket["trials"] - bucket["errors"]

        bucket["valid_trials"] = valid_trials
        bucket["correct_largest_residual_rate"] = (
            bucket["correct_largest_residual"] / valid_trials
            if valid_trials else None
        )

        if valid_trials:
            bucket["mean_residual_by_satellite_m"] = [
                value / valid_trials
                for value in bucket["mean_residual_by_satellite_m"]
            ]
            bucket["mean_absolute_residual_by_satellite_m"] = [
                value / valid_trials
                for value in bucket["mean_absolute_residual_by_satellite_m"]
            ]

    report = {
        "experiment": "GNSS post-fit residual pattern analysis",
        "seed": 42,
        "trials_per_faulty_satellite": trials,
        "noise_std_m": noise_std_m,
        "injected_fault_m": fault_size_m,
        "threshold_m": threshold_m,
        "results_by_actual_faulty_satellite": results,
        "limitations": [
            "Synthetic measurements and fixed satellite geometry.",
            "One positive pseudorange fault is injected per trial.",
            "Post-fit residuals share effects from the estimated position "
            "and clock bias.",
            "This analysis does not implement complete RAIM.",
            "Results do not establish real-world receiver performance.",
        ],
    }

    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "residual_pattern_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
