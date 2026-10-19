
"""Evaluate GNSS fault detection over repeated synthetic trials."""

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
    rng = np.random.default_rng(42)

    clean_false_alarms = 0
    fault_detections = 0
    correct_isolations = 0
    clean_errors = 0
    fault_errors = 0

    for trial in range(trials):
        noise = rng.normal(0.0, noise_std_m, size=len(satellites))

        clean_ranges = calculate_pseudoranges(
            receiver,
            satellites,
            receiver_clock_bias_m=125.0,
            noise_m=noise,
        )

        initial_position = receiver + np.array([100.0, -80.0, 50.0])

        try:
            clean_result = detect_pseudorange_fault(
                satellites,
                clean_ranges,
                initial_position_m=initial_position,
                residual_threshold_m=threshold_m,
                initial_clock_bias_m=125.0,
            )
            clean_false_alarms += int(clean_result["detected"])
        except (ValueError, RuntimeError):
            clean_errors += 1

        corrupted_ranges = clean_ranges.copy()
        faulty_index = trial % len(satellites)
        corrupted_ranges[faulty_index] += fault_size_m

        try:
            fault_result = detect_pseudorange_fault(
                satellites,
                corrupted_ranges,
                initial_position_m=initial_position,
                residual_threshold_m=threshold_m,
                initial_clock_bias_m=125.0,
            )

            if fault_result["detected"]:
                fault_detections += 1
                if fault_result["suspected_satellite_index"] == faulty_index:
                    correct_isolations += 1

        except (ValueError, RuntimeError):
            fault_errors += 1

    valid_clean_trials = trials - clean_errors
    valid_fault_trials = trials - fault_errors

    report = {
        "experiment": "Repeated GNSS fault detection evaluation",
        "seed": 42,
        "trials_per_scenario": trials,
        "noise_std_m": noise_std_m,
        "injected_fault_m": fault_size_m,
        "residual_threshold_m": threshold_m,
        "clean_scenario": {
            "valid_trials": valid_clean_trials,
            "errors": clean_errors,
            "false_alarms": clean_false_alarms,
            "false_alarm_rate": (
                clean_false_alarms / valid_clean_trials
                if valid_clean_trials else None
            ),
        },
        "fault_scenario": {
            "valid_trials": valid_fault_trials,
            "errors": fault_errors,
            "detections": fault_detections,
            "detection_rate": (
                fault_detections / valid_fault_trials
                if valid_fault_trials else None
            ),
            "correct_isolations": correct_isolations,
            "isolation_rate_over_all_valid_trials": (
                correct_isolations / valid_fault_trials
                if valid_fault_trials else None
            ),
        },
        "limitations": [
            "Synthetic measurements and one fixed satellite geometry.",
            "One large pseudorange fault is injected per fault trial.",
            "Residual thresholding is not full RAIM.",
            "Results do not establish real-world receiver performance.",
        ],
    }

    output_path = Path("results/fault_detection_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
