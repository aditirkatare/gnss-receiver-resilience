
"""Compare residual detection with leave-one-satellite-out diagnostics."""

import json
from pathlib import Path

import numpy as np

from gnss_resilience.fault_detection import detect_pseudorange_fault
from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


def leave_one_out_diagnostic(
    satellites,
    pseudoranges,
    initial_position_m,
    initial_clock_bias_m,
    threshold_m,
):
    """Score each satellite using its prediction error when excluded."""
    scores = []

    for excluded_index in range(len(satellites)):
        keep = np.arange(len(satellites)) != excluded_index

        position, clock_bias = estimate_position(
            satellites[keep],
            pseudoranges[keep],
            initial_position_m=initial_position_m,
            initial_clock_bias_m=initial_clock_bias_m,
        )

        predicted_range = (
            np.linalg.norm(satellites[excluded_index] - position)
            + clock_bias
        )
        prediction_error = float(
            pseudoranges[excluded_index] - predicted_range
        )
        scores.append(prediction_error)

    scores = np.asarray(scores)
    suspected_index = int(np.argmax(np.abs(scores)))
    max_score = float(np.max(np.abs(scores)))

    return {
        "detected": max_score > threshold_m,
        "suspected_satellite_index": (
            suspected_index if max_score > threshold_m else None
        ),
        "max_excluded_satellite_error_m": max_score,
        "scores_m": scores.tolist(),
    }


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
    initial_position = receiver + np.array([100.0, -80.0, 50.0])

    methods = {
        "baseline": {
            "clean_false_alarms": 0,
            "clean_errors": 0,
            "fault_detections": 0,
            "fault_errors": 0,
            "correct_isolations": 0,
        },
        "leave_one_out": {
            "clean_false_alarms": 0,
            "clean_errors": 0,
            "fault_detections": 0,
            "fault_errors": 0,
            "correct_isolations": 0,
        },
    }

    for trial in range(trials):
        noise = rng.normal(0.0, noise_std_m, size=len(satellites))
        clean_ranges = calculate_pseudoranges(
            receiver,
            satellites,
            receiver_clock_bias_m=125.0,
            noise_m=noise,
        )

        faulty_index = trial % len(satellites)
        corrupted_ranges = clean_ranges.copy()
        corrupted_ranges[faulty_index] += fault_size_m

        for method_name, detector in (
            ("baseline", detect_pseudorange_fault),
            ("leave_one_out", leave_one_out_diagnostic),
        ):
            detector_kwargs = (
                {
                    "initial_position_m": initial_position,
                    "initial_clock_bias_m": 125.0,
                    "residual_threshold_m": threshold_m,
                }
                if method_name == "baseline"
                else {
                    "initial_position_m": initial_position,
                    "initial_clock_bias_m": 125.0,
                    "threshold_m": threshold_m,
                }
            )

            try:
                clean_result = detector(
                    satellites,
                    clean_ranges,
                    **detector_kwargs,
                )
                methods[method_name]["clean_false_alarms"] += int(
                    clean_result["detected"]
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                methods[method_name]["clean_errors"] += 1

            try:
                fault_result = detector(
                    satellites,
                    corrupted_ranges,
                    **detector_kwargs,
                )

                if fault_result["detected"]:
                    methods[method_name]["fault_detections"] += 1
                    if (
                        fault_result["suspected_satellite_index"]
                        == faulty_index
                    ):
                        methods[method_name]["correct_isolations"] += 1

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                methods[method_name]["fault_errors"] += 1

    report = {
        "experiment": "Baseline vs leave-one-out GNSS fault diagnostics",
        "seed": 42,
        "trials_per_scenario": trials,
        "noise_std_m": noise_std_m,
        "injected_fault_m": fault_size_m,
        "threshold_m": threshold_m,
        "results": {},
        "limitations": [
            "Synthetic data with fixed satellite geometry.",
            "Leave-one-out uses four satellites to fit four unknowns.",
            "This is a diagnostic comparison, not a complete RAIM implementation.",
            "Thresholds need calibration for different geometries and noise levels.",
        ],
    }

    for name, values in methods.items():
        valid_clean = trials - values["clean_errors"]
        valid_fault = trials - values["fault_errors"]

        report["results"][name] = {
            **values,
            "clean_false_alarm_rate": (
                values["clean_false_alarms"] / valid_clean
                if valid_clean else None
            ),
            "fault_detection_rate": (
                values["fault_detections"] / valid_fault
                if valid_fault else None
            ),
            "correct_isolation_rate": (
                values["correct_isolations"] / valid_fault
                if valid_fault else None
            ),
        }

    output_path = Path("results/leave_one_out_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
