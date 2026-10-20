"""Compare baseline and leave-one-out GNSS diagnostics across thresholds."""

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
        scores.append(
            float(pseudoranges[excluded_index] - predicted_range)
        )

    scores = np.asarray(scores)
    suspected_index = int(np.argmax(np.abs(scores)))
    max_score = float(np.max(np.abs(scores)))

    return {
        "detected": max_score > threshold_m,
        "suspected_satellite_index": (
            suspected_index if max_score > threshold_m else None
        ),
    }


def main():
    receiver = np.array([1e6, 2e6, 3e6])
    satellites = np.array([
        [20e6, 0, 0],
        [0, 21e6, 0],
        [0, 0, 22e6],
        [-15e6, -16e6, -17e6],
        [18e6, -19e6, 12e6],
    ])

    trials = 200
    noise_std_m = 1.0
    fault_size_m = 1000.0
    thresholds_m = [5.0, 10.0, 20.0, 50.0, 100.0]
    rng = np.random.default_rng(42)

    initial_position = receiver + np.array([100.0, -80.0, 50.0])
    initial_clock_bias_m = 125.0

    results = {}

    for threshold_m in thresholds_m:
        results[str(int(threshold_m))] = {}

        for method in ("baseline", "leave_one_out"):
            metrics = {
                "clean_trials": 0,
                "clean_errors": 0,
                "clean_false_alarms": 0,
                "fault_trials": 0,
                "fault_errors": 0,
                "fault_detections": 0,
                "correct_isolations": 0,
            }

            for trial in range(trials):
                noise = rng.normal(0.0, noise_std_m, len(satellites))

                clean_ranges = calculate_pseudoranges(
                    satellites,
                    receiver,
                    clock_bias_m=initial_clock_bias_m,
                ) + noise


                faulty_index = trial % len(satellites)
                faulty_ranges = clean_ranges.copy()
                faulty_ranges[faulty_index] += fault_size_m

                for scenario, ranges in (
                    ("clean", clean_ranges),
                    ("fault", faulty_ranges),
                ):
                    try:
                        if method == "baseline":
                            detection = detect_pseudorange_fault(
                                satellites,
                                ranges,
                                initial_position_m=initial_position,
                                initial_clock_bias_m=initial_clock_bias_m,
                                residual_threshold_m=threshold_m,
                            )
                        else:
                            detection = leave_one_out_diagnostic(
                                satellites,
                                ranges,
                                initial_position,
                                initial_clock_bias_m,
                                threshold_m,
                            )

                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        metrics[f"{scenario}_errors"] += 1
                        continue

                    metrics[f"{scenario}_trials"] += 1

                    if scenario == "clean":
                        if detection["detected"]:
                            metrics["clean_false_alarms"] += 1
                    else:
                        if detection["detected"]:
                            metrics["fault_detections"] += 1
                            if (
                                detection["suspected_satellite_index"]
                                == faulty_index
                            ):
                                metrics["correct_isolations"] += 1

            clean_trials = metrics["clean_trials"]
            fault_trials = metrics["fault_trials"]

            metrics["clean_false_alarm_rate"] = (
                metrics["clean_false_alarms"] / clean_trials
                if clean_trials else None
            )
            metrics["fault_detection_rate"] = (
                metrics["fault_detections"] / fault_trials
                if fault_trials else None
            )
            metrics["correct_isolation_rate"] = (
                metrics["correct_isolations"] / fault_trials
                if fault_trials else None
            )

            results[str(int(threshold_m))][method] = metrics

    report = {
        "experiment": "GNSS fault diagnostic threshold sensitivity",
        "seed": 42,
        "trials_per_scenario_per_threshold": trials,
        "noise_std_m": noise_std_m,
        "injected_fault_m": fault_size_m,
        "thresholds_m": thresholds_m,
        "results": results,
        "limitations": [
            "Synthetic measurements with fixed satellite geometry.",
            "Leave-one-out fits four satellites to four unknowns.",
            "Thresholds are evaluated on the same fixed geometry.",
            "This is not a complete RAIM implementation.",
            "Results do not establish real-world receiver performance.",
        ],
    }

    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "threshold_sensitivity_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
