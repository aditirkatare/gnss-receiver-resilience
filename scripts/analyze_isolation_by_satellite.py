
"""Analyze GNSS fault isolation performance by satellite index."""

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
    """Score each satellite using a position fit that excludes it."""
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
        "detected": bool(max_score > threshold_m),
        "suspected_satellite_index": (
            suspected_index if max_score > threshold_m else None
        ),
        "scores_m": scores.tolist(),
    }


def new_fault_metrics():
    return {
        "trials": 0,
        "errors": 0,
        "detections": 0,
        "correct_isolations": 0,
        "wrong_isolations": 0,
        "missed_detections": 0,
    }


def add_rates(metrics):
    valid = metrics["trials"] - metrics["errors"]
    metrics["valid_trials"] = valid

    metrics["detection_rate"] = (
        metrics["detections"] / valid if valid else None
    )
    metrics["correct_isolation_rate"] = (
        metrics["correct_isolations"] / valid if valid else None
    )
    metrics["isolation_rate_when_detected"] = (
        metrics["correct_isolations"] / metrics["detections"]
        if metrics["detections"] else None
    )


def main():
    receiver = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])

    satellites = np.array([
        [20_000_000.0, 0.0, 0.0],
        [0.0, 21_000_000.0, 0.0],
        [0.0, 0.0, 22_000_000.0],
        [-15_000_000.0, -16_000_000.0, -17_000_000.0],
        [18_000_000.0, -19_000_000.0, 12_000_000.0],
    ])

    trials_per_satellite = 200
    noise_std_m = 1.0
    fault_size_m = 1000.0
    threshold_m = 10.0
    clock_bias_m = 125.0
    rng = np.random.default_rng(42)

    initial_position = receiver + np.array([100.0, -80.0, 50.0])

    method_names = ("baseline", "leave_one_out")

    clean_metrics = {
        method: {
            "trials": 0,
            "errors": 0,
            "false_alarms": 0,
        }
        for method in method_names
    }

    fault_metrics = {
        method: {
            str(index): new_fault_metrics()
            for index in range(len(satellites))
        }
        for method in method_names
    }

    def evaluate(method, ranges):
        if method == "baseline":
            return detect_pseudorange_fault(
                satellites,
                ranges,
                initial_position_m=initial_position,
                initial_clock_bias_m=clock_bias_m,
                residual_threshold_m=threshold_m,
            )

        return leave_one_out_diagnostic(
            satellites,
            ranges,
            initial_position,
            clock_bias_m,
            threshold_m,
        )

    for _ in range(trials_per_satellite):
        noise = rng.normal(
            0.0, noise_std_m, size=len(satellites)
        )

        clean_ranges = calculate_pseudoranges(
            receiver,
            satellites,
            receiver_clock_bias_m=clock_bias_m,
            noise_m=noise,
        )

        # Evaluate clean measurements for the false-alarm baseline.
        for method in method_names:
            try:
                result = evaluate(method, clean_ranges)
                clean_metrics[method]["trials"] += 1
                clean_metrics[method]["false_alarms"] += int(
                    result["detected"]
                )
            except (
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
            ):
                clean_metrics[method]["errors"] += 1

        # Inject a fault into each satellite, using the same clean
        # measurements so the methods can be compared fairly.
        for faulty_index in range(len(satellites)):
            faulty_ranges = clean_ranges.copy()
            faulty_ranges[faulty_index] += fault_size_m

            for method in method_names:
                metrics = fault_metrics[method][str(faulty_index)]
                metrics["trials"] += 1

                try:
                    result = evaluate(method, faulty_ranges)
                except (
                    ValueError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                ):
                    metrics["errors"] += 1
                    continue

                if not result["detected"]:
                    metrics["missed_detections"] += 1
                    continue

                metrics["detections"] += 1

                if result["suspected_satellite_index"] == faulty_index:
                    metrics["correct_isolations"] += 1
                else:
                    metrics["wrong_isolations"] += 1

    for method in method_names:
        clean = clean_metrics[method]
        valid_clean = clean["trials"]

        clean["false_alarm_rate"] = (
            clean["false_alarms"] / valid_clean
            if valid_clean else None
        )

        for metrics in fault_metrics[method].values():
            add_rates(metrics)

    report = {
        "experiment": "GNSS fault isolation by satellite index",
        "seed": 42,
        "trials_per_satellite": trials_per_satellite,
        "noise_std_m": noise_std_m,
        "injected_fault_m": fault_size_m,
        "threshold_m": threshold_m,
        "satellite_count": len(satellites),
        "clean_scenario": clean_metrics,
        "fault_results_by_satellite": fault_metrics,
        "limitations": [
            "Synthetic measurements with fixed satellite geometry.",
            "Each satellite is tested with the same generated clean samples.",
            "One positive 1000 m pseudorange fault is injected per trial.",
            "Leave-one-out fits four satellites to four unknowns.",
            "Residual and leave-one-out thresholds are not calibrated "
            "for general operating conditions.",
            "This is not a complete RAIM implementation.",
            "Results do not establish real-world receiver performance.",
        ],
    }

    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "isolation_by_satellite_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
