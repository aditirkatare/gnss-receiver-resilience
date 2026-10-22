
"""Evaluate GNSS fault detection across injected fault magnitudes."""

import json
from pathlib import Path

import numpy as np

from gnss_resilience.position import estimate_position

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORT_PATH = RESULTS_DIR / "fault_magnitude_report.json"

RECEIVER_POSITION_M = np.array([1e6, 2e6, 3e6])
CLOCK_BIAS_M = 125.0
NOISE_STD_M = 1.0
FAULT_MAGNITUDES_M = (50.0, 100.0, 250.0, 500.0, 1000.0)
SATELLITE_COUNTS = (5, 6, 7, 8)
CALIBRATION_TRIALS = 1000
CLEAN_TEST_TRIALS = 1000
FAULT_TRIALS_PER_SATELLITE = 200
TARGET_FALSE_ALARM_RATE = 0.01
SEED = 2026

SATELLITE_POSITIONS_M = np.array([
    [20e6, 0, 0],
    [0, 21e6, 0],
    [0, 0, 22e6],
    [-15e6, -16e6, -17e6],
    [18e6, -19e6, 12e6],
    [-20e6, 14e6, 8e6],
    [11e6, 17e6, -19e6],
    [-12e6, -8e6, 21e6],
], dtype=float)


def safe_scores(positions, rng, fault_m=0.0, faulty_index=None):
    try:
        ranges = (
            np.linalg.norm(positions - RECEIVER_POSITION_M, axis=1)
            + CLOCK_BIAS_M
        )
        pseudoranges = ranges + rng.normal(
            0.0, NOISE_STD_M, len(positions)
        )
        if faulty_index is not None:
            pseudoranges[faulty_index] += fault_m

        estimate, clock_bias = estimate_position(
            satellite_positions_m=positions,
            pseudoranges_m=pseudoranges,
            initial_position_m=RECEIVER_POSITION_M
            + np.array([100.0, -100.0, 100.0]),
            initial_clock_bias_m=0.0,
        )

        geometric_ranges = np.linalg.norm(
            positions - estimate, axis=1
        )
        residuals = pseudoranges - (geometric_ranges + clock_bias)
        los = (estimate - positions) / geometric_ranges[:, None]
        design = np.column_stack([los, np.ones(len(positions))])
        projection = design @ np.linalg.pinv(design)
        leverage = np.clip(
            np.diag(projection), 0.0, 1.0 - 1e-10
        )
        scores = np.abs(residuals) / np.sqrt(1.0 - leverage)

        return scores if np.all(np.isfinite(scores)) else None
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None


def calibrate_threshold(positions, rng):
    maxima = []
    errors = 0

    for _ in range(CALIBRATION_TRIALS):
        scores = safe_scores(positions, rng)
        if scores is None:
            errors += 1
        else:
            maxima.append(float(np.max(scores)))

    if len(maxima) < 100:
        raise RuntimeError("Too few valid calibration trials.")

    threshold = float(np.quantile(
        maxima, 1.0 - TARGET_FALSE_ALARM_RATE, method="higher"
    ))
    return threshold, len(maxima), errors


def evaluate_clean(positions, rng, threshold):
    valid = errors = alarms = 0

    for _ in range(CLEAN_TEST_TRIALS):
        scores = safe_scores(positions, rng)
        if scores is None:
            errors += 1
            continue
        valid += 1
        alarms += int(float(np.max(scores)) > threshold)

    return {
        "valid_trials": valid,
        "errors": errors,
        "false_alarms": alarms,
        "false_alarm_rate": alarms / valid if valid else None,
    }


def evaluate_fault(positions, rng, threshold, magnitude):
    valid = errors = detections = correct = detected_correct = 0

    for faulty_index in range(len(positions)):
        for _ in range(FAULT_TRIALS_PER_SATELLITE):
            scores = safe_scores(
                positions, rng, fault_m=magnitude,
                faulty_index=faulty_index,
            )
            if scores is None:
                errors += 1
                continue

            valid += 1
            alarm = float(np.max(scores)) > threshold
            candidate = int(np.argmax(scores))

            detections += int(alarm)
            correct += int(candidate == faulty_index)
            detected_correct += int(alarm and candidate == faulty_index)

    return {
        "fault_magnitude_m": magnitude,
        "valid_trials": valid,
        "errors": errors,
        "detections": detections,
        "detection_rate": detections / valid if valid else None,
        "correct_isolations": correct,
        "isolation_accuracy": correct / valid if valid else None,
        "detected_and_correct": detected_correct,
        "correct_isolation_given_detection": (
            detected_correct / detections if detections else None
        ),
    }


def main():
    rng = np.random.default_rng(SEED)
    report = {
        "experiment": "GNSS fault-magnitude sensitivity",
        "noise_std_m": NOISE_STD_M,
        "fault_magnitudes_m": list(FAULT_MAGNITUDES_M),
        "satellite_counts": list(SATELLITE_COUNTS),
        "calibration_trials_requested": CALIBRATION_TRIALS,
        "clean_test_trials_requested": CLEAN_TEST_TRIALS,
        "fault_trials_per_magnitude_per_satellite":
            FAULT_TRIALS_PER_SATELLITE,
        "target_false_alarm_rate": TARGET_FALSE_ALARM_RATE,
        "seed": SEED,
        "scenarios": [],
        "limitations": [
            "Synthetic measurements and fixed satellite geometry.",
            "Positive faults are injected into one satellite per trial.",
            "Thresholds are calibrated per satellite count and held "
            "fixed across fault magnitudes.",
            "Calibration and evaluation use separate random draws.",
            "The target false-alarm rate is not a guarantee.",
            "Results do not establish real-world GNSS performance.",
        ],
    }

    for count in SATELLITE_COUNTS:
        positions = SATELLITE_POSITIONS_M[:count]
        threshold, calibration_valid, calibration_errors = (
            calibrate_threshold(positions, rng)
        )
        clean = evaluate_clean(positions, rng, threshold)

        fault_results = [
            evaluate_fault(positions, rng, threshold, magnitude)
            for magnitude in FAULT_MAGNITUDES_M
        ]

        report["scenarios"].append({
            "satellite_count": count,
            "threshold": threshold,
            "calibration_valid_trials": calibration_valid,
            "calibration_errors": calibration_errors,
            "clean_test": clean,
            "fault_magnitude_results": fault_results,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("GNSS FAULT-MAGNITUDE SENSITIVITY")
    print(f"Report saved to: {REPORT_PATH}")

    for scenario in report["scenarios"]:
        print(f"\nSatellites: {scenario['satellite_count']}")
        print(f"Threshold: {scenario['threshold']:.3f}")
        print(
            "Clean false-alarm rate: "
            f"{scenario['clean_test']['false_alarm_rate']}"
        )
        for result in scenario["fault_magnitude_results"]:
            print(
                f"Fault {result['fault_magnitude_m']:g} m: "
                f"detection={result['detection_rate']}, "
                f"isolation={result['isolation_accuracy']}, "
                f"errors={result['errors']}"
            )


if __name__ == "__main__":
    main()
