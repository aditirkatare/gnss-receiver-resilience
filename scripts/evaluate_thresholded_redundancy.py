
import json
from pathlib import Path

import numpy as np

from gnss_resilience.position import estimate_position


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORT_PATH = RESULTS_DIR / "thresholded_redundancy_report.json"

RECEIVER_POSITION_M = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])
CLOCK_BIAS_M = 125.0
NOISE_STD_M = 1.0
INJECTED_FAULT_M = 1000.0

SATELLITE_COUNTS = (5, 6, 7, 8)
CALIBRATION_TRIALS = 2000
TEST_TRIALS = 2000
TARGET_FALSE_ALARM_RATE = 0.01
SEED = 42

SATELLITE_POSITIONS_M = np.array(
    [
        [20e6, 0.0, 0.0],
        [0.0, 21e6, 0.0],
        [0.0, 0.0, 22e6],
        [-15e6, -16e6, -17e6],
        [18e6, -19e6, 12e6],
        [-20e6, 14e6, 8e6],
        [11e6, 17e6, -19e6],
        [-12e6, -8e6, 21e6],
    ],
    dtype=float,
)


def make_measurements(positions, rng, faulty_index=None):
    ranges = (
        np.linalg.norm(positions - RECEIVER_POSITION_M, axis=1)
        + CLOCK_BIAS_M
    )
    pseudoranges = ranges + rng.normal(0.0, NOISE_STD_M, len(positions))

    if faulty_index is not None:
        pseudoranges[faulty_index] += INJECTED_FAULT_M

    return pseudoranges


def normalized_residual_scores(positions, pseudoranges):
    estimate, clock_bias = estimate_position(
        satellite_positions_m=positions,
        pseudoranges_m=pseudoranges,
        initial_position_m=RECEIVER_POSITION_M
        + np.array([100.0, -100.0, 100.0]),
        initial_clock_bias_m=0.0,
    )

    geometric_ranges = np.linalg.norm(positions - estimate, axis=1)
    predicted = geometric_ranges + clock_bias
    residuals = pseudoranges - predicted

    line_of_sight = (
        estimate - positions
    ) / geometric_ranges[:, None]

    design_matrix = np.column_stack(
        [line_of_sight, np.ones(len(positions))]
    )
    projection = design_matrix @ np.linalg.pinv(design_matrix)
    leverage = np.clip(np.diag(projection), 0.0, 1.0 - 1e-10)

    return np.abs(residuals) / np.sqrt(1.0 - leverage)


def get_scores(positions, rng, faulty_index=None):
    pseudoranges = make_measurements(positions, rng, faulty_index)
    return normalized_residual_scores(positions, pseudoranges)


def safe_scores(positions, rng, faulty_index=None):
    try:
        scores = get_scores(positions, rng, faulty_index)
        if np.all(np.isfinite(scores)):
            return scores
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    return None


def calibrate_threshold(positions, rng):
    clean_max_scores = []

    for _ in range(CALIBRATION_TRIALS):
        scores = safe_scores(positions, rng)
        if scores is not None:
            clean_max_scores.append(float(np.max(scores)))

    if len(clean_max_scores) < 100:
        raise RuntimeError(
            f"Too few valid calibration trials: {len(clean_max_scores)}"
        )

    # Empirical upper quantile of the maximum score across all satellites.
    # This calibrates the system-level alarm rate, not each satellite
    # independently.
    threshold = float(
        np.quantile(
            clean_max_scores,
            1.0 - TARGET_FALSE_ALARM_RATE,
            method="higher",
        )
    )

    return threshold, len(clean_max_scores)


def evaluate_clean(positions, rng, threshold):
    valid = 0
    errors = 0
    false_alarms = 0

    for _ in range(TEST_TRIALS):
        scores = safe_scores(positions, rng)

        if scores is None:
            errors += 1
            continue

        valid += 1
        if float(np.max(scores)) > threshold:
            false_alarms += 1

    return {
        "valid_trials": valid,
        "errors": errors,
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarms / valid if valid else None,
    }


def evaluate_fault(positions, rng, threshold, faulty_index):
    valid = 0
    errors = 0
    detected = 0
    correctly_isolated = 0
    detected_and_correct = 0

    for _ in range(TEST_TRIALS):
        scores = safe_scores(positions, rng, faulty_index)

        if scores is None:
            errors += 1
            continue

        valid += 1
        candidate = int(np.argmax(scores))
        alarm = float(np.max(scores)) > threshold

        if alarm:
            detected += 1

            if candidate == faulty_index:
                detected_and_correct += 1

        if candidate == faulty_index:
            correctly_isolated += 1

    return {
        "faulty_satellite_index": faulty_index,
        "valid_trials": valid,
        "errors": errors,
        "alarms": detected,
        "detection_rate": detected / valid if valid else None,
        "correct_isolations": correctly_isolated,
        "isolation_accuracy": (
            correctly_isolated / valid if valid else None
        ),
        "detected_and_correct": detected_and_correct,
        "correct_isolation_given_detection": (
            detected_and_correct / detected if detected else None
        ),
    }


def main():
    rng = np.random.default_rng(SEED)

    report = {
        "experiment": "Threshold-calibrated satellite redundancy",
        "receiver_position_m": RECEIVER_POSITION_M.tolist(),
        "receiver_clock_bias_m": CLOCK_BIAS_M,
        "noise_std_m": NOISE_STD_M,
        "injected_fault_m": INJECTED_FAULT_M,
        "calibration_trials_requested": CALIBRATION_TRIALS,
        "test_trials_requested_per_scenario": TEST_TRIALS,
        "target_false_alarm_rate": TARGET_FALSE_ALARM_RATE,
        "seed": SEED,
        "scenarios": [],
        "limitations": [
            "Synthetic fixed satellite geometries only.",
            "The empirical threshold is calibrated separately for each satellite count.",
            "Calibration and test trials use the same noise model but separate draws.",
            "The requested false-alarm rate is a target, not a guarantee.",
            "The normalized residual is a diagnostic score, not a proof of fault cause.",
        ],
    }

    for count in SATELLITE_COUNTS:
        positions = SATELLITE_POSITIONS_M[:count]

        threshold, calibration_valid = calibrate_threshold(positions, rng)
        clean = evaluate_clean(positions, rng, threshold)

        faults = [
            evaluate_fault(positions, rng, threshold, faulty_index)
            for faulty_index in range(count)
        ]

        detection_rates = [
            item["detection_rate"]
            for item in faults
            if item["detection_rate"] is not None
        ]
        isolation_rates = [
            item["isolation_accuracy"]
            for item in faults
            if item["isolation_accuracy"] is not None
        ]
        conditional_rates = [
            item["correct_isolation_given_detection"]
            for item in faults
            if item["correct_isolation_given_detection"] is not None
        ]

        report["scenarios"].append(
            {
                "satellite_count": count,
                "threshold": threshold,
                "calibration_valid_trials": calibration_valid,
                "clean_test": clean,
                "fault_scenarios": faults,
                "mean_detection_rate": (
                    float(np.mean(detection_rates))
                    if detection_rates else None
                ),
                "mean_isolation_accuracy": (
                    float(np.mean(isolation_rates))
                    if isolation_rates else None
                ),
                "mean_correct_isolation_given_detection": (
                    float(np.mean(conditional_rates))
                    if conditional_rates else None
                ),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("THRESHOLD-CALIBRATED REDUNDANCY EVALUATION")
    print(f"Report saved to: {REPORT_PATH}")
    print()

    for scenario in report["scenarios"]:
        print(f"Satellites: {scenario['satellite_count']}")
        print(f"  Threshold: {scenario['threshold']:.3f}")
        print(
            "  Clean false-alarm rate: "
            f"{scenario['clean_test']['false_alarm_rate']:.3f}"
        )
        print(
            "  Mean fault detection rate: "
            f"{scenario['mean_detection_rate']:.3f}"
        )
        print(
            "  Mean isolation accuracy: "
            f"{scenario['mean_isolation_accuracy']:.3f}"
        )
        print(
            "  Correct isolation given detection: "
            f"{scenario['mean_correct_isolation_given_detection']:.3f}"
        )


if __name__ == "__main__":
    main()
