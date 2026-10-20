
import json
from pathlib import Path

import numpy as np

from gnss_resilience.position import estimate_position


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORT_PATH = RESULTS_DIR / "geometry_robustness_report.json"

RECEIVER_POSITION_M = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])
CLOCK_BIAS_M = 125.0
NOISE_STD_M = 1.0
INJECTED_FAULT_M = 1000.0

SATELLITE_COUNTS = (5, 6, 7, 8)
CALIBRATION_TRIALS = 1000
TEST_TRIALS = 1000
TARGET_FALSE_ALARM_RATE = 0.01
SEED = 2026

BASE_POSITIONS = np.array(
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


def build_geometries():
    # Geometry A: original arrangement.
    original = BASE_POSITIONS.copy()

    # Geometry B: rotate the satellite directions around the z-axis.
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = BASE_POSITIONS @ rotation.T

    # Geometry C: move the satellite directions closer together.
    # This intentionally creates a more challenging geometry.
    receiver = RECEIVER_POSITION_M
    directions = BASE_POSITIONS - receiver
    center_direction = np.mean(
        directions / np.linalg.norm(directions, axis=1)[:, None],
        axis=0,
    )
    center_direction /= np.linalg.norm(center_direction)

    compressed_directions = []
    for direction in directions:
        unit = direction / np.linalg.norm(direction)
        blended = 0.35 * unit + 0.65 * center_direction
        blended /= np.linalg.norm(blended)
        compressed_directions.append(blended)

    compressed = receiver + np.array(compressed_directions) * 20e6

    return {
        "original": original,
        "rotated": rotated,
        "compressed": compressed,
    }


def make_measurements(positions, rng, faulty_index=None):
    true_ranges = (
        np.linalg.norm(positions - RECEIVER_POSITION_M, axis=1)
        + CLOCK_BIAS_M
    )
    pseudoranges = true_ranges + rng.normal(0.0, NOISE_STD_M, len(positions))

    if faulty_index is not None:
        pseudoranges[faulty_index] += INJECTED_FAULT_M

    return pseudoranges


def calculate_scores(positions, pseudoranges):
    estimate, clock_bias = estimate_position(
        satellite_positions_m=positions,
        pseudoranges_m=pseudoranges,
        initial_position_m=RECEIVER_POSITION_M
        + np.array([100.0, -100.0, 100.0]),
        initial_clock_bias_m=0.0,
    )

    ranges = np.linalg.norm(positions - estimate, axis=1)
    residuals = pseudoranges - (ranges + clock_bias)

    line_of_sight = (estimate - positions) / ranges[:, None]
    design = np.column_stack([line_of_sight, np.ones(len(positions))])
    projection = design @ np.linalg.pinv(design)
    leverage = np.clip(np.diag(projection), 0.0, 1.0 - 1e-10)

    return np.abs(residuals) / np.sqrt(1.0 - leverage)


def safe_scores(positions, rng, faulty_index=None):
    try:
        pseudoranges = make_measurements(positions, rng, faulty_index)
        scores = calculate_scores(positions, pseudoranges)
        if np.all(np.isfinite(scores)):
            return scores
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    return None


def geometry_condition_number(positions):
    direction = (
        RECEIVER_POSITION_M - positions
    ) / np.linalg.norm(positions - RECEIVER_POSITION_M, axis=1)[:, None]
    design = np.column_stack([direction, np.ones(len(positions))])
    return float(np.linalg.cond(design))


def calibrate_threshold(positions, rng):
    maximum_scores = []

    for _ in range(CALIBRATION_TRIALS):
        scores = safe_scores(positions, rng)
        if scores is not None:
            maximum_scores.append(float(np.max(scores)))

    if len(maximum_scores) < 100:
        raise RuntimeError("Too few valid calibration trials.")

    threshold = float(
        np.quantile(
            maximum_scores,
            1.0 - TARGET_FALSE_ALARM_RATE,
            method="higher",
        )
    )
    return threshold, len(maximum_scores)


def evaluate_clean(positions, rng, threshold):
    valid = 0
    errors = 0
    alarms = 0

    for _ in range(TEST_TRIALS):
        scores = safe_scores(positions, rng)
        if scores is None:
            errors += 1
            continue

        valid += 1
        if np.max(scores) > threshold:
            alarms += 1

    return {
        "valid_trials": valid,
        "errors": errors,
        "false_alarms": alarms,
        "false_alarm_rate": alarms / valid if valid else None,
    }


def evaluate_fault(positions, rng, threshold, faulty_index):
    valid = 0
    errors = 0
    alarms = 0
    correct = 0
    detected_and_correct = 0

    for _ in range(TEST_TRIALS):
        scores = safe_scores(positions, rng, faulty_index)
        if scores is None:
            errors += 1
            continue

        valid += 1
        alarm = float(np.max(scores)) > threshold
        candidate = int(np.argmax(scores))

        if alarm:
            alarms += 1
            if candidate == faulty_index:
                detected_and_correct += 1

        if candidate == faulty_index:
            correct += 1

    return {
        "faulty_satellite_index": faulty_index,
        "valid_trials": valid,
        "errors": errors,
        "detections": alarms,
        "detection_rate": alarms / valid if valid else None,
        "correct_isolations": correct,
        "isolation_accuracy": correct / valid if valid else None,
        "detected_and_correct": detected_and_correct,
        "correct_isolation_given_detection": (
            detected_and_correct / alarms if alarms else None
        ),
    }


def main():
    rng = np.random.default_rng(SEED)
    geometries = build_geometries()

    report = {
        "experiment": "Geometry robustness of thresholded normalized residuals",
        "receiver_position_m": RECEIVER_POSITION_M.tolist(),
        "noise_std_m": NOISE_STD_M,
        "injected_fault_m": INJECTED_FAULT_M,
        "calibration_trials_requested": CALIBRATION_TRIALS,
        "test_trials_requested_per_scenario": TEST_TRIALS,
        "target_false_alarm_rate": TARGET_FALSE_ALARM_RATE,
        "seed": SEED,
        "scenarios": [],
        "limitations": [
            "All geometries are synthetic.",
            "Rotating the original geometry is a coordinate transformation, "
            "not an independent geometry-quality test.",
            "The compressed geometry is a deliberately constructed challenge, "
            "not a model of a specific real constellation.",
            "Thresholds are calibrated separately for every geometry and "
            "satellite count.",
            "A single injected fault of fixed magnitude is tested.",
        ],
    }

    for geometry_name, full_positions in geometries.items():
        for count in SATELLITE_COUNTS:
            positions = full_positions[:count]
            threshold, calibration_valid = calibrate_threshold(positions, rng)
            clean = evaluate_clean(positions, rng, threshold)

            faults = [
                evaluate_fault(positions, rng, threshold, index)
                for index in range(count)
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
                    "geometry": geometry_name,
                    "satellite_count": count,
                    "geometry_condition_number": geometry_condition_number(
                        positions
                    ),
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

    print("GEOMETRY ROBUSTNESS EVALUATION")
    print(f"Report saved to: {REPORT_PATH}")
    print()

    for item in report["scenarios"]:
        print(
            f"{item['geometry']} | "
            f"{item['satellite_count']} satellites"
        )
        print(
            f"  Condition number: "
            f"{item['geometry_condition_number']:.3f}"
        )
        print(
            f"  False-alarm rate: "
            f"{item['clean_test']['false_alarm_rate']:.3f}"
        )
        print(
            f"  Detection rate: {item['mean_detection_rate']:.3f}"
        )
        print(
            f"  Isolation accuracy: "
            f"{item['mean_isolation_accuracy']:.3f}"
        )
        print(
            f"  Correct isolation given detection: "
            f"{item['mean_correct_isolation_given_detection']:.3f}"
        )


if __name__ == "__main__":
    main()
