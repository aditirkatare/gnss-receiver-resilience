
import json
from pathlib import Path

import numpy as np

from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORT_PATH = RESULTS_DIR / "satellite_redundancy_report.json"

RECEIVER_POSITION_M = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])
CLOCK_BIAS_M = 125.0
NOISE_STD_M = 1.0
INJECTED_FAULT_M = 1000.0
TRIALS_PER_SCENARIO = 200
SEED = 42

# The first five satellites match the existing project geometry.
# Three additional satellites let us evaluate increasing redundancy.
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


def estimate_with_satellites(satellite_positions_m, pseudoranges_m):
    """Estimate receiver position using a fixed initial guess."""
    return estimate_position(
        satellite_positions_m=satellite_positions_m,
        pseudoranges_m=pseudoranges_m,
        initial_position_m=RECEIVER_POSITION_M + np.array([100.0, -100.0, 100.0]),
        initial_clock_bias_m=0.0,
    )


def normalized_residual_scores(satellite_positions_m, pseudoranges_m):
    """Return absolute post-fit residuals normalized by leverage."""
    position, clock_bias = estimate_with_satellites(
        satellite_positions_m, pseudoranges_m
    )

    predicted_ranges = (
        np.linalg.norm(satellite_positions_m - position, axis=1) + clock_bias
    )
    residuals = pseudoranges_m - predicted_ranges

    line_of_sight = (
        position - satellite_positions_m
    ) / np.linalg.norm(satellite_positions_m - position, axis=1)[:, None]

    design_matrix = np.column_stack(
        [line_of_sight, np.ones(len(satellite_positions_m))]
    )
    projection = design_matrix @ np.linalg.pinv(design_matrix)
    leverage = np.clip(np.diag(projection), 0.0, 1.0 - 1e-10)

    scores = np.abs(residuals) / np.sqrt(1.0 - leverage)
    return scores


def run_scenario(satellite_count, faulty_satellite, rng):
    """Evaluate one clean or single-fault scenario."""
    positions = SATELLITE_POSITIONS_M[:satellite_count]
    correct_isolations = 0
    detections = 0
    valid_trials = 0
    errors = 0
    candidate_counts = np.zeros(satellite_count, dtype=int)

    for _ in range(TRIALS_PER_SCENARIO):
        noise = rng.normal(0.0, NOISE_STD_M, satellite_count)

        true_ranges = (
            np.linalg.norm(positions - RECEIVER_POSITION_M, axis=1)
            + CLOCK_BIAS_M
        )
        pseudoranges = true_ranges + noise

        if faulty_satellite is not None:
            pseudoranges[faulty_satellite] += INJECTED_FAULT_M

        try:
            scores = normalized_residual_scores(positions, pseudoranges)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            errors += 1
            continue

        if not np.all(np.isfinite(scores)):
            errors += 1
            continue

        valid_trials += 1
        candidate = int(np.argmax(scores))
        candidate_counts[candidate] += 1

        # Detection here means a fault was injected and the diagnostic
        # produced a candidate. This is not a threshold-based detector.
        if faulty_satellite is not None:
            detections += 1
            if candidate == faulty_satellite:
                correct_isolations += 1

    return {
        "faulty_satellite_index": faulty_satellite,
        "valid_trials": valid_trials,
        "errors": errors,
        "candidate_counts_by_satellite_index": candidate_counts.tolist(),
        "fault_detections": detections,
        "detection_rate": (
            detections / valid_trials if valid_trials else None
        ),
        "correct_isolations": correct_isolations,
        "correct_isolation_rate": (
            correct_isolations / valid_trials if valid_trials else None
        ),
    }


def main():
    rng = np.random.default_rng(SEED)
    report = {
        "experiment": "Satellite redundancy and normalized residual isolation",
        "receiver_position_m": RECEIVER_POSITION_M.tolist(),
        "receiver_clock_bias_m": CLOCK_BIAS_M,
        "noise_std_m": NOISE_STD_M,
        "injected_fault_m": INJECTED_FAULT_M,
        "trials_per_scenario": TRIALS_PER_SCENARIO,
        "seed": SEED,
        "scenarios": [],
        "notes": [
            "Satellite counts use the first N entries of one fixed geometry.",
            "Isolation accuracy measures whether the highest normalized residual "
            "belongs to the injected faulty satellite.",
            "Fault detection is not threshold-based in this experiment.",
            "Synthetic results do not establish real-world GNSS performance.",
        ],
    }

    for satellite_count in (5, 6, 7, 8):
        clean_result = run_scenario(satellite_count, None, rng)
        fault_results = [
            run_scenario(satellite_count, faulty_index, rng)
            for faulty_index in range(satellite_count)
        ]

        isolation_rates = [
            result["correct_isolation_rate"]
            for result in fault_results
            if result["correct_isolation_rate"] is not None
        ]

        report["scenarios"].append(
            {
                "satellite_count": satellite_count,
                "clean": clean_result,
                "fault_scenarios": fault_results,
                "mean_correct_isolation_rate": (
                    float(np.mean(isolation_rates)) if isolation_rates else None
                ),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("GNSS SATELLITE REDUNDANCY EVALUATION")
    print(f"Report saved to: {REPORT_PATH}")
    print()

    for scenario in report["scenarios"]:
        print(f"Satellites: {scenario['satellite_count']}")
        print(
            "  Mean correct isolation rate: "
            f"{scenario['mean_correct_isolation_rate']:.3f}"
        )
        print(
            "  Clean candidate counts: "
            f"{scenario['clean']['candidate_counts_by_satellite_index']}"
        )
        print(
            "  Fault-scenario valid trials: "
            f"{sum(item['valid_trials'] for item in scenario['fault_scenarios'])}"
        )


if __name__ == "__main__":
    main()

