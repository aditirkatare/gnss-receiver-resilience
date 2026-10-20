
"""Evaluate geometry-normalized residual fault isolation.

Compares the largest raw post-fit residual with a normalized residual
score based on the diagonal of the least-squares residual sensitivity matrix.

This is a synthetic research experiment, not a complete RAIM implementation.
"""

import json
from pathlib import Path

import numpy as np

from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


def geometry_and_residuals(receiver_estimate, satellites, ranges, clock_bias):
    """Return residuals and the local pseudorange design matrix."""
    differences = receiver_estimate - satellites
    distances = np.linalg.norm(differences, axis=1)

    if np.any(distances <= 0):
        raise ValueError("Receiver estimate coincides with a satellite.")

    # Derivative of predicted pseudorange with respect to x, y, z, clock bias.
    design = np.column_stack((differences / distances[:, None], np.ones(len(satellites))))

    predicted = distances + clock_bias
    residuals = ranges - predicted

    return residuals, design


def normalized_residual_scores(residuals, design):
    """Calculate absolute residuals normalized by residual sensitivity."""
    projection = design @ np.linalg.pinv(design)
    residual_sensitivity = 1.0 - np.diag(projection)

    # Very small sensitivity means a residual is not reliably distinguishable.
    minimum_sensitivity = 1e-8
    scores = []

    for residual, sensitivity in zip(residuals, residual_sensitivity):
        if sensitivity <= minimum_sensitivity:
            scores.append(None)
        else:
            scores.append(float(abs(residual) / np.sqrt(sensitivity)))

    return scores, residual_sensitivity.tolist()


def fit_and_score(satellites, ranges, initial_position, clock_bias):
    estimate, estimated_clock = estimate_position(
        satellites,
        ranges,
        initial_position_m=initial_position,
        initial_clock_bias_m=clock_bias,
    )

    residuals, design = geometry_and_residuals(
        estimate, satellites, ranges, estimated_clock
    )
    normalized_scores, sensitivities = normalized_residual_scores(
        residuals, design
    )

    return {
        "position": estimate,
        "clock_bias": estimated_clock,
        "residuals": residuals,
        "normalized_scores": normalized_scores,
        "sensitivities": sensitivities,
    }


def choose_largest(values):
    valid = [
        (index, value)
        for index, value in enumerate(values)
        if value is not None and np.isfinite(value)
    ]
    if not valid:
        return None
    return max(valid, key=lambda item: item[1])[0]


def main():
    receiver = np.array([1e6, 2e6, 3e6], dtype=float)
    satellites = np.array(
        [
            [20e6, 0, 0],
            [0, 21e6, 0],
            [0, 0, 22e6],
            [-15e6, -16e6, -17e6],
            [18e6, -19e6, 12e6],
        ],
        dtype=float,
    )

    trials = 200
    noise_std = 1.0
    fault_size = 1000.0
    clock_bias = 125.0
    rng = np.random.default_rng(42)
    initial_position = receiver + np.array([100.0, -80.0, 50.0])
    count = len(satellites)

    scenarios = {str(fault): [] for fault in range(-1, count)}

    for _ in range(trials):
        noise = rng.normal(0.0, noise_std, size=count)
        clean_ranges = calculate_pseudoranges(
            receiver,
            satellites,
            receiver_clock_bias_m=clock_bias,
            noise_m=noise,
        )

        for actual_fault in range(-1, count):
            ranges = clean_ranges.copy()
            if actual_fault >= 0:
                ranges[actual_fault] += fault_size

            try:
                result = fit_and_score(
                    satellites, ranges, initial_position, clock_bias
                )
                residual_scores = np.abs(result["residuals"]).tolist()
                normalized_scores = result["normalized_scores"]

                scenarios[str(actual_fault)].append(
                    {
                        "raw_candidate": choose_largest(residual_scores),
                        "normalized_candidate": choose_largest(normalized_scores),
                        "raw_scores_m": residual_scores,
                        "normalized_scores_m": normalized_scores,
                        "residual_sensitivity": result["sensitivities"],
                        "position_error_m": float(
                            np.linalg.norm(result["position"] - receiver)
                        ),
                    }
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                scenarios[str(actual_fault)].append(
                    {
                        "raw_candidate": None,
                        "normalized_candidate": None,
                        "raw_scores_m": None,
                        "normalized_scores_m": None,
                        "residual_sensitivity": None,
                        "position_error_m": None,
                    }
                )

    summary = {}
    for scenario, records in scenarios.items():
        actual_fault = int(scenario)
        summary[scenario] = {}

        for method, key in (
            ("raw_residual", "raw_candidate"),
            ("normalized_residual", "normalized_candidate"),
        ):
            candidates = [
                record[key] for record in records if record[key] is not None
            ]
            summary[scenario][method] = {
                "valid_trials": len(candidates),
                "candidate_counts": {
                    str(index): candidates.count(index) for index in range(count)
                },
                "correct_isolations": (
                    sum(candidate == actual_fault for candidate in candidates)
                    if actual_fault >= 0
                    else None
                ),
                "isolation_accuracy": (
                    float(
                        sum(candidate == actual_fault for candidate in candidates)
                        / len(candidates)
                    )
                    if candidates and actual_fault >= 0
                    else None
                ),
            }

        errors = [
            record["position_error_m"]
            for record in records
            if record["position_error_m"] is not None
        ]
        summary[scenario]["position_error"] = {
            "valid_trials": len(errors),
            "rmse_m": (
                float(np.sqrt(np.mean(np.square(errors)))) if errors else None
            ),
        }

    report = {
        "experiment": "Normalized residual fault-isolation evaluation",
        "seed": 42,
        "trials_per_scenario": trials,
        "noise_std_m": noise_std,
        "injected_fault_m": fault_size,
        "satellite_count": count,
        "scenario_key": "-1 means clean; 0-4 identify the faulty satellite.",
        "results_by_actual_fault": summary,
        "method_note": (
            "Scores use local least-squares residual sensitivity. The largest "
            "normalized residual is a candidate ranking, not a confirmed fault."
        ),
        "limitations": [
            "Synthetic measurements and fixed satellite geometry.",
            "Local linearization may be inaccurate for large faults.",
            "A five-satellite, four-unknown solution has only one residual degree "
            "of freedom, limiting fault isolation.",
            "No real-world receiver performance is established.",
            "This is not a complete RAIM implementation.",
        ],
    }

    output = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "normalized_residual_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
