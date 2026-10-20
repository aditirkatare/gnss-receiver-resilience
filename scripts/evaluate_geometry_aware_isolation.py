
"""Evaluate geometry-aware satellite fault-isolation strategies.

Compares:
1. Largest post-fit residual.
2. Largest leave-one-out pseudorange prediction discrepancy.
3. Candidate exclusion validated by position-error improvement.

Synthetic experiment only. Known receiver truth is used for evaluation,
not as an operational input to the isolation decision.
"""

import json
from pathlib import Path

import numpy as np

from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


def build_geometry():
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
    return receiver, satellites


def fit_solution(satellites, ranges, initial_position, clock_bias):
    """Return estimated position, clock bias, and residuals."""
    estimate, estimated_clock_bias = estimate_position(
        satellites,
        ranges,
        initial_position_m=initial_position,
        initial_clock_bias_m=clock_bias,
    )

    predicted = (
        np.linalg.norm(satellites - estimate, axis=1)
        + estimated_clock_bias
    )
    residuals = ranges - predicted
    return estimate, estimated_clock_bias, residuals


def evaluate_scenario(
    ranges,
    receiver,
    satellites,
    initial_position,
    clock_bias,
):
    count = len(satellites)
    full_fit = None
    full_fit_error = None
    residual_scores = None
    loo_scores = []
    subset_position_errors = []
    subset_failures = []

    try:
        estimate, estimated_clock_bias, residuals = fit_solution(
            satellites,
            ranges,
            initial_position,
            clock_bias,
        )
        full_fit = {
            "position_error_m": float(
                np.linalg.norm(estimate - receiver)
            ),
            "estimated_clock_bias_m": float(estimated_clock_bias),
        }
        residual_scores = np.abs(residuals).tolist()
        full_fit_error = None
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        full_fit_error = str(exc)

    for excluded in range(count):
        keep = np.arange(count) != excluded

        # Candidate-exclusion subset position error is an evaluation metric
        # only. In deployment, the true receiver position is unknown.
        try:
            subset_position, subset_clock, subset_residuals = fit_solution(
                satellites[keep],
                ranges[keep],
                initial_position,
                clock_bias,
            )
            subset_error = float(
                np.linalg.norm(subset_position - receiver)
            )
            subset_position_errors.append(subset_error)
            subset_failures.append(None)

            # Predict the omitted satellite range from the four-satellite fit.
            predicted_omitted = (
                np.linalg.norm(
                    satellites[excluded] - subset_position
                )
                + subset_clock
            )
            loo_scores.append(
                float(abs(ranges[excluded] - predicted_omitted))
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            subset_position_errors.append(None)
            subset_failures.append(str(exc))
            loo_scores.append(None)

    residual_candidate = (
        int(np.argmax(residual_scores))
        if residual_scores is not None
        else None
    )

    valid_loo = [
        (index, score)
        for index, score in enumerate(loo_scores)
        if score is not None and np.isfinite(score)
    ]
    loo_candidate = (
        max(valid_loo, key=lambda item: item[1])[0]
        if valid_loo
        else None
    )

    # This candidate is selected by the lowest position error against known
    # synthetic truth. It is an oracle benchmark, NOT an operational method.
    valid_position_errors = [
        (index, error)
        for index, error in enumerate(subset_position_errors)
        if error is not None and np.isfinite(error)
    ]
    oracle_candidate = (
        min(valid_position_errors, key=lambda item: item[1])[0]
        if valid_position_errors
        else None
    )

    return {
        "full_fit": full_fit,
        "full_fit_error": full_fit_error,
        "absolute_postfit_residuals_m": residual_scores,
        "residual_rank_candidate": residual_candidate,
        "leave_one_out_scores_m": loo_scores,
        "leave_one_out_candidate": loo_candidate,
        "subset_position_errors_m": subset_position_errors,
        "subset_fit_failures": subset_failures,
        "oracle_best_exclusion": oracle_candidate,
    }


def main():
    receiver, satellites = build_geometry()
    trials = 200
    noise_std = 1.0
    fault_size = 1000.0
    clock_bias = 125.0
    rng = np.random.default_rng(42)
    initial_position = receiver + np.array([100.0, -80.0, 50.0])
    count = len(satellites)

    scenarios = {
        str(actual_fault): []
        for actual_fault in range(-1, count)
    }

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

            result = evaluate_scenario(
                ranges,
                receiver,
                satellites,
                initial_position,
                clock_bias,
            )
            result["actual_fault_satellite"] = actual_fault
            scenarios[str(actual_fault)].append(result)

    summary = {}

    for scenario, records in scenarios.items():
        actual_fault = int(scenario)
        summary[scenario] = {}

        for method, key in [
            ("postfit_residual", "residual_rank_candidate"),
            ("leave_one_out", "leave_one_out_candidate"),
            ("oracle_best_exclusion", "oracle_best_exclusion"),
        ]:
            candidates = [
                record[key]
                for record in records
                if record[key] is not None
            ]
            correct = (
                sum(candidate == actual_fault for candidate in candidates)
                if actual_fault >= 0
                else sum(candidate is None for candidate in candidates)
            )
            summary[scenario][method] = {
                "valid_trials": len(candidates),
                "candidate_counts": {
                    str(index): candidates.count(index)
                    for index in range(count)
                },
                "correct_isolations": correct if actual_fault >= 0 else None,
                "isolation_accuracy": (
                    float(correct / len(candidates))
                    if candidates and actual_fault >= 0
                    else None
                ),
            }

        full_errors = [
            record["full_fit"]["position_error_m"]
            for record in records
            if record["full_fit"] is not None
        ]
        summary[scenario]["full_fit_position_error"] = {
            "valid_trials": len(full_errors),
            "rmse_m": (
                float(np.sqrt(np.mean(np.square(full_errors))))
                if full_errors
                else None
            ),
        }

        summary[scenario]["subset_position_error_rmse_m"] = {}
        for excluded in range(count):
            values = [
                record["subset_position_errors_m"][excluded]
                for record in records
                if record["subset_position_errors_m"][excluded] is not None
            ]
            summary[scenario]["subset_position_error_rmse_m"][
                str(excluded)
            ] = (
                float(np.sqrt(np.mean(np.square(values))))
                if values
                else None
            )

    clean_records = scenarios["-1"]
    clean_loo_candidates = [
        record["leave_one_out_candidate"]
        for record in clean_records
        if record["leave_one_out_candidate"] is not None
    ]
    clean_residual_candidates = [
        record["residual_rank_candidate"]
        for record in clean_records
        if record["residual_rank_candidate"] is not None
    ]

    report = {
        "experiment": "Geometry-aware fault-isolation evaluation",
        "seed": 42,
        "trials_per_scenario": trials,
        "noise_std_m": noise_std,
        "injected_fault_m": fault_size,
        "satellite_count": count,
        "scenario_key": (
            "-1 means clean measurements; 0-4 identify the faulty satellite."
        ),
        "results_by_actual_fault": summary,
        "clean_data_candidate_selection": {
            "leave_one_out_candidate_counts": {
                str(index): clean_loo_candidates.count(index)
                for index in range(count)
            },
            "postfit_residual_candidate_counts": {
                str(index): clean_residual_candidates.count(index)
                for index in range(count)
            },
        },
        "limitations": [
            "Synthetic measurements and fixed satellite geometry only.",
            "The oracle best-exclusion benchmark uses known receiver truth "
            "and is not an operational isolation algorithm.",
            "Candidate counts alone do not establish a safe fault decision.",
            "This is not a complete RAIM implementation.",
            "No real-world receiver performance is established.",
        ],
    }

    output = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "geometry_aware_isolation_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

