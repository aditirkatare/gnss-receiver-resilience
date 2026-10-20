
"""Analyze leave-one-out GNSS fault-isolation scores."""

import json
from pathlib import Path

import numpy as np

from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


def score_excluded_satellites(satellites, ranges, initial_position, clock_bias):
    scores = []

    for excluded in range(len(satellites)):
        keep = np.arange(len(satellites)) != excluded
        position, bias = estimate_position(
            satellites[keep],
            ranges[keep],
            initial_position_m=initial_position,
            initial_clock_bias_m=clock_bias,
        )
        predicted = np.linalg.norm(satellites[excluded] - position) + bias
        scores.append(float(ranges[excluded] - predicted))

    return np.asarray(scores)


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
    noise_std = 1.0
    fault_size = 1000.0
    clock_bias = 125.0
    rng = np.random.default_rng(42)
    initial_position = receiver + np.array([100.0, -80.0, 50.0])
    count = len(satellites)

    results = {}

    for faulty in range(count):
        scores_by_candidate = [[] for _ in range(count)]
        correct_top = 0
        errors = 0
        top_counts = {str(i): 0 for i in range(count)}

        for _ in range(trials):
            noise = rng.normal(0.0, noise_std, size=count)
            clean = calculate_pseudoranges(
                receiver,
                satellites,
                receiver_clock_bias_m=clock_bias,
                noise_m=noise,
            )
            ranges = clean.copy()
            ranges[faulty] += fault_size

            try:
                scores = score_excluded_satellites(
                    satellites, ranges, initial_position, clock_bias
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                errors += 1
                continue

            absolute_scores = np.abs(scores)
            predicted = int(np.argmax(absolute_scores))
            top_counts[str(predicted)] += 1
            correct_top += int(predicted == faulty)

            for candidate, score in enumerate(absolute_scores):
                scores_by_candidate[candidate].append(float(score))

        valid = trials - errors
        mean_scores = [
            float(np.mean(values)) if values else None
            for values in scores_by_candidate
        ]

        competing_maxima = []
        for trial_index in range(valid):
            competing = [
                scores_by_candidate[i][trial_index]
                for i in range(count)
                if i != faulty
            ]
            if competing:
                competing_maxima.append(max(competing))

        results[str(faulty)] = {
            "trials": trials,
            "valid_trials": valid,
            "errors": errors,
            "correct_top_score": correct_top,
            "correct_top_score_rate": correct_top / valid if valid else None,
            "top_score_counts": top_counts,
            "mean_absolute_score_by_excluded_satellite_m": mean_scores,
            "mean_correct_score_m": mean_scores[faulty],
            "mean_largest_competing_score_m": (
                float(np.mean(competing_maxima)) if competing_maxima else None
            ),
        }

    report = {
        "experiment": "Leave-one-out score analysis",
        "seed": 42,
        "trials_per_faulty_satellite": trials,
        "noise_std_m": noise_std,
        "injected_fault_m": fault_size,
        "results_by_actual_faulty_satellite": results,
        "limitations": [
            "Synthetic measurements and fixed satellite geometry.",
            "One positive pseudorange fault is injected per trial.",
            "Leave-one-out fits four satellites to four unknowns.",
            "This is a diagnostic, not a complete RAIM implementation.",
            "Results do not establish real-world receiver performance.",
        ],
    }

    output = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "leave_one_out_score_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
