
"""Measure leave-one-out position error under clean and faulty measurements."""

import json
from pathlib import Path

import numpy as np

from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


def main():
    receiver = np.array([1e6, 2e6, 3e6])
    satellites = np.array([
        [20e6, 0, 0],
        [0, 21e6, 0],
        [0, 0, 22e6],
        [-15e6, -16e6, -17e6],
        [18e6, -19e6, 12e6],
    ], dtype=float)

    trials = 200
    noise_std = 1.0
    fault_size = 1000.0
    clock_bias = 125.0
    rng = np.random.default_rng(42)
    initial_position = receiver + np.array([100.0, -80.0, 50.0])
    count = len(satellites)

    errors = {
        str(actual_fault): {
            str(excluded): []
            for excluded in range(count)
        }
        for actual_fault in range(-1, count)
    }

    failures = {
        str(actual_fault): {
            str(excluded): 0
            for excluded in range(count)
        }
        for actual_fault in range(-1, count)
    }

    for _ in range(trials):
        noise = rng.normal(0.0, noise_std, size=count)
        clean = calculate_pseudoranges(
            receiver,
            satellites,
            receiver_clock_bias_m=clock_bias,
            noise_m=noise,
        )

        for actual_fault in range(-1, count):
            ranges = clean.copy()
            if actual_fault >= 0:
                ranges[actual_fault] += fault_size

            for excluded in range(count):
                keep = np.arange(count) != excluded
                key = str(actual_fault)

                try:
                    estimate, _ = estimate_position(
                        satellites[keep],
                        ranges[keep],
                        initial_position_m=initial_position,
                        initial_clock_bias_m=clock_bias,
                    )
                    error_m = float(np.linalg.norm(estimate - receiver))
                    errors[key][str(excluded)].append(error_m)
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    failures[key][str(excluded)] += 1

    summary = {}
    for actual_fault in range(-1, count):
        key = str(actual_fault)
        summary[key] = {}

        for excluded in range(count):
            values = errors[key][str(excluded)]
            summary[key][str(excluded)] = {
                "valid_trials": len(values),
                "errors": failures[key][str(excluded)],
                "mean_position_error_m": (
                    float(np.mean(values)) if values else None
                ),
                "rmse_position_error_m": (
                    float(np.sqrt(np.mean(np.square(values))))
                    if values else None
                ),
                "max_position_error_m": (
                    float(np.max(values)) if values else None
                ),
            }

    report = {
        "experiment": "Leave-one-out subset position error analysis",
        "seed": 42,
        "trials_per_scenario": trials,
        "noise_std_m": noise_std,
        "injected_fault_m": fault_size,
        "scenario_key": (
            "-1 means clean measurements; 0-4 identify the faulty satellite."
        ),
        "results_by_actual_fault": summary,
        "limitations": [
            "Synthetic measurements and fixed satellite geometry.",
            "Position errors are evaluated against the known synthetic receiver.",
            "This is a diagnostic, not a complete RAIM implementation.",
            "Results do not establish real-world receiver performance.",
        ],
    }

    output = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "subset_position_error_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
