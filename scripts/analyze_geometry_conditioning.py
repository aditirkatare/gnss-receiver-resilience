
"""Analyze geometry conditioning for leave-one-out GNSS subsets."""

import json
from pathlib import Path

import numpy as np


def main():
    satellites = np.array([
        [20e6, 0, 0],
        [0, 21e6, 0],
        [0, 0, 22e6],
        [-15e6, -16e6, -17e6],
        [18e6, -19e6, 12e6],
    ], dtype=float)

    receiver = np.array([1e6, 2e6, 3e6], dtype=float)
    results = {}

    for excluded in range(len(satellites)):
        keep = np.arange(len(satellites)) != excluded
        subset = satellites[keep]

        # Linearized pseudorange Jacobian at the known receiver position.
        differences = receiver - subset
        distances = np.linalg.norm(differences, axis=1)
        jacobian = np.column_stack((
            differences / distances[:, None],
            np.ones(len(subset)),
        ))

        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jacobian))
        condition_number = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 0 else None
        )

        results[str(excluded)] = {
            "excluded_satellite_index": excluded,
            "remaining_satellite_indices": np.flatnonzero(keep).tolist(),
            "jacobian_rank": rank,
            "unknown_count": 4,
            "singular_values": singular_values.tolist(),
            "condition_number": condition_number,
            "geometry_solvable": rank == 4,
        }

    report = {
        "experiment": "Leave-one-out geometry conditioning",
        "receiver_position_m": receiver.tolist(),
        "satellite_count": len(satellites),
        "method": (
            "Condition numbers of linearized pseudorange Jacobians "
            "evaluated at the known synthetic receiver position."
        ),
        "results_by_excluded_satellite": results,
        "limitations": [
            "Synthetic fixed satellite geometry.",
            "Linearized geometry evaluated at the known receiver position.",
            "Condition number alone does not prove the cause of isolation errors.",
            "This is not a complete RAIM implementation.",
        ],
    }

    output = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "geometry_conditioning_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
