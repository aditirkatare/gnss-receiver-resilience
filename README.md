# GNSS Receiver & Resilience Simulator

A simulation-based Global Navigation Satellite System (GNSS) positioning project exploring pseudorange measurements, receiver position estimation, measurement noise, and positioning resilience.

## Objectives

- Simulate satellite positions and receiver measurements.
- Estimate receiver position from pseudorange observations.
- Evaluate the impact of measurement noise on positioning accuracy.
- Produce reproducible experiments, plots, and evaluation reports.
- Validate core functionality with automated tests.

## Project Status

**Phase 1: Repository setup**

The initial implementation will use synthetic satellite and measurement data. Physical GNSS hardware and live satellite reception are outside the scope of the initial version.

## Planned Technology

- Python
- NumPy and SciPy
- Matplotlib
- pytest
- GitHub Actions

## Planned Repository Structure

```text
gnss-receiver-resilience/
├── src/
├── tests/
├── notebooks/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducibility

Experiments will use documented assumptions and fixed random seeds where randomness is involved. Automated tests will check expected behavior, and evaluation results will distinguish simulated performance from real-world receiver performance.

## Scope and Limitations

This project is an educational software simulation. Results generated from synthetic data do not establish the accuracy or resilience of a physical GNSS receiver.

## License

MIT License
