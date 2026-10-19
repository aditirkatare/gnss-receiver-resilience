
# GNSS Receiver & Resilience Simulator

A simulation-based Global Navigation Satellite System (GNSS) project exploring pseudorange measurements, receiver position estimation, satellite geometry, measurement noise, and positioning resilience.

## Current Features

- Simulated pseudorange measurements with receiver clock bias and noise.
- Receiver position and clock-bias estimation using nonlinear least squares.
- Position Dilution of Precision (PDOP), Time Dilution of Precision (TDOP), and Geometric Dilution of Precision (GDOP).
- Reproducible Gaussian pseudorange-noise experiments.
- Automated unit tests and GitHub Actions continuous integration.

## Noise Experiment

The noise experiment evaluates positioning error under six pseudorange-noise standard deviations: 0, 0.5, 1, 2, 5, and 10 metres.

Each noise level uses 500 trials, with a fixed random seed and fixed synthetic satellite geometry.

### Results

| Noise standard deviation | Position RMSE |
|---:|---:|
| 0 m | Approximately 0 m |
| 0.5 m | 0.709 m |
| 1 m | 1.419 m |
| 2 m | 2.837 m |
| 5 m | 7.093 m |
| 10 m | 14.186 m |

All 3,000 requested trials completed successfully in this experiment.

### Interpretation

Positioning RMSE increased as simulated pseudorange noise increased. With zero injected noise, the estimator recovered the known receiver position to within numerical precision.

The experiment uses synthetic measurements and fixed satellite geometry. Its results demonstrate the behaviour of this implementation under the stated conditions; they do not establish real-world GNSS receiver accuracy or resilience.

## Reproducibility

- Python 3.12
- Dependencies are pinned in `requirements.txt`.
- Randomized experiments use a fixed seed.
- Unit tests run through GitHub Actions.
- The noise experiment runs in the CI workflow.

To run the experiment in a configured Python environment:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python scripts/run_noise_experiment.py
```

## Limitations and Future Work

- Satellite positions and measurements are synthetic.
- The current noise experiment uses one fixed satellite geometry.
- Multipath, atmospheric effects, satellite clock and orbit errors, and real receiver data are not yet modelled.
- Future work can compare satellite geometries, study signal degradation, and evaluate resilience under multiple error sources.

## License

MIT
