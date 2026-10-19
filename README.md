
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
## Satellite Geometry Experiment

This experiment compares positioning performance under two synthetic satellite configurations: good geometry and poor geometry.

### Experimental Setup

- **Trials:** 500 per geometry, 1,000 total
- **Pseudorange noise standard deviation:** 2.0 m
- **Random seed:** 42
- **Satellites:** 6 in each configuration
- **Receiver clock bias:** 125 m
- **Noise sampling:** The same noise vector is used for corresponding trials across both geometries.

### Results

| Metric | Good Geometry | Poor Geometry |
|---|---:|---:|
| PDOP | 11.266 | 122.192 |
| TDOP | 9.660 | 120.457 |
| GDOP | 14.840 | 171.583 |
| Mean position error | 18.424 m | 194.484 m |
| Position RMSE | 22.576 m | 239.898 m |
| Median position error | 15.190 m | 166.458 m |
| Maximum position error | 63.086 m | 765.358 m |
| Successful trials | 500/500 | 500/500 |
| Failed trials | 0 | 0 |

### Interpretation

The poor-geometry configuration produced approximately **10.63 times the position RMSE** of the good-geometry configuration under the same pseudorange-noise standard deviation.

Its PDOP was approximately 10.85 times higher, indicating substantially weaker satellite geometry for positioning. The maximum observed position error also increased from 63.086 m to 765.358 m.

These results illustrate how satellite geometry can amplify the effect of pseudorange measurement noise on position estimates. The comparison is based on a controlled synthetic simulation and does not establish real-world GNSS accuracy.

### Limitations

- Satellite positions and measurements are synthetic rather than derived from real orbital data.
- Both configurations use six satellites and independent Gaussian pseudorange noise.
- Multipath, atmospheric effects, satellite clock errors, and orbital errors are not modelled.
- The findings apply to the simulated geometries, estimator, noise assumptions, and trial configuration. They should not be generalized to all real-world GNSS conditions.

## Satellite Availability Experiment

This experiment evaluates how the number of available synthetic satellites affects positioning performance.

### Experimental Setup

- **Trials:** 200 per satellite count
- **Satellite counts tested:** 6, 5, 4, and 3
- **Pseudorange noise standard deviation:** 2.0 m
- **Random seed:** 42
- **Receiver clock bias:** 125 m
- **Measurements:** Synthetic pseudoranges with independent Gaussian noise
- **Geometry:** Six normalized synthetic satellite direction vectors; each trial uses the first N satellites for the tested satellite count

### Results

| Available satellites | Successful trials | Failed trials | Position RMSE |
|---:|---:|---:|---:|
| 6 | 200/200 | 0 | 3.415 m |
| 5 | 200/200 | 0 | 4.077 m |
| 4 | 200/200 | 0 | 4.666 m |
| 3 | 0/200 | 200 | Not available |

### Interpretation

Under this experiment's synthetic geometry and noise assumptions, position RMSE increased as the number of available satellites decreased from six to four. All 200 trials succeeded for each of these three configurations.

All trials with three satellites failed because the estimator requires at least four satellites to estimate three-dimensional position and receiver clock bias.

Four satellites can be sufficient when the measurement geometry provides four independent constraints. The results demonstrate that the four-satellite subset used in this revised experiment satisfies the estimator's geometry requirements.

### Failure Analysis

The experiment records exception types and messages for failed trials. In this configuration, all 200 three-satellite trials failed with `ValueError: At least four satellites are required.` No failures occurred with four, five, or six satellites.

### Limitations

- Satellite positions and measurements are synthetic, not derived from real orbital data.
- Only independent Gaussian pseudorange noise is modelled.
- Multipath, atmospheric effects, satellite clock errors, and orbital errors are not modelled.
- Results apply to the selected satellite subsets, estimator, noise assumptions, and trial configuration. They should not be generalized to all real-world GNSS conditions.

## License

MIT
