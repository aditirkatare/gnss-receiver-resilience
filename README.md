# GNSS Receiver & Resilience Simulator

A reproducible Python-based simulator for evaluating Global Navigation Satellite System (GNSS) positioning accuracy, satellite availability, measurement faults, and fault-isolation strategies under controlled synthetic conditions.

The project investigates how measurement noise, satellite geometry, constellation size, and residual-based diagnostics influence receiver positioning and fault detection.

> **Research scope:** This is a synthetic simulation and evaluation project. Results demonstrate behavior under specified experimental assumptions, not validated performance on real GNSS hardware or live satellite measurements.

## Project Objectives

- Simulate GNSS pseudorange measurements and receiver positioning.
- Evaluate positioning error under different measurement-noise levels.
- Investigate how satellite availability affects positioning accuracy.
- Detect injected measurement faults using residual-based diagnostics.
- Evaluate faulty-satellite isolation using normalized residuals.
- Compare baseline and leave-one-out diagnostic approaches.
- Examine satellite redundancy and geometry-dependent isolation behavior.
- Calibrate detection thresholds against a target false-alarm rate.
- Produce reproducible experiment reports and automated test results.

## Research Questions

1. How does pseudorange measurement noise affect receiver positioning accuracy?
2. How does the number of available satellites influence positioning performance?
3. Can residual-based diagnostics detect large injected measurement faults?
4. Does detecting a fault reliably identify the satellite responsible?
5. How does satellite count affect fault-isolation accuracy?
6. How do constructed satellite geometries influence detection and isolation results?
7. Can threshold calibration help control false alarms while preserving fault detection?

## System Overview

The simulator uses synthetic satellite positions and pseudorange measurements to study receiver behavior under controlled conditions.

```text
Synthetic Satellite Geometry
             |
             v
Pseudorange Measurement Generation
             |
             v
Receiver Position Estimation
             |
             v
Residual-Based Fault Diagnostics
             |
             v
Detection and Isolation Evaluation
             |
             v
Metrics, JSON Reports and Tests
```

The evaluation separates several outcomes that should not be treated as interchangeable:

- **Positioning accuracy:** How close the estimated receiver position is to the known simulated position.
- **Fault detection:** Whether a diagnostic method identifies a fault condition.
- **Fault isolation:** Whether the method identifies the satellite associated with the injected fault.
- **False-alarm rate:** How often a clean scenario is incorrectly flagged.
- **Geometry conditioning:** A numerical indicator of the sensitivity of a positioning solution to measurement and geometry perturbations.

## Experiments

### 1. Positioning Noise

Investigates how different pseudorange noise levels affect receiver position estimates while satellite geometry is held fixed.

Reported metrics include:

- Mean position error
- Position RMSE
- Median position error
- Maximum position error
- Number of successful trials

This experiment establishes a baseline for understanding the relationship between measurement noise and positioning error.

### 2. Satellite Geometry

Compares positioning behavior under different satellite configurations.

The experiment reports geometry-related metrics, including position and timing dilution of precision (PDOP, TDOP), geometric dilution of precision (GDOP), and positioning error.

A key limitation is that favorable or unfavorable behavior in a particular synthetic geometry does not establish how a real satellite constellation will perform.

### 3. Satellite Availability

Evaluates positioning performance as the number of available satellites changes.

The experiment uses synthetic satellite configurations and pseudorange measurements to compare positioning error across satellite-count scenarios.

The results help investigate the relationship between measurement redundancy and positioning performance. They should not be interpreted as a universal performance guarantee for a particular number of satellites.

### 4. Fault Detection and Baseline Isolation

Injects a large pseudorange fault into otherwise synthetic measurements and evaluates residual-based diagnostics.

A baseline evaluation used the following configuration:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Trials per scenario | 200 |
| Pseudorange noise standard deviation | 1 m |
| Injected fault magnitude | 1,000 m |
| Residual threshold | 10 m |

In this experiment, the baseline method recorded:

| Metric | Result |
|---|---:|
| Clean false alarms | 0 |
| Clean false-alarm rate | 0% |
| Fault detection rate | 100% |
| Correct isolation rate across all valid fault trials | 20% |

These figures describe this particular simulation configuration. A 1,000 m injected fault is substantially different from many subtle or realistic GNSS error conditions, so the detection result should not be generalized to smaller faults or operational environments.

### 5. Leave-One-Out Diagnostics

Compares baseline residual diagnostics with a leave-one-out approach for evaluating candidate satellite exclusions.

In the recorded 200-trial-per-scenario evaluation:

| Metric | Baseline | Leave-one-out |
|---|---:|---:|
| Clean false alarms | 0 | 22 |
| Clean false-alarm rate | 0% | 11% |
| Fault detection rate | 100% | 100% |
| Correct isolation rate | 20% | 20% |

The leave-one-out diagnostic detected all injected faults in the tested fault scenario, but it did not improve the measured isolation rate. It also introduced additional false alarms in clean trials.

**Interpretation:** A diagnostic method that identifies suspicious measurements does not necessarily identify the faulty satellite correctly. More elaborate diagnostics must be evaluated for both detection performance and false-alarm behavior.

### 6. Satellite Redundancy and Normalized Residual Isolation

Evaluates normalized-residual isolation as the number of available satellites increases.

The experiment used:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Trials per scenario | 200 |
| Noise standard deviation | 1 m |
| Injected fault magnitude | 1,000 m |
| Satellite counts | 5, 6, 7 and 8 |

Recorded mean isolation accuracy:

| Available satellites | Mean isolation accuracy |
|---:|---:|
| 5 | 20.9% |
| 6 | 100% |
| 7 | 100% |
| 8 | 100% |

In this experiment, the isolation score evaluated whether the satellite with the highest normalized residual corresponded to the injected faulty satellite.

The fault detection outcome was not threshold-based in this experiment. The table therefore describes isolation accuracy, not a calibrated operational detection rate.

The results suggest that additional satellite redundancy can substantially change isolation behavior in the tested configuration. However, the experiment uses a fixed geometry and synthetic measurements, so this pattern requires broader validation.

### 7. Threshold-Calibrated Satellite Redundancy

Evaluates threshold calibration using a target clean false-alarm rate of 1%.

The experiment used:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Noise standard deviation | 1 m |
| Injected fault magnitude | 1,000 m |
| Calibration trials per scenario | 2,000 |
| Clean test trials per scenario | 2,000 |
| Target false-alarm rate | 1% |
| Satellite counts | 5, 6, 7 and 8 |

The detection thresholds were calibrated separately for each satellite-count scenario.

| Satellites | Clean false alarms | Observed false-alarm rate | Mean fault detection rate | Mean isolation accuracy |
|---:|---:|---:|---:|---:|
| 5 | 18 / 2,000 | 0.90% | 100% | 19.84% |
| 6 | 23 / 2,000 | 1.15% | 100% | 100% |
| 7 | 18 / 2,000 | 0.90% | 100% | 100% |
| 8 | 15 / 2,000 | 0.75% | 100% | 100% |

The observed false-alarm rates are close to the 1% calibration target, although the six-satellite scenario exceeds it slightly. Calibration to a target rate does not guarantee that every independent test sample will attain that rate.

The reported 100% detection rates apply to the injected fault and noise conditions evaluated. They do not establish detection performance for smaller faults, different noise distributions, correlated errors, or real GNSS measurements.

### 8. Geometry Robustness of Thresholded Normalized Residuals

Examines thresholded normalized-residual diagnostics under three constructed geometry conditions:

- **Original:** The reference synthetic geometry.
- **Rotated:** A coordinate-rotated version of the reference geometry.
- **Compressed:** A constructed geometry with substantially worse numerical conditioning.

The experiment used 1 m noise, a 1,000 m injected fault, a target false-alarm rate of 1%, and 1,000 calibration trials plus 1,000 clean test trials per scenario.

#### Fault detection

Mean detection rate was 100% across all 12 combinations of geometry condition and satellite count.

#### Clean false-alarm rates

| Geometry | 5 satellites | 6 satellites | 7 satellites | 8 satellites |
|---|---:|---:|---:|---:|
| Original | 0.5% | 0.5% | 1.3% | 0.7% |
| Rotated | 1.2% | 0.3% | 1.7% | 0.6% |
| Compressed | 1.1% | 0.4% | 0.6% | 1.0% |

#### Mean isolation accuracy

| Geometry | 5 satellites | 6 satellites | 7 satellites | 8 satellites |
|---|---:|---:|---:|---:|
| Original | 19.92% | 100% | 100% | 100% |
| Rotated | 20.18% | 100% | 100% | 100% |
| Compressed | 20.46% | 100% | 100% | 100% |

#### Geometry conditioning

The reported condition numbers were:

| Satellites | Original | Rotated | Compressed |
|---:|---:|---:|---:|
| 5 | 2.1507 | 2.1507 | 33.2835 |
| 6 | 2.0718 | 2.0718 | 35.8999 |
| 7 | 2.1231 | 2.1231 | 37.0655 |
| 8 | 2.2466 | 2.2466 | 36.1722 |

The rotated geometry preserves the reference geometry's conditioning, as expected for a coordinate rotation. It is therefore a coordinate-invariance check rather than an independent test of geometry quality.

The compressed geometry produces substantially larger condition numbers, making it a useful constructed challenge case. Nevertheless, the measured detection and isolation outcomes remained unchanged under the particular fault and noise settings used.

These results do not establish that geometry is irrelevant to fault isolation. Only one fixed fault magnitude was tested, thresholds were calibrated separately for each geometry and satellite count, and the geometries were synthetic.

## Key Findings

The current experiments support the following limited conclusions:

1. **Detection and isolation are different problems.** Detecting an injected fault does not guarantee that the correct satellite is identified.
2. **Satellite count mattered in the tested scenarios.** Five-satellite configurations produced approximately 20% isolation accuracy, while six-, seven-, and eight-satellite configurations produced 100% isolation accuracy in the redundancy and geometry-robustness experiments.
3. **Threshold calibration helps quantify false alarms.** Observed false-alarm rates were generally close to the 1% target, but individual scenarios varied.
4. **More complex diagnostics can introduce trade-offs.** The recorded leave-one-out experiment had an 11% clean false-alarm rate without improving isolation accuracy.
5. **Synthetic results need broader validation.** Large injected faults and fixed geometries cannot establish performance under realistic GNSS operating conditions.

These are findings from controlled simulation experiments, not claims of validated real-world receiver performance.

## Technology Stack

The project uses Python and a reproducible experiment-and-test workflow.

| Component | Purpose |
|---|---|
| Python | Simulation and evaluation logic |
| NumPy | Numerical operations and array handling |
| SciPy | Scientific computation |
| Matplotlib | Experiment visualization |
| pytest | Automated testing |
| GitHub Actions | Continuous integration and experiment execution |
| JSON | Machine-readable experiment reports |

Dependency versions are pinned in `requirements.txt`. The exact available scripts and module names are defined by the repository itself.

## Repository Structure

The repository contains simulation code, experiment runners, automated tests, and machine-readable reports or report-generation workflows.

Expected project artifacts include:

- Positioning-noise experiment results
- Satellite-geometry comparison results
- Satellite-availability experiment results
- Fault-detection evaluation
- Leave-one-out diagnostic evaluation
- Normalized-residual isolation results
- Threshold-calibrated redundancy results
- Geometry-robustness results

Use the repository's actual directory tree and GitHub Actions artifacts as the source of truth for individual file paths and generated outputs.

## Getting Started

### Prerequisites

- Python version compatible with the pinned dependencies
- Git
- A terminal or command-line environment

### 1. Clone the repository

```bash
git clone https://github.com/aditirkatare/gnss-receiver-resilience.git
cd gnss-receiver-resilience
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the automated tests

```bash
python -m pytest
```

### 5. Run experiments

Use the experiment entry points and commands defined by the repository. Check the project files and workflow configuration before running an individual experiment, since command names and output paths depend on the implemented scripts.

Generated JSON reports can be inspected to review trial counts, parameters, positioning metrics, false alarms, detection rates, and isolation accuracy.

## Reproducibility

The experiments use explicit random seeds, including seed `42` in several evaluations and seed `2026` in the geometry-robustness experiment.

Reproducibility depends on more than a random seed. It also requires retaining the relevant code version, dependency versions, experiment parameters, and output artifacts.

For each experiment, record:

- Random seed
- Satellite geometry and satellite count
- Measurement-noise model and parameters
- Injected fault magnitude and location
- Number of calibration and evaluation trials
- Detection threshold or calibration procedure
- Metric definitions
- Successful, failed, and excluded trials
- Generated output files

Threshold-calibration experiments should keep calibration data separate from clean evaluation data. Results should report the actual observed false-alarm rate rather than assuming it equals the target rate.

## Limitations

The following limitations apply to the reported experiments:

- Measurements and satellite configurations are synthetic.
- Satellite geometries are fixed within individual scenarios.
- The principal fault-isolation experiments use a large, 1,000 m injected fault.
- Some experiments use relatively small trial counts.
- Thresholds are calibrated separately for specific scenario configurations.
- The noise model does not establish robustness to all real measurement errors.
- The compressed geometry is a constructed challenge, not a validated model of a particular satellite constellation.
- The coordinate-rotation experiment is not an independent geometry-quality test.
- The current findings do not demonstrate performance with live GNSS signals or physical receiver hardware.

Consequently, the reported rates should be interpreted as experimental observations under defined assumptions, not operational guarantees.

## Future Work

Potential next experiments include:

- Evaluate a range of injected fault magnitudes, such as 50 m, 100 m, 250 m, 500 m, and 1,000 m.
- Increase trial counts and report uncertainty intervals for detection, false-alarm, and isolation metrics.
- Evaluate multiple independent satellite geometries rather than relying on one fixed geometry.
- Investigate different fault locations and simultaneous faults.
- Test alternative noise distributions and measurement-error models.
- Evaluate whether geometry-aware isolation scoring improves results without increasing false alarms.
- Separate threshold calibration, parameter selection, and final evaluation more rigorously.
- Where suitable datasets or receiver measurements are available, compare simulated behavior against real observations.

These are proposed extensions, not completed features.

## Continuous Integration

The repository uses GitHub Actions to run automated project checks and experiments.

[View workflow runs](https://github.com/aditirkatare/gnss-receiver-resilience/actions)

Successful workflow runs provide evidence that the configured checks completed for a particular commit. They do not independently validate the scientific assumptions or establish real-world GNSS performance.

Generated workflow artifacts may be available for a limited retention period. Download and archive important reports if they need to be preserved.

## Author

**Aditi Rajendra Katare**

GitHub: [@aditirkatare](https://github.com/aditirkatare)

## Project Status

**Status:** Ongoing research and engineering project.

The current implementation focuses on reproducible synthetic experiments for GNSS positioning, fault detection, and satellite fault isolation. Further statistical evaluation and validation are needed before drawing conclusions about operational receiver performance.

## License

No license is specified here. Check the repository for an existing `LICENSE` file before redistributing or reusing the project.
