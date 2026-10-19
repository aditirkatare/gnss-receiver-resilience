
import pytest

from gnss_resilience.simulation import run_noise_experiment


@pytest.fixture
def simulation_inputs():
    receiver = [1_000_000.0, 2_000_000.0, 3_000_000.0]

    satellites = [
        [20_000_000.0, 0.0, 0.0],
        [0.0, 21_000_000.0, 0.0],
        [0.0, 0.0, 22_000_000.0],
        [-15_000_000.0, -16_000_000.0, -17_000_000.0],
        [18_000_000.0, -19_000_000.0, 12_000_000.0],
    ]

    return receiver, satellites


def test_zero_noise_produces_negligible_position_error(simulation_inputs):
    receiver, satellites = simulation_inputs

    result = run_noise_experiment(
        receiver, satellites, noise_std_m=0.0, trials=5, seed=7
    )

    assert result["trials_succeeded"] == 5
    assert result["trials_failed"] == 0
    assert result["rmse_m"] < 1e-3


def test_simulation_is_reproducible(simulation_inputs):
    receiver, satellites = simulation_inputs

    first = run_noise_experiment(
        receiver, satellites, noise_std_m=2.0, trials=10, seed=123
    )
    second = run_noise_experiment(
        receiver, satellites, noise_std_m=2.0, trials=10, seed=123
    )

    assert first == second


def test_error_statistics_are_non_negative(simulation_inputs):
    receiver, satellites = simulation_inputs

    result = run_noise_experiment(
        receiver, satellites, noise_std_m=1.0, trials=10, seed=42
    )

    for key in ("mean_error_m", "rmse_m", "median_error_m", "max_error_m"):
        assert result[key] >= 0

    assert result["trials_succeeded"] + result["trials_failed"] == 10


def test_invalid_noise_level_is_rejected(simulation_inputs):
    receiver, satellites = simulation_inputs

    with pytest.raises(ValueError, match="Noise standard deviation"):
        run_noise_experiment(receiver, satellites, noise_std_m=-1.0)


def test_invalid_trial_count_is_rejected(simulation_inputs):
    receiver, satellites = simulation_inputs

    with pytest.raises(ValueError, match="trials must"):
        run_noise_experiment(receiver, satellites, noise_std_m=1.0, trials=0)
