
import numpy as np
import pytest

from gnss_resilience.measurements import calculate_pseudoranges
from gnss_resilience.position import estimate_position


@pytest.fixture
def simulated_gnss():
    receiver = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])

    satellites = np.array([
        [20_000_000.0, 0.0, 0.0],
        [0.0, 21_000_000.0, 0.0],
        [0.0, 0.0, 22_000_000.0],
        [-15_000_000.0, -16_000_000.0, -17_000_000.0],
        [18_000_000.0, -19_000_000.0, 12_000_000.0],
    ])

    true_clock_bias = 125.0

    pseudoranges = calculate_pseudoranges(
        receiver,
        satellites,
        receiver_clock_bias_m=true_clock_bias,
    )

    return receiver, satellites, pseudoranges, true_clock_bias


def test_recovers_known_receiver_position(simulated_gnss):
    receiver, satellites, ranges, clock_bias = simulated_gnss

    estimated_position, estimated_bias = estimate_position(
        satellites,
        ranges,
        initial_position_m=receiver + np.array([1000.0, -800.0, 500.0]),
    )

    np.testing.assert_allclose(
        estimated_position, receiver, atol=1e-3
    )
    assert estimated_bias == pytest.approx(clock_bias, abs=1e-3)


def test_recovers_position_from_different_initial_guess(simulated_gnss):
    receiver, satellites, ranges, clock_bias = simulated_gnss

    estimated_position, estimated_bias = estimate_position(
        satellites,
        ranges,
        initial_position_m=receiver + np.array([-500.0, 300.0, 700.0]),
        initial_clock_bias_m=50.0,
    )

    np.testing.assert_allclose(
        estimated_position, receiver, atol=1e-3
    )
    assert estimated_bias == pytest.approx(clock_bias, abs=1e-3)


def test_requires_at_least_four_satellites():
    satellites = np.array([
        [20_000_000.0, 0.0, 0.0],
        [0.0, 21_000_000.0, 0.0],
        [0.0, 0.0, 22_000_000.0],
    ])

    with pytest.raises(ValueError, match="At least four satellites"):
        estimate_position(
            satellites,
            np.array([20e6, 21e6, 22e6]),
            initial_position_m=np.zeros(3),
        )


def test_rejects_wrong_pseudorange_count(simulated_gnss):
    _, satellites, ranges, _ = simulated_gnss

    with pytest.raises(ValueError, match="Pseudoranges must have shape"):
        estimate_position(
            satellites,
            ranges[:-1],
            initial_position_m=np.zeros(3),
        )


def test_rejects_non_finite_pseudoranges(simulated_gnss):
    _, satellites, ranges, _ = simulated_gnss
    invalid_ranges = ranges.copy()
    invalid_ranges[0] = np.nan

    with pytest.raises(ValueError, match="finite and positive"):
        estimate_position(
            satellites,
            invalid_ranges,
            initial_position_m=np.zeros(3),
        )
def test_position_estimation_rejects_fewer_than_four_satellites():
    import numpy as np
    import pytest

    from gnss_resilience.position import estimate_position

    satellite_positions_m = np.array([
        [20_000_000.0, 0.0, 0.0],
        [0.0, 20_000_000.0, 0.0],
        [0.0, 0.0, 20_000_000.0],
    ])
    pseudoranges_m = np.array([20_000_000.0] * 3)
    initial_position_m = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])

    with pytest.raises(ValueError, match="At least four satellites"):
        estimate_position(
            satellite_positions_m,
            pseudoranges_m,
            initial_position_m,
        )


def test_position_estimation_rejects_exactly_four_invalid_shape_ranges():
    import numpy as np
    import pytest

    from gnss_resilience.position import estimate_position

    satellite_positions_m = np.array([
        [20_000_000.0, 0.0, 0.0],
        [0.0, 20_000_000.0, 0.0],
        [0.0, 0.0, 20_000_000.0],
        [-20_000_000.0, 0.0, 0.0],
    ])

    with pytest.raises(ValueError, match="Pseudoranges must have shape"):
        estimate_position(
            satellite_positions_m,
            np.array([20_000_000.0] * 3),
            np.array([1_000_000.0, 2_000_000.0, 3_000_000.0]),
        )
