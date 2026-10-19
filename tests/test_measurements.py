
import numpy as np
import pytest

from gnss_resilience.measurements import calculate_pseudoranges


def test_exact_geometric_ranges():
    receiver = np.array([0.0, 0.0, 0.0])
    satellites = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 12.0],
    ])

    result = calculate_pseudoranges(receiver, satellites)

    np.testing.assert_allclose(result, [3.0, 4.0, 12.0])


def test_clock_bias_is_added_to_each_measurement():
    receiver = np.zeros(3)
    satellites = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
    ])

    result = calculate_pseudoranges(
        receiver, satellites, receiver_clock_bias_m=5.0
    )

    np.testing.assert_allclose(result, [15.0, 25.0])


def test_measurement_noise_is_applied():
    receiver = np.zeros(3)
    satellites = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
    ])

    result = calculate_pseudoranges(
        receiver, satellites, noise_m=np.array([1.0, -2.0])
    )

    np.testing.assert_allclose(result, [11.0, 18.0])


def test_scalar_noise_broadcasts_to_all_satellites():
    receiver = np.zeros(3)
    satellites = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
    ])

    result = calculate_pseudoranges(
        receiver, satellites, noise_m=2.0
    )

    np.testing.assert_allclose(result, [12.0, 22.0])


def test_invalid_receiver_shape_is_rejected():
    with pytest.raises(ValueError, match="receiver_position_m"):
        calculate_pseudoranges(
            np.array([1.0, 2.0]),
            np.array([[10.0, 0.0, 0.0]]),
        )


def test_empty_satellite_array_is_rejected():
    with pytest.raises(ValueError, match="At least one satellite"):
        calculate_pseudoranges(
            np.zeros(3), np.empty((0, 3))
        )


def test_non_finite_coordinates_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        calculate_pseudoranges(
            np.array([np.nan, 0.0, 0.0]),
            np.array([[10.0, 0.0, 0.0]]),
        )


def test_noise_length_must_match_satellite_count():
    with pytest.raises(ValueError, match="Noise must be"):
        calculate_pseudoranges(
            np.zeros(3),
            np.array([
                [10.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
            ]),
            noise_m=np.array([1.0, 2.0, 3.0]),
        )
