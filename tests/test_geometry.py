
import numpy as np
import pytest

from gnss_resilience.geometry import calculate_dop


@pytest.fixture
def receiver_and_satellites():
    receiver = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])
    satellites = np.array([
        [20_000_000.0, 0.0, 0.0],
        [0.0, 21_000_000.0, 0.0],
        [0.0, 0.0, 22_000_000.0],
        [-15_000_000.0, -16_000_000.0, -17_000_000.0],
        [18_000_000.0, -19_000_000.0, 12_000_000.0],
    ])
    return receiver, satellites


def test_dop_values_are_positive_and_finite(receiver_and_satellites):
    receiver, satellites = receiver_and_satellites

    result = calculate_dop(receiver, satellites)

    assert set(result) == {"PDOP", "TDOP", "GDOP"}
    assert all(np.isfinite(value) and value > 0 for value in result.values())


def test_gdop_is_consistent_with_pdop_and_tdop(receiver_and_satellites):
    receiver, satellites = receiver_and_satellites

    result = calculate_dop(receiver, satellites)

    assert result["GDOP"] ** 2 == pytest.approx(
        result["PDOP"] ** 2 + result["TDOP"] ** 2
    )


def test_requires_four_satellites():
    with pytest.raises(ValueError, match="At least four satellites"):
        calculate_dop(
            np.zeros(3),
            np.array([
                [20_000_000.0, 0.0, 0.0],
                [0.0, 21_000_000.0, 0.0],
                [0.0, 0.0, 22_000_000.0],
            ]),
        )


def test_rejects_non_finite_coordinates():
    with pytest.raises(ValueError, match="finite"):
        calculate_dop(
            np.array([np.nan, 0.0, 0.0]),
            np.array([
                [20_000_000.0, 0.0, 0.0],
                [0.0, 21_000_000.0, 0.0],
                [0.0, 0.0, 22_000_000.0],
                [-15_000_000.0, -16_000_000.0, -17_000_000.0],
            ]),
        )
