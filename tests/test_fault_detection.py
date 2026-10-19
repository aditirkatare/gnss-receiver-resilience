
import numpy as np
import pytest

from gnss_resilience.fault_detection import detect_pseudorange_fault
from gnss_resilience.measurements import calculate_pseudoranges


@pytest.fixture
def gnss_measurements():
    receiver = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])

    satellites = np.array([
        [20_000_000.0, 0.0, 0.0],
        [0.0, 21_000_000.0, 0.0],
        [0.0, 0.0, 22_000_000.0],
        [-15_000_000.0, -16_000_000.0, -17_000_000.0],
        [18_000_000.0, -19_000_000.0, 12_000_000.0],
    ])

    ranges = calculate_pseudoranges(
        receiver,
        satellites,
        receiver_clock_bias_m=125.0,
    )

    return receiver, satellites, ranges


def test_clean_measurements_do_not_trigger_fault(gnss_measurements):
    receiver, satellites, ranges = gnss_measurements

    result = detect_pseudorange_fault(
        satellites,
        ranges,
        initial_position_m=receiver + np.array([100.0, -80.0, 50.0]),
        residual_threshold_m=10.0,
        initial_clock_bias_m=125.0,
    )

    assert result["detected"] is False
    assert result["suspected_satellite_index"] is None
    assert result["max_abs_residual_m"] < 10.0


def test_large_pseudorange_error_triggers_fault(gnss_measurements):
    receiver, satellites, ranges = gnss_measurements
    corrupted_ranges = ranges.copy()

    corrupted_index = 0
    corrupted_ranges[corrupted_index] += 1_000.0

    result = detect_pseudorange_fault(
        satellites,
        corrupted_ranges,
        initial_position_m=receiver + np.array([100.0, -80.0, 50.0]),
        residual_threshold_m=10.0,
        initial_clock_bias_m=125.0,
    )

    assert result["detected"] is True
    assert result["max_abs_residual_m"] > 10.0
    assert result["suspected_satellite_index"] is not None


def test_rejects_invalid_residual_threshold(gnss_measurements):
    receiver, satellites, ranges = gnss_measurements

    with pytest.raises(ValueError, match="Residual threshold"):
        detect_pseudorange_fault(
            satellites,
            ranges,
            initial_position_m=receiver,
            residual_threshold_m=0.0,
        )
