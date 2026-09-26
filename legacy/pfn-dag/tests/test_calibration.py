import numpy as np

from pfn_dag_verify.calibration import build_calibration_panel


def test_calibration_panel_is_deterministic_and_length_30():
    first = build_calibration_panel(3)
    second = build_calibration_panel(3)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 30, 2)
