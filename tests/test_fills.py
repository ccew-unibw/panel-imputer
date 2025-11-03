import numpy as np
import pandas as pd

from panel_imputer import PanelImputer

from helpers import df_from_values


def test_bfill_imputation_basic():
    """Backward-fill each location and confirm NaNs resolve to later observations."""
    locs, times = ["A", "B"], [0, 1, 2, 3]
    values = {
        "A": [1.0, np.nan, np.nan, 4.0],
        "B": [np.nan, 2.0, np.nan, np.nan],
    }
    df = df_from_values(locs, times, values)

    imp = PanelImputer(location_index="loc", time_index="time", imputation_method="bfill")
    out = imp.fit_transform(df)

    expected_A = [1.0, 4.0, 4.0, 4.0]
    expected_B = [2.0, 2.0, np.nan, np.nan]

    assert np.array_equal(out.xs("A", level="loc")["v"].values, expected_A, equal_nan=True)
    assert np.array_equal(out.xs("B", level="loc")["v"].values, expected_B, equal_nan=True)


def test_ffill_imputation_basic():
    """Forward-fill each location and confirm NaNs inherit earlier observations."""
    locs, times = ["A", "B"], [0, 1, 2, 3]
    values = {
        "A": [1.0, np.nan, np.nan, 4.0],
        "B": [np.nan, 2.0, np.nan, np.nan],
    }
    df = df_from_values(locs, times, values)

    imp = PanelImputer(location_index="loc", time_index="time", imputation_method="ffill")
    out = imp.fit_transform(df)

    expected_A = [1.0, 1.0, 1.0, 4.0]
    expected_B = [np.nan, 2.0, 2.0, 2.0]

    assert np.array_equal(out.xs("A", level="loc")["v"].values, expected_A, equal_nan=True)
    assert np.array_equal(out.xs("B", level="loc")["v"].values, expected_B, equal_nan=True)


def test_fill_all_imputation_basic():
    """Combine forward/back fill and verify every gap closes where data exists."""
    locs, times = ["A", "B"], [0, 1, 2, 3]
    values = {
        "A": [1.0, np.nan, np.nan, 4.0],
        "B": [np.nan, 2.0, np.nan, np.nan],
    }
    df = df_from_values(locs, times, values)

    imp = PanelImputer(location_index="loc", time_index="time", imputation_method="fill_all")
    out = imp.fit_transform(df)

    expected_A = [1.0, 4.0, 4.0, 4.0]
    expected_B = [2.0, 2.0, 2.0, 2.0]

    assert np.array_equal(out.xs("A", level="loc")["v"].values, expected_A, equal_nan=True)
    assert np.array_equal(out.xs("B", level="loc")["v"].values, expected_B, equal_nan=True)

