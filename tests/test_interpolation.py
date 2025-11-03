import numpy as np
import pandas as pd
import pytest

from panel_imputer import PanelImputer

from helpers import df_from_values


def test_interpolate_tail_fill():
    """Ensure interpolate fills tails with boundary values when tail_behavior='fill'."""
    locs, times = ["A"], [0, 1, 2, 3, 4]
    values = {"A": [np.nan, 1.0, np.nan, 3.0, np.nan]}
    df = df_from_values(locs, times, values)

    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="interpolate",
        interp_method="linear",
        tail_behavior="fill",
    )
    out = imp.fit_transform(df)
    expected = [1.0, 1.0, 2.0, 3.0, 3.0]
    assert np.array_equal(out["v"].values, expected, equal_nan=True)


def test_interpolate_tail_none():
    """Ensure interpolate leaves tails as NaN when tail_behavior='None'."""
    locs, times = ["A"], [0, 1, 2, 3, 4]
    values = {"A": [np.nan, 1.0, np.nan, 3.0, np.nan]}
    df = df_from_values(locs, times, values)

    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="interpolate",
        interp_method="linear",
        tail_behavior="None",
    )
    out = imp.fit_transform(df)
    expected = [np.nan, 1.0, 2.0, 3.0, np.nan]
    assert np.array_equal(out["v"].values, expected, equal_nan=True)


def test_interpolate_tail_extrapolate():
    """Ensure interpolate extrapolates tails when tail_behavior='extrapolate'."""
    locs, times = ["A"], [0, 1, 2, 3, 4]
    values = {"A": [np.nan, 1.0, np.nan, 3.0, np.nan]}
    df = df_from_values(locs, times, values)

    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="interpolate",
        interp_method="linear",
        tail_behavior="extrapolate",
    )
    out = imp.fit_transform(df)
    expected = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert np.array_equal(out["v"].values, expected, equal_nan=True)


def test_interpolate_different_tails():
    """Ensure per-tail behavior list works when mixing 'None' and 'fill' options."""
    locs, times = ["A"], [0, 1, 2, 3, 4]
    values = {"A": [np.nan, 1.0, np.nan, 3.0, np.nan]}
    df = df_from_values(locs, times, values)

    imp1 = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="interpolate",
        interp_method="linear",
        tail_behavior=["None", "fill"],
    )
    out1 = imp1.fit_transform(df)
    expected1 = [np.nan, 1.0, 2.0, 3.0, 3.0]
    
    imp2 = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="interpolate",
        interp_method="linear",
        tail_behavior=["fill", "None"],
    )
    out2 = imp2.fit_transform(df)
    expected2 = [1.0, 1.0, 2.0, 3.0, np.nan]

    assert np.array_equal(out1["v"].values, expected1, equal_nan=True)
    assert np.array_equal(out2["v"].values, expected2, equal_nan=True)



