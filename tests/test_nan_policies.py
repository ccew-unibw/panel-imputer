import numpy as np
import pandas as pd
import pytest

import panel_imputer as pim_module
from panel_imputer import PanelImputer

from helpers import df_from_values



def test_all_nan_policy_drop_column():
    """Ensure drop policy removes all-NaN columns and leaves others intact."""
    locs, times = ["A", "B"], [0, 1]
    idx = pd.MultiIndex.from_product([locs, times], names=["loc", "time"])
    df = pd.DataFrame(
        index=idx,
        data={
            "drop_me": [np.nan, np.nan, np.nan, np.nan],
            "v": [np.nan, np.nan, 1.0, 2.0],
            "w": [10, 20, 30, 40],
        },
    )

    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="ffill",
        all_nan_policy="drop",
    )
    out = imp.fit_transform(df)

    assert "drop_me" not in out.columns
    pd.testing.assert_series_equal(out["w"], df["w"])
    assert "v" in out.columns


def test_nan_loc_policy_mean():
    """Ensure nan_loc_policy='mean' fills entire missing location from cross-sectional means."""
    locs, times = ["A", "B", "C", "Target"], [0, 1, 2]
    idx = pd.MultiIndex.from_product([locs, times], names=["loc", "time"])
    df = pd.DataFrame(
        index=idx,
        data={
            "v": [
                1.0, 2.0, 3.0,        # A
                1.0, 2.0, 3.0,        # B
                10.0, 10.0, 10.0,     # C
                np.nan, np.nan, np.nan,  # Target (all NaN)
            ]
        },
    )

    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        nan_loc_policy="mean",
    )
    out = imp.fit_transform(df)

    expected_target = [4.0, 14.0 / 3.0, 16.0 / 3.0]
    assert np.allclose(out.xs("Target", level="loc")["v"].values, expected_target)


def test_nan_loc_policy_median():
    """Ensure nan_loc_policy='median' fills missing location using cross-sectional medians."""
    locs, times = ["A", "B", "C", "Target"], [0, 1, 2]
    idx = pd.MultiIndex.from_product([locs, times], names=["loc", "time"])
    df = pd.DataFrame(
        index=idx,
        data={
            "v": [
                1.0, 2.0, 3.0,        # A
                1.0, 2.0, 3.0,        # B
                10.0, 10.0, 10.0,     # C
                np.nan, np.nan, np.nan,  # Target (all NaN)
            ]
        },
    )

    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        nan_loc_policy="median",
    )
    out = imp.fit_transform(df)

    expected_target = [1.0, 2.0, 3.0]
    assert np.array_equal(
        out.xs("Target", level="loc")["v"].values, expected_target, equal_nan=True
    )


def test_nan_loc_policy_knnimpute():
    """Ensure KNN imputation clones nearest neighbor series when distances are identical."""
    locs, times = ["A", "B"], [0, 1, 2]
    idx = pd.MultiIndex.from_product([locs, times], names=["loc", "time"])
    v_A = [1.0, 2.0, 3.0]
    w_A = [10.0, 20.0, 30.0]
    w_B = [10.0, 20.0, 30.0]
    v_B = [np.nan, np.nan, np.nan]
    df = pd.DataFrame(index=idx, data={
        "v": v_A + v_B,
        "w": w_A + w_B,
    })

    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="ffill",
        nan_loc_policy="knnimpute",
        knn_kwargs={"n_neighbors": 1, "weights": "uniform"},
    )
    out = imp.fit_transform(df)

    assert np.array_equal(out.xs("B", level="loc")["v"].values, v_A, equal_nan=True)


def test_nan_loc_policy_respects_all_nan_times():
    """Ensure nan_loc_policy leaves globally all-NaN timestamps untouched (example uses mean/ffill)."""
    locs, times = ["A", "B"], [0, 1, 2]
    idx = pd.MultiIndex.from_product([locs, times], names=["loc", "time"])
    df = pd.DataFrame(index=idx, data={
        "v": [np.nan, 1.0, np.nan, np.nan, np.nan, np.nan]
    })

    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="ffill",
        nan_loc_policy="mean",
    )
    out = imp.fit_transform(df)

    b_vals = out.xs("B", level="loc")["v"].values
    assert np.isnan(b_vals[0])


def test_nan_loc_policy_respects_tail_behavior():
    """Ensure nan_loc_policy does not fill tails in ffill/bfill or in conflict with tail_behavior."""
    idx = pd.MultiIndex.from_product([["A", "B", "Target"], [0, 1, 2, 3]], names=["loc", "time"])
    base = [np.nan, 1.0, 2.0, np.nan]
    df = pd.DataFrame(index=idx, data={"v": base + base + [np.nan] * 4})

    # ffill leaves the leading all-NA time NaN; only times 1-3 are filled.
    out = PanelImputer(
        location_index="loc", time_index="time",
        imputation_method="ffill", nan_loc_policy="mean",
    ).fit_transform(df)
    assert np.array_equal(
        out.xs("Target", level="loc")["v"].values,
        [np.nan, 1.0, 2.0, 2.0], equal_nan=True,
    )

    # bfill leaves the trailing all-NA time NaN; only times 0-2 are filled.
    out = PanelImputer(
        location_index="loc", time_index="time",
        imputation_method="bfill", nan_loc_policy="mean",
    ).fit_transform(df)
    assert np.array_equal(
        out.xs("Target", level="loc")["v"].values,
        [1.0, 1.0, 2.0, np.nan], equal_nan=True,
    )

    # fill_all fills both tails, so no all-NA time survives and every time is filled.
    out = PanelImputer(
        location_index="loc", time_index="time",
        imputation_method="fill_all", nan_loc_policy="mean",
    ).fit_transform(df)
    assert np.array_equal(
        out.xs("Target", level="loc")["v"].values,
        [1.0, 1.0, 2.0, 2.0], equal_nan=True,
    )

    # interpolate with tail_behavior="None" leaves both tail all-NA times NaN; only times 1-2 are filled.
    out = PanelImputer(
        location_index="loc", time_index="time",
        imputation_method="interpolate", interp_method="linear",
        tail_behavior="None", nan_loc_policy="mean",
    ).fit_transform(df)
    assert np.array_equal(
        out.xs("Target", level="loc")["v"].values,
        [np.nan, 1.0, 2.0, np.nan], equal_nan=True,
    )

    # interpolate with tail_behavior="fill" fills both tails, so every time is filled.
    out = PanelImputer(
        location_index="loc", time_index="time",
        imputation_method="interpolate", interp_method="linear",
        tail_behavior="fill", nan_loc_policy="mean",
    ).fit_transform(df)
    assert np.array_equal(
        out.xs("Target", level="loc")["v"].values,
        [1.0, 1.0, 2.0, 2.0], equal_nan=True,
    )

    # List-form tail_behavior=["None", "fill"] drops only the "None" side (time 0); the
    # "fill" side keeps time 3 filled. Guards the "in" membership check that supports both
    # string and list tail_behavior.
    out = PanelImputer(
        location_index="loc", time_index="time",
        imputation_method="interpolate", interp_method="linear",
        tail_behavior=["None", "fill"], nan_loc_policy="mean",
    ).fit_transform(df)
    assert np.array_equal(
        out.xs("Target", level="loc")["v"].values,
        [np.nan, 1.0, 2.0, 2.0], equal_nan=True,
    )

