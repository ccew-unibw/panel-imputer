import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError

from panel_imputer import PanelImputer

from helpers import df_from_values

# init checks

def test_init_rejects_unknown_imputation_method():
    """Ensure constructor rejects unsupported imputation strategies."""
    with pytest.raises(AssertionError):
        PanelImputer(
            location_index="loc",
            time_index="time",
            imputation_method="unknown",
        )


def test_interpolate_requires_interp_method():
    """Ensure interpolate mode requires specifying interp_method."""
    with pytest.raises(AssertionError):
        PanelImputer(
            location_index="loc",
            time_index="time",
            imputation_method="interpolate",
        )


def test_tail_behavior_invalid_string_rejected():
    """Ensure invalid tail behavior choices raise during init."""
    with pytest.raises(AssertionError):
        PanelImputer(
            location_index="loc",
            time_index="time",
            imputation_method="ffill",
            tail_behavior="invalid",
        )


def test_tail_behavior_sequence_requires_two_entries():
    """Ensure tail behavior list must contain two valid options."""
    with pytest.raises(AssertionError):
        PanelImputer(
            location_index="loc",
            time_index="time",
            imputation_method="ffill",
            tail_behavior=("fill",),
        )


def test_invalid_nan_loc_policy_rejected():
    """Ensure unsupported nan_loc_policy values fail fast."""
    with pytest.raises(AssertionError):
        PanelImputer(
            location_index="loc",
            time_index="time",
            nan_loc_policy="meanish",
        )


def test_invalid_all_nan_policy_rejected():
    """Ensure unsupported all_nan_policy values fail fast."""
    with pytest.raises(AssertionError):
        PanelImputer(
            location_index="loc",
            time_index="time",
            all_nan_policy="keep",
        )


def test_parallel_kwargs_verbose_defaults_to_verbose():
    """Ensure verbose level populates the parallel kwargs default."""
    imp = PanelImputer(location_index="loc", time_index="time", parallelize=True)
    assert imp.parallel_kwargs["verbose"] == 0
    assert imp.parallel_kwargs["n_jobs"] == -2

    imp2 = PanelImputer(
        location_index="loc",
        time_index="time",
        parallelize=True,
        verbose=3,
        parallel_kwargs={"n_jobs": -1},
    )
    assert imp2.parallel_kwargs["verbose"] == 3
    assert imp2.parallel_kwargs["n_jobs"] == -1

    imp3 = PanelImputer(
        location_index="loc",
        time_index="time",
        parallelize=True,
        verbose=1,
        parallel_kwargs={"n_jobs": -1, "verbose": 10},
    )
    assert imp3.parallel_kwargs["verbose"] == 10
    assert imp3.parallel_kwargs["n_jobs"] == -1
    
    imp4 = PanelImputer(
        location_index="loc",
        time_index="time",
        parallelize=True,
        parallel_kwargs={"backend": "loky"},
    )
    assert imp4.parallel_kwargs["backend"] == "loky"
    assert imp4.parallel_kwargs["verbose"] == 0
    assert imp4.parallel_kwargs["n_jobs"] == -2


# checks (mostly validate_input_)

def test_interpolate_raises_on_single_non_nan_column():
    """Ensure interpolate fit fails when a column has only one non-NaN value."""
    locs, times = ["A"], [0, 1, 2, 3, 4]
    values = {"A": [np.nan, 1.0, np.nan, np.nan, np.nan]}
    df = df_from_values(locs, times, values)
    
    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="interpolate",
        interp_method="linear",
        tail_behavior="fill",
    )
    with pytest.raises(ValueError):
        imp.fit(df)
        
        
def test_all_nan_policy_error():
    """Ensure all_nan_policy='error' raises when any location is entirely missing."""
    locs, times = ["A", "B"], [0, 1]
    idx = pd.MultiIndex.from_product([locs, times], names=["loc", "time"])
    df = pd.DataFrame(
        index=idx,
        data={
            "all_nan": [np.nan, np.nan, np.nan, np.nan],
            "valid": [1.0, 2.0, 3.0, 4.0],
        },
    )
    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        all_nan_policy="error",
    )
    with pytest.raises(ValueError):
        imp.fit(df)


def test_parallel_n_jobs_excess_warns_and_adjusts():
    """Ensure excessive parallel n_jobs triggers a warning and adjustment."""
    locs, times = ["A", "B", "C"], [0, 1]
    df = df_from_values(locs, times, {
        "A": [1.0, np.nan],
        "B": [np.nan, 2.0],
        "C": [3.0, np.nan],
    })

    imp = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="fill_all",
        parallelize=True,
        parallel_kwargs={"n_jobs": 64},
    )
    with pytest.warns(UserWarning, match="reducing n_jobs"):
        imp.fit(df)
    assert imp.parallel_kwargs["n_jobs"] == len(locs)


def test_transform_requires_fit():
    """Ensure transform raises NotFittedError when called before fit."""
    locs, times = ["A"], [0, 1]
    df = df_from_values(locs, times, {"A": [1, np.nan]})
    imp = PanelImputer(location_index="loc", time_index="time")
    with pytest.raises(NotFittedError):
        imp.transform(df)


# basic functionality

def test_no_missing_values_returns_input_unchanged():
    """Ensure transform is a no-op when the input panel contains no gaps."""
    locs, times = ["A", "B"], [0, 1, 2]
    df = df_from_values(locs, times, {"A": [1.0, 1.0, 1.0], "B": [2.0, 2.0, 2.0]})
    imp = PanelImputer(location_index="loc", time_index="time", imputation_method="bfill")
    out = imp.fit_transform(df)
    pd.testing.assert_frame_equal(df, out)


def test_index_order_preserved_after_transform():
    """Ensure transform restores the original order of MultiIndex levels."""
    locs, times = ["A", "B"], [0, 1, 2]
    values = {"A": [1.0, np.nan, 3.0], "B": [np.nan, 2.0, np.nan]}
    df = df_from_values(locs, times, values)
    df = df.reorder_levels(["time", "loc"])

    imp = PanelImputer(location_index="loc", time_index="time", imputation_method="fill_all")
    out = imp.fit_transform(df)
    assert out.index.names == df.index.names


def test_index_stability():
    """Ensure multi-level time indexes fill correctly, with unsorted index and 
    index levels, and the orginal index is preserved in the output."""
    locs = ["A", "B"]
    months = [1, 2, 3]
    years = [2021, 2022]
    idx = pd.MultiIndex.from_product([locs, months, years], names=["loc", "month", "year"])
    df = pd.DataFrame(index=idx, data={
        "v": [1.0, 5.0, np.nan, np.nan, np.nan, np.nan] * 2
    })
    df = df.sample(frac=1.0)

    imp = PanelImputer(location_index="loc", time_index=["year", "month"], imputation_method="ffill")
    out = imp.fit_transform(df)

    assert out.index.equals(df.index)
    a_2021 = out.xs(("A", 2021), level=("loc", "year"))["v"].values
    assert np.array_equal(a_2021, [1.0, 1.0, 1.0], equal_nan=True)


def test_missing_values_parameter_replacement():
    """Ensure configured sentinel values are converted to NaN before imputation."""
    locs, times = ["A"], [0, 1, 2]
    df = df_from_values(locs, times, {"A": [0, 1.0, 0]})

    imp = PanelImputer(location_index="loc", time_index="time", imputation_method="fill_all", missing_values=0)
    out = imp.fit_transform(df)
    assert np.array_equal(out["v"].values, [1.0, 1.0, 1.0], equal_nan=True)
        

def test_parallel_equivalence_to_sequential():
    """Ensure parallelized execution matches sequential imputation results."""
    locs = ["A", "B", "C", "D"]
    times = list(range(6))
    values = {}
    for i, loc in enumerate(locs):
        base = i + 1
        vals = [base if t % 2 == 0 else np.nan for t in times]
        values[loc] = vals
    df = df_from_values(locs, times, values)

    imp_seq = PanelImputer(
        location_index="loc", 
        time_index="time", 
        imputation_method="fill_all", 
        parallelize=False
    )
    out_seq = imp_seq.fit_transform(df)

    imp_par = PanelImputer(
        location_index="loc",
        time_index="time",
        imputation_method="fill_all",
        parallelize=True,
    )
    with pytest.warns(UserWarning, match="reducing n_jobs"):
        out_par = imp_par.fit_transform(df)

    pd.testing.assert_frame_equal(out_seq, out_par)
