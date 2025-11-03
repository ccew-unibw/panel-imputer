import pandas as pd


def df_from_values(locs, times, values_by_loc, columns=("v",)):
    """Build a panel-shaped DataFrame with MultiIndex (loc, time) from value dict."""
    idx = pd.MultiIndex.from_product([locs, times], names=["loc", "time"])
    df = pd.DataFrame(index=idx, columns=list(columns), dtype=float)
    for loc, values in values_by_loc.items():
        for t, val in zip(times, values):
            df.loc[(loc, t), columns[0]] = val
    return df

