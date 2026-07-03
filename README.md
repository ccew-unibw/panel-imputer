# Description
Political science data often comes in panels, with separate time series data for each location or unit. This custom Imputer for panel data make it easy to deal with missing values in your panel, or gap-fill e.g. yearly observations in a monthly observation. Imputation is performed on a location-by-location basis, currently assuming independence between locations, without having to deal with looping through everything manually. This works as a standalone tool, but can also be used in sklearn Pipeline objects for machine learning tasks, offering protection from data leakage due to improper imputation.

## Installation
Install via
```
pip install panel-imputer
```
## Configuration

**Args:**

```
location_index: str
    name of the index with the location information

time_index: str|[str]
    Information the time component in the index, which is used to sort the data.
    Accepts lists for multi-level time information (e.g. "year", "quarter").

missing_values: float|int default=np.nan
    Value of missing values. If not np.nan, all values in df matching missing_values are
    replaced when calling transform method.

imputation_method: Literal['bfill', 'ffill', 'fill_all', 'interpolate'], default='bfill'
    Imputation is performed on a location-by-location basis.

    Available options:
    'bfill': Imputation using only bfill where newer data is available. Leaves NA's
        after the most recent data in place.
    'ffill': Imputation using only ffill where older data is available. Leaves NA's
        before the earliest datapoint in place.
    'fill_all': Combination of 'bfill' and ffill where no data for backfilling is available.
    'interpolate': Imputation using pandas interpolate. Needs at least 2 non-nan values,
        skipping all-NA locations.

interp_method: str, default=None
    Interpolation method parameter to be passed to pandas.DataFrame.interpolate. Only
    used and required in case 'imputation_method'='interpolate'.

    Please note that only 'linear' interpolation is currently supported,
    others may or may not work.

tail_behavior: str, [str], possible values: ['fill', 'None', 'extrapolate']
    Fill behavior for nan tails. Can either be a single string, which applies to both
    ends, or a list/tuple of length 2 for end-specific behavior.

    Available options:
    'None': Do nothing
    'fill': Fill with last non-nan value in the respective direction.
    'extrapolate': Extrapolate from given observations according to the chosen interpolation method.

nan_loc_policy: Literal[None, 'mean', 'median', 'knnimpute'], default=None
    Fill strategy for all-NA locations, after initial imputation step. Defaults to None,
    leaving all-NA locations as is.

    Available options:
    None: Do nothing
    'mean': Use the mean value of all other locations on a time-by-time basis
    'median': Use the median value of all other locations on a time-by-time basis.
    'knnimpute': Impute using sklearn's KNNImputer. Note that this is a performance
        bottleneck since KNNImputer does not support parallelization. Using this option
        may multiply imputation time by a large factor.

knn_kwargs: dict, default=None
    Dictionary with kwargs to be passed to sklearn's KNNImputer in case nan_interp_policy='knnimpute'.
    If no kwargs are passed, uses KNNImputer(weights='distance').

knn_performance: bool, default=False
    Flag to indicate wether to speed up KNN imputation by splitting the data into its
    time steps and imputing them independently (Sequentially or in parallel if
    parallelize=True). This speeds up performance at the cost of imputation accuracy,
    since this means there are only the datapoints from the same time step available to
    impute from.

all_nan_policy: str, default='drop', possible values: ['drop', 'error']
    Whether to drop columns with all-nan values and proceed with imputation or raise an
    error instead.

verbose: int, default=0
    Controls progress bars and other informational outputs. Values > 0 enable
    progress bars and messages; 0 disables them. Also used as the default for
    joblib.Parallel's `verbose` kwarg when parallelization is enabled.

parallelize: bool, default=False
    Whether to use parallelization with joblib Parallel. Creates chunks based on the
    location index.

parallel_kwargs: dict, default=None
    Dictionary with kwargs to be passed to joblib Parallel. Unless otherwise specified,
    default values used are `n_jobs = -2`, `verbose = verbose`.

```

**Methods:**

```
fit(self, X, y=None): Performs input checks.
    Returns: None
transform(self, X, y=None): Imputes missing values based on the instance's configuration.
    Returns: imputed pd.DataFrame
fit_transform(self, X, y=None): Inherited combination of fit and transform in one step.
    Returns: imputed pd.DataFrame
```

## Example use:

```
from panel_imputer import PanelImputer

#1: use fit_transform for imputation with prepared dataframe
df = read_some_panel_data_with_missing_values()

imp = PanelImputer(
    location_index='country',
    time_index=['year', 'month'],
    imputation_method='bfill'
)

df_imputed = imp.fit_transform(df)

#2: use in a pipeline
pipe = Pipeline(
    [('impute', imp),
    ('model', RandomForestClassifier())]
)
X, y = df[features], df[target]

pipe.fit(X, y)
```    

For more examples, see the jupyter notebook.


## Changelog:

### 0.8.4

Bugfixes:
- KNNImputer now always uses keep_empty_features=True to prevent index mismatches during the all-NA location fill step, which caused the imputation to fail in some cases when using the KNN method.

### 0.8.3

Improvements:
- Added a "Performance" mode for KNN imputation for all-NA locations, where imputation is performed for each timestep independently. This reduces imputation accuracy but significantly speeds up the process. Avalaible though the `knn_performance` flag.

Bugfixes:
- Fixed a bug where NA values were not correctly preserved during all-NA location imputation for the 'bfill' strategy.
- Fixed a bug where the wrong fallback value was used during median all-NA location imputation in case of unique timestamps for a location.

### 0.8.2

Improvements:
- Added `verbose` option to control whether to print progress information.
- Parallel will automatically reduce the number of `n_jobs` to prevent crashing if used with too few locations.
- Added a test suite for the PanelImputer class.
- DataFrame index is now preserved through imputation.

Bugfixes:
- Provided NA value through `missing_values` now correctly applied.
- Fixes erroneous warning related to tail_behavior.

### 0.8.1

Bugfixes:
- fixed incorrect tail_behavior input parsing
- now correctly leaves all-NA rows unimputed
- fixed bug with mean/median NA-loc filling strategy in case of no other datapoints for a timestep.

### 0.8.0

- __Potentially breaking:__ `time_index` no longer optional, but required.
- Added functionality: Ability to fill all-NaN locations using mean/median statistics or sklearn's KNNImputer. New `PanelImputer` parameters: `nan_loc_policy`, `knn_kwargs`.
- Improved code documentation.
- Improved argument validation.
- Updated naming for internal methods.


### 0.7.1

- Parallelization performance improved massively for certain use cases.
- Parallelization turned off by default.
- If `parallelize` parameter is True and no `parallel_kwargs` are specified by the user, PanelImputer now uses `Parallel(n_jobs = -2)` by default.

### 0.7.0

Initial release via CCEW.
