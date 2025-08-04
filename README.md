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

time_index (optional): str|[str], default=None
    Information the time component in the index, which is used to sort the data if provided. Make sure to either
    provide this or pass an already sorted dataframe. Accepts lists for multi-indices

imputation_method: str, default='bfill', possible values: ['bfill', 'ffill', 'interpolate']
    Imputation is performed on a location-by-location basis. For correct results, input df needs to be constructed
    with a time and a location index. Df either needs to be sorted by time or the time index needs to be passed to
    the imputer, so the imputation can be performed separately for each location.

    Available options:
    'bfill': Imputation using only bfill where newer data is available. Leaves NA's after the most recent data in
            place.
    'ffill': Imputation using only ffill where older data is available. Leaves NA's before the earliest datapoint in
            place.
    'fill_all': Combination of 'bfill' and ffill where no data for backfilling is available.
    'interpolate': Imputation using pandas interpolate. Needs at least 2 non-nan values.

interp_method: str
    Interpolation method parameter to be passed for pandas.DataFrame.interpolate. Please note that only linear
    interpolation is fully tested.

tail_behavior: str, [str], possible values: ['fill', 'None', 'extrapolate']
    Fill behaviour for nan tails. Can either be a single string, which applies to both ends, or a list/tuple of
    length 2 for end-specific behavior.

    'fill': Fill with last non-nan value in the respective direction.
    'extrapolate': Extrapolate from given observations according to the chosen interpolation method.

missing_values: float|int default=np.nan
    Value of missing values. If not np.nan, all values in df matching missing_values are replaced
    when calling transform method.

all_nan_policy: str, default='drop', possible values: ['drop', 'error']
    Whether to drop columns with all-nan values and proceed with imputation or raise an error instead.

parallelize: bool, default=False
    Whether to use parallelization with joblib Parallel. Creates chunks based on the location
    index. Whether or not parallelization speeds up things may depend on the input data structure.

parallel_kwargs: dict, default=None
    Dictionary with kwargs to be passed to joblib Parallel.
```

**Methods:**

```
fit(self, X, y=None): Performs input checks.
    Returns: None
transform(self, X, y=None): Imputes missing values based on the configuration in init.
    Returns: imputed pd.DataFrame
fit_transform(self, X, y=None): Inherited combination of fit and transform in one step.
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

### 0.7.1

- Parallelization performance improved massively for certain use cases.
- Parallelization turned off by default.
- If `parallelize` parameter is True and no `parallel_kwargs` are specified by the user, PanelImputer now uses `Parallel(n_jobs = -2)` by default.

### 0.7.0

Initial release via CCEW.