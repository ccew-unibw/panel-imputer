from typing import Literal, List
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
import warnings

from tqdm import tqdm


class PanelImputer(BaseEstimator, TransformerMixin):
    '''
    Custom Imputer compatible with sklearn pipelines to fill missing values in panel data in a pd.DataFrame

    Parameters
    ----------
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

    parallelize: bool, default=True
        Whether to use parallelization with joblib Parallel.

    parallel_kwargs: dict, default=None
        Dictionary with kwargs to be passed to joblibs Parallel.
    '''

    def __init__(
            self,
            location_index: str,
            time_index: str|List[str] = None,
            imputation_method: str = 'bfill',
            interp_method: str = None,
            tail_behavior: str|List[str] = None,
            missing_values: float|int = np.nan,
            all_nan_policy: Literal['drop', 'error'] = 'drop',
            parallelize: bool = True,
            parallel_kwargs: dict = None
    ):
        self.location_index = location_index
        self.time_index = time_index
        self.missing_values = missing_values
        self.imputation_method = imputation_method
        self.interp_method = interp_method
        self.tail_behavior = tail_behavior
        self.all_nan_policy = all_nan_policy
        self.parallelize = parallelize
        if parallel_kwargs is None:
            parallel_kwargs = {}
        self.parallel_kwargs = parallel_kwargs

    def _validate_input(self, X, in_fit):
        # validity check
        try:
            assert type(X) == pd.DataFrame
        except:
            if type(X) == pd.Series:
                X = X.to_frame()
            else:
                raise AssertionError("please pass a pandas DataFrame or Series")
        assert self.location_index in X.index.names
        assert self.imputation_method in ['bfill', 'ffill', 'fill_all', 'interpolate']

        if any(X.replace(self.missing_values, np.nan).isna().all()):
            all_nan_cols = X.columns[X.isna().all()].tolist()
            if self.all_nan_policy == 'error':
                raise ValueError(f'Cannot impute all-nan columns {all_nan_cols}. Set all_nan_policy="drop" to drop columns.')

        if self.imputation_method != 'interpolate' and (self.interp_method is not None or self.tail_behavior is not None):
            message = (f'interp_method and tail_behavior are only relevant for interpolation, not for chosen imputation'
                       f'method <"{self.imputation_method}"> and therefore have no effect. ')
            warnings.warn(message, UserWarning)

        if self.imputation_method == 'interpolate':
            assert (type(self.tail_behavior) is str) or (len(self.tail_behavior) == 2)
            if len(self.tail_behavior) == 2:
                assert all([tail in ['None', 'fill', 'extrapolate'] for tail in self.tail_behavior])

            if any(X.isna().sum() == len(X)-1):
                single_nan_cols = X.columns[X.isna().sum() == len(X)-1].tolist()
                raise ValueError(f'Cannot interpolate columns with only 1 non-nan value: {single_nan_cols}.')

            if (self.tail_behavior != 'fill') and (self.interp_method == 'linear'):
                warnings.warn(
                    'Chosen interpolation method "linear" used with pandas.DataFrame.interpolate() leads to unexpected '
                    'results for tail behavior other than "fill". Using scipy.interpolate.interp1d\'s "slinear" '
                    'interpolation instead.'
                )
                self.interp_method = 'slinear'

            if self.interp_method not in ['linear', 'slinear']:
                warnings.warn(
                    'Class only tested for linear interpolation, please doublecheck whether imputation leads to desired '
                    'results.'
                )

        if in_fit:
            # just validate input, the actual work is done in transform
            self.fit_checks_done_=True
            return

        else:
            # process input
            df = X.copy()
            if not np.isnan(self.missing_values):
                df.replace(self.missing_values, np.nan)

            if any(df.isna().all()) and self.all_nan_policy == 'drop':
                all_nan_cols = df.columns[df.isna().all()].tolist()
                df = df.drop(columns=all_nan_cols)

            if self.time_index is not None:
                time_index_list = [self.time_index] if type(self.time_index) is str else list(dict.fromkeys(self.time_index))
                sort_levels = [self.location_index] + time_index_list
                df = df.sort_index(level=sort_levels)

            return df

    def fit(self, X, y=None):
        """
        only does input checks - actual imputation does not really require fit, only for sklearn structure
        """
        self._validate_input(X, in_fit=True)
        return self

    def _get_update_map(self, df):
        if not df.isna().any().any():
            return df
        df = df.sort_index()
        if self.parallelize:
            update_maps = Parallel(**self.parallel_kwargs)(
                delayed(self._parallel_interpolate)(df, loc)
                for loc in df.index.get_level_values(self.location_index).unique()
            )
        else:
            update_maps = [self._parallel_interpolate(df, loc)
                           for loc in tqdm(df.index.get_level_values(self.location_index).unique())]
        update_map = pd.concat(update_maps)
        return update_map

    def _parallel_interpolate(self, df, loc):
        df_loc = df.xs(loc, level=self.location_index, drop_level=False)
        if self.imputation_method == 'bfill':
            loc_map = df_loc.bfill()
        elif self.imputation_method == 'ffill':
            loc_map = df_loc.ffill()
        elif self.imputation_method == 'fill_all':
            loc_map = df_loc.bfill().ffill()
        elif self.imputation_method == 'interpolate':
            loc_map = self._local_fit_interpolate(df_loc)
        else:
            # check are performed before, should not happen
            raise NotImplementedError
        return loc_map

    def _local_fit_interpolate(self, df_loc):
        def get_fill_values():
            fill_values = (
                df_loc.loc[df_loc[col].first_valid_index(), col],
                df_loc.loc[df_loc[col].last_valid_index(), col]
            )
            if 'None' in self.tail_behavior:
                if self.tail_behavior == 'None':
                    fill_values = (np.nan, np.nan)
                else:
                    fill_values = tuple(
                        [np.nan if tail == 'None'
                         else tail if tail=='extrapolate'
                         else fill_values[i] for i, tail in enumerate(self.tail_behavior)]
                    )
            return fill_values

        def uniform_tails_fill():
            # Fill performed in nested function to improve code readability
            if (~df_loc[col].isna()).sum() == 1:
                # message = (
                #     f'Only 1 non-nan data point for location <{df_loc.index.get_level_values(self.location_index)[0]}'
                #     f'>, column <{col}>, imputation only performed via filling where tail behavior != "None".'
                # )
                # if we want to fill/extrapolate in 1 direction but only have 1 value, we default to filling
                # according to the specified tail behavior
                if self.tail_behavior in ['fill', 'extrapolate']:
                    loc_map[col] = df_loc[col].bfill().ffill()
                else:
                    # this is simply the same data without imputation
                    loc_map[col] = df_loc[col]
                # warnings.warn(message, UserWarning)
            else:
                if self.tail_behavior in ['fill', 'None']:
                    fill_value = get_fill_values()
                    loc_map[col] = df_loc.reset_index()[col].interpolate(
                        method=self.interp_method, limit_direction='both', fill_value=fill_value
                    ).values
                else:
                    loc_map[col] = df_loc.reset_index()[col].interpolate(
                        method=self.interp_method, limit_direction='both', fill_value='extrapolate'
                    ).values
            return

        def different_tails_fill():
            # Fill performed in nested function to improve code readability
            if (~df_loc[col].isna()).sum() == 1:
                # message = (
                #     f'Only 1 non-nan data point for location <{df_loc.index.get_level_values(self.location_index)[0]}'
                #     f'>, column <{col}>, imputation performed via filling where tail behavior is not "None".'
                # )
                df_temp = df_loc[col]
                if self.tail_behavior[0] != 'None':
                    df_temp = df_temp.bfill()
                if self.tail_behavior[1] != 'None':
                    df_temp = df_temp.ffill()
                loc_map[col] = df_temp
                # warnings.warn(message, UserWarning)
            else:
                fill_value = get_fill_values()
                loc_map[col] = df_loc.reset_index()[col].interpolate(method=self.interp_method, limit_area='inside').values
                limit_direction = ('backward', 'forward')
                for i in range(2):
                    if self.tail_behavior[i] in ['fill', 'None']:
                        update_data = df_loc.reset_index()[col].interpolate(
                            method=self.interp_method, limit_direction=limit_direction[i], fill_value=fill_value[i]
                        ).values
                    else:
                        update_data = df_loc.reset_index()[col].interpolate(
                            method=self.interp_method, limit_direction=limit_direction[i], fill_value='extrapolate'
                        ).values
                    update_series = pd.Series(index=loc_map.index, data=update_data, name=col)
                    loc_map.update(update_series, overwrite=False)
            return

        interp_cols = df_loc.columns[df_loc.isna().any()].tolist()
        loc_map = pd.DataFrame(index=df_loc.index)

        for col in interp_cols:
            # check if we can even interpolate (we need more than 1 non-nan value for location)
            if df_loc[col].isna().all():
                # message = f'All nan data for location <{df_loc.index.get_level_values(self.location_index)[0]}' \
                #           f'>, column <{col}>, imputation locally not possible.'
                # warnings.warn(message, UserWarning)
                # this is simply the all-nan data in this case
                loc_map[col] = df_loc[col]
            else:
                if type(self.tail_behavior) is str:
                    uniform_tails_fill()
                else:
                    different_tails_fill()
        return loc_map

    def transform(self, X, y=None):
        # make sure that the imputer was fitted
        check_is_fitted(self, 'fit_checks_done_')

        df = self._validate_input(X, in_fit=False)
        update_map = self._get_update_map(df)

        df.update(update_map, overwrite=False)
        return df


