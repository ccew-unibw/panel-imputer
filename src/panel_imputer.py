import multiprocessing
from typing import Literal, List, overload
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm


class PanelImputer(BaseEstimator, TransformerMixin):
    """
    Custom Imputer compatible with sklearn pipelines to fill missing values in panel data in a pd.DataFrame.

    Implements the sklearn-interface via `fit()` and `transform()`, inheriting the combined `fit_transform()`
    """

    def __init__(
        self,
        location_index: str,
        time_index: str | List[str],
        missing_values: float | int = np.nan,
        imputation_method: Literal["bfill", "ffill", "fill_all", "interpolate"] = "bfill",
        interp_method: str = None,
        tail_behavior: str | List[str] = None,
        nan_loc_policy: Literal[None, "mean", "median", "knnimpute"] = None,
        knn_kwargs: dict = None,
        all_nan_policy: Literal["drop", "error"] = "drop",
        parallelize: bool = False,
        parallel_kwargs: dict = None,
    ):
        """
        Initializes the PanelImputer instance and validates arguments.
        
        Sets up the configuration for the imputation process, including identifiers
        for location and time, the imputation strategy, and policies for handling 
        various edge cases such as all-NaN slices.

        Args:
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
                Interpolation method parameter to be passed for pandas.DataFrame.interpolate. Only
                used and required in case 'imputation_method' is one of ['interpolate',
                'nan_interpolate']. Please note that only linear interpolation is fully tested.

            tail_behavior: str, [str], possible values: ['fill', 'None', 'extrapolate']
                Fill behaviour for nan tails. Can either be a single string, which applies to both
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

            all_nan_policy: str, default='drop', possible values: ['drop', 'error']
                Whether to drop columns with all-nan values and proceed with imputation or raise an
                error instead.

            parallelize: bool, default=False
                Whether to use parallelization with joblib Parallel. Creates chunks based on the
                location index.

            parallel_kwargs: dict, default=None
                Dictionary with kwargs to be passed to joblib Parallel. If `parallelize=True` and
                parallel_kwargs is None, set to `{"n_jobs": -2}`.
        """
        self.location_index = location_index
        self.time_index = time_index
        self.missing_values = missing_values
        assert imputation_method in [
            "bfill",
            "ffill",
            "fill_all",
            "interpolate",
            "nan_interpolate",
        ]
        self.imputation_method = imputation_method
        self.tail_behavior = tail_behavior
        if "interpolate" in imputation_method:
            # tail_behavior
            if isinstance(tail_behavior, str):
                assert tail_behavior in ["None", "fill", "extrapolate"]
            else:
                assert (
                    all(isinstance(e, str) for e in tail_behavior)
                    and (len(tail_behavior) == 2)
                    and all(
                        [
                            tail in ["None", "fill", "extrapolate"]
                            for tail in tail_behavior
                        ]
                    )
                )
            # interp_method
            assert interp_method is not None
            if interp_method not in ["linear", "slinear"]:
                warnings.warn(
                    "Class only tested for linear interpolation, please doublecheck whether imputation leads to desired "
                    "results."
                )
            if (tail_behavior != "fill") and (interp_method == "linear"):
                warnings.warn(
                    'Chosen interpolation method "linear" used with pandas.DataFrame.interpolate() leads to unexpected '
                    'results for tail behavior other than "fill". Using scipy.interpolate.interp1d\'s "slinear" '
                    "interpolation instead."
                )
                interp_method = "slinear"
            # nan_interp_policy
        assert nan_loc_policy in [None, "mean", "median", "knnimpute"]
        self.nan_loc_policy = nan_loc_policy
        if nan_loc_policy == "knnimpute" and knn_kwargs is None:
            knn_kwargs = {"weights": "distance"}
        self.knn_kwargs = knn_kwargs
        self.interp_method = interp_method
        if "interpolate" not in imputation_method and (
            interp_method is not None or tail_behavior is not None
        ):
            message = (
                f"interp_method and tail_behavior are only relevant for interpolation, not for chosen imputation"
                f'method <"{imputation_method}"> and therefore have no effect. '
            )
            warnings.warn(message, UserWarning)
        assert all_nan_policy in ["drop", "error"]
        self.all_nan_policy = all_nan_policy
        self.parallelize = parallelize
        if parallel_kwargs is None and parallelize:
            parallel_kwargs = {"n_jobs": -2}
        self.parallel_kwargs = parallel_kwargs

    def fit(self, X: pd.DataFrame | pd.Series, y=None):
        """Validates the input data and prepares the imputer.

        This method conforms to the scikit-learn API. It performs essential
        input checks on the provided DataFrame `X` to ensure it meets the
        requirements for panel data imputation, such as having the specified
        location and time indices. The actual imputation logic is contained
        within the `transform` method.

        Args:
            X: The input pandas DataFrame with a MultiIndex containing location
                and time information and missing values to be imputed.
            y: Ignored. Present for API consistency.

        Returns:
            The fitted PanelImputer instance.
        """
        self._validate_input(X, in_fit=True)
        return self

    def transform(self, X: pd.DataFrame | pd.Series, y=None) -> pd.DataFrame:
        """Applies the imputation to the input data.

        This method orchestrates the imputation workflow for the provided
        DataFrame or Series `X`. It first validates and prepares the data, then 
        generates the imputed values using the configured strategies, and finally
        updates the original DataFrame with these new values.

        Args:
            X: The input pandas DataFrame with a MultiIndex containing location
                and time information and missing values to be imputed.
            y: Ignored. Present for API consistency.

        Returns:
            A pandas DataFrame with missing values imputed according to the
            instance's configuration.
        """
        # make sure that the imputer was fitted
        check_is_fitted(self, "fit_checks_done_")
        original_index_levels = X.index.names
        df = self._validate_input(X, in_fit=False)
        update_map = self._get_update_map(df)
        df.update(update_map, overwrite=False)
        # level order may be modified during input validation
        df = df.reorder_levels(original_index_levels)
        return df

    @overload
    def _validate_input(self, X, in_fit: Literal[False]) -> pd.DataFrame: ...

    @overload
    def _validate_input(self, X, in_fit: Literal[True]) -> None: ...

    def _validate_input(self, X, in_fit: bool) -> pd.DataFrame | None:
        """Validates and prepares the input DataFrame.

        This internal method serves two purposes based on the `in_fit` flag.
        When called from `fit()`, it performs validation checks on the input `X`,
        ensuring it is a DataFrame with the necessary index structure, and set a
        check flag.
        When called from `transform()`, it performs the same checks and also
        prepares the data for imputation by replacing `missing_values` with
        np.nan, dropping all-NaN columns depending on the `all_nan_policy` flag,
        and sorting the data by the location and time indices.

        Args:
            X: The input pandas DataFrame or Series to be validated.
            in_fit: A boolean flag to indicate if the call is from `fit` (True)
                or `transform` (False).

        Returns:
            If `in_fit` is True, returns None. If `in_fit` is False, returns
            the prepared and sorted pandas DataFrame.
        """
        # validity check
        try:
            assert isinstance(X, pd.DataFrame)
        except:
            if isinstance(X, pd.Series):
                X = X.to_frame()
            else:
                raise AssertionError("please pass a pandas DataFrame or Series")
        assert self.location_index in X.index.names

        time_index_list = (
            [self.time_index]
            if isinstance(self.time_index, str)
            else list(self.time_index)
        )
        assert all(time_index in X.index.names for time_index in time_index_list)

        if any(X.replace(self.missing_values, np.nan).isna().all()):
            all_nan_cols = X.columns[X.isna().all()].tolist()
            if self.all_nan_policy == "error":
                raise ValueError(
                    f'Cannot impute all-nan columns {all_nan_cols}. Set all_nan_policy="drop" to drop columns.'
                )

        if self.imputation_method == "interpolate":
            if any(X.isna().sum() == len(X) - 1):
                single_nan_cols = X.columns[X.isna().sum() == len(X) - 1].tolist()
                raise ValueError(
                    f"Cannot interpolate columns with only 1 non-nan value: {single_nan_cols}."
                )

        if in_fit:
            # just validate input, the actual work is done in transform
            self.fit_checks_done_ = True
            return

        else:
            # process input
            df = X.copy()
            if not np.isnan(self.missing_values):
                df.replace(self.missing_values, np.nan)

            if any(df.isna().all()) and self.all_nan_policy == "drop":
                all_nan_cols = df.columns[df.isna().all()].tolist()
                df = df.drop(columns=all_nan_cols)

            sort_levels = [self.location_index] + time_index_list
            df = df.reorder_levels(sort_levels)
            df = df.sort_index()
            return df

    def _get_update_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates a DataFrame of imputed values.

        This method orchestrates the core imputation process. It divides the
        input DataFrame by location and applies the chosen imputation method to
        each location's time series. If `parallelize` is set to True, this
        process is chunked and performed via joblib.Parallel. After the primary
        imputation, it handles any locations that are still entirely NaN 
        according to the specified `nan_loc_policy`.

        Args:
            df: The pre-processed and sorted input DataFrame.

        Returns:
            A DataFrame with the same index as the input, containing the imputed
            values.
        """
        if not df.isna().any().any():
            return df
        if self.parallelize:
            with Parallel(**self.parallel_kwargs) as parallel:
                # does not necessarily correspond, but quick n_jobs estimation via multiprocessing
                # to create chunks
                if parallel.n_jobs < 0:
                    n_chunks = multiprocessing.cpu_count() + parallel.n_jobs + 1
                else:
                    n_chunks = parallel.n_jobs
                unique_locs = list(
                    df.index.get_level_values(self.location_index).unique()
                )
                chunk_idxs = np.linspace(0, len(unique_locs), n_chunks + 1, dtype=int)
                chunk_locs_list = [
                    unique_locs[chunk_idxs[i] : chunk_idxs[i + 1]]
                    for i in range(n_chunks)
                ]
                update_maps = parallel(
                    delayed(self._locs_interpolate)(
                        df.loc[df.index.get_level_values(self.location_index).isin(locs)], False
                    )
                    for locs in chunk_locs_list
                )
                update_map = pd.concat(update_maps)
        else:
            update_map = self._locs_interpolate(df)
        if self.nan_loc_policy is not None:
            fill_df = self._fill_nan_locs(update_map)
            update_map.update(fill_df, overwrite=False)
        return update_map.sort_index()

    def _locs_interpolate(self, df_interp: pd.DataFrame, progress_bar: bool = True) -> pd.DataFrame:
        """Applies the specified imputation method to each location.

        This method iterates through each unique location present in the input
        DataFrame. For each location, it applies the imputation method defined
        during initialization ['bfill', 'ffill', 'fill_all', 'interpolate']. 
        This forms the first pass of the imputation process which is always
        performed, focused on filling gaps within each individual time series.

        Args:
            df_interp: A subset of the preprocessed DataFrame containing 
                some or all locations.

        Returns:
            A DataFrame containing the imputed values for the provided locations.
        """

        def impute_loc(loc) -> pd.DataFrame:
            df_loc = df_interp.xs(loc, level=self.location_index, drop_level=False)
            if self.imputation_method == "bfill":
                loc_map = df_loc.bfill()
            elif self.imputation_method == "ffill":
                loc_map = df_loc.ffill()
            elif self.imputation_method == "fill_all":
                loc_map = df_loc.bfill().ffill()
            elif self.imputation_method == "interpolate":
                loc_map = self._local_interpolate(df_loc)
            else:
                # check are performed before, should not happen
                raise NotImplementedError
            return loc_map

        locs = df_interp.index.get_level_values(self.location_index).unique()
        desc = "step 1" if self.nan_loc_policy is not None else None
        if progress_bar:
            update_maps = [impute_loc(loc) for loc in tqdm(locs, desc=desc)]
        else:
            update_maps = [impute_loc(loc) for loc in locs]
        update_map = pd.concat(update_maps)
        return update_map

    def _local_interpolate(self, df_loc: pd.DataFrame) -> pd.DataFrame:
        """Performs interpolation for a single location's time series.

        This method is called when `imputation_method` is 'interpolate'. It
        handles the logic for interpolating missing values within a single
        location. It also manages the behavior outside the known values for the
        series (the "tails") based on the `tail_behavior` parameter, allowing
        for filling, extrapolation, or leaving tails as NaN.

        Args:
            df_loc: A DataFrame containing the time series data for a single location.

        Returns:
            A DataFrame with interpolated values for that location.
        """
        def get_fill_values():
            fill_values = (
                df_loc.loc[df_loc[col].first_valid_index(), col],
                df_loc.loc[df_loc[col].last_valid_index(), col],
            )
            if "None" in self.tail_behavior:
                if self.tail_behavior == "None":
                    fill_values = (np.nan, np.nan)
                else:
                    fill_values = tuple(
                        [
                            np.nan
                            if tail == "None"
                            else tail
                            if tail == "extrapolate"
                            else fill_values[i]
                            for i, tail in enumerate(self.tail_behavior)
                        ]
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
                if self.tail_behavior in ["fill", "extrapolate"]:
                    loc_map[col] = df_loc[col].bfill().ffill()
                else:
                    # this is simply the same data without imputation
                    loc_map[col] = df_loc[col]
                # warnings.warn(message, UserWarning)
            else:
                if self.tail_behavior in ["fill", "None"]:
                    fill_value = get_fill_values()
                    loc_map[col] = (
                        df_loc.reset_index()[col]
                        .interpolate(
                            method=self.interp_method,
                            limit_direction="both",
                            fill_value=fill_value,
                        )
                        .values
                    )
                else:
                    loc_map[col] = (
                        df_loc.reset_index()[col]
                        .interpolate(
                            method=self.interp_method,
                            limit_direction="both",
                            fill_value="extrapolate",
                        )
                        .values
                    )
            return

        def different_tails_fill():
            # Fill performed in nested function to improve code readability
            if (~df_loc[col].isna()).sum() == 1:
                # message = (
                #     f'Only 1 non-nan data point for location <{df_loc.index.get_level_values(self.location_index)[0]}'
                #     f'>, column <{col}>, imputation performed via filling where tail behavior is not "None".'
                # )
                df_temp = df_loc[col]
                if self.tail_behavior[0] != "None":
                    df_temp = df_temp.bfill()
                if self.tail_behavior[1] != "None":
                    df_temp = df_temp.ffill()
                loc_map[col] = df_temp
                # warnings.warn(message, UserWarning)
            else:
                fill_value = get_fill_values()
                loc_map[col] = (
                    df_loc.reset_index()[col]
                    .interpolate(method=self.interp_method, limit_area="inside")
                    .values
                )
                limit_direction = ("backward", "forward")
                for i in range(2):
                    if self.tail_behavior[i] in ["fill", "None"]:
                        update_data = (
                            df_loc.reset_index()[col]
                            .interpolate(
                                method=self.interp_method,
                                limit_direction=limit_direction[i],
                                fill_value=fill_value[i],
                            )
                            .values
                        )
                    else:
                        update_data = (
                            df_loc.reset_index()[col]
                            .interpolate(
                                method=self.interp_method,
                                limit_direction=limit_direction[i],
                                fill_value="extrapolate",
                            )
                            .values
                        )
                    update_series = pd.Series(
                        index=loc_map.index, data=update_data, name=col
                    )
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

    def _fill_nan_locs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes locations that are entirely composed of NaN values.

        After the initial within-location imputation, some locations might still
        be all-NaN. This method handles these cases based on the `nan_loc_policy`.
        It can fill these values using cross-sectional statistics (mean or median
        of other locations at the same time point) or by using the KNNImputer
        for a more sophisticated, multivariate approach. For consistency with
        the tail behavior of the first interpolation, will keep time points NaN
        where all other locations are all-NaN after the first pass.

        Args:
            df: The DataFrame after the first pass of imputation, which may
                still contain all-NaN locations.

        Returns:
            A DataFrame with imputed values for the previously all-NaN locations.
        """
        def impute_nan_loc(loc) -> pd.DataFrame:
            df_loc = df.xs(loc, level=self.location_index, drop_level=False)
            cols = df_loc.columns[df_loc.isna().any()].tolist()
            if len(cols) == 0:
                return None
            else:
                loc_map = pd.DataFrame(index=df_loc.index)
                for col in cols:
                    if self.nan_loc_policy == "mean":
                        try:
                            loc_map[col] = lookup_df_time.loc[
                                df_loc.reset_index()[self.time_index], (col, "mean")
                            ].to_list()
                        except KeyError:
                            loc_map[col] = lookup_df_all.loc[("mean", col)].to_list()
                    elif self.nan_loc_policy == "median":
                        try:
                            loc_map[col] = lookup_df_time.loc[
                                df_loc.reset_index()[self.time_index], (col, "median")
                            ].to_list()
                        except KeyError:
                            loc_map[col] = lookup_df_all.loc[("mean", col)].to_list()
                    else:
                        raise NotImplementedError
                return loc_map
            
        # do not apply the NA location filling to all-NA times in already imputed/interpolated locs
        # the assumption is that these are not supposed to be filled in case of "None" tail_behavior
        # or the bfill/ffill filling strategy, where filling beyond the first/last available date is
        # not desired
        if "None" in self.tail_behavior or self.imputation_method in [
            "bffill",
            "ffill",
        ]:
            all_na_times = df.isna().all(axis=1).groupby(self.time_index).all()
            all_na_filter = df.reset_index().apply(
                lambda x: all_na_times.loc[x[self.time_index]], axis=1
            )
            # drop from the dataframe
            df.loc[(~all_na_filter).to_list()]
            
        if self.nan_loc_policy in ["mean", "median"]:
            locs = df.index.get_level_values(self.location_index).unique()
            # creating a lookup df to do the mean/median based on the point in time if possible
            lookup_df_time = (
                df.groupby(self.time_index).agg(["mean", "median"]).dropna(how="all")
            )
            # fallback option in case of unequal time series in the panel
            lookup_df_all = df.agg(["mean", "median"]).dropna(how="all")
            update_dfs = [impute_nan_loc(loc) for loc in locs]
            update_df = pd.concat(
                [df_loc for df_loc in update_dfs if df_loc is not None]
            )
        elif self.nan_loc_policy == "knnimpute":
            print("KNN imputation, this may take a while...")
            imputer = KNNImputer(**self.knn_kwargs)
            update_df = pd.DataFrame(
                imputer.fit_transform(df), index=df.index, columns=df.columns
            )
            update_df = update_df.astype(df.dtypes)
        else:
            raise NotImplementedError
        return update_df