import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from sklearn.linear_model import LinearRegression
import logging
import os

from boruta import BorutaPy
from sklearn.impute import SimpleImputer
from xgboost import XGBRanker

# restore the deprecated aliases
np.int = int
np.bool = bool
np.float = float


# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Formatter: add timestamp, logger name, function, line number, message
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(funcName)s - line %(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# File handler
file_handler = logging.FileHandler("logs/pipeline.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
class Preprocessor:
    """
    A class for preprocessing data, including feature engineering, imputations, and encodings.
    Based on the kieran_process.py file.
    """

    def __init__(self, df: pd.DataFrame = None):
        """
        Initialize the Preprocessor with an optional dataframe.

        Parameters:
        -----------
        df : pd.DataFrame, optional
            The dataframe to preprocess
        """
        self.df = df.copy() if df is not None else None
        self.encoder = None

        self.target_cols = ['position', 'click_bool', 'booking_bool',
                        'gross_bookings_usd', 'target']


            # List of competitor rate percent diff columns
        self.comp_rate_cols = [
                "comp1_rate_percent_diff", "comp2_rate_percent_diff", "comp3_rate_percent_diff",
                "comp4_rate_percent_diff", "comp5_rate_percent_diff", "comp6_rate_percent_diff",
                "comp7_rate_percent_diff", "comp8_rate_percent_diff",
            ]


    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set the dataframe to preprocess.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to preprocess
        """
        self.df = df.copy()


    ###### HELPER CLEAN FUNCTIONS
    def _log_transform_columns(self, log_transform_cols: List[str] = None):
        for col in log_transform_cols:
            if col in self.df.columns:
                self.df[f"log_{col}"] = np.log1p(self.df[col])
                self.df.drop(columns=[col], inplace=True)
                logger.info(f"Log-transformed {col}")
            else:
                logger.info(f"Skipped log-transforming {col} (not in dataframe)")

    def _create_training_target(self):
        if 'booking_bool' in self.df.columns and 'click_bool' in self.df.columns:
            self.df["target"] = self.df["booking_bool"] * 5 + (self.df["click_bool"] & ~self.df["booking_bool"]) * 1
            logger.info("Created training target variable")

    def _filter_gross_bookings(self):
        if 'gross_bookings_usd' in self.df.columns:
            self.df = self.df[(self.df["gross_bookings_usd"].isna()) |
                            ((self.df["gross_bookings_usd"] >= 10) & (self.df["gross_bookings_usd"] <= 2000))]
            logger.info(f"Filtered gross_bookings_usd: min={self.df['gross_bookings_usd'].min()}, max={self.df['gross_bookings_usd'].max()}")

    def _filter_unrealistic_prices(self):
        if 'price_usd' in self.df.columns:
            self.df = self.df[(self.df["price_usd"] >= 10) & (self.df["price_usd"] <= 1000)]
            logger.info(f"Filtered price_usd: min={self.df['price_usd'].min()}, max={self.df['price_usd'].max()}")

    def _filter_competitor_rates(self):
        for col in self.comp_rate_cols:
            if col in self.df.columns:
                self.df = self.df[(self.df[col].isna()) | (self.df[col] <= 60)]
                logger.info(f"Filtered {col}: min={self.df[col].min()}, max={self.df[col].max()}")

    def _recode_zero_to_nan(self, cols: List[str] = None):
        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(0, np.nan)
                logger.info(f"Recoded 0 to NA in {col}")

    def clean_data(self, is_test: bool = False, name: str = "train") -> pd.DataFrame:
        """
        Clean the input data.

        Parameters:
        -----------
        is_test : bool, optional
            Whether this is test data (to prevent data leakage)

        Returns:
        --------
        pd.DataFrame
            The cleaned dataframe with optional target column for training data
        """
        if self.df is None:
            raise ValueError("No dataframe to clean. Please set a dataframe first.")
        logger.info("Cleaning data")

        df = self.df.copy()
        self.df = df  # make sure all sub-methods work on self.df

        self._add_datetime_features()
        self._log_transform_columns(log_transform_cols=["orig_destination_distance"])

        self._recode_zero_to_nan(cols=["prop_review_score", "prop_starrating"])

        if is_test == False:
            self._filter_gross_bookings()
            self._filter_unrealistic_prices()
            self._filter_competitor_rates()
        if name != "test": # also for val
            self._create_training_target()


        return self.df

    ############# IMPUTATIONS #########
    def impute_data(self,
                    sum_imp: List[str] = None,
                    mean_imp: List[str] = None,
                    mode_imp: List[str] = None,
                    interp_imp: List[str] = None,
                    regression_imputation: List[List[str]] = None
                    ) -> pd.DataFrame:
        """
        Impute missing values in the dataframe.

        Parameters:
        -----------
        sum_imp : List[str], optional
            Columns to impute with 0
        mean_imp : List[str], optional
            Columns to impute with mean
        mode_imp : List[str], optional
            Columns to impute with mode
        interp_imp : List[str], optional
            Columns to impute with interpolation

        Returns:
        --------
        pd.DataFrame
            The imputed dataframe
        """
        if self.df is None:
            raise ValueError("No dataframe to impute. Please set a dataframe first.")
        
        logger.info("Imputing data")

        df = self.df.copy()

        # Impute with 0 for sum categories
        if sum_imp:
            for col in sum_imp:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        # Impute with mean
        if mean_imp:
            for col in mean_imp:
                # if "prop" is in name, first try imputing filtered on prop_id
                if "prop" in col:
                    df[col] = df.groupby("prop_id")[col].transform(lambda x: x.fillna(x.mean()))
                    logger.info(f"Imputed {col} with mean per prop_id")
                else:
                    df[col] = df[col].fillna(df[col].mean())
                    logger.info(f"Imputed {col} with global mean")


        # Impute with mode
        if mode_imp:
            for col in mode_imp:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # Impute with interpolation
        if interp_imp:
            for col in interp_imp:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')

        if regression_imputation:
            for cols in regression_imputation:
                # Check if all columns in the list exist in the dataframe
                if all(col in df.columns for col in cols):
                    df = self._regression_imputation(cols[:-1], cols[-1])

        self.df = df
        return df

    def _regression_imputation(self, predictors: List[str], target: str) -> pd.DataFrame:
        """
        Impute missing values in a column using regression.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe containing the column to impute
        col : str
            The column to impute

        Returns:
        --------
        pd.DataFrame
            The dataframe with the column imputed
        """
        # mask over missing values
        mask = self.df[target].isnull()

        # fit the model on the non-missing values
        X = self.df.loc[~mask, predictors]
        y = self.df.loc[~mask, target]
        model = LinearRegression().fit(X, y)

        # predict the missing values
        X_missing = self.df.loc[mask, predictors]
        self.df.loc[mask, target] = model.predict(X_missing)

        logger.info(f"Imputed {target} with regression using {predictors}")

        return self.df

    #### FEATURE ENGINEERING ################
    def _add_datetime_features(self):
        if 'date_time' in self.df.columns:
            self.df['date_time'] = pd.to_datetime(self.df['date_time'])
            self.df['month'] = self.df['date_time'].dt.month
            self.df['weekday'] = self.df['date_time'].dt.weekday
            logger.info("Added month and weekday features")

    def _add_vacation_features(self):
        if all(col in self.df.columns for col in ['date_time', 'srch_booking_window']):
            self.df["vacation_date"] = pd.to_datetime(self.df["date_time"]) + pd.to_timedelta(self.df["srch_booking_window"], unit="D")
            self.df["vacation_day_of_week"] = self.df["vacation_date"].dt.dayofweek
            self.df["vacation_month"] = self.df["vacation_date"].dt.month
            # self.df["vacation_year"] = self.df["vacation_date"].dt.year
            logger.info("Added vacation time features")
        else:
            logger.info("Skipped vacation features (missing columns)")

    def _add_desirability_features(self):
        if all(col in self.df.columns for col in ['prop_location_score1', 'prop_location_score2', 'prop_review_score', 'prop_starrating', 'price_usd']):
            self.df['prop_desirability'] = (
                self.df['prop_location_score1'] +
                self.df['prop_location_score2'] +
                self.df['prop_review_score'] +
                self.df['prop_starrating']
            ) / self.df['price_usd']
            logger.info("Added 'prop_desirability' feature")
        else:
            logger.info("Skipped 'prop_desirability' (missing columns)")

        if all(col in self.df.columns for col in ['price_usd', 'prop_log_historical_price']):
            self.df['price_desirability'] = self.df['price_usd'] / self.df['prop_log_historical_price']
            logger.info("Added 'price_desirability' feature")
        else:
            logger.info("Skipped 'price_desirability' (missing columns)")

    def _add_price_per_person_per_room(self):
        if 'price_usd' not in self.df.columns:
            logger.info("Skipped relative price features (price_usd missing)")
            return

        if all(col in self.df.columns for col in ['srch_adults_count', 'srch_children_count', 'srch_room_count']):
            self.df["price_per_person_per_room"] = self.df["price_usd"] / (
                (self.df["srch_adults_count"] + self.df["srch_children_count"]) * self.df["srch_room_count"]
            )
            logger.info("Added 'price_per_person_per_room'")

    def _add_seasonality_features(self, relative_to: str = "srch_destination_id"):
        if relative_to == "srch_destination_id" and "price_per_person_per_room" not in self.df.columns:
            self._add_price_per_person_per_room()
        if "vacation_month" not in self.df.columns:
            self._add_vacation_features()

        # ADD PRICE RELATIVE TO SEASON AVERAGE (Season-aware z-score):
        if all(col in self.df.columns for col in ['price_per_person_per_room', 'vacation_month', relative_to]):
            month_means = self.df.groupby([relative_to, "vacation_month"])["price_per_person_per_room"].transform("mean")
            prop_stds = self.df.groupby(relative_to)["price_per_person_per_room"].transform("std")
            self.df["rel_hotel_price_season_aware"] = (self.df["price_per_person_per_room"] - month_means) / prop_stds
            logger.info(f"Added 'rel_hotel_price_per_date_z relative to {relative_to}'")
        else:
            logger.info("Skipped 'rel_hotel_price_season_aware' (missing columns)")

        # (Season-agnostic z-score)
        if 'price_per_person_per_room' in self.df.columns:
            self.df["rel_hotel_price_season_agnostic"] = (
                self.df["price_per_person_per_room"] - self.df.groupby("prop_id")["price_per_person_per_room"].transform("mean")
            ) / self.df.groupby("prop_id")["price_per_person_per_room"].transform("std")
            logger.info("Added 'rel_hotel_price_season_agnostic'")

    def _add_new_customer_flag(self):
        if 'visitor_hist_starrating' in self.df.columns:
            self.df["new_customer"] = np.where(self.df["visitor_hist_starrating"] == 0, 1, 0)
            logger.info("Added 'new_customer' flag")

    def _add_domestic_trip_flag(self):
        if all(col in self.df.columns for col in ['prop_country_id', 'visitor_location_country_id']):
            self.df["domestic_trip"] = np.where(self.df["prop_country_id"] == self.df["visitor_location_country_id"], 1, 0)
            logger.info("Added 'domestic_trip' flag")

    def _add_price_surprise(self):
        if all(col in self.df.columns for col in ['price_usd', 'visitor_hist_adr_usd']):
            self.df['price_surprise'] = self.df['price_usd'] / self.df['visitor_hist_adr_usd']
            logger.info("Added 'price_surprise' feature")
        else:
            logger.info("Skipped 'price_surprise' (missing columns)")



    def add_relative_features(self, feature_specs: Dict = None, group_key: str = "srch_id") -> pd.DataFrame:
        """
        Add relative features (rank, z-score, etc.) to the dataframe.

        Parameters:
        -----------
        feature_specs : Dict, optional
            Dictionary mapping base feature to list of methods
        group_key : str, optional
            Key to group by (default: 'srch_id')

        Returns:
        --------
        pd.DataFrame
            The dataframe with relative features added
        """
        if self.df is None:
            raise ValueError("No dataframe to add features to. Please set a dataframe first.")

        logger.info(f"Adding relative features with {feature_specs}, grouped by {group_key}")

        df = self.df.copy()
        feature_specs = feature_specs or self.feature_methods

        for base_feat, methods in feature_specs.items():
            if base_feat not in df.columns:
                logger.info(f"Skipping '{base_feat}' (not in DataFrame)")
                continue

            group = df.groupby(group_key)[base_feat]

            for method in methods:
                new_col = f"{base_feat}_{method}"
                if new_col in df.columns:
                    continue  # Skip if already exists

                if method == "pct_rank":
                    df[new_col] = group.rank(pct=True)
                elif method == "minmax":
                    min_ = group.transform("min")
                    max_ = group.transform("max")
                    df[new_col] = (df[base_feat] - min_) / (max_ - min_ + 1e-6)
                elif method == "zscore":
                    mean = group.transform("mean")
                    std = group.transform("std").replace(0, 1e-6)
                    df[new_col] = (df[base_feat] - mean) / std
                elif method == "rank":
                    df[new_col] = group.rank(method="min")
                else:
                    logger.info(f"Unknown method '{method}' for feature '{base_feat}'")

        self.df = df
        return df

    def add_statistics_per_group(self, selected_features: List[str] = None,
                              stat_func: str = 'median',
                              with_respect_to: str = "prop_id") -> pd.DataFrame:
        """
        Add property ID statistics to the dataframe.

        Parameters:
        -----------
        selected_features : List[str], optional
            List of features to compute stats on
        stat_func : str, optional
            Statistic function to use ('mean', 'median', or 'std')

        Returns:
        --------
        pd.DataFrame
            The dataframe with property ID statistics added
        """
        if self.df is None:
            raise ValueError("No dataframe to add features to. Please set a dataframe first.")

        df = self.df.copy()

        # Select numeric columns to aggregate
        exclude_cols = ['srch_id', with_respect_to, 'position', 'click_bool', 'booking_bool',
                        'gross_bookings_usd', 'target']

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Limit to selected features if provided
        if selected_features:
            numeric_cols = [col for col in numeric_cols if col in selected_features]

        # Compute statistics per prop_id
        prop_stats = df.groupby(with_respect_to)[numeric_cols].agg(stat_func)

        # Rename columns to reflect the stat used
        prop_stats.columns = [f"{col}_{stat_func}" for col in prop_stats.columns]

        # Map each column back to df using prop_id lookup
        for col in prop_stats.columns:
            df[col] = df[with_respect_to].map(prop_stats[col])
        
        logger.info(f"Added {stat_func} statistics for {with_respect_to} to {len(prop_stats.columns)} columns")

        self.df = df
        return df



    ####### ENCODINGS #######

    def probability_encode(self, cols: List[str], is_test: bool = False, encoder: Dict = None) -> pd.DataFrame:
        """
        Probability encode categorical columns based on target.

        Parameters:
        -----------
        cols : List[str]
            List of columns to encode
        is_test : bool, optional
            Whether this is test data (to prevent data leakage)
        encoder : Dict, optional
            Pre-existing encoder for test data

        Returns:
        --------
        pd.DataFrame
            The dataframe with categorical columns encoded
        """
        if self.df is None:
            raise ValueError("No dataframe to encode. Please set a dataframe first.")

        df = self.df.copy()

        # If in test mode, ensure we're only using the pre-existing encoder
        if is_test:
            if encoder is None:
                raise ValueError("Encoder must be provided for test data to prevent leakage")

            for col in cols:
                # Use the encoder to look up the probabilities for each categorical value in the column
                if col in encoder:
                    # Retrieve the probability maps from encoder
                    mean_click = encoder[f"{col}_prob_click"]
                    mean_book = encoder[f"{col}_prob_book"]

                    # Map the probabilities to the dataframe
                    df[f"{col}_prob_click"] = df[col].map(mean_click)
                    df[f"{col}_prob_book"] = df[col].map(mean_book)
                else:
                    logger.info(f"Warning: Encoder for {col} not found. Skipping probability encoding for this column.")

            # After encoding, impute missing values (NaN) using the column mean for all modes
            for col in cols:
                logger.info(f"Imputing missing values in {col}_prob_click and {col}_prob_book with column mean")
                # Impute missing values in the 'prob_click' column with the column's mean
                global_mean_click = df[f"{col}_prob_click"].mean()
                df[f"{col}_prob_click"].fillna(global_mean_click, inplace=True)

                # Impute missing values in the 'prob_book' column with the column's mean
                global_mean_book = df[f"{col}_prob_book"].mean()
                df[f"{col}_prob_book"].fillna(global_mean_book, inplace=True)

            # Don't drop prop_id or srch_id
            drop_cols = [col for col in cols if col != "prop_id" and col != "srch_id"]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                logger.info(f"Encoded cols with mean of CLICK and BOOK probabilities {drop_cols}")
        else:
            # Training mode - create encoder
            encoder = {}
            for col in cols:
                # Get the mean of the target column for each unique value in the categorical column
                mean_click = df.groupby(col)["click_bool"].mean()
                mean_book = df.groupby(col)["booking_bool"].mean()

                # Map the mean to the categorical column
                df[f"{col}_prob_click"] = df[col].map(mean_click)
                df[f"{col}_prob_book"] = df[col].map(mean_book)

                # Add original and encoded columns to encoder dictionary
                encoder[col] = mean_click
                encoder[f"{col}_prob_click"] = mean_click
                encoder[f"{col}_prob_book"] = mean_book

                # Drop the original column
                # Don't drop prop_id or srch_id
                if col != "prop_id" and col != "srch_id":
                    df = df.drop(columns=[col])

            logger.info(f"Encoded cols with mean of CLICK and BOOK probabilities {cols}")
            # Store the encoder
            self.encoder = encoder

        self.df = df
        return df

    def _frequency_encode(self, cols: List[str], drop_original: bool = True) -> pd.DataFrame:
        """
        Frequency encode categorical columns in the dataframe.

        This version avoids block fragmentation by building all new columns
        at once and concatenating them in one go.
        """
        if self.df is None:
            raise ValueError("No dataframe to encode. Please set a dataframe first.")

        # Filter down to only cols that actually exist
        valid_cols = [c for c in cols if c in self.df.columns]
        if not valid_cols:
            logger.info("No columns available for frequency encoding.")
            return self.df

        # Compute all frequency maps
        freq_maps: Dict[str, pd.Series] = {}
        for col in valid_cols:
            freq = self.df[col].value_counts(normalize=True)
            freq_maps[f"{col}_freq"] = self.df[col].map(freq)
            logger.info(f"Prepared freq-encoding for '{col}'")

        # Build a small DataFrame of all new columns
        freq_df = pd.DataFrame(freq_maps, index=self.df.index)

        # Concatenate once
        self.df = pd.concat([self.df, freq_df], axis=1)
        logger.info(f"Added {len(freq_maps)} frequency-encoded columns: {list(freq_maps.keys())}")

        # Drop originals if desired
        if drop_original:
            self.df.drop(columns=valid_cols, inplace=True)
            logger.info(f"Dropped original columns after frequency encoding: {valid_cols}")

        return self.df


    def encode(self,
            prob_encode_cols: List[str] = None,
            one_hot_encode_cols: List[str] = None,
            freq_encode_cols: List[str] = None,
            is_test: bool = False,
            encoder: Dict = None
            ) -> pd.DataFrame:
        """
        Encode categorical columns in the dataframe.

        Returns:
        --------
        pd.DataFrame
            The dataframe with categorical columns encoded
        """
        if self.df is None:
            raise ValueError("No dataframe to encode. Please set a dataframe first.")

        df = self.df.copy()
        logger.info(f"Encoding columns")

        # Probability encode high cardinality columns
        prob_encode_cols = [col for col in prob_encode_cols if col in df.columns]
        if prob_encode_cols and all(col in df.columns for col in ['click_bool', 'booking_bool']):
            self.probability_encode(prob_encode_cols, is_test=is_test, encoder=encoder)
            # df = df.drop(columns=prob_encode_cols) # already dropped in probability_encode
            logger.info(f"Encoded cols with mean of CLICK and BOOK probabilities {prob_encode_cols}")

        # Frequency encode high cardinality columns
        freq_encode_cols = [col for col in freq_encode_cols if col in df.columns]
        if freq_encode_cols:
            self._frequency_encode(freq_encode_cols, drop_original=True)
            logger.info(f"Encoded col with frequency encoding {freq_encode_cols}")
        
        # One-hot encode low cardinality columns
        if one_hot_encode_cols:
            df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True)
            logger.info(f"Encoded cols with one-hot encoding {one_hot_encode_cols}")

        self.df = df
        return df

    def filter_features_based_on_importance(self, feature_importances: List[str], top_n: int = 40, min_importance: float = 0.005) -> pd.DataFrame:
        """
        Filter features based on their importance.
        """

        if self.df is None:
            raise ValueError("No dataframe to filter. Please set a dataframe first.")
        
        if feature_importances is None:
            raise ValueError("No feature importances provided. Please provide feature importances.")
        
        # Filter features based on top_n featres
        top_features = feature_importances[:top_n]
        self.df = self.df[top_features]
        logger.info(f"Filtered features based on top {top_n} features")
        return self.df

    def remove_highly_correlated_features(self, threshold: float = 0.95, to_drop: List[str] = None) -> pd.DataFrame:
        """
        Remove features that are highly correlated with any other feature,
        but only drop one from each correlated pair: the one with higher
        average absolute correlation to all other features.
        """
        if self.df is None:
            raise ValueError("No dataframe to filter. Please set a dataframe first.")
        
        if to_drop:
            logger.info(f"Removing highly correlated features based on previous run")
        
        else:
            logger.info(f"Removing highly correlated features with threshold {threshold}")
            # Select only numeric features (exclude targets)
            df_num = self.df.select_dtypes(include=[np.number]).copy()
            df_num = df_num.drop(columns=[c for c in self.target_cols if c in df_num], errors='ignore')
            
            # Compute correlation matrix and absolute values
            corr = df_num.corr().abs()
            # Create mask for upper triangle
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            
            # Find all pairs exceeding threshold
            to_drop = set()
            for col in upper.columns:
                correlated = upper.index[upper[col] > threshold].tolist()
                for other in correlated:
                    # compare avg abs-correlation
                    mean_col = corr[col].mean()
                    mean_other = corr[other].mean()
                    # mark whichever has the higher mean correlation
                    to_drop.add(col if mean_col > mean_other else other)
        
            # Drop them once
        if to_drop:
            self.df.drop(columns=list(to_drop), inplace=True)
            logger.info(f"Dropped {len(to_drop)} highly correlated features: {sorted(to_drop)}")
        else:
            logger.info("No highly correlated features found")
        self.dropped_cols = list(to_drop)
        return self.df, self.dropped_cols


    def boruta_feature_selection_xgbranker(self,
                                           X: pd.DataFrame,
                                           y: Union[pd.Series, np.ndarray],
                                           group: List[int],
                                           max_iter: int = 100,
                                           random_state: int = 42,
                                           impute_strategy: str = 'median',
                                           clip_inf: bool = True,
                                           inf_replace_value: Optional[float] = None
                                           ) -> List[str]:
        """
        Run Boruta feature selection using XGBRanker (with NDCG@5) 
        as the internal estimator. Requires a group-size list.

        Parameters:
        -----------
        X : DataFrame
            Numeric feature matrix.
        y : array-like
            Target relevance scores.
        group : list of int
            Group sizes (number of rows per query) in the same order as X, y.
        max_iter : int
            Boruta max iterations.
        random_state : int
            Seed for reproducibility.
        impute_strategy : str
            SimpleImputer strategy: 'mean', 'median', etc.
        clip_inf : bool
            Convert ±inf to NaN before imputation.
        inf_replace_value : float|None
            If not None, use this value in place of ±inf.

        Returns:
        --------
        confirmed : List[str]
            List of features confirmed important by Boruta.
        """
        # 1) Replace infinities
        if clip_inf:
            X_clean = X.replace([np.inf, -np.inf], 
                                 inf_replace_value if inf_replace_value is not None else np.nan)
            logger.info(f"Replaced ±inf with {inf_replace_value or 'NaN'}")
        else:
            X_clean = X.copy()

        # 2) Impute missing
        imputer = SimpleImputer(strategy=impute_strategy)
        X_imp = pd.DataFrame(
            imputer.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        logger.info(f"Imputed missing values using strategy='{impute_strategy}'")

        class _XGBRankerWrapper:
            def __init__(self, params, group, random_state):
                self._params = params.copy()
                # Add eval_metric to params during initialization
                self._params['eval_metric'] = 'ndcg@5'
                self._group = group
                self.random_state = random_state
                self.model = None

            def fit(self, X_arr, y_arr):
                self.model = XGBRanker(**self._params)
                self.model.fit(
                    X_arr, y_arr,
                    group=self._group,
                    verbose=False
                )
                return self

            @property
            def feature_importances_(self):
                return self.model.feature_importances_

            def get_params(self, deep=True):
                # BorutaPy uses this to inspect max_depth, n_estimators, etc.
                return self._params.copy()

            def set_params(self, **kwargs):
                # BorutaPy sometimes updates params via set_params
                self._params.update(kwargs)
                return self

        # … previous cleaning & imputation steps …

        # 3) Prepare sklearn-style wrapper around XGBRanker
        xgb_params = {
            'objective': 'rank:ndcg',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'random_state': random_state,
            'n_jobs': -1
        }
        wrapper = _XGBRankerWrapper(xgb_params, group, random_state)

        boruta = BorutaPy(
            estimator=wrapper,
            n_estimators='auto',
            max_iter=max_iter,
            random_state=random_state,
            verbose=2
        )

        # 4) Run Boruta
        boruta.fit(X_imp.values, y)
        confirmed = X.columns[boruta.support_].tolist()

        logger.info(f"Boruta (XGBRanker) confirmed {len(confirmed)} features: {confirmed}")
        return confirmed


    def run_pipeline(self, is_test: bool = False, encoder: Dict = None, save: bool = False, name: str = "train", dropped_columns: List[str] = None) -> pd.DataFrame:
        """
        Run the entire preprocessing pipeline.

        Parameters:
        -----------
        is_test : bool, optional
            Whether this is test data (to prevent data leakage)
        encoder : Dict, optional
            Pre-existing encoder for test data

        Returns:
        --------
        pd.DataFrame
            The preprocessed dataframe
        """
        if self.df is None:
            raise ValueError("No dataframe to preprocess. Please set a dataframe first.")


        logger.info(f"Running preprocessing pipeline for {name}")

        # Clean data
        self.clean_data(is_test=is_test, name=name)


        # Impute data
        self.impute_data(
                    sum_imp=None,
                    mean_imp=["prop_review_score", "prop_starrating"],
                    mode_imp=None,
                    interp_imp=None,
                    regression_imputation=[["prop_location_score1", "prop_review_score"]]
                    )

        #### FEATURES
        # Add custom features
        self._add_datetime_features()
        self._add_vacation_features()

        self._add_desirability_features()
        self._add_price_per_person_per_room()
        self._add_price_surprise()
        self._add_new_customer_flag()
        self._add_domestic_trip_flag()
        self._add_seasonality_features(relative_to="srch_destination_id")

        # Define feature methods for relative features
        feature_methods = {
            "price_usd": ["pct_rank", "minmax", "zscore", "rank"],
            "prop_review_score": ["pct_rank", "zscore", "rank"],
            "prop_location_score1": ["pct_rank", "minmax", "zscore", "rank"],
            "prop_starrating": ["pct_rank", "minmax", "zscore", "rank"],
            "prop_location_score2": ["pct_rank", "minmax", "zscore", "rank"],
            "prop_log_historical_price": ["pct_rank", "minmax", "zscore", "rank"],
            "comp5_rate_percent_diff": ["zscore"],
            "comp8_rate": ["pct_rank", "zscore"],
            "srch_query_affinity_score": ["pct_rank", "minmax", "zscore", "rank"],
            "log_orig_destination_distance": ["pct_rank", "minmax", "zscore", "rank"],
            "visitor_hist_adr_usd": ["pct_rank", "minmax", "zscore", "rank"],
            "visitor_hist_starrating": ["pct_rank", "minmax", "zscore", "rank"],
        }
        # add all feature methods to all columns
        # feature_methods = {col: ["pct_rank", "minmax", "zscore", "rank"] for col in self.df.columns}

        self.add_relative_features(feature_methods)


        # PROP ID STATS

                # Define top features for property ID statistics
        top_means = [
            'price_usd', 'random_bool', 'prop_location_score2', 'srch_query_affinity_score',
            'srch_length_of_stay', 'prop_log_historical_price', 'promotion_flag', 'comp5_rate',
            'orig_destination_distance', 'srch_room_count', 'srch_booking_window',
            'srch_saturday_night_bool', 'comp3_inv', 'srch_children_count', 'comp8_rate',
            'comp8_inv', 'comp3_rate_percent_diff', 'comp8_rate_percent_diff',
            'comp5_rate_percent_diff', 'comp2_rate_percent_diff'
        ]

        top_stds = [
            'price_usd', 'random_bool', 'prop_location_score2', 'srch_query_affinity_score',
            'comp8_inv', 'srch_room_count', 'prop_log_historical_price', 'promotion_flag',
            'srch_length_of_stay', 'visitor_location_country_id', 'comp3_rate_percent_diff',
            'comp8_rate', 'comp5_rate_percent_diff', 'comp8_rate_percent_diff', 'comp3_rate',
            'orig_destination_distance', 'month', 'comp2_rate_percent_diff', 'comp5_rate',
            'srch_children_count'
        ]

        top_medians = [
            'price_usd', 'prop_log_historical_price', 'prop_location_score2',
            'srch_query_affinity_score', 'orig_destination_distance', 'srch_booking_window',
            'comp3_rate_percent_diff', 'comp5_rate_percent_diff', 'comp8_rate_percent_diff',
            'comp2_rate_percent_diff', 'visitor_hist_starrating', 'srch_length_of_stay',
            'visitor_hist_adr_usd', 'comp5_rate', 'promotion_flag'
        ]

        # Add property ID statistics to all columns
        all_cols = self.df.select_dtypes(include=[np.number]).columns.tolist() # date_time and vacation_date are not needed anymore

        # self.add_statistics_per_group(all_cols,
        #                             stat_func='median',
        #                             with_respect_to="prop_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='mean',
        #                             with_respect_to="prop_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='std',
        #                             with_respect_to="prop_id")
        # self.add_statistics_per_group(top_means,
        #                             stat_func='median',
        #                             with_respect_to="srch_destination_id")
        # self.add_statistics_per_group(top_stds,
        #                             stat_func='mean',
        #                             with_respect_to="srch_destination_id")
        # self.add_statistics_per_group(top_medians,
        #                             stat_func='std',
        #                             with_respect_to="srch_destination_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='median',
        #                             with_respect_to="site_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='mean',
        #                             with_respect_to="site_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='std',
        #                             with_respect_to="site_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='median',
        #                             with_respect_to="visitor_location_country_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='mean',
        #                             with_respect_to="visitor_location_country_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='std',
        #                             with_respect_to="visitor_location_country_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='median',
        #                             with_respect_to="prop_country_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='mean',
        #                             with_respect_to="prop_country_id")
        # self.add_statistics_per_group(all_cols,
        #                             stat_func='std',
        #                             with_respect_to="prop_country_id")

        self.add_statistics_per_group(top_means,
                                    stat_func='median',
                                    with_respect_to="prop_id")
        self.add_statistics_per_group(top_stds,
                                    stat_func='mean',
                                    with_respect_to="prop_id")
        self.add_statistics_per_group(top_medians,
                                    stat_func='std',
                                    with_respect_to="prop_id")
        #TODO do with respect to srch_destination_id, site_id ....


        #### ENCODINGS

        unique_counts = self.df.nunique()
        binary_cols = unique_counts[unique_counts == 2].index.tolist()
        low_cardinality_cols = unique_counts[(unique_counts > 2) & (unique_counts < 10)].index.tolist()
        high_cardinality_cols = unique_counts[unique_counts > 10].index.tolist()

        logger.info(f"Binary categorical columns: {binary_cols}")
        logger.info(f"Low cardinality categorical columns: {low_cardinality_cols}")
        logger.info(f"High cardinality categorical columns: {high_cardinality_cols}")

        categorical_columns = [
            "site_id",
            "visitor_location_country_id",
            "prop_country_id",
            "prop_id",
            "srch_destination_id",
        ]

        # one-hot encode:
        one_hot_encode_cols = ["month", "weekday", "vacation_day_of_week"]

        self.encode(
                    prob_encode_cols=[],
                    one_hot_encode_cols=one_hot_encode_cols,
                    freq_encode_cols=categorical_columns,
                    is_test=is_test,
                    encoder=encoder  # for probability encoding
                    )


        # remove highly correlated features

        # remove non-numeric columns
        self.df = self.df.select_dtypes(include=[np.number]) # date_time and vacation_date are not needed anymore


        if save:
            self.df.to_csv(f"data/processed_{name}.csv", index=False)
            logger.info(f"Saved preprocessed data to data/processed_{name}.csv")

        # if self.drop_cols doesnt exist, set it to none
        if not hasattr(self, 'dropped_cols'):
            self.dropped_cols = None
        # Return the preprocessed dataframe and encoder if available
        if not is_test:
            return self.df, self.encoder, self.dropped_cols
        else:
            return self.df
