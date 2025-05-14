import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor
from modeler import Modeler
import os


def load_data_and_split(force_raw: bool = False, save_processed_data: bool = False):
        # ---------- TRAINING DATA ----------
    if os.path.exists("data/processed_train.csv") and not force_raw and os.path.exists("data/processed_val.csv"):  # val has to exist cuz of dropped columns
        print("Loading saved processed training data...")
        processed_train_df = pd.read_csv("data/processed_train.csv")
        encoder = None  # placeholder if you want to return it
    else:
        print("Loading raw training data...")
        train_df = pd.read_csv("data/training_set_VU_DM.csv")

        print("Splitting data into train and validation sets...")
        all_search_ids = train_df["srch_id"].unique()
        train_ids, val_ids = train_test_split(all_search_ids, test_size=0.2, random_state=42)

        train_data = train_df[train_df["srch_id"].isin(train_ids)].copy()
        print(f"Training data shape: {train_data.shape}")

        print("Preprocessing training data...")
        train_preprocessor = Preprocessor(train_data)
        processed_train_df, encoder, dropped_columns = train_preprocessor.run_pipeline(
            is_test=False, save=save_processed_data, name="train", dropped_columns=None
        )
        _, dropped_columns = train_preprocessor.remove_highly_correlated_features(threshold=0.90, to_drop=None)


    # ---------- VALIDATION DATA ----------
    if os.path.exists("data/processed_val.csv") and not force_raw:
        print("Loading saved processed validation data...")
        processed_val_df = pd.read_csv("data/processed_val.csv")
    else:
        if 'train_df' not in locals():
            print("Loading raw training data for validation split...")
            train_df = pd.read_csv("data/training_set_VU_DM.csv")
        all_search_ids = train_df["srch_id"].unique()
        _, val_ids = train_test_split(all_search_ids, test_size=0.2, random_state=42)

        val_data = train_df[train_df["srch_id"].isin(val_ids)].copy()
        print(f"Validation data shape: {val_data.shape}")

        print("Preprocessing validation data...")
        val_preprocessor = Preprocessor(val_data)
        processed_val_df = val_preprocessor.run_pipeline(
            is_test=True, encoder=encoder, save=save_processed_data, name="val", dropped_columns=dropped_columns
        )
        val_preprocessor.remove_highly_correlated_features(threshold=0.90, to_drop=dropped_columns)

    # remove non numeric columns
    processed_train_df = processed_train_df.select_dtypes(include=[np.number])
    processed_val_df = processed_val_df.select_dtypes(include=[np.number])


    # check if all columns are the same apart from target cols
    target_cols = ['position', 'click_bool', 'booking_bool',
                        'gross_bookings_usd', 'target']
    train_cols = processed_train_df.columns.to_list()
    val_cols = processed_val_df.columns.to_list()
    
    for col in train_cols:
        if col not in val_cols and col not in target_cols:
            raise ValueError(f"Column {col} is in train but not in val")

    # Initialize modeler with processed train and validation data
    # check if target in train and val
    if "target" not in processed_train_df.columns:
        raise ValueError("Target not in train data")
    if "target" not in processed_val_df.columns:
        raise ValueError("Target not in val data")

    return processed_train_df, processed_val_df, encoder


def train_data(save_processed_data: bool = False):

    processed_train_df, processed_val_df, encoder = load_data_and_split(force_raw=False, save_processed_data=save_processed_data)

    # features from feature selection
    # features = ['site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'random_bool', 'comp2_rate_percent_diff', 'comp5_rate_percent_diff', 'log_orig_destination_distance', 'prop_desirability', 'price_per_person_per_room', 'price_surprise', 'domestic_trip', 'rel_hotel_price_season_aware', 'rel_hotel_price_season_agnostic', 'price_usd_minmax', 'price_usd_rank', 'prop_review_score_zscore', 'prop_review_score_rank', 'prop_location_score1_pct_rank', 'prop_location_score1_zscore', 'prop_location_score1_rank', 'prop_starrating_minmax', 'prop_starrating_zscore', 'prop_starrating_rank', 'prop_location_score2_minmax', 'prop_location_score2_zscore', 'prop_location_score2_rank', 'prop_log_historical_price_pct_rank', 'prop_log_historical_price_minmax', 'prop_log_historical_price_rank', 'comp5_rate_percent_diff_zscore', 'srch_query_affinity_score_pct_rank', 'srch_query_affinity_score_minmax', 'srch_query_affinity_score_zscore', 'srch_query_affinity_score_rank', 'log_orig_destination_distance_minmax', 'log_orig_destination_distance_zscore', 'log_orig_destination_distance_rank', 'prop_location_score2_median', 'prop_log_historical_price_median', 'price_usd_median', 'promotion_flag_median', 'srch_length_of_stay_median', 'srch_booking_window_median', 'srch_saturday_night_bool_median', 'srch_query_affinity_score_median', 'comp5_rate_median', 'comp8_rate_percent_diff_median', 'visitor_location_country_id_mean', 'prop_log_historical_price_mean', 'srch_children_count_mean', 'srch_room_count_mean', 'random_bool_mean', 'comp5_rate_mean', 'comp8_inv_mean', 'prop_location_score2_std', 'prop_log_historical_price_std', 'price_usd_std', 'promotion_flag_std', 'srch_length_of_stay_std', 'srch_query_affinity_score_std', 'comp2_rate_percent_diff_std', 'comp3_rate_percent_diff_std']

    # features = features + ["srch_id", "target"]
    # processed_train_df = processed_train_df[features]
    # processed_val_df = processed_val_df[features]
    modeler = Modeler(train_df=processed_train_df, val_df=processed_val_df)

    print("Training model...")
    model = modeler.train_model()

    # Plot training metrics
    modeler.plot_training_metrics()

    top_features = modeler.get_feature_importance(top_n=20)
    print(f"Top 20 features: {top_features}")


def select_important_features(save_processed_data: bool = False):
    processed_df = pd.read_csv("data/processed_train_whole.csv")
    preprocessor = Preprocessor()
    preprocessor.set_dataframe(processed_df)
    # Boruta feature selection
    # Suppose df_proc has 'srch_id', 'prop_id', 'target' + your features
    to_drop = preprocessor.target_cols + ['srch_id']
    X = processed_df.drop(columns=to_drop)
    y = processed_df['target'].values

    # Build group sizes in order
    group_sizes = processed_df.groupby("srch_id").size().tolist()

    confirmed_feats = preprocessor.boruta_feature_selection_xgbranker(
        X, y, group=group_sizes, max_iter=50
    )
    print(f"Confirmed features: {confirmed_feats}")




def test_data(save_processed_data: bool = False):
    ##### PROCESSED FULL TRAINING DATA ###########
    if os.path.exists("data/processed_train_whole.csv"):
        print("Loading saved processed full training data...")
        processed_whole_train_df = pd.read_csv("data/processed_train_whole.csv")
    else:
        print("Processing full training data for encoder fitting...")
        train_df = pd.read_csv("data/training_set_VU_DM.csv")
        whole_train_preprocessor = Preprocessor(train_df)
        processed_whole_train_df, encoder = whole_train_preprocessor.run_pipeline(
            is_test=False, save=save_processed_data, name="train_whole"
        )

    ##### PROCESSED TEST DATA ###########
    if os.path.exists("data/processed_test.csv"):
        print("Loading saved processed test data...")
        processed_test_df = pd.read_csv("data/processed_test.csv")
    else:
        print("Processing raw test data...")
        test_df = pd.read_csv("data/test_set_VU_DM.csv")

        # If encoder hasn't been fitted above, load and process training data here
        if encoder is None:
            print("Loading training data to fit encoder for test preprocessing...")
            train_df = pd.read_csv("data/training_set_VU_DM.csv")
            whole_train_preprocessor = Preprocessor(train_df)
            _, encoder = whole_train_preprocessor.run_pipeline(
                is_test=False, save=save_processed_data, name="train_whole"
            )

        test_preprocessor = Preprocessor(test_df)
        processed_test_df = test_preprocessor.run_pipeline(
            is_test=True, encoder=encoder, save=save_processed_data, name="test"
        )

    # remove non numeric columns
    processed_whole_train_df = processed_whole_train_df.select_dtypes(include=[np.number])
    processed_test_df = processed_test_df.select_dtypes(include=[np.number])

    # For final predictions, train on the full dataset (train+val)
    print("Training final model on full dataset...")
    modeler = Modeler()
    modeler.train_full_model(full_df=processed_whole_train_df)

    # Make predictions
    print("Making predictions...")
    predictions = modeler.predict(processed_test_df)

    # Sort and save predictions
    print("Sorting and saving predictions...")
    ranked = modeler.sort_and_save_predictions(predictions, output_file="predictions.csv")

    print("Done! Predictions saved to predictions.csv")


def hyperparameter_tuning_example(save_processed_data: bool = False):
    # Example of hyperparameter tuning with proper train/validation split

    processed_train_df, processed_val_df, encoder = load_data_and_split()

    features = ['site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'random_bool', 'comp2_rate_percent_diff', 'comp5_rate_percent_diff', 'log_orig_destination_distance', 'prop_desirability', 'price_per_person_per_room', 'price_surprise', 'domestic_trip', 'rel_hotel_price_season_aware', 'rel_hotel_price_season_agnostic', 'price_usd_minmax', 'price_usd_rank', 'prop_review_score_zscore', 'prop_review_score_rank', 'prop_location_score1_pct_rank', 'prop_location_score1_zscore', 'prop_location_score1_rank', 'prop_starrating_minmax', 'prop_starrating_zscore', 'prop_starrating_rank', 'prop_location_score2_minmax', 'prop_location_score2_zscore', 'prop_location_score2_rank', 'prop_log_historical_price_pct_rank', 'prop_log_historical_price_minmax', 'prop_log_historical_price_rank', 'comp5_rate_percent_diff_zscore', 'srch_query_affinity_score_pct_rank', 'srch_query_affinity_score_minmax', 'srch_query_affinity_score_zscore', 'srch_query_affinity_score_rank', 'log_orig_destination_distance_minmax', 'log_orig_destination_distance_zscore', 'log_orig_destination_distance_rank', 'prop_location_score2_median', 'prop_log_historical_price_median', 'price_usd_median', 'promotion_flag_median', 'srch_length_of_stay_median', 'srch_booking_window_median', 'srch_saturday_night_bool_median', 'srch_query_affinity_score_median', 'comp5_rate_median', 'comp8_rate_percent_diff_median', 'visitor_location_country_id_mean', 'prop_log_historical_price_mean', 'srch_children_count_mean', 'srch_room_count_mean', 'random_bool_mean', 'comp5_rate_mean', 'comp8_inv_mean', 'prop_location_score2_std', 'prop_log_historical_price_std', 'price_usd_std', 'promotion_flag_std', 'srch_length_of_stay_std', 'srch_query_affinity_score_std', 'comp2_rate_percent_diff_std', 'comp3_rate_percent_diff_std']
    features = features + ["srch_id", "target"]
    processed_train_df = processed_train_df[features]
    processed_val_df = processed_val_df[features]
    

    # Initialize modeler with processed train and validation data
    modeler = Modeler(train_df=processed_train_df, val_df=processed_val_df)

    # Tune hyperparameters
    print("Tuning hyperparameters...")
    best_params = modeler.hyperparameter_tuning(n_trials=10, visualize=True, dashboard=True)
    print(f"Best parameters: {best_params}")

    # Train model with best parameters
    print("Training model with best parameters...")
    model = modeler.train_model()

    # Plot training metrics
    modeler.plot_training_metrics()

    # Get feature importance
    top_features = modeler.get_feature_importance(top_n=50)
    print(f"Top 20 features: {top_features}")

if __name__ == "__main__":
    train_data(save_processed_data=True)
    # test_data(save_processed_data=True)
    # hyperparameter_tuning_example(save_processed_data=False)
    # select_important_features(save_processed_data=True)
