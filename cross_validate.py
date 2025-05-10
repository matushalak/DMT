import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from itertools import product


############### clean input data ###############
def train_clean(df):
    # Load the data

    # Add relevant information from date-time, and remove the original date-time column
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['month'] = df['date_time'].dt.month
    df['weekday'] = df['date_time'].dt.weekday
    df.drop(columns=['date_time'], inplace=True)

    # log transform orig_destination_distance
    df["log_orig_destination_distance"] = np.log1p(df["orig_destination_distance"])

    # Create label for training set: 5 for booking, 1 for click, 0 for nothing
    df["label"] = df["booking_bool"] * 5 + (df["click_bool"] & ~df["booking_bool"]) * 1

    # Filter unrealistic gross_bookings_usd (only where it's not missing)
    df = df[(df["gross_bookings_usd"].isna()) | ((df["gross_bookings_usd"] >= 10) & (df["gross_bookings_usd"] <= 2000))]

    # Filter unrealistic prices (price_usd)
    df = df[(df["price_usd"] >= 10) & (df["price_usd"] <= 1000)]

    # List of competitor rate percent diff columns
    comp_rate_cols = [
        "comp1_rate_percent_diff",
        "comp2_rate_percent_diff",
        "comp3_rate_percent_diff",
        "comp4_rate_percent_diff",
        "comp5_rate_percent_diff",
        "comp6_rate_percent_diff",
        "comp7_rate_percent_diff",
        "comp8_rate_percent_diff",
    ]

    # Loop to apply filtering to each competitor rate column
    for col in comp_rate_cols:
        # Cap unrealistic values but keep NaNs
            df = df[(df[col].isna()) | (df[col] <= 50)]

    return df

############### clean input data ###############
def val_clean(df):

    # Add relevant information from date-time, and remove the original date-time column
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['month'] = df['date_time'].dt.month
    df['weekday'] = df['date_time'].dt.weekday
    df.drop(columns=['date_time'], inplace=True)

    # log transform orig_destination_distance
    df["log_orig_destination_distance"] = np.log1p(df["orig_destination_distance"])

    # df["label"] = df["booking_bool"] * 5 + (df["click_bool"] & ~df["booking_bool"]) * 1

    # # Filter unrealistic gross_bookings_usd (only where it's not missing)
    # df = df[(df["gross_bookings_usd"].isna()) | ((df["gross_bookings_usd"] >= 10) & (df["gross_bookings_usd"] <= 2000))]

    # Filter unrealistic prices (price_usd)
    df = df[(df["price_usd"] >= 10) & (df["price_usd"] <= 1000)]

    return df


##################### Adding features #####################
# add found features to dataframe
def add_found_features(df, feature_specs, group_key="srch_id"):
    """
    Ensure that all desired relative features (e.g., rank/zscore/etc.) exist in df.
    Only compute missing ones.
    
    Parameters:
    - df: your DataFrame
    - feature_specs: dict mapping base feature -> list of methods (e.g. ['pct_rank', 'zscore'])
    - group_key: key to group by (usually 'srch_id')
    
    Returns:
    - df with new features added
    """
    df = df.copy()
    
    for base_feat, methods in feature_specs.items():
        if base_feat not in df.columns:
            print(f"Skipping '{base_feat}' (not in DataFrame)")
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
                print(f"Unknown method '{method}' for feature '{base_feat}'")

    return df

# found rank feature methods
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
    "orig_destination_distance": ["pct_rank", "minmax", "zscore", "rank"],
    "visitor_hist_adr_usd": ["pct_rank", "minmax", "zscore", "rank"],
    "visitor_hist_starrating": ["pct_rank", "minmax", "zscore", "rank"],
}

def filter_final_features(df: pd.DataFrame) -> pd.DataFrame:
    allowed_columns = [
        'srch_id', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating',
        'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating',
        'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
        'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
        'srch_length_of_stay', 'srch_booking_window', 'srch_query_affinity_score',
        'orig_destination_distance', 'random_bool', 'comp8_rate_percent_diff',
        'click_bool', 'gross_bookings_usd', 'booking_bool', 'label',
        'price_usd_pct_rank', 'price_usd_minmax', 'price_usd_zscore', 'price_usd_rank',
        'prop_review_score_pct_rank', 'prop_review_score_zscore', 'prop_review_score_rank',
        'prop_location_score1_pct_rank', 'prop_location_score1_minmax',
        'prop_location_score1_zscore', 'prop_location_score1_rank',
        'prop_starrating_pct_rank', 'prop_starrating_minmax', 'prop_starrating_zscore',
        'prop_starrating_rank', 'prop_location_score2_pct_rank', 'prop_location_score2_minmax',
        'prop_location_score2_zscore', 'prop_location_score2_rank',
        'prop_log_historical_price_pct_rank', 'prop_log_historical_price_minmax',
        'prop_log_historical_price_zscore', 'prop_log_historical_price_rank',
        'comp5_rate_percent_diff_zscore', 'comp8_rate_pct_rank', 'comp8_rate_zscore',
        'srch_query_affinity_score_pct_rank', 'srch_query_affinity_score_minmax',
        'srch_query_affinity_score_zscore', 'srch_query_affinity_score_rank',
        'orig_destination_distance_pct_rank', 'orig_destination_distance_minmax',
        'orig_destination_distance_zscore', 'orig_destination_distance_rank',
        'visitor_hist_adr_usd_pct_rank', 'visitor_hist_adr_usd_minmax',
        'visitor_hist_adr_usd_zscore', 'visitor_hist_adr_usd_rank',
        'visitor_hist_starrating_pct_rank', 'visitor_hist_starrating_minmax',
        'visitor_hist_starrating_zscore', 'visitor_hist_starrating_rank', "log_orig_destination_distance"
    ]
    
    return df[[col for col in df.columns if col in allowed_columns]]


def add_prop_id_statistics(df, selected_features=None, stat_func='median'):
    """
    Adds per-prop_id aggregated statistics (mean, median, or std) of numeric features efficiently.

    Parameters:
    - df: pandas DataFrame
    - stat_func: 'mean', 'median', or 'std' (default is 'mean')
    - selected_features: List of features to compute stats on (default is all numeric except exclusions)
    """
    df = df.copy()

    # Select numeric columns to aggregate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['srch_id', 'prop_id', 'position', 'click_bool', 'booking_bool', 'gross_bookings_usd', 'label']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Limit to selected features if provided
    if selected_features:
        numeric_cols = [col for col in numeric_cols if col in selected_features]

    # Compute statistics per prop_id
    prop_stats = df.groupby('prop_id')[numeric_cols].agg(stat_func)

    # Rename columns to reflect the stat used
    prop_stats.columns = [f"{col}_{stat_func}" for col in prop_stats.columns]

    # Map each column back to df using prop_id lookup
    for col in prop_stats.columns:
        df[col] = df['prop_id'].map(prop_stats[col])

    return df

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




######################## Training and Testing ########################
# training model with own test data to find the best features
def split_and_test():
    # import data
    df = pd.read_csv(f"data/training_set_VU_DM.csv")

    # Split into train/val
    all_search_ids = df["srch_id"].unique()
    train_ids, val_ids = train_test_split(all_search_ids, test_size=0.2, random_state=42)

    train_df = df[df["srch_id"].isin(train_ids)].copy()
    val_df   = df[df["srch_id"].isin(val_ids)].copy()
    train_df = train_df.sort_values(by="srch_id")
    val_df   = val_df.sort_values(by="srch_id")

    # clean datasets differently
    train_df = train_clean(train_df)
    val_df   = val_clean(val_df)

    # add features
    train_df = add_found_features(train_df, feature_methods)
    val_df = add_found_features(val_df, feature_methods)

    train_df = filter_final_features(train_df)              
    train_df = add_prop_id_statistics(train_df, top_means, 'mean') 
    train_df = add_prop_id_statistics(train_df, top_stds, 'std')
    train_df = add_prop_id_statistics(train_df, top_medians, 'median')

    val_df = filter_final_features(val_df)              
    val_df = add_prop_id_statistics(val_df, top_means, 'mean') 
    val_df = add_prop_id_statistics(val_df, top_stds, 'std')
    val_df = add_prop_id_statistics(val_df, top_medians, 'median')

    # Create group sizes
    group_sizes_train = train_df.groupby("srch_id").size().tolist()
    group_sizes_val = val_df.groupby("srch_id").size().tolist()

    # Drop unwanted features
    to_drop = ["srch_id", "label", "position", "click_bool", "booking_bool", "gross_bookings_usd"]
    existing_cols = [col for col in to_drop if col in val_df.columns]

    # Final train/val sets
    train_x = train_df.drop(columns=existing_cols)
    val_x = val_df.drop(columns=existing_cols)
    train_y = train_df["label"]
    val_y = val_df["label"]

    # Train LightGBM ranker
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        n_estimators=125,
        num_leaves=80,
        learning_rate=0.1,
        min_child_samples=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X=train_x,
        y=train_y,
        group=group_sizes_train,
        eval_set=[(train_x, train_y), (val_x, val_y)],
        eval_group=[group_sizes_train, group_sizes_val],
        eval_at=[5],
        eval_metric="ndcg",
        eval_names=["train", "val"],
    )

    # Plot NDCG@5 over rounds
    evals_result = model.evals_result_

    plt.figure(figsize=(10, 5))
    plt.plot(evals_result["train"]["ndcg@5"], label="Train NDCG@5")
    plt.plot(evals_result["val"]["ndcg@5"], label="Validation NDCG@5")
    plt.xlabel("Boosting Round")
    plt.ylabel("NDCG@5")
    plt.title("NDCG@5 over Boosting Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print final score
    final_val_score = evals_result["val"]["ndcg@5"][-1]
    print(f"Final NDCG@5 on validation set: {final_val_score:.5f}")

    return model





# train and predict on training data with new test data
def model_trainer(df):
    # Drop unwanted features
    drop_cols = ["srch_id", "position", "click_bool", "booking_bool", "gross_bookings_usd", "label"]
    existing_cols = [col for col in drop_cols if col in df.columns]

    # Final training set
    train_x = df.drop(columns=existing_cols)
    train_y = df["label"]
    group_sizes_train = df.groupby("srch_id").size().tolist()

    # Train model on all training data
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        n_estimators=200,
        num_leaves=100,
        learning_rate=0.1,
        min_child_samples=10,
        random_state=42
    )

    model.fit(
        X=train_x,
        y=train_y,
        group=group_sizes_train
    )

    return model


# ############# feature importance #############
def plot_feature_importance(model, top_n=40):
    """
    Plots and prints the top_n most important features of a trained LGBMRanker model.

    Parameters:
    - model: Trained LGBMRanker model
    - top_n: Number of top features to display (default 40)
    """
    importances = model.booster_.feature_importance(importance_type='gain')
    feature_names = model.booster_.feature_name()

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp["feature"], feat_imp["importance"], color="skyblue")
    plt.xlabel("Feature importance (gain)")
    plt.title(f"Top {top_n} Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Print as list
    top_features_list = feat_imp["feature"].tolist()
    print(f"📝 Top {top_n} Features:\n{top_features_list}")

    return top_features_list


# predict ranks of the test data
def get_predictions(model: lgb.LGBMRanker, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts relevance scores for the test set using the trained LGBMRanker model,
    ranks hotels per search, and returns the formatted submission dataframe.
    """
    # Drop srch_id/prop_id for prediction
    X_test = test_df.drop(columns=['srch_id'], errors='ignore')

    # Predict
    predicted_relevance = model.predict(X_test)
    test_df = test_df.copy()
    test_df["predicted_relevance"] = predicted_relevance

    # Sort by srch_id and predicted score (descending)
    ranked = test_df.sort_values(by=["srch_id", "predicted_relevance"], ascending=[True, False]).reset_index(drop=True)
    return ranked[["srch_id", "prop_id"]]



################ process and run model on test set for submission ################
if __name__ == '__main__':
    # Prepare and process training data
    train_df = pd.read_csv(f"data/training_set_VU_DM.csv")
    train_df = train_clean(train_df)
    train_df = add_found_features(train_df, feature_methods)
    train_df = filter_final_features(train_df)
    train_df = add_prop_id_statistics(train_df, top_means, 'mean')
    train_df = add_prop_id_statistics(train_df, top_stds, 'std')
    train_df = add_prop_id_statistics(train_df, top_medians, 'median')

    # Train model
    model = model_trainer(train_df)

    # Plot feature importance
    plot_feature_importance(model)

    # Prepare and process test data
    test_df = pd.read_csv(f"data/test_set_VU_DM.csv")
    test_df = val_clean(test_df)
    test_df = add_found_features(test_df, feature_methods)
    test_df = filter_final_features(test_df)
    test_df = add_prop_id_statistics(test_df, top_means, 'mean')
    test_df = add_prop_id_statistics(test_df, top_stds, 'std')
    test_df = add_prop_id_statistics(test_df, top_medians, 'median')

    # Get predictions
    TEST_pred = get_predictions(model, test_df)
    TEST_pred.to_csv('VU-DM-2025-Group-100.csv', index=False)
    print("✅ Submission file saved as 'VU-DM-2025-Group-100.csv'")




# # for testing
# if __name__ == '__main__':
#     final_model = split_and_test()
#     plot_feature_importance(final_model)






#################### Hyperparameter tuning with grid search ####################    
# def grid_search_lgbm_ranker(train_df, param_grid):
#     """
#     Performs a grid search on LGBMRanker hyperparameters.

#     Parameters:
#     - train_df: processed training DataFrame
#     - param_grid: dictionary with lists of hyperparameters to try

#     Returns:
#     - List of (params, final_val_ndcg) tuples sorted by NDCG
#     """
#     keys, values = zip(*param_grid.items())
#     all_combinations = [dict(zip(keys, v)) for v in product(*values)]
#     results = []

#     for i, params in enumerate(all_combinations, 1):
#         print(f"\n🚀 Testing combination {i}/{len(all_combinations)}: {params}")
        
#         model = lgb.LGBMRanker(
#             objective="lambdarank",
#             metric="ndcg",
#             importance_type="gain",
#             **params
#         )

#         all_search_ids = train_df["srch_id"].unique()
#         train_ids, val_ids = train_test_split(all_search_ids, test_size=0.2, random_state=42)
#         train_set = train_df[train_df["srch_id"].isin(train_ids)]
#         val_set = train_df[train_df["srch_id"].isin(val_ids)]

#         group_train = train_set.groupby("srch_id").size().tolist()
#         group_val = val_set.groupby("srch_id").size().tolist()

#         X_train = train_set.drop(columns=["srch_id", "label", "position", "click_bool", "booking_bool", "gross_bookings_usd"])
#         y_train = train_set["label"]
#         X_val = val_set.drop(columns=["srch_id", "label", "position", "click_bool", "booking_bool", "gross_bookings_usd"])
#         y_val = val_set["label"]

#         model.fit(
#             X_train, y_train,
#             group=group_train,
#             eval_set=[(X_val, y_val)],
#             eval_group=[group_val],
#             eval_at=[5],
#             eval_metric="ndcg",
#             eval_names=["val"],
#         )

#         final_val_ndcg = model.evals_result_["val"]["ndcg@5"][-1]
#         print(f"✅ Final NDCG@5: {final_val_ndcg:.5f}")
#         results.append((params, final_val_ndcg))

#     results.sort(key=lambda x: x[1], reverse=True)
#     return results




