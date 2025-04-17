from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
# def normalize_data_and_split(
#     df: pd.DataFrame,
#     features: List[str],
#     target_col: str,
#     id_col: str,
#     timestamp_col: Optional[str] = None,  # Timestamp column
#     date_col: Optional[str] = None,     # Explicitly use date column
#     per_participant_normalization: bool = False,
#     scaler_type: str = "StandardScaler",
#     test_size: float = 0.2,
#     val_size: float = 0.1,
#     random_state: int = 42
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
#            pd.Series, pd.Series, pd.Series, Dict, Dict]:
#     """
#     Splits the dataframe into training, validation, and test sets and normalizes only the features.
#     The scaler(s) are fit on the training set and then applied to the validation and test sets.
#     Preserves metadata needed for timeseries plotting.
#     IMPORTANT: Ensures all measurements from the same date stay in the same split to prevent data leakage.

#     Args:
#         df: pandas DataFrame containing all the data.
#         features: List of column names to be used as features.
#         target_col: Column name for the target variable.
#         id_col: Column name used to uniquely identify participants (will not be normalized).
#         timestamp_col: Optional column name for timestamps (will be preserved for plotting).
#         date_col: Column name containing date information. Required to ensure data from the same
#                  date stays together to prevent data leakage.
#         per_participant_normalization: If True, performs normalization separately for each participant.
#         scaler_type: Type of scaler to use ("StandardScaler" or "MinMaxScaler").
#         test_size: Fraction of the whole data to hold out for testing.
#         val_size: Fraction of the non-test data to hold out for validation.
#         random_state: Random seed for reproducibility.

#     Returns:
#         A tuple containing:
#           - X_train: Normalized features for the training set.
#           - X_val: Normalized features for the validation set.
#           - X_test: Normalized features for the test set.
#           - y_train: Target variable for training.
#           - y_val: Target variable for validation.
#           - y_test: Target variable for test.
#           - scalers: Dictionary containing the scaler(s) used.
#           - metadata: Dictionary containing information for reconstruction and plotting.
#     """
#     # Initialize scalers dict and metadata dict
#     scalers = {}
#     metadata = {
#         'original_index': df.index.copy(),
#         'id_col': id_col
#     }
    
#     # Store timestamp information if provided
#     if timestamp_col:
#         metadata['timestamp_col'] = timestamp_col
#         metadata['timestamps'] = df[timestamp_col].copy()
    
#     # Helper function to return a new scaler instance based on scaler_type
#     def get_new_scaler():
#         if scaler_type == "StandardScaler":
#             return StandardScaler()
#         elif scaler_type == "MinMaxScaler":
#             return MinMaxScaler()
#         else:
#             raise ValueError(f"Unknown scaler type: {scaler_type}")

#     # Create a copy of the original dataframe with indices to track original positions
#     df_with_index = df.copy()
#     df_with_index['_original_index'] = np.arange(len(df))
    
#     # Handle date information for grouping by day
#     if date_col is None and timestamp_col is not None:
#         # Extract date from timestamp if no explicit date column is provided
#         try:
#             # Check if timestamp is already a datetime type
#             if not pd.api.types.is_datetime64_any_dtype(df_with_index[timestamp_col]):
#                 df_with_index['_date'] = pd.to_datetime(df_with_index[timestamp_col]).dt.date
#             else:
#                 df_with_index['_date'] = df_with_index[timestamp_col].dt.date
#             date_col = '_date'
#         except:
#             raise ValueError("Could not extract date from timestamp. Please provide a valid date_col to prevent data leakage.")
    
#     if date_col is None:
#         raise ValueError("date_col must be provided to prevent data leakage across dates.")
    
#     # Separate features (X) and target (y)
#     X = df_with_index[features + ['_original_index']].copy()
#     X[date_col] = df_with_index[date_col]
#     y = df_with_index[target_col]
    
#     # Columns to normalize (exclude id_col and _original_index and date_col)
#     feature_cols = [col for col in features if col != id_col and col != date_col]

#     # Group by date to prevent data leakage
#     # Get unique dates
#     unique_dates = df_with_index[date_col].unique()
    
#     # Create a mapping of dates to indices
#     date_to_indices = {date: df_with_index[df_with_index[date_col] == date].index.tolist() 
#                       for date in unique_dates}
    
#     print(f"Found {len(unique_dates)} unique dates")
    
#     # Split the dates into train, validation, and test sets
#     dates_train_val, dates_test = train_test_split(
#         unique_dates, test_size=test_size, random_state=random_state
#     )
#     dates_train, dates_val = train_test_split(
#         dates_train_val, test_size=val_size/(1-test_size), random_state=random_state
#     )
    
#     print(f"Train dates: {len(dates_train)}, Validation dates: {len(dates_val)}, Test dates: {len(dates_test)}")
    
#     # Get indices for each split
#     train_indices = [idx for date in dates_train for idx in date_to_indices[date]]
#     val_indices = [idx for date in dates_val for idx in date_to_indices[date]]
#     test_indices = [idx for date in dates_test for idx in date_to_indices[date]]
    
#     # Split the data according to the indices
#     X_train = X.loc[train_indices].copy()
#     X_val = X.loc[val_indices].copy()
#     X_test = X.loc[test_indices].copy()
    
#     y_train = y.loc[train_indices].copy()
#     y_val = y.loc[val_indices].copy()
#     y_test = y.loc[test_indices].copy()
    
#     # Store indices for each split in metadata for reconstruction
#     metadata['train_indices'] = X_train['_original_index'].values
#     metadata['val_indices'] = X_val['_original_index'].values
#     metadata['test_indices'] = X_test['_original_index'].values

#     if per_participant_normalization:
#         # Global dict for scalers, keyed by participant id under the "features" key
#         scalers['features'] = {}
        
#         # Get unique participant IDs in the training set.
#         train_ids = X_train[id_col].unique()
        
#         # Fit a separate scaler on the training data for each participant
#         for p_id in train_ids:
#             mask = X_train[id_col] == p_id
#             scaler = get_new_scaler()
#             # Fit the scaler on training subset (only the feature columns)
#             scaler.fit(X_train.loc[mask, feature_cols])
#             scalers['features'][p_id] = scaler
            
#             # Transform training, validation, and test sets for the given participant (if present)
#             for dataset in [X_train, X_val, X_test]:
#                 ds_mask = dataset[id_col] == p_id
#                 if ds_mask.sum() > 0:
#                     dataset.loc[ds_mask, feature_cols] = scaler.transform(dataset.loc[ds_mask, feature_cols])
#     else:
#         # Global normalization: fit a single scaler on the training set
#         scaler = get_new_scaler()
#         scaler.fit(X_train[feature_cols])
#         scalers['features'] = scaler
        
#         # Apply the same scaler to training, validation, and test sets
#         X_train[feature_cols] = scaler.transform(X_train[feature_cols])
#         X_val[feature_cols] = scaler.transform(X_val[feature_cols])
#         X_test[feature_cols] = scaler.transform(X_test[feature_cols])
    
#     # Store original dataframe for reconstruction
#     cols_to_store = [id_col, target_col]
#     if timestamp_col:
#         cols_to_store.append(timestamp_col)
#     if date_col and date_col != '_date':  # Don't store the auto-created date column
#         cols_to_store.append(date_col)
#     metadata['df_original'] = df[cols_to_store].copy()
    
#     # Store indices separately in metadata
#     metadata['X_train_indices'] = X_train['_original_index'].copy()
#     metadata['X_val_indices'] = X_val['_original_index'].copy()
#     metadata['X_test_indices'] = X_test['_original_index'].copy()
    
#     # Store date to split mapping for verification
#     metadata['dates_train'] = dates_train
#     metadata['dates_val'] = dates_val
#     metadata['dates_test'] = dates_test
    
#     # Drop the non-feature columns from the feature sets before returning
#     columns_to_drop = ['_original_index', date_col]
#     if id_col in X_train.columns:
#         columns_to_drop.append(id_col)
    
#     X_train.drop(columns=columns_to_drop, inplace=True, errors='ignore')
#     X_val.drop(columns=columns_to_drop, inplace=True, errors='ignore')  
#     X_test.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
#     print("Normalization complete.")
#     print(f"Train set size: {len(X_train)}")
#     print(f"Validation set size: {len(X_val)}")
#     print(f"Test set size: {len(X_test)}")
#     print("Features:", feature_cols)
#     print("Target:", target_col)
#     print(f"Data from {len(dates_train)} dates in train, {len(dates_val)} in validation, and {len(dates_test)} in test")
   
#     return X_train, X_val, X_test, y_train, y_val, y_test, scalers, metadata
def normalize_data_and_split(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    id_col: str,
    timestamp_col: Optional[str] = None,  # Timestamp column
    date_col: Optional[str] = None,     # Explicitly use date column
    per_participant_normalization: bool = False,
    scaler_type: str = "StandardScaler",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
           pd.Series, pd.Series, pd.Series, Dict, Dict]:
    """
    Splits the dataframe into training, validation, and test sets and normalizes only the features.
    The scaler(s) are fit on the training set and then applied to the validation and test sets.
    Preserves metadata needed for timeseries plotting.
    IMPORTANT: Ensures all measurements from the same participant on the same day stay in the same split
    to prevent data leakage.

    Args:
        df: pandas DataFrame containing all the data.
        features: List of column names to be used as features.
        target_col: Column name for the target variable.
        id_col: Column name used to uniquely identify participants (will not be normalized).
        timestamp_col: Optional column name for timestamps (will be preserved for plotting).
        date_col: Column name containing date information. Required to ensure data from the same
                 participant-day stays together to prevent data leakage.
        per_participant_normalization: If True, performs normalization separately for each participant.
        scaler_type: Type of scaler to use ("StandardScaler" or "MinMaxScaler").
        test_size: Fraction of the whole data to hold out for testing.
        val_size: Fraction of the non-test data to hold out for validation.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple containing:
          - X_train: Normalized features for the training set.
          - X_val: Normalized features for the validation set.
          - X_test: Normalized features for the test set.
          - y_train: Target variable for training.
          - y_val: Target variable for validation.
          - y_test: Target variable for test.
          - scalers: Dictionary containing the scaler(s) used.
          - metadata: Dictionary containing information for reconstruction and plotting.
    """
    # Initialize scalers dict and metadata dict
    scalers = {}
    metadata = {
        'original_index': df.index.copy(),
        'id_col': id_col
    }
    
    # Store timestamp information if provided
    if timestamp_col:
        metadata['timestamp_col'] = timestamp_col
        metadata['timestamps'] = df[timestamp_col].copy()
    
    # Helper function to return a new scaler instance based on scaler_type
    def get_new_scaler():
        if scaler_type == "StandardScaler":
            return StandardScaler()
        elif scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    # Create a copy of the original dataframe with indices to track original positions
    df_with_index = df.copy()
    df_with_index['_original_index'] = np.arange(len(df))
    
    # Handle date information for grouping by participant-day
    if date_col is None and timestamp_col is not None:
        # Extract date from timestamp if no explicit date column is provided
        try:
            # Check if timestamp is already a datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_with_index[timestamp_col]):
                df_with_index['_date'] = pd.to_datetime(df_with_index[timestamp_col]).dt.date
            else:
                df_with_index['_date'] = df_with_index[timestamp_col].dt.date
            date_col = '_date'
        except:
            raise ValueError("Could not extract date from timestamp. Please provide a valid date_col to prevent data leakage.")
    
    if date_col is None:
        raise ValueError("date_col must be provided to prevent data leakage across dates.")
    
    # Separate features (X) and target (y)
    X = df_with_index[features + ['_original_index']].copy()
    X[date_col] = df_with_index[date_col]
    X[id_col] = df_with_index[id_col]  # Ensure id_col is in X for participant-day grouping
    y = df_with_index[target_col]
    
    # Columns to normalize (exclude id_col and _original_index and date_col)
    feature_cols = [col for col in features if col != id_col and col != date_col]

    # Create a participant-day identifier to prevent leakage
    df_with_index['_participant_day'] = df_with_index[id_col].astype(str) + '_' + df_with_index[date_col].astype(str)
    X['_participant_day'] = df_with_index['_participant_day']
    
    # Get unique participant-days
    unique_participant_days = df_with_index['_participant_day'].unique()
    
    # Create a mapping of participant-days to indices
    participant_day_to_indices = {p_day: df_with_index[df_with_index['_participant_day'] == p_day].index.tolist() 
                               for p_day in unique_participant_days}
    
    print(f"Found {len(unique_participant_days)} unique participant-days")
    
    # Split the participant-days into train, validation, and test sets
    p_days_train_val, p_days_test = train_test_split(
        unique_participant_days, test_size=test_size, random_state=random_state
    )
    p_days_train, p_days_val = train_test_split(
        p_days_train_val, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    print(f"Train participant-days: {len(p_days_train)}, Validation participant-days: {len(p_days_val)}, Test participant-days: {len(p_days_test)}")
    
    # Get indices for each split
    train_indices = [idx for p_day in p_days_train for idx in participant_day_to_indices[p_day]]
    val_indices = [idx for p_day in p_days_val for idx in participant_day_to_indices[p_day]]
    test_indices = [idx for p_day in p_days_test for idx in participant_day_to_indices[p_day]]
    
    # Split the data according to the indices
    X_train = X.loc[train_indices].copy()
    X_val = X.loc[val_indices].copy()
    X_test = X.loc[test_indices].copy()
    
    y_train = y.loc[train_indices].copy()
    y_val = y.loc[val_indices].copy()
    y_test = y.loc[test_indices].copy()
    
    # Store indices for each split in metadata for reconstruction
    metadata['train_indices'] = X_train['_original_index'].values
    metadata['val_indices'] = X_val['_original_index'].values
    metadata['test_indices'] = X_test['_original_index'].values

    # Store participant-day information in metadata
    metadata['participant_days_train'] = p_days_train
    metadata['participant_days_val'] = p_days_val
    metadata['participant_days_test'] = p_days_test

    if per_participant_normalization:
        # Global dict for scalers, keyed by participant id under the "features" key
        scalers['features'] = {}
        
        # Get unique participant IDs in the training set.
        train_ids = X_train[id_col].unique()
        
        # Fit a separate scaler on the training data for each participant
        for p_id in train_ids:
            mask = X_train[id_col] == p_id
            scaler = get_new_scaler()
            # Fit the scaler on training subset (only the feature columns)
            scaler.fit(X_train.loc[mask, feature_cols])
            scalers['features'][p_id] = scaler
            
            # Transform training, validation, and test sets for the given participant (if present)
            for dataset in [X_train, X_val, X_test]:
                ds_mask = dataset[id_col] == p_id
                if ds_mask.sum() > 0:
                    dataset.loc[ds_mask, feature_cols] = scaler.transform(dataset.loc[ds_mask, feature_cols])
    else:
        # Global normalization: fit a single scaler on the training set
        scaler = get_new_scaler()
        scaler.fit(X_train[feature_cols])
        scalers['features'] = scaler
        
        # Apply the same scaler to training, validation, and test sets
        X_train[feature_cols] = scaler.transform(X_train[feature_cols])
        X_val[feature_cols] = scaler.transform(X_val[feature_cols])
        X_test[feature_cols] = scaler.transform(X_test[feature_cols])
    
    # Store original dataframe for reconstruction
    cols_to_store = [id_col, target_col]
    if timestamp_col:
        cols_to_store.append(timestamp_col)
    if date_col and date_col != '_date':  # Don't store the auto-created date column
        cols_to_store.append(date_col)
    metadata['df_original'] = df[cols_to_store].copy()
    
    # Store indices separately in metadata
    metadata['X_train_indices'] = X_train['_original_index'].copy()
    metadata['X_val_indices'] = X_val['_original_index'].copy()
    metadata['X_test_indices'] = X_test['_original_index'].copy()
    
    # Drop the non-feature columns from the feature sets before returning
    columns_to_drop = ['_original_index', date_col, '_participant_day']
    if id_col in features:
        # Keep id_col if it's in features list
        columns_to_drop = [col for col in columns_to_drop if col != id_col]
    else:
        # Otherwise drop it
        columns_to_drop.append(id_col)
    
    X_train.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    X_val.drop(columns=columns_to_drop, inplace=True, errors='ignore')  
    X_test.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    print("Normalization complete.")
    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print("Features:", feature_cols)
    print("Target:", target_col)
    print(f"Data from {len(p_days_train)} participant-days in train, {len(p_days_val)} in validation, and {len(p_days_test)} in test")
   
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, metadata

def plot_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: Dict,
    dataset: str = 'test',
    figsize: Tuple[int, int] = (12, 6),
    title: str = 'Predicted vs Actual Values',
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot all timeseries predictions for the specified dataset on a single plot,
    coloring the background spans for each participant, without including participants in the legend.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        metadata: Metadata dictionary from normalize_data_and_split function.
        dataset: Which dataset to plot ('train', 'val', or 'test').
        figsize: Figure size.
        title: Plot title.
        save_path: If provided, save the plot to this path.
        show_plot: Whether to display the plot.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Retrieve indices for desired dataset
    indices_key = f'X_{dataset}_indices'
    if indices_key not in metadata:
        raise ValueError(f"Dataset '{dataset}' not found in metadata")
    indices = metadata[indices_key]
    if isinstance(indices, pd.Series):
        indices = indices.values

    # Subset and sort original dataframe
    df = metadata['df_original'].copy()
    df = df.loc[indices].copy()
    if 'timestamp_col' in metadata:
        ts = metadata['timestamp_col']
        if ts in df.columns:
            df = df.sort_values([metadata['id_col'], ts])

    # Attach predictions and actuals (align lengths)
    n = min(len(df), len(y_pred))
    df = df.iloc[:n].copy()
    df['Predicted'] = y_pred[:n]
    df['Actual']    = y_true.values[:n]

    # Identify participants in display order
    participants = df[metadata['id_col']].unique()
    participant_col = metadata['id_col']

    # Create overall plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))

    # Color spans per participant
    cmap = plt.get_cmap('tab20', len(participants))
    for idx, pid in enumerate(participants):
        mask = df[participant_col] == pid
        inds = np.where(mask)[0]
        start, end = inds[0], inds[-1]
        ax.axvspan(start, end, color=cmap(idx), alpha=0.2)

    # Plot lines
    ax.plot(x, df['Actual'], linestyle='-',  label='Actual')
    ax.plot(x, df['Predicted'], linestyle='--', label='Predicted')

    # Legend for lines only
    line_handles = [
        Line2D([0], [0], color='black', linestyle='-', label='Actual'),
        Line2D([0], [0], color='black', linestyle='--', label='Predicted')
    ]
    ax.legend(handles=line_handles, loc='upper right')

    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.set_title(f"{title} - {dataset.capitalize()}")
    ax.grid(True)
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(f"{save_path}_{dataset}.png")

    if show_plot:
        plt.show()
    else:
        plt.close()
