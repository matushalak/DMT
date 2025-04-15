from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
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
    IMPORTANT: Ensures all measurements from the same date stay in the same split to prevent data leakage.

    Args:
        df: pandas DataFrame containing all the data.
        features: List of column names to be used as features.
        target_col: Column name for the target variable.
        id_col: Column name used to uniquely identify participants (will not be normalized).
        timestamp_col: Optional column name for timestamps (will be preserved for plotting).
        date_col: Column name containing date information. Required to ensure data from the same
                 date stays together to prevent data leakage.
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
    
    # Handle date information for grouping by day
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
    y = df_with_index[target_col]
    
    # Columns to normalize (exclude id_col and _original_index and date_col)
    feature_cols = [col for col in features if col != id_col and col != date_col]

    # Group by date to prevent data leakage
    # Get unique dates
    unique_dates = df_with_index[date_col].unique()
    
    # Create a mapping of dates to indices
    date_to_indices = {date: df_with_index[df_with_index[date_col] == date].index.tolist() 
                      for date in unique_dates}
    
    print(f"Found {len(unique_dates)} unique dates")
    
    # Split the dates into train, validation, and test sets
    dates_train_val, dates_test = train_test_split(
        unique_dates, test_size=test_size, random_state=random_state
    )
    dates_train, dates_val = train_test_split(
        dates_train_val, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    print(f"Train dates: {len(dates_train)}, Validation dates: {len(dates_val)}, Test dates: {len(dates_test)}")
    
    # Get indices for each split
    train_indices = [idx for date in dates_train for idx in date_to_indices[date]]
    val_indices = [idx for date in dates_val for idx in date_to_indices[date]]
    test_indices = [idx for date in dates_test for idx in date_to_indices[date]]
    
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
    
    # Store date to split mapping for verification
    metadata['dates_train'] = dates_train
    metadata['dates_val'] = dates_val
    metadata['dates_test'] = dates_test
    
    # Drop the non-feature columns from the feature sets before returning
    columns_to_drop = ['_original_index', date_col]
    if id_col in X_train.columns:
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
    print(f"Data from {len(dates_train)} dates in train, {len(dates_val)} in validation, and {len(dates_test)} in test")
   
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers, metadata


def plot_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: Dict,
    dataset: str = 'test',
    figsize: Tuple[int, int] = (12, 6),
    title: str = 'Predicted vs Actual Values',
    save_path: Optional[str] = None,
    show_plot: bool = True,
    max_participants: Optional[int] = None
) -> None:
    """
    Plot timeseries predictions against actual values.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        metadata: Metadata dictionary from normalize_data_and_split function.
        dataset: Which dataset to plot ('train', 'val', or 'test').
        figsize: Figure size.
        title: Plot title.
        save_path: If provided, save the plot to this path.
        show_plot: Whether to display the plot.
        max_participants: Maximum number of participants to plot (None for all).
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Dict, Tuple, Optional
    
    # Get indices for the specified dataset
    indices_key = f'X_{dataset}_indices'
    if indices_key not in metadata:
        raise ValueError(f"Dataset '{dataset}' not found in metadata")
    
    indices = metadata[indices_key]
    
    # Get the original dataframe
    df_original = metadata['df_original'].copy()
    
    # Create DataFrame for plotting
    # First ensure indices are in the correct format
    if isinstance(indices, pd.Series):
        indices = indices.values
    
    # Check if indices exist in the original dataframe index
    valid_indices = [idx for idx in indices if idx in df_original.index]
    if len(valid_indices) < len(indices):
        print(f"Warning: {len(indices) - len(valid_indices)} indices not found in original dataframe")
    
    try:
        # Try to create the plotting dataframe
        df_plot = df_original.loc[valid_indices].copy()
        
        # Add predictions and actual values
        if len(y_pred) != len(df_plot):
            print(f"Warning: Length mismatch between predictions ({len(y_pred)}) and dataframe ({len(df_plot)})")
            # Use the smaller length
            min_len = min(len(y_pred), len(df_plot))
            df_plot = df_plot.iloc[:min_len].copy()
            df_plot['Predicted'] = y_pred[:min_len]
            df_plot['Actual'] = y_true.values[:min_len]
        else:
            df_plot['Predicted'] = y_pred
            df_plot['Actual'] = y_true.values
        
        # Ensure the id_col exists
        if metadata['id_col'] not in df_plot.columns:
            print(f"Warning: ID column '{metadata['id_col']}' not found in dataframe")
            # Create a dummy ID column
            df_plot[metadata['id_col']] = 'participant_1'
        
        # Try to sort by timestamp if available
        if 'timestamp_col' in metadata and metadata['timestamp_col'] in df_plot.columns:
            timestamp_col = metadata['timestamp_col']
            # Check for duplicates in columns before sorting
            if df_plot.columns.duplicated().any():
                # Make column names unique using a simple approach
                print("Warning: Duplicate column names found, making them unique")
                
                # Custom function to make column names unique
                def make_unique_cols(columns):
                    seen = {}
                    result = []
                    for item in columns:
                        if item in seen:
                            seen[item] += 1
                            result.append(f"{item}_{seen[item]}")
                        else:
                            seen[item] = 0
                            result.append(item)
                    return result
                
                # Apply the function to create unique column names
                new_columns = make_unique_cols(df_plot.columns)
                df_plot.columns = new_columns
                
                # Update the column name if it was changed
                if metadata['id_col'] not in df_plot.columns:
                    possible_id_cols = [col for col in df_plot.columns if metadata['id_col'] in col]
                    if possible_id_cols:
                        metadata['id_col'] = possible_id_cols[0]
                        print(f"ID column renamed to {metadata['id_col']}")
                if timestamp_col not in df_plot.columns:
                    possible_ts_cols = [col for col in df_plot.columns if timestamp_col in col]
                    if possible_ts_cols:
                        timestamp_col = possible_ts_cols[0]
                        print(f"Timestamp column renamed to {timestamp_col}")
            
            # Now try sorting
            try:
                df_plot = df_plot.sort_values(by=[metadata['id_col'], timestamp_col])
            except Exception as e:
                print(f"Warning: Error while sorting by {metadata['id_col']} and {timestamp_col}: {str(e)}")
                print("Columns available:", df_plot.columns.tolist())
        
        # Get unique participants
        participants = df_plot[metadata['id_col']].unique()
        
        # Limit the number of participants if specified
        if max_participants is not None and len(participants) > max_participants:
            print(f"Limiting to {max_participants} participants out of {len(participants)}")
            participants = participants[:max_participants]
        
        # Plot separately for each participant
        for participant_id in participants:
            participant_data = df_plot[df_plot[metadata['id_col']] == participant_id]
            
            if len(participant_data) == 0:
                print(f"Warning: No data for participant {participant_id}")
                continue
                
            plt.figure(figsize=figsize)
            
            # Determine x-axis
            if 'timestamp_col' in metadata and metadata['timestamp_col'] in participant_data.columns:
                x_values = participant_data[metadata['timestamp_col']]
                x_label = metadata['timestamp_col']
            else:
                x_values = range(len(participant_data))
                x_label = 'Data Point'
            
            # Plot the data
            plt.plot(x_values, participant_data['Actual'], 'b-', label='Actual')
            plt.plot(x_values, participant_data['Predicted'], 'r--', label='Predicted')
            
            # Calculate metrics for this participant
            mae = np.mean(np.abs(participant_data['Actual'] - participant_data['Predicted']))
            rmse = np.sqrt(np.mean((participant_data['Actual'] - participant_data['Predicted'])**2))
            
            # Add metrics to plot
            plt.text(0.02, 0.95, f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}", 
                     transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))
            
            plt.xlabel(x_label)
            plt.ylabel('Value')
            plt.title(f"{title} - Participant {participant_id}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            if save_path:
                # Ensure the participant_id is string and clean for filenames
                safe_id = str(participant_id).replace('/', '_').replace('\\', '_')
                plt.savefig(f"{save_path}_{dataset}_{safe_id}.png")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
    except Exception as e:
        print(f"Error plotting predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Print diagnostic information
        print("\nDiagnostic information:")
        print(f"Dataset: {dataset}")
        print(f"Indices shape: {np.shape(indices)}")
        print(f"y_true shape: {np.shape(y_true)}")
        print(f"y_pred shape: {np.shape(y_pred)}")
        print(f"Metadata keys: {list(metadata.keys())}")
        print(f"Original dataframe shape: {metadata['df_original'].shape}")
        print(f"Original dataframe columns: {metadata['df_original'].columns.tolist()}")