from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def normalize_data_and_split(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    id_col: str,
    per_participant_normalization: bool = False,
    scaler_type: str = "StandardScaler",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
           pd.Series, pd.Series, pd.Series, Dict]:
    """
    Splits the dataframe into training, validation, and test sets and normalizes only the features.
    The scaler(s) are fit on the training set and then applied to the validation and test sets.

    Args:
        df: pandas DataFrame containing all the data.
        features: List of column names to be used as features.
        target_col: Column name for the target variable.
        id_col: Column name used to uniquely identify participants (will not be normalized).
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
    """
    # Initialize scalers dict (empty by default)
    scalers = {}
    
    # Helper function to return a new scaler instance based on scaler_type
    def get_new_scaler():
        if scaler_type == "StandardScaler":
            return StandardScaler()
        elif scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    # Separate features (X) and target (y)
    X = df[features].copy()
    y = df[target_col]
    
    # Columns to normalize (exclude id_col)
    feature_cols = [col for col in features if col != id_col]

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Then split the remaining data into training and validation sets.
    # Note: val_size is a fraction of the remaining data (not the full dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

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
    

    # drop the id_col from the feature sets
    X_train.drop(columns=[id_col], inplace=True)
    X_val.drop(columns=[id_col], inplace=True)
    X_test.drop(columns=[id_col], inplace=True)
    
    print("Normalization complete.")
    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print("features:", feature_cols)
    print("target:", target_col)

    # breakpoint()

    
    return X_train, X_val, X_test, y_train, y_val, y_test, scalers