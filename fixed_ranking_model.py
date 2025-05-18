import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow_ranking as tfr

from sklearn.metrics import ndcg_score

import matplotlib.pyplot as plt
from typing import Literal

from preprocess import preprocess_df, add_features

def train_val_split(DF:pd.DataFrame, 
                    train_prop:float = 0.7
                    )->tuple[
                        # Train (Query sizes, X_train, y_train)
                        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                        # Validate (Query sizes, X_val, y_val)
                        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    # Train/val split by srch_id
    all_searchIDs = DF['srch_id'].unique()
    np.random.seed(42)  # Add seed for reproducibility
    train_sids = np.random.choice(all_searchIDs, size=int(train_prop * len(all_searchIDs)), replace=False)
    val_sids   = np.setdiff1d(all_searchIDs, train_sids)

    TRAIN = DF[DF['srch_id'].isin(train_sids)].copy()
    VAL = DF[DF['srch_id'].isin(val_sids)].copy()

    # Compute search id group sizes
    query_size_train = TRAIN.groupby('srch_id').size().to_numpy()
    query_size_val   = VAL.groupby('srch_id').size().to_numpy()

    # Features and labels
    X_train = TRAIN.drop(columns=['srch_id', 'relevance', 
                                'position', 'click_bool', 'booking_bool', 'gross_bookings_usd'])
    y_train = TRAIN['relevance']

    X_val = VAL.drop(columns=['srch_id', 'relevance', 
                            'position', 'click_bool', 'booking_bool', 'gross_bookings_usd'])
    y_val = VAL['relevance']

    return ((query_size_train, X_train, y_train),
            (query_size_val, X_val, y_val))

def check_data_quality(X, y, feature_names):
    """Check for data quality issues that could cause NaN loss"""
    print("=== Data Quality Check ===")
    
    # Check for NaN/inf values in features
    nan_counts = X[feature_names].isna().sum()
    inf_counts = np.isinf(X[feature_names]).sum()
    
    if nan_counts.sum() > 0:
        print(f"WARNING: Found {nan_counts.sum()} NaN values in features")
        print("NaN counts by feature:")
        print(nan_counts[nan_counts > 0])
    
    if inf_counts.sum() > 0:
        print(f"WARNING: Found {inf_counts.sum()} infinite values in features")
        print("Inf counts by feature:")
        print(inf_counts[inf_counts > 0])
    
    # Check for NaN values in labels
    nan_labels = y.isna().sum()
    if nan_labels > 0:
        print(f"WARNING: Found {nan_labels} NaN values in labels")
    
    # Check feature statistics
    print("\nFeature statistics:")
    feature_stats = X[feature_names].describe()
    print(feature_stats)
    
    # Check for constant features
    constant_features = []
    for col in feature_names:
        if X[col].nunique() == 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"WARNING: Found constant features: {constant_features}")
    
    # Check label distribution
    print(f"\nLabel statistics:")
    print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean():.3f}, Std: {y.std():.3f}")
    print(f"Label distribution:\n{y.value_counts().sort_index()}")
    
    return nan_counts.sum() == 0 and inf_counts.sum() == 0 and nan_labels == 0

def clean_data(X, y, feature_names):
    """Clean data by handling NaN/inf values"""
    print("=== Cleaning Data ===")
    
    # Handle NaN values in features (fill with 0 or median)
    X_clean = X.copy()
    for col in feature_names:
        if X_clean[col].isna().sum() > 0:
            # For numerical features, fill with median
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)
            print(f"Filled {col} NaNs with median value: {median_val}")
    
    # Handle infinite values (replace with 99th percentile)
    for col in feature_names:
        if np.isinf(X_clean[col]).sum() > 0:
            percentile_99 = X_clean[col].replace([np.inf, -np.inf], np.nan).quantile(0.99)
            X_clean[col] = X_clean[col].replace([np.inf, -np.inf], percentile_99)
            print(f"Replaced inf values in {col} with 99th percentile: {percentile_99}")
    
    # Handle NaN labels (remove these rows entirely)
    mask = ~y.isna()
    X_clean = X_clean[mask]
    y_clean = y[mask]
    
    # Ensure labels are non-negative (required for NDCG)
    if y_clean.min() < 0:
        print(f"WARNING: Found negative labels. Shifting by {-y_clean.min()}")
        y_clean = y_clean - y_clean.min()
    
    print(f"Data cleaning complete. Original size: {len(y)}, Clean size: {len(y_clean)}")
    return X_clean, y_clean

def make_ranking_dataset(X: pd.DataFrame,
                         y: pd.Series,
                         group_sizes: np.ndarray,
                         feature_names: list,
                         MAX_LIST_SIZE: int,
                         shuffle: bool = True,
                         batch_size: int = 32,
                         seed: int = 42):
    
    # Convert to float32 and ensure no NaN/inf values
    feature_arrays = {}
    for col in feature_names:
        arr = X[col].to_numpy(dtype=np.float32)
        # Double-check for any remaining NaN/inf
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"ERROR: Found NaN/inf in feature {col} after cleaning!")
            raise ValueError(f"Feature {col} contains NaN or inf values")
        feature_arrays[col] = arr
    
    labels_array = y.to_numpy(dtype=np.float32)
    
    # Check labels
    if np.any(np.isnan(labels_array)) or np.any(np.isinf(labels_array)):
        print("ERROR: Found NaN/inf in labels after cleaning!")
        raise ValueError("Labels contain NaN or inf values")

    examples = []
    cursor = 0
    for size in group_sizes:
        q_feats = {k: v[cursor:cursor+size] for k, v in feature_arrays.items()}
        q_labels = labels_array[cursor:cursor+size]
        
        # Create a mask: 1 for real items, 0 for padded items.
        q_mask = np.ones_like(q_labels, dtype=np.float32)

        pad_size = MAX_LIST_SIZE - size
        if pad_size < 0:
             # Truncate instead of raising error
             print(f"WARNING: Query size {size} exceeds MAX_LIST_SIZE {MAX_LIST_SIZE}. Truncating.")
             for k_feat in q_feats:
                 q_feats[k_feat] = q_feats[k_feat][:MAX_LIST_SIZE]
             q_labels = q_labels[:MAX_LIST_SIZE]
             q_mask = q_mask[:MAX_LIST_SIZE]
             pad_size = 0

        # Pad features
        for k_feat in q_feats:
            q_feats[k_feat] = np.concatenate(
                [q_feats[k_feat], np.zeros(pad_size, dtype=np.float32)]
            )
        
        # Pad labels with -1 (invalid label for ranking)
        q_labels = np.concatenate(
            [q_labels, np.full(pad_size, -1.0, dtype=np.float32)]
        )
        
        # Pad mask with 0
        q_mask = np.concatenate(
            [q_mask, np.zeros(pad_size, dtype=np.float32)]
        )

        examples.append((q_feats, q_labels, q_mask))
        cursor += size

    def gen():
        for feats, labs, msk in examples:
            yield feats, labs, msk

    # Create output signature
    output_signature=(
        {k: tf.TensorSpec(shape=(MAX_LIST_SIZE,), dtype=tf.float32) for k in feature_names},
        tf.TensorSpec(shape=(MAX_LIST_SIZE,), dtype=tf.float32),  # Labels
        tf.TensorSpec(shape=(MAX_LIST_SIZE,), dtype=tf.float32)   # Mask
    )

    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(examples), 1000), seed=seed, reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_keras_model(feature_names: list, max_list_size: int) -> keras.Model:
    # Define input layers for each feature
    inputs = {
        name: keras.Input(shape=(max_list_size,), name=name, dtype=tf.float32)
        for name in feature_names
    }
    
    # Stack features
    expanded_feature_tensors = [tf.expand_dims(inputs[name], axis=-1) for name in feature_names]
    stacked_features = keras.layers.Concatenate(axis=-1)(expanded_feature_tensors)
    
    # Add batch normalization to help with training stability
    stacked_features = keras.layers.BatchNormalization()(stacked_features)
    
    # Define the scoring network with dropout for regularization
    hidden_layer = keras.layers.Dense(128, activation='relu', name='hidden_layer')(stacked_features)
    hidden_layer = keras.layers.Dropout(0.2)(hidden_layer)
    
    # Add another hidden layer
    hidden_layer_2 = keras.layers.Dense(64, activation='relu', name='hidden_layer_2')(hidden_layer)
    hidden_layer_2 = keras.layers.Dropout(0.2)(hidden_layer_2)
    
    scores = keras.layers.Dense(1, name='scores_output')(hidden_layer_2)
    scores = tf.squeeze(scores, axis=-1)  # Remove the last dimension

    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=scores, name="ranking_model")
    return model

def train_model_keras(DF_path: str = 'training_set_VU_DM.csv', 
                      epochs: int = 10, 
                      batch_size: int = 32,
                      learning_rate: float = 0.001):
    
    print("Loading and preprocessing data...")
    # Load & preprocess
    DF = pd.read_csv(DF_path)
    DF = preprocess_df(DF)
    DF = add_features(DF)
    
    # Filter to queries with positive relevance
    good_sids = (
        DF.groupby("srch_id")["relevance"]
          .sum()
          .loc[lambda s: s > 0]
          .index
    )
    DF = DF[DF["srch_id"].isin(good_sids)].reset_index(drop=True)
    print(f'Filtered to {len(good_sids)} queries with positive relevance')

    # Define feature columns
    non_feature_cols = ['srch_id', 'relevance', 'position', 'click_bool', 'booking_bool', 'gross_bookings_usd']
    ALL_FEATURE_NAMES = [col for col in DF.columns if col not in non_feature_cols]
    print(f"Using {len(ALL_FEATURE_NAMES)} features")

    # Train / Val split
    T_V_split = train_val_split(DF, train_prop=0.7)
    query_size_train, X_train, y_train = T_V_split[0]
    query_size_val, X_val, y_val = T_V_split[1]
    print('Train-val split done!')

    # Ensure X_train, X_val only contain feature columns
    X_train = X_train[ALL_FEATURE_NAMES]
    X_val = X_val[ALL_FEATURE_NAMES]

    # Check and clean data quality
    print("\n=== TRAINING DATA ===")
    train_is_clean = check_data_quality(X_train, y_train, ALL_FEATURE_NAMES)
    if not train_is_clean:
        X_train, y_train = clean_data(X_train, y_train, ALL_FEATURE_NAMES)
        # Recompute group sizes after cleaning
        train_df_clean = pd.concat([X_train, y_train], axis=1)
        train_df_clean['srch_id'] = DF.loc[train_df_clean.index, 'srch_id']
        query_size_train = train_df_clean.groupby('srch_id').size().to_numpy()

    print("\n=== VALIDATION DATA ===")
    val_is_clean = check_data_quality(X_val, y_val, ALL_FEATURE_NAMES)
    if not val_is_clean:
        X_val, y_val = clean_data(X_val, y_val, ALL_FEATURE_NAMES)
        # Recompute group sizes after cleaning
        val_df_clean = pd.concat([X_val, y_val], axis=1)
        val_df_clean['srch_id'] = DF.loc[val_df_clean.index, 'srch_id']
        query_size_val = val_df_clean.groupby('srch_id').size().to_numpy()

    MAX_query_size = int(min(max(query_size_train.max(), query_size_val.max()), 50))  # Cap at 50
    print(f"Max query size (capped): {MAX_query_size}")

    # Create datasets
    print("Creating training dataset...")
    train_DS = make_ranking_dataset(X_train, y_train, query_size_train, ALL_FEATURE_NAMES,
                                    MAX_query_size, shuffle=True, batch_size=batch_size)
    
    print("Creating validation dataset...")
    valid_DS = make_ranking_dataset(X_val, y_val, query_size_val, ALL_FEATURE_NAMES,
                                    MAX_query_size, shuffle=False, batch_size=batch_size)

    # Create the Keras model
    print("Creating model...")
    keras_model = create_keras_model(ALL_FEATURE_NAMES, MAX_query_size)
    keras_model.summary()

    # Compile the model with appropriate settings
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),  # Gradient clipping
        loss=tfr.keras.losses.SoftmaxLoss(),  # Start with simpler loss
        metrics=[tfr.keras.metrics.NDCGMetric(name='ndcg_5', topn=5)]
    )

    # Add callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    # Train the model
    print("Starting model training...")
    history = keras_model.fit(
        train_DS,
        validation_data=valid_DS,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        workers=4  # Reduced workers to avoid memory issues
    )
    
    print('Training finished.')
    print('Final validation metrics:', {k: v[-1] for k, v in history.history.items() if 'val_' in k})
    
    return keras_model, ALL_FEATURE_NAMES, MAX_query_size

# Updated prediction function to work with the new model structure
def get_predictions(model, feature_names, max_query_size) -> pd.DataFrame:
    print("Loading test data...")
    TESTDF = pd.read_csv('test_set_VU_DM.csv')
    TESTDF = preprocess_df(TESTDF, TEST=True)
    TESTDF = add_features(TESTDF)
    
    # Clean test data
    X_test = TESTDF[feature_names]
    X_test_clean, _ = clean_data(X_test, pd.Series(np.zeros(len(X_test))), feature_names)
    
    # Get query sizes
    query_sizes = TESTDF.groupby('srch_id').size().to_numpy()
    srch_ids = TESTDF['srch_id'].unique()
    
    # Predict by batches of queries
    all_predictions = []
    cursor = 0
    
    for i, size in enumerate(query_sizes):
        # Get features for this query
        query_features = {col: X_test_clean.iloc[cursor:cursor+size][col].to_numpy(dtype=np.float32) 
                         for col in feature_names}
        
        # Pad to max_query_size
        pad_size = max_query_size - size
        if pad_size > 0:
            for col in feature_names:
                query_features[col] = np.concatenate([
                    query_features[col], 
                    np.zeros(pad_size, dtype=np.float32)
                ])
        elif pad_size < 0:
            # Truncate if necessary
            for col in feature_names:
                query_features[col] = query_features[col][:max_query_size]
            size = max_query_size
        
        # Reshape for batch prediction
        batch_features = {col: feat.reshape(1, -1) for col, feat in query_features.items()}
        
        # Predict
        scores = model.predict(batch_features, verbose=0)[0]
        
        # Take only the relevant scores (not padded ones)
        relevant_scores = scores[:size]
        all_predictions.extend(relevant_scores)
        
        cursor += len(TESTDF[TESTDF['srch_id'] == srch_ids[i]])
    
    # Add predictions to dataset
    TESTDF['predicted_relevance'] = all_predictions
    
    # Produce ranking
    RANKED = TESTDF.sort_values(['srch_id', 'predicted_relevance'], 
                               ascending=[True, False]).reset_index(drop=True)
    
    return RANKED[['srch_id', 'prop_id']]

if __name__ == '__main__':
    MODEL, FEATURE_NAMES, MAX_QUERY_SIZE = train_model_keras(epochs=20, learning_rate=0.0001)
    TEST_pred = get_predictions(MODEL, FEATURE_NAMES, MAX_QUERY_SIZE)
    TEST_pred.to_csv('VU-DM-2025-Group-100.csv', index=False)