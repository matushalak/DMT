import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor
from modeler import Modeler


def train_data(save_processed_data: bool = False):
    if os.path.exists("data/processed_train.csv") and os.path.exists("data/processed_val.csv"):
        print("Loading saved processed training data...")
        processed_train_df = pd.read_csv("data/processed_train.csv")
        processed_val_df = pd.read_csv("data/processed_val.csv")
    else:
        print("Loading raw training data data...")
        train_df = pd.read_csv("data/training_set_VU_DM.csv")

        print("Splitting data into train and validation sets...")
        all_search_ids = train_df["srch_id"].unique()
        train_ids, val_ids = train_test_split(all_search_ids, test_size=0.2, random_state=42)

        train_data = train_df[train_df["srch_id"].isin(train_ids)].copy()
        val_data = train_df[train_df["srch_id"].isin(val_ids)].copy()

        print(f"Training data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")

        print("Preprocessing training data...")
        train_preprocessor = Preprocessor(train_data)
        processed_train_df, encoder = train_preprocessor.run_pipeline(is_test=False, save=save_processed_data, name="train")

        print("Preprocessing validation data...")
        val_preprocessor = Preprocessor(val_data)
        processed_val_df = val_preprocessor.run_pipeline(is_test=True, encoder=encoder, save=save_processed_data, name="val")

    # Initialize modeler with processed train and validation data
    modeler = Modeler(train_df=processed_train_df, val_df=processed_val_df)

    print("Training model...")
    model = modeler.train_model()

    # Plot training metrics
    modeler.plot_training_metrics()

    top_features = modeler.get_feature_importance(top_n=20)
    print(f"Top 20 features: {top_features}")



def test_data(save_processed_data: bool = False):
    ##### TEST DATA ###########
    if os.path.exists("data/processed_test.csv") and os.path.exists("data/processed_train_whole.csv"):
        print("Loading saved processed test data...")
        processed_test_df = pd.read_csv("data/processed_test.csv")
        processed_whole_train_df = pd.read_csv("data/processed_train_whole.csv")
    else:
        print("Loading and preprocessing test data...")
        train_df = pd.read_csv("data/training_set_VU_DM.csv")
        test_df = pd.read_csv("data/test_set_VU_DM.csv")

        # Initialize preprocessor for test data
        whole_train_preprocessor = Preprocessor(train_df)
        processed_whole_train_df, encoder = whole_train_preprocessor.run_pipeline(is_test=False, save=save_processed_data, name="train_whole")
        test_preprocessor = Preprocessor(test_df)

        # Run preprocessing pipeline for test data
        processed_test_df = test_preprocessor.run_pipeline(is_test=True, encoder=encoder, save=save_processed_data, name="test")

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

    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv("data/training_set_VU_DM.csv")

    # Split data by search_id to prevent data leakage
    print("Splitting data into train and validation sets...")
    all_search_ids = train_df["srch_id"].unique()
    train_ids, val_ids = train_test_split(all_search_ids, test_size=0.2, random_state=42)

    train_data = train_df[train_df["srch_id"].isin(train_ids)].copy()
    val_data = train_df[train_df["srch_id"].isin(val_ids)].copy()

    # Preprocess training data
    print("Preprocessing training data...")
    train_preprocessor = Preprocessor(train_data)
    processed_train_df, encoder = train_preprocessor.run_pipeline(is_test=False, save=save_processed_data, name="train")

    # Preprocess validation data using the same encoder
    print("Preprocessing validation data...")
    val_preprocessor = Preprocessor(val_data)
    processed_val_df = val_preprocessor.run_pipeline(is_test=True, encoder=encoder, save=save_processed_data, name="val")

    # Initialize modeler with processed train and validation data
    modeler = Modeler(train_df=processed_train_df, val_df=processed_val_df)

    # Tune hyperparameters
    print("Tuning hyperparameters...")
    best_params = modeler.hyperparameter_tuning(n_trials=10)

    print(f"Best parameters: {best_params}")

    # Train model with best parameters
    print("Training model with best parameters...")
    model = modeler.train_model()

    # Plot training metrics
    modeler.plot_training_metrics()

    # Get feature importance
    top_features = modeler.get_feature_importance(top_n=20)
    print(f"Top 20 features: {top_features}")

if __name__ == "__main__":
    train_data(save_processed_data=True)
    # test_data(save_processed_data=True)
