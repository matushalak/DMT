import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn
import torch.optim as optim
import time
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from lstm_utils import MultiParticipantDataset, LSTMModel, SimpleRNNModel, GRUModel
from lstm_utils import normalize, predict_and_plot, create_data_split, train_and_evaluate
from lstm_utils import save_model, load_model, simple_hyperparameter_tuning


# --- Mac acceleration ---
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("Using Mac acceleration")
else:
    mps_device = torch.device("cpu")
    print("Using CPU")


if __name__ == "__main__":
    # Load the dataset
    dataset_name = "mean_mode_imputation_combinedAppCat"
    dropped_vars = [""]
    imputation = "mean_mode"
    df = pd.read_csv(f'tables/preprocessed/{dataset_name}.csv')

    do_hyperparameter_tuning = False  # Set to True to enable tuning
    
    
    # Default hyperparameters
    config = {
        "seq_length": 7,
        "batch_size": 32,
        "hidden_dim": 128,
        "num_layers": 2,
        "learning_rate": 0.0001,
        "dropout": 0.2,
        "model_type": "SimpleRNN",  # Options: "LSTM", "GRU", "SimpleRNN"
        "num_epochs": 20,
        "transform_target": True,
        "scaler_type": "MinMaxScaler",
        "clip_gradients": True,
        "max_grad_norm": 1.0
    }
    
    # Create directories for outputs
    os.makedirs("tables/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # Data split
    train_df, val_df, test_df = create_data_split(
        df, 
        proportion_train=0.7, 
        proportion_val=0.15, 
        split_within_participants=True,
        seq_length=config["seq_length"]
    )
    
    # Print dataset sizes
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Print mood descriptives
    print("\nTrain mood descriptives")
    print(train_df["mood"].describe())
    print("\nValidation mood descriptives")
    print(val_df["mood"].describe())
    print("\nTest mood descriptives")
    print(test_df["mood"].describe())
    
    # Choose whether to perform hyperparameter tuning
    
    if do_hyperparameter_tuning:

        param_grid = {
            "seq_length": [7, 8, 9],
            "batch_size": [32],
            "hidden_dim": [64, 128, 256],
            "num_layers": [1, 2],
            "learning_rate": [0.0001, 0.001, 0.01],
            "dropout": [0.1, 0.2, 0.3],
            "model_type": ["SimpleRNN"],
            "transform_target": [True],
            "scaler_type": ["MinMaxScaler"],
            "clip_gradients": [True],
            "max_grad_norm": [1.0]
        }
        # Run hyperparameter tuning
        best_config = simple_hyperparameter_tuning(
            train_df, val_df, test_df,
            dataset_name, imputation, dropped_vars,
            default_config=config,
            param_grid=param_grid,
            )
        
        
        # Use the best configuration
        config = best_config
        print(f"Using best configuration: {config}")
    
    # Train and evaluate with the selected configuration
    metrics = train_and_evaluate(
        config, 
        train_df=train_df, 
        val_df=val_df, 
        test_df=test_df,
        dataset_name=dataset_name,
        imputation=imputation,
        dropped_vars=dropped_vars,
    )
    
    print("Final evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")