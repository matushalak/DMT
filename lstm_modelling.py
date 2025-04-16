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

from lstm_utils import LSTMModel, SimpleRNNModel, GRUModel
from lstm_utils import train_and_evaluate
from lstm_utils import save_model, load_model, simple_hyperparameter_tuning, save_predictions_to_csv, load_and_prepare_csv, plot_timeseries_subplots_date


# --- Mac acceleration ---
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("Using Mac acceleration")
else:
    mps_device = torch.device("cpu")
    print("Using CPU")


if __name__ == "__main__":
    # Load the dataset
    dataset_name = "df_ready_date"
    dropped_vars = ["appCat"]
    imputation = "mean_mode"
    df = pd.read_csv(f'tables/imputed/{dataset_name}.csv')

    df.drop(columns=["categorical_target"], inplace=True)



    # combine all app.Categorical features into one
    df['appCat'] = df[df.columns[df.columns.str.contains('appCat')]].sum(axis=1)
    # Drop individual app.Categorical features
    df.drop(columns=df.columns[df.columns.str.contains('appCat.')], inplace=True)
    
    # select features
    # features = ["id_num", "date", "target",'mood_last_daily', 'circumplex.QUADRANT_daily_2', 'weekday_1', 'weekday_4', 'weekday_3', 'month_5', 'mva7_circumplex.valence_std.slide_daily', 'change_screen_sum_daily', 'circumplex.valence_std.slide_daily', 'activity_max_daily', 'circumplex.QUADRANT_daily_3', 'mood_std.slide_daily', 'circumplex.valence_mean_daily', 'mva7_change_screen_sum_daily', 'mood_first_daily', 'circumplex.valence_last_daily', 'mva7_mood_mean_daily']

    most_correlated_selection = True
    if most_correlated_selection:
        # select 15 most correlated features absolutely (but only numerical features)
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        # Calculate the correlation matrix
        corr = df[numerical_features].corr()        
        corr_target = corr["target"].abs().sort_values(ascending=False)
        # Select the top 15 features
        top_15_features = corr_target.index[:15].tolist()
        print(f"Top 15 features: {top_15_features}")

        # add id_num, "target", "date" to the top 15 features
        top_15_features = ["id_num", "date", "target"] + top_15_features
        top_15_features = list(dict.fromkeys(top_15_features)) # remove duplicates
        # add time_of_day if it exists in df
        if "time_of_day" in df.columns:
            top_15_features.append("time_of_day")


        # df = df[top_15_features]
        features = top_15_features


    # Select only the relevant columns
    # df = df[features]

    do_hyperparameter_tuning = False  # Set to True to enable tuning
    
    
    # Default hyperparameters
    if dataset_name == "df_ready_date":
        config = {
            # Data parameters
            "seq_length": 5,                        # Number of days to use for prediction
            "batch_size": 32,                       # Batch size for training
            
            # Model architecture
            "model_type": "GRU",                   # Model type (LSTM, GRU, or SimpleRNN)
            "hidden_dim": 32,                       # Size of LSTM hidden layer
            "num_layers": 2,                        # Number of LSTM layers
            "dropout": 0.2,                         # Dropout rate
            
            # Training parameters
            "learning_rate": 0.001,                 # Learning rate for Adam optimizer
            "num_epochs": 2,                       # Maximum number of training epochs
            "clip_gradients": False,                 # Whether to use gradient clipping
            "max_grad_norm": 1.0,                   # Maximum gradient norm if clipping
            
            # Data processing
            "transform_features": True,             # Whether to normalize features
            "transform_target": True,               # Whether to normalize target
            "scaler_type": "MinMaxScaler",        # Scaler type (StandardScaler or MinMaxScaler)
            "shuffle_data": True,                   # Whether to shuffle training data
            
            # Additional options for padding support
            "per_participant_normalization": True  # Whether to normalize per participant
        }
    elif dataset_name == "df_ready_both":
        config = {
            # Data parameters
            "seq_length": 21,                        # Number of days to use for prediction
            "batch_size": 32,                       # Batch size for training
            
            # Model architecture
            "model_type": "GRU",                   # Model type (LSTM, GRU, or SimpleRNN)
            "hidden_dim": 16,                       # Size of LSTM hidden layer
            "num_layers": 2,                        # Number of LSTM layers
            "dropout": 0.3,                         # Dropout rate
            
            # Training parameters
            "learning_rate": 0.001,                 # Learning rate for Adam optimizer
            "num_epochs": 20,                       # Maximum number of training epochs
            "clip_gradients": False,                 # Whether to use gradient clipping
            "max_grad_norm": 1.0,                   # Maximum gradient norm if clipping
            
            # Data processing
            "transform_features": True,             # Whether to normalize features
            "transform_target": True,               # Whether to normalize target
            "scaler_type": "MinMaxScaler",        # Scaler type (StandardScaler or MinMaxScaler)
            "shuffle_data": True,                   # Whether to shuffle training data
            
            "per_participant_normalization": True  # Whether to normalize per participant
        }
    
    # Create directories for outputs
    os.makedirs("tables/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    

    
    if do_hyperparameter_tuning:

        param_grid = {
            # Sequence parameters
            "seq_length": [15, 21, 27],                 # Test shorter and longer sequence windows
            
            # Model architecture parameters
            "model_type": ["GRU", "SimpleRNN", "LSTM"],            # Focus on the two stronger model types
            "hidden_dim": [16, 32],             # Test different capacities
            "num_layers": [1, 2],                     # More than 2 layers rarely helps for this task
            "dropout": [0.2, 0.3],               # Test different regularization strengths
            
            # Training parameters
            "batch_size": [32],                       # Keep batch size constant
            "learning_rate": [0.001],  # Test around the typical Adam default
            "num_epochs": [20],                       # Fix epochs and use early stopping
            "clip_gradients": [False],                 # Always use gradient clipping for stability
            "max_grad_norm": [1.0],              # Test different clipping thresholds
            
            # Data processing parameters
            "transform_features": [True],             # Always normalize features
            "transform_target": [True],               # Always normalize target
            "scaler_type": ["MinMaxScaler"],        # StandardScaler typically works better for LSTM
            "shuffle_data": [True],                   # Always shuffle training data
            
            # Additional options
            "per_participant_normalization": [True]  # Test both global and per-participant normalization
        }
        # Run hyperparameter tuning
        best_config = simple_hyperparameter_tuning(
            df,
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
        df, 
        checkpoint_dir=None, 
        dataset_name=None, 
        imputation=imputation, 
        dropped_vars=dropped_vars, 
        save_fig=True
    )
    
    print("Final evaluation metrics:")
    for key, value in metrics.items():
        # dont print predictions_
        if "predictions_" in key or "actuals_" in key:
            continue
        print(f"{key}: {value}")


    csv_files = {
    "Train": "predictions1/train_predictions.csv",
    "Val": "predictions1/val_predictions.csv",
    "Test": "predictions1/test_predictions.csv"
}

    # Process each CSV file and plot the time series.
    for split_name, csv_path in csv_files.items():
        if os.path.exists(csv_path):
            df = load_and_prepare_csv(csv_path)
            if dataset_name == "df_ready_both":
                plot_timeseries_subplots_both(df, split_name)
            else:
                plot_timeseries_subplots_date(df, split_name)
        else:
            print(f"File not found: {csv_path}")