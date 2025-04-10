import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn
import torch.optim as optim
import time

from DMT_functions import MultiParticipantDataset, LSTMModel, SimpleRNNModel, GRUModel
from DMT_functions import normalize, predict_and_plot


# --- Mac acceleration ---

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("Using Mac acceleration")
else:
    mps_device = torch.device("cpu")
    print("Using CPU")





################################## Script ######################################
# Load the dataset

# dataset and label details
dataset_name = "mean_mode_imputation_combinedAppCat"
dropped_vars = [""]
imputation = "mean_mode"
df = pd.read_csv(f'tables/preprocessed/{dataset_name}.csv')

# hyperparameters for dataset
seq_length = 5
batch_size = 32

# hyperparameters for the normalization
TRANSFORM_TARGET = True
SCALER_TYPE = "MinMaxScaler"

# hyperparameters for the train test split
SPLIT_WITHIN_PARTICIPANTS = True
proportion_train = 0.8
SHUFFLE_TRAIN = True
SHUFFLE_TEST = False

# hyperparameters for the model
model_type = "LSTM"  # Choose from 'LSTM', 'SimpleRNN', or 'GRU'
hidden_dim = 64             # Number of LSTM units
num_layers = 2
output_dim = 1                   # For regression (predicting a single value)
num_epochs = 20
learning_rate = 0.001
dropout = 0.2

clip_grad = True  # Whether to clip gradients
max_grad_norm = 1.0  # Maximum norm for gradient clipping


# ------------------------- Train/test/val split

dfs_train = []
dfs_test = []

if SPLIT_WITHIN_PARTICIPANTS:
    # split within participants
    for participant, group in df.groupby('id_num'):
        group = group.sort_values(by='day')
        # Calculate the split index based on the proportion
        split_idx = int(len(group) * proportion_train)
        # Ensure that the test set has at least seq_length + 1 samples
        if len(group) - split_idx < seq_length + 1:
            # Adjust the split index accordingly
            split_idx = len(group) - (seq_length + 1)
        if split_idx <= 0:
            # Option: Skip this participant if not enough data
            continue
        dfs_train.append(group.iloc[:split_idx])
        dfs_test.append(group.iloc[split_idx:])

    train_df = pd.concat(dfs_train)
    test_df = pd.concat(dfs_test)

    # get the start end end dates per participant per df
    train_start_dates = train_df.groupby('id_num')['day'].min()
    train_end_dates = train_df.groupby('id_num')['day'].max()
    test_start_dates = test_df.groupby('id_num')['day'].min()
    test_end_dates = test_df.groupby('id_num')['day'].max()

    # put in a dataframe with participant train start end and test start end
    dates_df = pd.DataFrame({
        "participant": train_start_dates.index,
        "train_start": train_start_dates.values,
        "train_end": train_end_dates.values,
        "test_start": test_start_dates.values,
        "test_end": test_end_dates.values,
    })

    dates_df.to_csv("tables/training_dates_split.csv", index=False)

# SPLIT ACROSS PARTICIPANTS
else:
    # Extract unique participant IDs
    participant_ids = df['id_num'].unique()

    # Split participants (e.g., 80% train, 20% test)
    train_ids, test_ids = train_test_split(participant_ids, test_size=0.2, random_state=42)
    print(f"Train IDs: {train_ids}")
    print(f"Test IDs: {test_ids}")

    # Filter the original DataFrame based on these IDs
    train_df = df[df['id_num'].isin(train_ids)].copy()
    test_df = df[df['id_num'].isin(test_ids)].copy()

    # Optional: sort your data by participant and day if not already sorted
    train_df.sort_values(by=['id_num', 'day'], inplace=True)
    test_df.sort_values(by=['id_num', 'day'], inplace=True)


# get mood descriptives
print("Train mood descriptives")
print(train_df["mood"].describe())
print("Test mood descriptives")
print(test_df["mood"].describe())


# Normalize the data ------------------

train_df_normalized, scaler, scaler_target = normalize(train_df, scaler=None, scaler_target=None, transform_target=TRANSFORM_TARGET, scaler_type=SCALER_TYPE)
test_df_normalized, _, _ = normalize(test_df, scaler=scaler, scaler_target=scaler_target, transform_target=TRANSFORM_TARGET, scaler_type=SCALER_TYPE)

print(f"Train shape: {train_df_normalized.shape}, Test shape: {test_df_normalized.shape}") # Train shape: (1230, 24), Test shape: (307, 24)


# Create datasets and dataloaders ------------------
# Create datasets
train_dataset = MultiParticipantDataset(train_df_normalized, seq_length=seq_length)
test_dataset = MultiParticipantDataset(test_df_normalized, seq_length=seq_length)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=SHUFFLE_TRAIN)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=SHUFFLE_TEST)

# Check the shape of the data
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")
for x, x_id, y in train_loader:
    print(f"x shape: {x.shape}, x_id shape: {x_id.shape}, y shape: {y.shape}")
    break



#### ----------------- Model training and evaluation -----------------

input_dim = len(train_dataset.features)  # e.g., 24
# Initialize model, loss function, and optimizer
if model_type == "LSTM":
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
elif model_type == "SimpleRNN":
    model = SimpleRNNModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
elif model_type == "GRU":
    model = GRUModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
else:
    raise ValueError("Invalid model type. Choose from 'LSTM', 'SimpleRNN', or 'GRU'.")

# Move model to the appropriate device
model = model.to(mps_device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # weight_decay=1e-5)

# Prepare lists to store loss values for plotting
train_epoch_losses = []
eval_epoch_losses = []
batch_losses = []
grad_norms = []  # list to store gradient norms for each batch
clipped_grad_norms = []  # list to store clipped gradient norms for each batch


# Training and evaluation loop
for epoch in range(num_epochs):
    model.train()
    train_loss_epoch = 0.0

    # --- Training ---
    for batch in train_loader:
        x_features, x_id, y = batch  # x_features: [batch, seq_length, input_dim]
        x_features = x_features.to(mps_device)
        x_id = x_id.to(mps_device)
        y = y.to(mps_device)  # y: [batch, output_dim]
        
        # Forward pass
        outputs = model(x_features)  # outputs shape: [batch, output_dim]
        loss = criterion(outputs.squeeze(), y)

        batch_losses.append(loss.item())
        
        # Backprop and optimization
        optimizer.zero_grad()
        loss.backward()

        if clip_grad:
            # Compute total gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            # Compute clipped gradient norm
            clipped_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    clipped_grad_norm += param_norm.item() ** 2
            clipped_grad_norm = clipped_grad_norm ** 0.5
            clipped_grad_norms.append(clipped_grad_norm)

        optimizer.step()
        
        train_loss_epoch += loss.item()
    
    avg_train_loss = train_loss_epoch / len(train_loader)
    train_epoch_losses.append(avg_train_loss)
    
    # --- Evaluation ---
    model.eval()
    eval_loss_epoch = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x_features, x_id, y = batch
            x_features = x_features.to(mps_device)
            x_id = x_id.to(mps_device)
            y = y.to(mps_device)
            outputs = model(x_features)
            loss = criterion(outputs.squeeze(), y)
            eval_loss_epoch += loss.item()
    
    avg_eval_loss = eval_loss_epoch / len(test_loader)
    eval_epoch_losses.append(avg_eval_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

# Plot the training and evaluation loss curves
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs+1), train_epoch_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs+1), eval_epoch_losses, label='Eval Loss', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training and Evaluation Loss Over Epochs")
plt.legend()
plt.show()

# plot batch losses
plt.figure(figsize=(12, 6))
plt.plot(batch_losses, label='Batch Loss')
plt.xlabel("Batch")
plt.ylabel("Loss (MSE)")
plt.title("Batch Loss Over Training")
plt.legend()
plt.show()

# plot gradient norms
plt.figure(figsize=(10, 5))
plt.plot(grad_norms)
plt.plot(clipped_grad_norms)
plt.legend(["Gradient Norm", "Clipped Gradient Norm"])
plt.xlabel("Batch iteration")
plt.ylabel("Gradient Norm (L2)")
plt.title("Gradient Norms during Training")
plt.show()



# append the csv with hyperparameters and losses
hyperparameters = {
    "dataset": dataset_name,
    "dropped_vars": dropped_vars,
    "imputation": imputation,
    "model": model_type,
    "train_size": len(train_df_normalized),
    "sequence_length": seq_length,
    "scaler": SCALER_TYPE,
    "scaler_target": TRANSFORM_TARGET,
    "batch_size": batch_size,
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "output_dim": output_dim,
    "dropout": dropout,
    "clip_grad": clip_grad,
    "max_grad_norm": max_grad_norm,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "train_loss": train_epoch_losses[-1],
    "eval_loss": eval_epoch_losses[-1],
    "Features": train_dataset.features,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
print(hyperparameters)


hyperparameters_df = pd.DataFrame([hyperparameters])
if not os.path.exists("tables/hyperparameters_lstm.csv"):
    # create the csv with the hyperparameters
    hyperparameters_df.to_csv("tables/hyperparameters_lstm.csv", index=False, header=True)
hyperparameters_df = pd.DataFrame([hyperparameters])
hyperparameters_df.to_csv("tables/hyperparameters_lstm.csv", mode='a', header=False, index=False)


# Call the function to predict and plot
df_results_train, mae_train, mse_train, rmse_train, r2_train = predict_and_plot(model, train_loader, train_dataset, target_scaler=scaler_target, show_plot=False, save_html=True, title="train", scaler_type=SCALER_TYPE)
df_results, mae, mse, rmse, r2 = predict_and_plot(model, test_loader, test_dataset, target_scaler=scaler_target, show_plot=False, save_html=True, title="test", scaler_type=SCALER_TYPE)


results = {}
results["lstm"] = {
        'R2': r2,
        'R2_train': r2_train,
        'MAE': mae,
        'MAE_train': mae_train,
        'MSE': mse,
        'MSE_train': mse_train,
        'RMSE': rmse,
        'RMSE_train': rmse_train,
        'dataset': dataset_name
    }

results_df = pd.DataFrame(results).T
if not os.path.exists('tables/results'):
    os.makedirs('tables/results', exist_ok=True)
    results_df.to_csv('tables/results/model_results.csv', index=True, header=True)
results_df.to_csv("tables/results/model_results.csv", mode='a', header=False, index=True)