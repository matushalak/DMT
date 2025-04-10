
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

import json
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from datetime import datetime


# Dataset class -------------------------
class MultiParticipantDataset(Dataset):
    def __init__(self, df, seq_length, target_col='mood', id_col='id_num', include_target_in_features=True):
        """
        df: pandas DataFrame sorted by time.
        seq_length: number of time steps in each sample.
        target_col: the column we want to predict.
        """
        df = df.drop(columns=["next_day", "next_day_mood"])
        
        self.seq_length = seq_length
        self.target_col = target_col
        self.id_col = id_col
        
        df.sort_values(by=[id_col, 'day'], inplace=True)
        self.data = df.reset_index(drop=True)

        if include_target_in_features:
            self.features = [col for col in self.data.columns if col not in [target_col, "day"]]
        else:
            self.features = [col for col in self.data.columns if col not in [target_col, id_col, "day"]]

        # Precompute valid indices where the sequence is within the same participant.
        self.valid_indices = []
        for i in range(len(self.data) - self.seq_length):
            participant_id = self.data.iloc[i][self.id_col]
            if all(self.data.iloc[i:i+self.seq_length][self.id_col] == participant_id):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Use precomputed valid index.
        real_idx = self.valid_indices[idx]
        row = self.data.iloc[real_idx]
        participant_id = row[self.id_col]
        
        x_features = self.data.iloc[real_idx:real_idx+self.seq_length][self.features].values.astype(np.float32)
        x_id = np.array([participant_id] * self.seq_length, dtype=np.int64)
        
        # The target is the next time step's mood
        y = self.data.iloc[real_idx+self.seq_length][self.target_col]
        
        return torch.tensor(x_features),torch.tensor(x_id), torch.tensor(y).float()


# Model classes -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Two-layer LSTM with dropout applied to outputs of each layer (except the last)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Fully-connected layer to output the final prediction
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM; out shape: [batch_size, seq_length, hidden_dim]
        out, _ = self.lstm(x, (h0, c0))
        
        # Use the last time step's output for prediction; shape: [batch_size, hidden_dim]
        out = out[:, -1, :]
        out = self.fc(out)  # shape: [batch_size, output_dim]
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Two-layer GRU with dropout applied between layers (if num_layers > 1)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout)
        
        # Fully-connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward propagate through GRU
        out, _ = self.gru(x, h0)  # out shape: [batch_size, seq_length, hidden_dim]
        
        # Use the output from the last time step for prediction
        out = out[:, -1, :]  # shape: [batch_size, hidden_dim]
        out = self.fc(out)   # shape: [batch_size, output_dim]
        return out


class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(SimpleRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Two-layer RNN with dropout applied between layers (if num_layers > 1)
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout)
        
        # Fully-connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward propagate through RNN
        out, _ = self.rnn(x, h0)  # out shape: [batch_size, seq_length, hidden_dim]
        
        # Use the output from the last time step for prediction
        out = out[:, -1, :]  # shape: [batch_size, hidden_dim]
        out = self.fc(out)   # shape: [batch_size, output_dim]
        return out
    
# ------------------------------------------------------


def normalize(df, scaler=None, scaler_target=None, transform_target=False, scaler_type="StandardScaler"):
    df = df.copy()
    features = [col for col in df.columns if col not in ['id_num', 'day', "date", "next_day_mood", "next_day", "mood"]]
    
    if scaler is None:
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
    
    # Scale the features
    df[features] = scaler.fit_transform(df[features])
    
    if transform_target:
        if scaler_target is None:
            if scaler_type == "StandardScaler":
                scaler_target = StandardScaler()
            elif scaler_type == "MinMaxScaler":
                scaler_target = MinMaxScaler()

        # Scale only the target column "mood"
        df["mood"] = scaler_target.fit_transform(df[["mood"]])

        # print("scaler properties:")
        # print(scaler.mean_)
        # print(scaler.scale_)
        if scaler_type == "StandardScaler":
            print("scaler properties:")
            print(scaler.mean_)
            print(scaler.scale_)
        
        return df, scaler, scaler_target
    else:
        return df, scaler, None
    

def predict_and_plot(model, data_loader, test_dataset, target_scaler=None, show_plot=True, save_html=True, title="predictions", scaler_type="StandardScaler"):
    """
    Runs predictions on the data_loader using model, builds a results DataFrame using the
    test_dataset's original data (which includes the 'day' and 'id_num' columns), and then plots
    real vs predicted values with Plotly using the 'day' column for the x-axis and a dropdown
    to select different participants.

    Parameters:
        model: Trained PyTorch model.
        data_loader: DataLoader for the dataset to predict on.
        test_dataset: The dataset instance (e.g., MultiParticipantDataset) used to create data_loader.
                      It must have a 'data' attribute containing the original DataFrame with a 'day' column.
        target_scaler: (Optional) Scaler used to normalize the target data.
    """
    model.eval()
    all_predictions = []
    all_targets = []

    # move everything to cpu
    model.to("cpu")

    
    # Run model predictions over the data_loader
    with torch.no_grad():
        for batch in data_loader:
            x_features, x_id, y = batch
            x_features = x_features.to("cpu")
            x_id = x_id.to("cpu")
            y = y.to("cpu")
            outputs = model(x_features)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate all predictions and targets into arrays.
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # print mean sd, min max of predictions and targets
    print("Predictions mean:", np.mean(all_predictions))
    print("Predictions sd:", np.std(all_predictions))
    print("Predictions min:", np.min(all_predictions))
    print("Predictions max:", np.max(all_predictions))
    print("Targets mean:", np.mean(all_targets))
    print("Targets sd:", np.std(all_targets))
    print("Targets min:", np.min(all_targets))
    print("Targets max:", np.max(all_targets))

    
    # Inverse transform if a target scaler is provided.
    if target_scaler is not None:
        if scaler_type == "StandardScaler":
            print("Target scaler mean:", target_scaler.mean_)
            print("Target scaler scale:", target_scaler.scale_)
        all_predictions = target_scaler.inverse_transform(all_predictions)
        all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1))
    

    # Compute the correct slice of the original DataFrame.
    # The i-th prediction corresponds to data row at index (i + seq_length)
    start_idx = test_dataset.seq_length
    end_idx = start_idx + len(test_dataset)
    df_results = test_dataset.data.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    # Add prediction and target columns to the results DataFrame.
    df_results['Real'] = all_targets.reshape(-1)
    df_results['Predicted'] = all_predictions.reshape(-1)
    
    # Get unique participant IDs from the results DataFrame.
    participant_col = test_dataset.id_col  # e.g., 'id_num'
    participants = df_results[participant_col].unique()
    
    # Build Plotly traces for each participant: two traces (real & predicted) per participant.
    traces = []
    for p in participants:
        df_p = df_results[df_results[participant_col] == p]
        traces.append(go.Scatter(
            x=df_p['day'],
            y=df_p['Real'],
            mode='lines',
            name=f'Real ({p})',
            visible=False  # We'll control visibility via the dropdown.
        ))
        traces.append(go.Scatter(
            x=df_p['day'],
            y=df_p['Predicted'],
            mode='lines',
            name=f'Predicted ({p})',
            visible=False
        ))
    
    total_traces = len(traces)  # Should be 2 * number of participants.
    
    # Create dropdown buttons. Each button sets visibility so that only the two traces for one participant are shown.
    dropdown_buttons = []
    for i, p in enumerate(participants):
        visibility = [False] * total_traces
        # For participant p, set traces at indices 2*i and 2*i+1 to True.
        visibility[2*i] = True
        visibility[2*i+1] = True
        button = dict(
            label=str(p),
            method="update",
            args=[{"visible": visibility},
                  {"title": f"Real vs Predicted Mood Values for Participant {p}",
                   "xaxis": {"title": "Day"},
                   "yaxis": {"title": "Mood Value"}}]
        )
        dropdown_buttons.append(button)
    
    # Set the initial visibility: show the first participant.
    initial_visibility = [False] * total_traces
    initial_visibility[0] = True
    initial_visibility[1] = True
    for i in range(total_traces):
        traces[i].visible = initial_visibility[i]
    
    # Build the figure with all traces and add the dropdown menu.
    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_buttons,
                x=1.1,
                y=1.0,
                showactive=True
            )
        ],
        title=f"Real vs Predicted Mood Values for Participant {participants[0]}",
        xaxis_title="Day",
        yaxis_title="Mood Value"
    )
    
    if show_plot:
        fig.show()
    if save_html:
        outdir = "figures/plotly/predictions"
        os.makedirs(outdir, exist_ok=True)
        fig.write_html(os.path.join(outdir, f"predictions_{title}.html"))

    # MAE RMSE R2
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")
    return df_results, mae, mse, rmse, r2
    
    

def create_data_split(df, proportion_train=0.7, proportion_val=0.15, split_within_participants=True, seq_length=5):
    """
    Split data into train, validation, and test sets
    
    Args:
        df: DataFrame containing the data
        proportion_train: Proportion of data for training
        proportion_val: Proportion of data for validation
        split_within_participants: Whether to split within participants or across participants
        seq_length: Sequence length required for model
        
    Returns:
        train_df, val_df, test_df: DataFrames for training, validation, and testing
    """
    
    dfs_train = []
    dfs_val = []
    dfs_test = []
    
    if split_within_participants:
        # Split within participants (chronologically)
        for participant, group in df.groupby('id_num'):
            group = group.sort_values(by='day')
            
            # Calculate split indices
            train_idx = int(len(group) * proportion_train)
            val_idx = int(len(group) * (proportion_train + proportion_val))
            
            # Ensure each set has at least seq_length + 1 samples
            min_samples = seq_length + 1
            
            # Adjust if there's not enough data
            if len(group) < 3 * min_samples:
                # Skip this participant if not enough data
                continue
                
            # Ensure train set has enough data
            if train_idx < min_samples:
                train_idx = min_samples
                
            # Ensure val set has enough data
            remaining = len(group) - train_idx
            val_samples = int(remaining * proportion_val / (1 - proportion_train))
            if val_samples < min_samples:
                val_samples = min_samples
            val_idx = train_idx + val_samples
            
            # Ensure test set has enough data
            if len(group) - val_idx < min_samples:
                val_idx = len(group) - min_samples
                
            # Add to respective sets
            dfs_train.append(group.iloc[:train_idx])
            dfs_val.append(group.iloc[train_idx:val_idx])
            dfs_test.append(group.iloc[val_idx:])
        
        train_df = pd.concat(dfs_train)
        val_df = pd.concat(dfs_val)
        test_df = pd.concat(dfs_test)
        
        # Record split dates for reference
        split_dates = []
        for participant, group in df.groupby('id_num'):
            if participant in train_df['id_num'].unique() and participant in val_df['id_num'].unique() and participant in test_df['id_num'].unique():
                participant_train = train_df[train_df['id_num'] == participant]
                participant_val = val_df[val_df['id_num'] == participant]
                participant_test = test_df[test_df['id_num'] == participant]
                
                split_dates.append({
                    "participant": participant,
                    "train_start": participant_train['day'].min(),
                    "train_end": participant_train['day'].max(),
                    "val_start": participant_val['day'].min(),
                    "val_end": participant_val['day'].max(),
                    "test_start": participant_test['day'].min(),
                    "test_end": participant_test['day'].max(),
                })
        
        dates_df = pd.DataFrame(split_dates)
        dates_df.to_csv("tables/training_dates_split.csv", index=False)
        
    else:
        # Split across participants
        participant_ids = df['id_num'].unique()
        
        # Split participants (70% train, 15% val, 15% test)
        train_val_ids, test_ids = train_test_split(participant_ids, test_size=1-proportion_train-proportion_val, random_state=42)
        train_ids, val_ids = train_test_split(train_val_ids, test_size=proportion_val/(proportion_train+proportion_val), random_state=42)
        
        print(f"Train IDs: {train_ids}")
        print(f"Validation IDs: {val_ids}")
        print(f"Test IDs: {test_ids}")
        
        # Filter the original DataFrame based on these IDs
        train_df = df[df['id_num'].isin(train_ids)].copy()
        val_df = df[df['id_num'].isin(val_ids)].copy()
        test_df = df[df['id_num'].isin(test_ids)].copy()
        
        # Sort by participant and day
        train_df.sort_values(by=['id_num', 'day'], inplace=True)
        val_df.sort_values(by=['id_num', 'day'], inplace=True)
        test_df.sort_values(by=['id_num', 'day'], inplace=True)
    
    return train_df, val_df, test_df


def save_model(model, hyperparams, metrics, path="models"):
    """
    Save model, hyperparameters, and metrics
    
    Args:
        model: Trained model
        hyperparams: Dictionary of hyperparameters
        metrics: Dictionary of evaluation metrics
        path: Directory path for saving
    """
    # Create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    # Create a timestamp-based model name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"{hyperparams['model']}_{timestamp}"
    model_path = os.path.join(path, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), os.path.join(model_path, "model_state.pt"))
    
    # Save hyperparameters
    with open(os.path.join(model_path, "hyperparams.json"), 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    # Save metrics
    with open(os.path.join(model_path, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model saved to {model_path}")
    return model_path


def load_model(model_path, device):
    """
    Load a saved model and its parameters
    
    Args:
        model_path: Path to saved model directory
        device: Device to load model to
        
    Returns:
        model: Loaded model
        hyperparams: Dictionary of hyperparameters
        metrics: Dictionary of evaluation metrics
    """

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # Load hyperparameters
    with open(os.path.join(model_path, "hyperparams.json"), 'r') as f:
        hyperparams = json.load(f)
    
    # Load metrics
    with open(os.path.join(model_path, "metrics.json"), 'r') as f:
        metrics = json.load(f)
    
    # Initialize model
    input_dim = hyperparams['input_dim']
    hidden_dim = hyperparams['hidden_dim']
    num_layers = hyperparams['num_layers']
    output_dim = hyperparams['output_dim']
    dropout = hyperparams['dropout']
    model_type = hyperparams['model']
    
    if model_type == "LSTM":
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    elif model_type == "SimpleRNN":
        model = SimpleRNNModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    elif model_type == "GRU":
        model = GRUModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Load model state
    model.load_state_dict(torch.load(os.path.join(model_path, "model_state.pt")))
    model = model.to(device)
    model.eval()
    
    return model, hyperparams, metrics


def train_and_evaluate(config, checkpoint_dir=None, train_df=None, val_df=None, test_df=None, 
                      dataset_name=None, imputation=None, dropped_vars=None):
    """
    Train and evaluate a model with given hyperparameters.
    This version has all Ray Tune dependencies removed.
    
    Args:
        config: Dictionary of hyperparameters
        checkpoint_dir: Directory for checkpoints (unused without Ray Tune)
        train_df, val_df, test_df: DataFrames for training, validation, and testing
        dataset_name, imputation, dropped_vars: Data information
    
    Returns:
        Dictionary of evaluation metrics
    """
    import sys
    import os
    import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    # Extract hyperparameters from config
    seq_length = config["seq_length"]
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    learning_rate = config["learning_rate"]
    dropout = config["dropout"]
    model_type = config["model_type"]
    num_epochs = config["num_epochs"]
    transform_target = config["transform_target"]
    scaler_type = config["scaler_type"]
    
    # Normalize data
    train_df_normalized, scaler, scaler_target = normalize(
        train_df, scaler=None, scaler_target=None, 
        transform_target=transform_target, scaler_type=scaler_type
    )
    val_df_normalized, _, _ = normalize(
        val_df, scaler=scaler, scaler_target=scaler_target, 
        transform_target=transform_target, scaler_type=scaler_type
    )
    test_df_normalized, _, _ = normalize(
        test_df, scaler=scaler, scaler_target=scaler_target, 
        transform_target=transform_target, scaler_type=scaler_type
    )
    
    # Create datasets
    train_dataset = MultiParticipantDataset(train_df_normalized, seq_length=seq_length)
    val_dataset = MultiParticipantDataset(val_df_normalized, seq_length=seq_length)
    test_dataset = MultiParticipantDataset(test_df_normalized, seq_length=seq_length)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for validation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for testing
    
    # Initialize model
    input_dim = len(train_dataset.features)
    output_dim = 1
    
    if model_type == "LSTM":
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    elif model_type == "SimpleRNN":
        model = SimpleRNNModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    elif model_type == "GRU":
        model = GRUModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Determine the appropriate device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load checkpoint if available (manual checkpoint handling)
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "model.pt")):
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
        print(f"Loaded model from {checkpoint_dir}")
    
    # Storage for losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5  # Early stopping patience
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x_features, x_id, y = batch
            x_features = x_features.to(device)
            x_id = x_id.to(device)
            y = y.to(device)
            
            # Forward pass
            outputs = model(x_features)
            loss = criterion(outputs.squeeze(), y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (using a different variable name to avoid conflicts)
            do_clip_grads = config.get("clip_gradients", True)
            if do_clip_grads:
                max_grad_norm = config.get("max_grad_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x_features, x_id, y = batch
                x_features = x_features.to(device)
                x_id = x_id.to(device)
                y = y.to(device)
                
                outputs = model(x_features)
                loss = criterion(outputs.squeeze(), y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model checkpoint (optional)
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pt')
        else:
            patience_counter += 1
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            x_features, x_id, y = batch
            x_features = x_features.to(device)
            x_id = x_id.to(device)
            y = y.to(device)
            
            outputs = model(x_features)
            loss = criterion(outputs.squeeze(), y)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # Detailed evaluation with metrics
    df_train_results, mae_train, mse_train, rmse_train, r2_train = predict_and_plot(
        model, train_loader, train_dataset, target_scaler=scaler_target, 
        show_plot=False, save_html=False, title="train", scaler_type=scaler_type
    )
    
    df_val_results, mae_val, mse_val, rmse_val, r2_val = predict_and_plot(
        model, val_loader, val_dataset, target_scaler=scaler_target, 
        show_plot=False, save_html=False, title="val", scaler_type=scaler_type
    )
    
    df_test_results, mae_test, mse_test, rmse_test, r2_test = predict_and_plot(
        model, test_loader, test_dataset, target_scaler=scaler_target, 
        show_plot=False, save_html=False, title="test", scaler_type=scaler_type
    )
    
    # Collect metrics
    metrics = {
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "test_loss": avg_test_loss,
        "mae_train": mae_train,
        "mse_train": mse_train,
        "rmse_train": rmse_train,
        "r2_train": r2_train,
        "mae_val": mae_val,
        "mse_val": mse_val,
        "rmse_val": rmse_val,
        "r2_val": r2_val,
        "mae_test": mae_test,
        "mse_test": mse_test,
        "rmse_test": rmse_test,
        "r2_test": r2_test
    }
    
    # Collect hyperparameters
    hyperparams = {
        "dataset": dataset_name,
        "dropped_vars": dropped_vars,
        "imputation": imputation,
        "model": model_type,
        "train_size": len(train_df_normalized),
        "val_size": len(val_df_normalized),
        "test_size": len(test_df_normalized),
        "sequence_length": seq_length,
        "scaler": scaler_type,
        "scaler_target": transform_target,
        "batch_size": batch_size,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "output_dim": output_dim,
        "dropout": dropout,
        "num_epochs": epoch + 1,  # Actual number of epochs run
        "learning_rate": learning_rate,
        "features": train_dataset.features,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save the model
    save_model(model, hyperparams, metrics)
    
    # Generate and save plots
    predict_and_plot(
        model, train_loader, train_dataset, target_scaler=scaler_target, 
        show_plot=False, save_html=True, title="train", scaler_type=scaler_type
    )
    predict_and_plot(
        model, val_loader, val_dataset, target_scaler=scaler_target, 
        show_plot=False, save_html=True, title="val", scaler_type=scaler_type
    )
    predict_and_plot(
        model, test_loader, test_dataset, target_scaler=scaler_target, 
        show_plot=False, save_html=True, title="test", scaler_type=scaler_type
    )
    
    # Save loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='o')
    plt.axhline(y=avg_test_loss, color='r', linestyle='-', label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training, Validation, and Test Loss")
    plt.legend()
    plt.savefig("figures/loss_curves.png")
    
    # Save results to CSV
    results = {}
    results[f"{model_type}_{time.strftime('%Y%m%d_%H%M%S')}"] = {
        'R2_train': r2_train,
        'R2_val': r2_val,
        'R2_test': r2_test,
        'MAE_train': mae_train,
        'MAE_val': mae_val,
        'MAE_test': mae_test,
        'MSE_train': mse_train,
        'MSE_val': mse_val,
        'MSE_test': mse_test,
        'RMSE_train': rmse_train,
        'RMSE_val': rmse_val,
        'RMSE_test': rmse_test,
        'dataset': dataset_name
    }
    
    results_df = pd.DataFrame(results).T
    if not os.path.exists('tables/results'):
        os.makedirs('tables/results', exist_ok=True)
        results_df.to_csv('tables/results/model_results.csv', index=True, header=True)
    else:
        results_df.to_csv('tables/results/model_results.csv', mode='a', header=False, index=True)
    
    return metrics


def simple_hyperparameter_tuning(train_df, val_df, test_df, dataset_name, imputation, dropped_vars, default_config, param_grid):
    """
    Run simple hyperparameter tuning without using Ray Tune
    
    Args:
        train_df, val_df, test_df: DataFrames for training, validation, and testing
        dataset_name, imputation, dropped_vars: Data information
    
    Returns:
        Best configuration of hyperparameters
    """
    
    print("Starting simple hyperparameter tuning...")
        
    
    # Create directory to store results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"tuning_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # For full grid search, we'd use this, but it would be too many combinations
    # all_combinations = list(itertools.product(*param_grid.values()))
    
    # Instead, we'll use a few key hyperparameters for our grid search
    # and keep others at default values
    key_params = {
        "model_type": param_grid["model_type"],
        "hidden_dim": param_grid["hidden_dim"],
        "seq_length": param_grid["seq_length"],
        "learning_rate": param_grid["learning_rate"]
    }
    
    # Create combinations of just the key parameters
    key_param_names = list(key_params.keys())
    key_param_values = list(key_params.values())
    key_combinations = list(itertools.product(*key_param_values))
    
    print(f"Testing {len(key_combinations)} hyperparameter combinations...")
    
    # Set up result tracking
    all_results = []
    best_val_loss = float('inf')
    best_config = None
    
    
    
    # Loop through combinations
    for i, combination in enumerate(key_combinations):
        # Create config with this combination
        config = default_config.copy()
        for j, param_name in enumerate(key_param_names):
            config[param_name] = combination[j]
        
        print(f"\nTrial {i+1}/{len(key_combinations)}: {config}")
        
        # Train and evaluate with this configuration
        try:
            start_time = time.time()
            metrics = train_and_evaluate(
                config, 
                train_df=train_df, 
                val_df=val_df, 
                test_df=test_df,
                dataset_name=dataset_name,
                imputation=imputation,
                dropped_vars=dropped_vars
            )
            end_time = time.time()
            
            # Calculate duration
            duration = end_time - start_time
            
            # Add trial info
            trial_info = {
                "trial": i+1,
                "config": config,
                "metrics": metrics,
                "duration_seconds": duration
            }
            
            all_results.append(trial_info)
            
            # Check if this is the best configuration
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_config = config.copy()
                print(f"New best validation loss: {best_val_loss:.6f}")
            
            # Save results so far to prevent data loss if the process is interrupted
            with open(os.path.join(results_dir, 'tuning_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)
                
            # Create summary DataFrame for easy analysis
            summary_data = []
            for result in all_results:
                row = {}
                # Add config parameters
                for param_name, param_value in result["config"].items():
                    if isinstance(param_value, (int, float, str, bool)):
                        row[param_name] = param_value
                
                # Add metrics
                for metric_name, metric_value in result["metrics"].items():
                    row[metric_name] = metric_value
                
                row["duration_seconds"] = result["duration_seconds"]
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(results_dir, 'tuning_summary.csv'), index=False)
            
            # Plot progress
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(range(1, len(summary_df) + 1), summary_df['val_loss'], 'bo-')
            plt.title('Validation Loss')
            plt.xlabel('Trial')
            plt.ylabel('Loss')
            
            plt.subplot(2, 2, 2)
            plt.plot(range(1, len(summary_df) + 1), summary_df['r2_val'], 'go-')
            plt.title('Validation R²')
            plt.xlabel('Trial')
            plt.ylabel('R²')
            
            plt.subplot(2, 2, 3)
            plt.plot(range(1, len(summary_df) + 1), summary_df['train_loss'], 'ro-')
            plt.plot(range(1, len(summary_df) + 1), summary_df['val_loss'], 'bo-')
            plt.title('Train vs Validation Loss')
            plt.xlabel('Trial')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'])
            
            plt.subplot(2, 2, 4)
            if len(summary_df) > 1:
                # Group by model_type and calculate mean validation loss
                plt.bar(summary_df['model_type'], summary_df['val_loss'])
                plt.title('Validation Loss by Model Type')
                plt.xlabel('Model Type')
                plt.ylabel('Avg. Validation Loss')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'tuning_progress.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error in trial {i+1}: {e}")
            # Log the error but continue with other hyperparameter combinations
            with open(os.path.join(results_dir, 'error_log.txt'), 'a') as f:
                f.write(f"Trial {i+1} failed with error: {str(e)}\n")
                f.write(f"Config: {config}\n\n")
    
    print("\nHyperparameter tuning completed!")
    print(f"Results saved to {results_dir}")
    
    if best_config:
        print("\nBest hyperparameters:")
        for param, value in best_config.items():
            print(f"  {param}: {value}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        # Save best config as a separate file
        with open(os.path.join(results_dir, 'best_config.json'), 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # Create comparison plots for key parameters
        if len(all_results) > 1:
            for param in key_param_names:
                plt.figure(figsize=(10, 6))
                
                # Group by parameter and calculate mean validation loss
                param_groups = summary_df.groupby(param)['val_loss'].mean().reset_index()
                
                plt.bar(param_groups[param].astype(str), param_groups['val_loss'])
                plt.title(f'Validation Loss by {param}')
                plt.xlabel(param)
                plt.ylabel('Avg. Validation Loss')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'param_impact_{param}.png'))
                plt.close()
    else:
        print("No successful trials completed.")
        best_config = default_config
    
    return best_config


def modify_train_and_evaluate_for_simple_tuning():
    """
    Modify the train_and_evaluate function to prevent checkpoint errors when used without Ray Tune.
    This function prints the code you should use to replace the Ray Tune reporting section in train_and_evaluate.
    """
    print("""
    Replace the Ray Tune reporting section in your train_and_evaluate function with this code:
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    # Skip Ray Tune reporting entirely - it causes errors
    # The simple_hyperparameter_tuning function doesn't need it
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    """)