
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
from datetime import datetime

import json
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Tuple, List, Optional, Union

import os
import time
import json
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataframe by participant ID, date, and time of date.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Sorted DataFrame
    """
    sort_columns = ['id_num', 'date']
    if 'time_of_date_non_encoded' in df.columns:
        sort_columns.append('time_of_date_non_encoded')
    
    return df.sort_values(by=sort_columns).reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    split_within_participants: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        split_within_participants: Whether to split within each participant or across participants
        
    Returns:
        train_df, val_df, test_df: DataFrames for training, validation, and testing
    """
    # Ensure the DataFrame is sorted
    df = sort_dataframe(df)
    
    # Calculate test ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Split within each participant
    if split_within_participants:
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for participant_id, group in df.groupby('id_num'):
            # Calculate split points
            total_points = len(group)
            
            # Skip participants with too few data points
            if total_points < 3:
                print(f"Skipping participant {participant_id} with only {total_points} data points")
                continue
            
            # Calculate split sizes (ensuring at least 1 point per split if possible)
            train_size = max(1, int(total_points * train_ratio))
            val_size = max(1, int(total_points * val_ratio))
            
            # Adjust to ensure we have at least 1 point for testing
            if train_size + val_size >= total_points:
                if val_size > 1:
                    val_size -= 1
                elif train_size > 1:
                    train_size -= 1
            
            # Split indices
            train_end = train_size
            val_end = train_size + val_size
            
            # Split the data
            train_dfs.append(group.iloc[:train_end])
            val_dfs.append(group.iloc[train_end:val_end])
            test_dfs.append(group.iloc[val_end:])
        
        # Combine splits
        train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
        val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
        test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()
        
    # Split across participants
    else:
        participant_ids = df['id_num'].unique()
        
        if len(participant_ids) < 3:
            print("Warning: Not enough participants for a proper split. Using within-participant split instead.")
            return split_data(df, train_ratio, val_ratio, split_within_participants=True)
        
        # Count data points per participant
        participant_counts = df.groupby('id_num').size()
        
        # Sort participants by data volume for better allocation
        sorted_participants = sorted(
            [(pid, participant_counts[pid]) for pid in participant_ids],
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Allocate participants to achieve target proportions
        train_ids, val_ids, test_ids = [], [], []
        train_points, val_points, test_points = 0, 0, 0
        
        # Greedy allocation to minimize deviation from target proportions
        for pid, count in sorted_participants:
            # Calculate which split would best approach target ratios
            current_total = train_points + val_points + test_points + count
            
            train_ratio_if_added = (train_points + count) / current_total
            val_ratio_if_added = (val_points + count) / current_total
            test_ratio_if_added = (test_points + count) / current_total
            
            # Calculate deviation from targets
            train_dev = abs(train_ratio_if_added - train_ratio)
            val_dev = abs(val_ratio_if_added - val_ratio)
            test_dev = abs(test_ratio_if_added - test_ratio)
            
            # Assign to minimize deviation
            if train_dev <= val_dev and train_dev <= test_dev:
                train_ids.append(pid)
                train_points += count
            elif val_dev <= train_dev and val_dev <= test_dev:
                val_ids.append(pid)
                val_points += count
            else:
                test_ids.append(pid)
                test_points += count
        
        # Ensure each split has at least one participant
        if not train_ids and participant_ids.size > 0:
            pid = participant_ids[0]
            train_ids.append(pid)
            if pid in val_ids:
                val_ids.remove(pid)
            elif pid in test_ids:
                test_ids.remove(pid)
        
        if not val_ids and participant_ids.size > 1:
            pid = participant_ids[1] if participant_ids[1] not in train_ids else participant_ids[0]
            val_ids.append(pid)
            if pid in test_ids:
                test_ids.remove(pid)
        
        if not test_ids and participant_ids.size > 2:
            for pid in participant_ids:
                if pid not in train_ids and pid not in val_ids:
                    test_ids.append(pid)
                    break
        
        # Create the split DataFrames
        train_df = df[df['id_num'].isin(train_ids)].copy()
        val_df = df[df['id_num'].isin(val_ids)].copy()
        test_df = df[df['id_num'].isin(test_ids)].copy()
    
    # Print achieved split ratios
    total_points = len(train_df) + len(val_df) + len(test_df)
    if total_points > 0:
        print(f"Data split complete:")
        print(f"  Train: {len(train_df)/total_points:.3f} ({len(train_df)} points)")
        print(f"  Validation: {len(val_df)/total_points:.3f} ({len(val_df)} points)")
        print(f"  Test: {len(test_df)/total_points:.3f} ({len(test_df)} points)")
    
    return train_df, val_df, test_df


def normalize_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    per_participant_normalization: bool = True,
    scaler_type: str = "StandardScaler",
    existing_scalers: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize data either globally or per participant.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        per_participant_normalization: Whether to normalize per participant
        scaler_type: Type of scaler to use ("StandardScaler" or "MinMaxScaler")
        existing_scalers: Pre-trained scalers to use
        
    Returns:
        Normalized DataFrame and dictionary of scalers
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # print("feature_cols", feature_cols)
    
    # Initialize scalers dictionary if not provided
    scalers = existing_scalers if existing_scalers is not None else {}
    
    # Create appropriate scaler
    def create_scaler():
        if scaler_type == "StandardScaler":
            return StandardScaler()
        elif scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Normalize per participant
    if per_participant_normalization:
        participant_ids = df['id_num'].unique()
        
        # Initialize scalers for features and target
        if 'features' not in scalers:
            scalers['features'] = {p_id: create_scaler() for p_id in participant_ids}
        if 'target' not in scalers:
            scalers['target'] = {p_id: create_scaler() for p_id in participant_ids}
        
        # Normalize each participant's data separately
        for p_id in participant_ids:
            mask = df['id_num'] == p_id
            
            # Skip if no data for this participant
            if not mask.any():
                continue
            
            # Normalize features
            features_data = df.loc[mask, feature_cols]
            df[feature_cols] = df[feature_cols].astype(float)
            if not features_data.empty:
                scaler = scalers['features'][p_id]
                df.loc[mask, feature_cols] = scaler.fit_transform(features_data)
            
            # Normalize target
            target_data = df.loc[mask, [target_col]]
            if not target_data.empty:
                scaler = scalers['target'][p_id]
                df.loc[mask, target_col] = scaler.fit_transform(target_data)
    
    # Global normalization
    else:
        # Initialize global scalers
        if 'features' not in scalers:
            scalers['features'] = create_scaler()
        if 'target' not in scalers:
            scalers['target'] = create_scaler()
        
        # Normalize features
        df[feature_cols] = scalers['features'].fit_transform(df[feature_cols])
        
        # Normalize target
        df[target_col] = scalers['target'].fit_transform(df[[target_col]])
    
    return df, scalers


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_length: int,
    id_col: str = 'id_num'
) -> Tuple[List[np.ndarray], List[int], List[float], List[np.ndarray]]:
    """
    Create padded sequences suitable for LSTM processing.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        seq_length: Number of time steps in each sequence
        id_col: Column that identifies participants
        
    Returns:
        sequences, participant_ids, targets, masks: Lists with prepared data
    """
    # Make sure data is sorted properly
    df = sort_dataframe(df)
    
    # Group data by participant
    grouped_data = df.groupby(id_col)
    
    # Initialize output lists
    sequences = []
    participant_ids = []
    targets = []
    masks = []
    
    # Process each participant's data
    for participant_id, participant_data in grouped_data:
        # Ensure data is sorted
        participant_data = participant_data.sort_values(by='date')
        
        # For each possible starting position
        max_idx = len(participant_data) - 1  # Need at least 1 target date
        
        for start_idx in range(max_idx):
            # Get available data for this sequence
            available_len = min(seq_length, max_idx - start_idx)
            end_idx = start_idx + available_len
            
            # Extract features
            x_seq = participant_data.iloc[start_idx:end_idx][feature_cols].values.astype(np.float32)
            
            # Create mask (1 for real data, 0 for padding)
            mask = np.ones(seq_length, dtype=np.float32)
            
            # Apply padding if needed
            if available_len < seq_length:
                padded_seq = np.zeros((seq_length, len(feature_cols)), dtype=np.float32)
                padded_seq[:available_len] = x_seq
                mask[available_len:] = 0
                x_seq = padded_seq
            
            # Get target (next date's value)
            if end_idx < len(participant_data):
                target = participant_data.iloc[end_idx][target_col]
                
                # Store the sequence data
                sequences.append(x_seq)
                participant_ids.append(participant_id)
                targets.append(target)
                masks.append(mask)
    
    return sequences, participant_ids, targets, masks


class LSTMDataset(Dataset):
    """
    Dataset for multi-participant time series prediction with LSTM.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_length: int,
        target_col: str = 'target',
        id_col: str = 'id_num',
        scaler_type: str = "StandardScaler",
        per_participant_normalization: bool = True,
        existing_scalers: Optional[Dict] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            df: Input DataFrame
            seq_length: Number of time steps in each sequence
            target_col: Column to predict
            id_col: Column that identifies participants
            scaler_type: Type of scaler ("StandardScaler" or "MinMaxScaler")
            per_participant_normalization: Whether to normalize per participant
            existing_scalers: Pre-trained scalers to use
        """
        # Store parameters
        self.seq_length = seq_length
        self.target_col = target_col
        self.id_col = id_col
        
        # Make a copy of the DataFrame
        df = df.copy()
        
        # Drop unnecessary columns
        if 'time_of_date_non_encoded' in df.columns:
            # Only used for sorting, then can be dropped
            df = sort_dataframe(df)
            df = df.drop(columns=['time_of_date_non_encoded'])
        
        # Define feature columns (excluding target, date and id columns)
        self.feature_cols = [col for col in df.columns if col not in [target_col, 'date', id_col, "id_num", "time_of_day_encoded", "next_date"]]
        
        # Normalize data
        df, self.scalers = normalize_data(
            df=df,
            feature_cols=self.feature_cols,
            target_col=target_col,
            per_participant_normalization=per_participant_normalization,
            scaler_type=scaler_type,
            existing_scalers=existing_scalers
        )
        
        # Create sequences
        sequences, participant_ids, targets, masks = prepare_sequences(
            df=df,
            feature_cols=self.feature_cols,
            target_col=target_col,
            seq_length=seq_length,
            id_col=id_col
        )
        
        # Store the prepared data
        self.sequences = sequences
        self.participant_ids = participant_ids
        self.targets = targets
        self.masks = masks
    
    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            features, participant_id, target, mask: Tensors with data
        """
        # Get data for this index
        features = self.sequences[idx]
        participant_id = self.participant_ids[idx]
        target = self.targets[idx]
        mask = self.masks[idx]
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        id_tensor = torch.tensor([participant_id] * self.seq_length, dtype=torch.int64)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        return features_tensor, id_tensor, target_tensor, mask_tensor
    
    def get_scalers(self) -> Dict:
        """Return the scalers used for normalization."""
        return self.scalers


def process_lstm_data(
    df: pd.DataFrame,
    seq_length: int = 7,
    target_col: str = 'target',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    split_within_participants: bool = True,
    scaler_type: str = "MinMaxScaler",
    per_participant_normalization: bool = True
) -> Tuple[LSTMDataset, LSTMDataset, LSTMDataset, Dict]:
    """
    Process data for LSTM model, handling splitting, normalization, and sequence creation.
    
    Args:
        df: Input DataFrame
        seq_length: Number of time steps in each sequence
        target_col: Column to predict
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        split_within_participants: Whether to split within participants
        scaler_type: Type of scaler to use
        per_participant_normalization: Whether to normalize per participant
        
    Returns:
        train_dataset, val_dataset, test_dataset, scalers: Datasets and normalization scalers
    """
    # Sort data
    df = sort_dataframe(df)
    
    # Split data
    train_df, val_df, test_df = split_data(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_within_participants=split_within_participants
    )

    # print shapes
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    # print target details
    print("Target column:", target_col)
    print("Target values in train set:", train_df[target_col].describe())
    print("Target values in validation set:", val_df[target_col].describe())
    print("Target values in test set:", test_df[target_col].describe())
    
    # Create training dataset (with new scalers)
    train_dataset = LSTMDataset(
        df=train_df,
        seq_length=seq_length,
        target_col=target_col,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_normalization
    )
    
    # Get trained scalers to use for validation and test sets
    trained_scalers = train_dataset.get_scalers()
    
    # Create validation dataset (using training scalers)
    val_dataset = LSTMDataset(
        df=val_df,
        seq_length=seq_length,
        target_col=target_col,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_normalization,
        existing_scalers=trained_scalers
    )
    
    # Create test dataset (using training scalers)
    test_dataset = LSTMDataset(
        df=test_df,
        seq_length=seq_length,
        target_col=target_col,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_normalization,
        existing_scalers=trained_scalers
    )
    
    return train_dataset, val_dataset, test_dataset, trained_scalers


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Two-layer LSTM with dropout applied to outputs of each layer (except the last)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Fully-connected layer to output the final prediction
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, participant_ids=None, mask=None):
        # x: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Handle padding if mask is provided
        if mask is not None:
            # Calculate sequence lengths from mask
            seq_lengths = mask.sum(dim=1).int()
            
            # Pack padded sequence
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Forward pass with packed sequence
            packed_output, _ = self.lstm(packed_input, (h0, c0))
            
            # Unpack the sequence
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Extract the last valid output for each sequence
            idx = (seq_lengths - 1).view(-1, 1).unsqueeze(1).expand(-1, 1, self.hidden_dim).long()
            last_out = out.gather(1, idx).squeeze(1)  # Shape: [batch_size, hidden_dim]
        else:
            # Standard forward pass without packing
            out, _ = self.lstm(x, (h0, c0))  # out shape: [batch_size, seq_length, hidden_dim]
            # Use the last time step's output
            last_out = out[:, -1, :]  # Shape: [batch_size, hidden_dim]
        
        # Final prediction through the fully connected layer
        fc_out = self.fc(last_out)  # Shape: [batch_size, 1]
        return fc_out  # This should be [batch_size, 1]

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-layer GRU with dropout applied between layers (if num_layers > 1)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout)
        
        # Fully-connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, participant_ids=None, mask=None):
        # x shape: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Handle padding if mask is provided
        if mask is not None:
            # Calculate sequence lengths from mask
            seq_lengths = mask.sum(dim=1).int()
            
            # Pack padded sequence
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Forward pass with packed sequence
            packed_output, _ = self.gru(packed_input, h0)
            
            # Unpack the sequence
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Extract the last valid output for each sequence
            # Fix: Convert index tensor to Long (int64) type
            idx = (seq_lengths - 1).view(-1, 1).unsqueeze(1).expand(-1, 1, self.hidden_dim).long()
            last_out = out.gather(1, idx).squeeze(1)
        else:
            # Standard forward pass without packing
            out, _ = self.gru(x, h0)
            # Use the last time step's output
            last_out = out[:, -1, :]
        
        # Final prediction
        out = self.fc(last_out)
        return out


class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(SimpleRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-layer RNN with dropout applied between layers (if num_layers > 1)
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout)
        
        # Fully-connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, participant_ids=None, mask=None):
        # x shape: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Handle padding if mask is provided
        if mask is not None:
            # Calculate sequence lengths from mask
            seq_lengths = mask.sum(dim=1).int()
            
            # Pack padded sequence
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Forward pass with packed sequence
            packed_output, _ = self.rnn(packed_input, h0)
            
            # Unpack the sequence
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Extract the last valid output for each sequence
            # Fix: Convert index tensor to Long (int64) type
            idx = (seq_lengths - 1).view(-1, 1).unsqueeze(1).expand(-1, 1, self.hidden_dim).long()
            last_out = out.gather(1, idx).squeeze(1)
        else:
            # Standard forward pass without packing
            out, _ = self.rnn(x, h0)
            # Use the last time step's output
            last_out = out[:, -1, :]
        
        # Final prediction
        out = self.fc(last_out)
        return out
# ------------------------------------------------------


def predict_and_plot(model, data_loader, test_dataset, target_scaler=None, show_plot=True, 
                  save_fig=True, title="predictions", scaler_type="StandardScaler"):
    """
    Runs predictions on the data_loader using model, builds a results DataFrame using the
    test_dataset's original data (which includes the 'date' and 'id_num' columns), and then plots
    real vs predicted values for all participants using matplotlib/seaborn.

    Parameters:
        model: Trained PyTorch model.
        data_loader: DataLoader for the dataset to predict on.
        test_dataset: The dataset instance (e.g., MultiParticipantDataset) used to create data_loader.
                      It must have a 'data' attribute containing the original DataFrame with a 'date' column.
        target_scaler: (Optional) Scaler used to normalize the target data.
        show_plot: Whether to display the plot.
        save_fig: Whether to save the figure to disk.
        title: Title prefix for the plot.
        scaler_type: Type of scaler used ("StandardScaler" or "MinMaxScaler").
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
    
    # Concatenate all predictions and targets into arrays
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    
    # Inverse transform if a target scaler is provided
    if target_scaler is not None:
        if scaler_type == "StandardScaler":
            print("Target scaler mean:", target_scaler.mean_)
            print("Target scaler scale:", target_scaler.scale_)
            print("Inverse transforming predictions and targets using StandardScaler")
        else:
            print("Inverse transforming predictions and targets using MinMaxScaler")

        all_predictions = target_scaler.inverse_transform(all_predictions)
        all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1))
    

    # Print statistics of predictions and targets
    print("Predictions for", title)
    print("Predictions mean:", np.mean(all_predictions))
    print("Predictions sd:", np.std(all_predictions))
    print("Predictions min:", np.min(all_predictions))
    print("Predictions max:", np.max(all_predictions))
    print("Targets mean:", np.mean(all_targets))
    print("Targets sd:", np.std(all_targets))
    print("Targets min:", np.min(all_targets))
    print("Targets max:", np.max(all_targets))
    
    # Compute the correct slice of the original DataFrame
    # The i-th prediction corresponds to data row at index (i + seq_length)
    start_idx = test_dataset.seq_length
    end_idx = start_idx + len(test_dataset)
    df_results = test_dataset.data.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    # Add prediction and target columns to the results DataFrame
    df_results['Real'] = all_targets.reshape(-1)
    df_results['Predicted'] = all_predictions.reshape(-1)
    
    # Get unique participant IDs from the results DataFrame
    participant_col = test_dataset.id_col  # e.g., 'id_num'
    participants = df_results[participant_col].unique()
    
    # Create a figure with subplots for each participant
    n_participants = len(participants)
    fig, axes = plt.subplots(n_participants, 1, figsize=(12, 4 * n_participants), constrained_layout=True)
    
    # If there's only one participant, axes won't be an array, so convert it to a list
    if n_participants == 1:
        axes = [axes]
    
    # Plot real and predicted values for each participant
    for i, p in enumerate(participants):
        df_p = df_results[df_results[participant_col] == p]
        
        # Sort by date to ensure proper line plotting
        df_p = df_p.sort_values('date')
        
        # Plot real values
        sns.lineplot(x='date', y='Real', data=df_p, ax=axes[i], label='Real', marker='o')
        
        # Plot predicted values
        sns.lineplot(x='date', y='Predicted', data=df_p, ax=axes[i], label='Predicted', marker='x')
        
        # Customize plot
        axes[i].set_title(f'Participant {p}')
        axes[i].set_xlabel('date')
        # rotate x-axis labels
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylabel('Mood Value')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add an overall title to the figure
    fig.suptitle(f'Real vs Predicted Mood Values\n{title}', fontsize=16)
    
    # Display the plot if requested
    if show_plot:
        plt.show()
    
    # Save the figure if requested
    if save_fig:
        outdir = "figures/matplotlib/predictions"
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"predictions_{title}.png"), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.join(outdir, f'predictions_{title}.png')}")
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")
    
    return df_results, mae, mse, rmse, r2



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



def train_and_evaluate(
    config, 
    df, 
    checkpoint_dir=None, 
    dataset_name=None, 
    imputation=None, 
    dropped_vars=None, 
    save_fig=True
):
    """
    Train and evaluate a model with given hyperparameters.
    
    Args:
        config: Dictionary of hyperparameters
        df: DataFrame with all data
        checkpoint_dir: Directory for checkpoints
        dataset_name: Name of the dataset (for logging)
        imputation: Imputation method used (for logging)
        dropped_vars: Variables dropped from dataset (for logging)
        save_fig: Whether to save evaluation figures
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Extract hyperparameters from config
    seq_length = config["seq_length"]
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    learning_rate = config["learning_rate"]
    dropout = config["dropout"]
    model_type = config["model_type"]
    num_epochs = config["num_epochs"]
    target_col = config.get("target_col", "target")
    
    transform_target = config.get("transform_target", True)  # Always transform by default
    per_participant_norm = config.get("per_participant_normalization", True)  # Per-participant by default
    scaler_type = config.get("scaler_type", "MinMaxScaler")
    
    shuffle_data = config.get("shuffle_data", True)
    train_ratio = config.get("train_ratio", 0.7)
    val_ratio = config.get("val_ratio", 0.15)
    split_within_participants = config.get("split_within_participants", True)
    
    # Process data - this handles normalization and splitting
    train_dataset, val_dataset, test_dataset, trained_scalers = process_lstm_data(
        df=df,
        seq_length=seq_length,
        target_col=target_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_within_participants=split_within_participants,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_norm
    )
    
    # Get target scaler for later use
    if per_participant_norm:
        scaler_target = trained_scalers.get('target', None)  # Dictionary of participant-specific scalers
    else:
        scaler_target = trained_scalers.get('target', None)  # Single scaler
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimensions from dataset
    sample_batch = next(iter(train_loader))
    x_features, x_id, _, _ = sample_batch
    input_dim = x_features.shape[2]  # [batch_size, seq_length, features]
    
    # Count number of unique participants
    num_participants = df['id_num'].nunique()
    output_dim = 1  # Predicting a single value
    
    # Initialize the model based on type
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
    
    # Load checkpoint if available
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "model.pt")):
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
        print(f"Loaded model from {checkpoint_dir}")
    
    # Storage for losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = config.get("patience", 5)  # Early stopping patience
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x_features, x_id, y, mask = [b.to(device) for b in batch]
            
            # Forward pass with mask
            outputs = model(x_features, x_id, mask)
            
            # Make sure output and target have the same shape
            output_squeezed = outputs.squeeze()

            if output_squeezed.ndim == 0 and y.ndim > 0:
                output_squeezed = output_squeezed.unsqueeze(0)
            
            # If squeezing made it a scalar but y is 1D, unsqueeze it back
            if output_squeezed.ndim == 0 and y.ndim > 0:
                output_squeezed = output_squeezed.unsqueeze(0)
                
            # Calculate loss
            loss = criterion(output_squeezed, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
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
                x_features, x_id, y, mask = [b.to(device) for b in batch]
                outputs = model(x_features, x_id, mask)
                # Make sure output and target have the same shape
                output_squeezed = outputs.squeeze()
                if output_squeezed.ndim == 0 and y.ndim > 0:
                    output_squeezed = output_squeezed.unsqueeze(0)
                
                # If squeezing made it a scalar but y is 1D, unsqueeze it back
                if output_squeezed.ndim == 0 and y.ndim > 0:
                    output_squeezed = output_squeezed.unsqueeze(0)
                    
                # Calculate loss
                loss = criterion(output_squeezed, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model checkpoint
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
    
    # Evaluate all datasets
    metrics = {}
    
    # Define a helper function for evaluation
    def evaluate_dataset(model, data_loader, dataset_name):
        model.eval()
        total_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in data_loader:
                x_features, x_id, y, mask = [b.to(device) for b in batch]
                outputs = model(x_features, x_id, mask)
                
                output_squeezed = outputs.squeeze()
                if output_squeezed.ndim == 0 and y.ndim > 0:
                    output_squeezed = output_squeezed.unsqueeze(0)
                loss = criterion(output_squeezed, y)
                total_loss += loss.item()
                
            # Store predictions and actuals, handling 0-d arrays properly
            outputs_np = outputs.squeeze().cpu().numpy()
            y_np = y.cpu().numpy()
            
            # Handle 0-d arrays (when batch size is 1 and we have a single value)
            if outputs_np.ndim == 0:  # It's a scalar
                predictions.append(float(outputs_np))
                actuals.append(float(y_np))
            else:  # It's a regular array
                predictions.extend(outputs_np)
                actuals.extend(y_np)
        
        avg_loss = total_loss / len(data_loader)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform if target was normalized
        if transform_target and scaler_target is not None:
            predictions_2d = predictions.reshape(-1, 1)
            actuals_2d = actuals.reshape(-1, 1)
            
            if isinstance(scaler_target, dict):
                # Using the first participant's scaler as a workaround
                # In a production system, we'd track participant IDs carefully
                first_key = list(scaler_target.keys())[0]
                predictions = scaler_target[first_key].inverse_transform(predictions_2d).flatten()
                actuals = scaler_target[first_key].inverse_transform(actuals_2d).flatten()
            else:
                predictions = scaler_target.inverse_transform(predictions_2d).flatten()
                actuals = scaler_target.inverse_transform(actuals_2d).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Print results
        print(f"{dataset_name} Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Create results dataframe
        df_results = pd.DataFrame({
            'Actual': actuals,
            'Predicted': predictions,
        })
        
        # Save plot if requested
        if save_fig:
            os.makedirs('figures', exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.scatter(actuals, predictions, alpha=0.5)
            plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
            plt.xlabel(f'Actual {target_col}')
            plt.ylabel(f'Predicted {target_col}')
            plt.title(f'{dataset_name} Set: Actual vs Predicted (R² = {r2:.4f})')
            plt.savefig(f'figures/{dataset_name.lower()}_predictions.png')
            plt.close()
        
        # Return metrics, ensuring scalar values for numerical metrics
        return {
            f'loss_{dataset_name.lower()}': float(avg_loss),
            f'mse_{dataset_name.lower()}': float(mse),
            f'rmse_{dataset_name.lower()}': float(rmse),
            f'mae_{dataset_name.lower()}': float(mae),
            f'r2_{dataset_name.lower()}': float(r2),
            f'predictions_{dataset_name.lower()}': predictions,
            f'actuals_{dataset_name.lower()}': actuals
        }
    
    # Evaluate each dataset
    train_metrics = evaluate_dataset(model, train_loader, "Train")
    val_metrics = evaluate_dataset(model, val_loader, "Val")
    test_metrics = evaluate_dataset(model, test_loader, "Test")
    
    # Combine all metrics, ensuring they are scalar values
    metrics = {}
    for k, v in train_metrics.items():
        # Convert numpy arrays to scalars where needed
        if isinstance(v, np.ndarray):
            if k.startswith(('predictions_', 'actuals_')):
                # Keep these as arrays
                metrics[k] = v
            else:
                # Convert metric arrays to scalar floats
                metrics[k] = float(v)
        else:
            metrics[k] = v
            
    for k, v in val_metrics.items():
        if isinstance(v, np.ndarray):
            if k.startswith(('predictions_', 'actuals_')):
                metrics[k] = v
            else:
                metrics[k] = float(v)
        else:
            metrics[k] = v
            
    for k, v in test_metrics.items():
        if isinstance(v, np.ndarray):
            if k.startswith(('predictions_', 'actuals_')):
                metrics[k] = v
            else:
                metrics[k] = float(v)
        else:
            metrics[k] = v
    
            # Save loss curves
    if save_fig:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='o')
        
        # Make sure test loss is a scalar
        test_loss = float(metrics['loss_test'])
        plt.axhline(y=test_loss, color='r', linestyle='-', label='Test Loss')
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training, Validation, and Test Loss")
        plt.legend()
        plt.savefig("figures/loss_curves.png")
        plt.close()
    
    # Collect hyperparameters
    hyperparams = {
        # Data information
        "dataset": dataset_name,
        "dropped_vars": dropped_vars,
        "imputation": imputation,
        
        # Model architecture
        "model": model_type,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "output_dim": output_dim,
        "dropout": dropout,
        
        # Dataset sizes
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        
        # Data processing
        "sequence_length": seq_length,
        "scaler": scaler_type,
        "transform_target": transform_target,
        "per_participant_normalization": per_participant_norm,
        
        # Training parameters
        "batch_size": batch_size,
        "shuffle_data": shuffle_data,
        "num_epochs": min(epoch + 1, num_epochs),  # Actual number of epochs run
        "learning_rate": learning_rate,
        "patience": patience,
        "clip_gradients": config.get("clip_gradients", True),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        
        # Metadata
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device.type,
    }
    
    # Save model
    os.makedirs('models', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = f'models/model_{model_type}_{timestamp}.pt'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'metrics': metrics
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Ensure all metrics are scalar values for the results DataFrame
    results_dict = {
        'R2_train': float(metrics['r2_train']),
        'R2_val': float(metrics['r2_val']),
        'R2_test': float(metrics['r2_test']),
        'MAE_train': float(metrics['mae_train']),
        'MAE_val': float(metrics['mae_val']),
        'MAE_test': float(metrics['mae_test']),
        'MSE_train': float(metrics['mse_train']),
        'MSE_val': float(metrics['mse_val']),
        'MSE_test': float(metrics['mse_test']),
        'RMSE_train': float(metrics['rmse_train']),
        'RMSE_val': float(metrics['rmse_val']),
        'RMSE_test': float(metrics['rmse_test']),
        'dataset': dataset_name
    }
    
    # Create DataFrame from the scalar values
    results_df = pd.DataFrame({f"{model_type}_{timestamp}": results_dict}).T
    
    os.makedirs('tables/results', exist_ok=True)
    
    # Append to existing results or create new file
    results_path = 'tables/results/model_results.csv'
    if not os.path.exists(results_path):
        results_df.to_csv(results_path, index=True, header=True)
    else:
        results_df.to_csv(results_path, mode='a', header=False, index=True)


    if save_fig:
        # Generate plots for each data split
        train_results, train_metrics = plot_participant_timeseries(
            model=model,
            data_loader=train_loader,
            dataset=train_dataset,
            target_col='target',
            target_scalers=trained_scalers,
            split_name="Train"
        )

        val_results, val_metrics = plot_participant_timeseries(
            model=model,
            data_loader=val_loader,
            dataset=val_dataset,
            target_col='target',
            target_scalers=trained_scalers,
            split_name="Validation"
        )

        test_results, test_metrics = plot_participant_timeseries(
            model=model,
            data_loader=test_loader,
            dataset=test_dataset,
            target_col='target',
            target_scalers=trained_scalers,
            split_name="Test"
        )

        # Create a comparison plot of metrics across splits
        metric_comparison = compare_split_metrics(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            target_col='target'
        )
    
    
    return metrics



def simple_hyperparameter_tuning(
    df, 
    dataset_name=None,
    imputation=None, 
    dropped_vars=None, 
    default_config=None, 
    param_grid=None
):
    """
    Run simple hyperparameter tuning with the improved training function
    
    Args:
        df: Full DataFrame with all data
        dataset_name: Name of the dataset (for logging)
        imputation: Imputation method used (for logging)
        dropped_vars: Variables dropped from dataset (for logging)
        default_config: Default configuration parameters
        param_grid: Grid of parameters to search
    
    Returns:
        Best configuration of hyperparameters
    """
    print("Starting simple hyperparameter tuning...")
    
    # Set default config if not provided
    if default_config is None:
        default_config = {
            "model_type": "LSTM",
            "hidden_dim": 64,
            "num_layers": 1,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 30,
            "seq_length": 7,
            "scaler_type": "MinMaxScaler",
            "transform_target": True,
            "per_participant_normalization": True,
            "split_within_participants": True,
            "shuffle_data": True,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "clip_gradients": True,
            "max_grad_norm": 1.0,
            "patience": 5,
            "target_col": "target"
        }
    
    # Set parameter grid if not provided
    if param_grid is None:
        param_grid = {
            "model_type": ["LSTM", "GRU", "SimpleRNN"],
            "hidden_dim": [32, 64, 128],
            "seq_length": [5, 7, 10],
            "learning_rate": [0.001, 0.0005]
        }
    
    # Create directory to store results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"tuning_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Focus on key hyperparameters for the grid search
    key_params = {k: v for k, v in param_grid.items() if k in [
        "model_type", "hidden_dim", "seq_length", "learning_rate"
    ]}
    
    # Create combinations of parameters
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
            
            # Call the improved train_and_evaluate function
            metrics = train_and_evaluate(
                config=config,
                df=df,
                dataset_name=dataset_name,
                imputation=imputation,
                dropped_vars=dropped_vars,
                save_fig=False
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Add trial info
            trial_info = {
                "trial": i+1,
                "config": config,
                "metrics": {k: float(v) if isinstance(v, (np.number, float)) and not isinstance(v, bool) else v
                           for k, v in metrics.items() if not isinstance(v, np.ndarray)},
                "duration_seconds": duration
            }
            
            all_results.append(trial_info)
            
            # Check if this is the best configuration
            val_loss = float(metrics["loss_val"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config.copy()
                print(f"New best validation loss: {best_val_loss:.6f}")
            
            # Save intermediate results
            _save_tuning_results(results_dir, all_results)
            
        except Exception as e:
            print(f"Error in trial {i+1}: {e}")
            # Log the error but continue
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
        
        # Save best config
        with open(os.path.join(results_dir, 'best_config.json'), 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # Create parameter comparison plots
        _create_parameter_plots(results_dir, all_results, key_param_names)
    else:
        print("No successful trials completed.")
        best_config = default_config
    
    return best_config


def _save_tuning_results(results_dir: str, all_results: List[Dict[str, Any]]):
    """
    Save intermediate tuning results to files.
    
    Args:
        results_dir: Directory for saving results
        all_results: List of result dictionaries from each trial
    """
    # Save detailed results as JSON
    with open(os.path.join(results_dir, 'tuning_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        row = {}
        # Add config parameters (only simple types)
        for param_name, param_value in result["config"].items():
            if isinstance(param_value, (int, float, str, bool)):
                row[param_name] = param_value
        
        # Add metrics (skip arrays)
        for metric_name, metric_value in result["metrics"].items():
            if not isinstance(metric_value, np.ndarray):
                row[metric_name] = metric_value
        
        row["duration_seconds"] = result["duration_seconds"]
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'tuning_summary.csv'), index=False)
    
    # Create progress plots
    _create_progress_plots(results_dir, summary_df)


def _create_progress_plots(results_dir: str, summary_df: pd.DataFrame):
    """
    Create plots showing tuning progress.
    
    Args:
        results_dir: Directory for saving plots
        summary_df: DataFrame with summary results
    """
    if len(summary_df) == 0:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Validation loss by trial
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(summary_df) + 1), summary_df['loss_val'], 'bo-')
    plt.title('Validation Loss')
    plt.xlabel('Trial')
    plt.ylabel('Loss')
    
    # Validation R² by trial
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(summary_df) + 1), summary_df['r2_val'], 'go-')
    plt.title('Validation R²')
    plt.xlabel('Trial')
    plt.ylabel('R²')
    
    # Train vs Validation loss
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(summary_df) + 1), summary_df['loss_train'], 'ro-')
    plt.plot(range(1, len(summary_df) + 1), summary_df['loss_val'], 'bo-')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Trial')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    
    # Model type comparison
    if len(summary_df) > 1 and 'model_type' in summary_df.columns:
        plt.subplot(2, 2, 4)
        model_stats = summary_df.groupby('model_type')['loss_val'].mean().reset_index()
        plt.bar(model_stats['model_type'], model_stats['loss_val'])
        plt.title('Validation Loss by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Avg. Validation Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tuning_progress.png'))
    plt.close()


def _create_parameter_plots(results_dir: str, all_results: List[Dict[str, Any]], key_param_names: List[str]):
    
    """
    Create plots showing impact of different parameters.
    
    Args:
        results_dir: Directory for saving plots
        all_results: List of result dictionaries from each trial
        key_param_names: Names of key parameters that were varied
    """
    if len(all_results) <= 1:
        return
    
    # Create DataFrame from results
    summary_data = []
    for result in all_results:
        row = {}
        for param_name, param_value in result["config"].items():
            if isinstance(param_value, (int, float, str, bool)):
                row[param_name] = param_value
        
        for metric_name, metric_value in result["metrics"].items():
            if not isinstance(metric_value, np.ndarray):
                row[metric_name] = metric_value
                
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create plots for each parameter
    for param in key_param_names:
        if param not in summary_df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Group by parameter and calculate mean validation loss
        param_groups = summary_df.groupby(param)['loss_val'].mean().reset_index()
        
        plt.bar(param_groups[param].astype(str), param_groups['loss_val'])
        plt.title(f'Validation Loss by {param}')
        plt.xlabel(param)
        plt.ylabel('Avg. Validation Loss')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'param_impact_{param}.png'))
        plt.close()



def _create_parameter_plots(results_dir: str, all_results: List[Dict[str, Any]], key_param_names: List[str]):
    
    """
    Create plots showing impact of different parameters.
    
    Args:
        results_dir: Directory for saving plots
        all_results: List of result dictionaries from each trial
        key_param_names: Names of key parameters that were varied
    """
    if len(all_results) <= 1:
        return
    
    # Create DataFrame from results
    summary_data = []
    for result in all_results:
        row = {}
        for param_name, param_value in result["config"].items():
            if isinstance(param_value, (int, float, str, bool)):
                row[param_name] = param_value
        
        for metric_name, metric_value in result["metrics"].items():
            if not isinstance(metric_value, np.ndarray):
                row[metric_name] = metric_value
                
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create plots for each parameter
    for param in key_param_names:
        if param not in summary_df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Group by parameter and calculate mean validation loss
        param_groups = summary_df.groupby(param)['loss_val'].mean().reset_index()
        
        plt.bar(param_groups[param].astype(str), param_groups['loss_val'])
        plt.title(f'Validation Loss by {param}')
        plt.xlabel(param)
        plt.ylabel('Avg. Validation Loss')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'param_impact_{param}.png'))
        plt.close()

def plot_participant_timeseries(model, 
                            data_loader, 
                            dataset,
                            target_col='target',
                            target_scalers=None, 
                            split_name="Test", 
                            scaler_type="MinMaxScaler",
                            show_plot=False,
                            save_path=None,
                            figsize=(12, 6),
                            date_format="%Y-%m-%d",
                            confidence_interval=False,
                            ci_alpha=0.3):
    """
    Creates detailed time series plots of actual vs predicted values for each participant,
    saving individual plots to separate files.
    
    Parameters:
        model: Trained PyTorch model
        data_loader: DataLoader for the dataset to predict on
        dataset: Dataset instance containing the data
        target_col: Name of the target column being predicted
        target_scalers: Scalers used to normalize target data (per participant or global)
        split_name: Dataset split name (e.g., "Train", "Validation", "Test")
        scaler_type: Type of scaler used ("StandardScaler" or "MinMaxScaler")
        show_plot: Whether to display the plots
        save_path: Base path for saving plots (if None, will use default path)
        figsize: Size of the figure for each participant
        date_format: Format for displaying dates
        confidence_interval: Whether to plot confidence intervals
        ci_alpha: Alpha value for confidence interval shading
    
    Returns:
        Tuple of (DataFrame with results, dictionary with metrics per participant)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    import pandas as pd
    import os
    import torch
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import seaborn as sns
    
    # Set style for plots
    sns.set_style("whitegrid")
    
    # Move model to CPU for inference
    model.eval()
    model.to("cpu")
    
    all_predictions = []
    all_targets = []
    all_participant_ids = []
    
    # Get predictions for each batch
    with torch.no_grad():
        for batch in data_loader:
            # Assuming batch contains (features, participant_ids, targets, masks)
            if len(batch) == 4:
                x_features, x_id, y, mask = batch
            else:
                x_features, x_id, y = batch
                mask = None
            
            # Move to CPU
            x_features = x_features.to("cpu")
            x_id = x_id.to("cpu")
            y = y.to("cpu")
            
            # Forward pass
            outputs = model(x_features, x_id, mask)
            
            # Store results
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            # Get participant IDs
            participant_ids = x_id[:, 0].cpu().numpy()
            all_participant_ids.extend(participant_ids)
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_participant_ids = np.array(all_participant_ids)
    
    # Create a results DataFrame
    results = pd.DataFrame({
        'participant_id': all_participant_ids,
        'predicted': all_predictions,
        'actual': all_targets
    })
    
    # Try to get original dates from dataset
    try:
        if hasattr(dataset, 'data') and 'date' in dataset.data.columns:
            # Get unique dates for each participant
            for p_id in results['participant_id'].unique():
                p_mask = results['participant_id'] == p_id
                p_count = p_mask.sum()
                
                # Get dates for this participant from the original dataset
                p_dates = dataset.data[dataset.data['id_num'] == p_id]['date'].sort_values().values
                
                if len(p_dates) >= p_count:
                    # Use the appropriate number of dates for this participant
                    results.loc[p_mask, 'date'] = p_dates[:p_count]
                else:
                    # Create dummy dates if not enough real dates
                    print(f"Warning: Not enough dates for participant {p_id}. Using dummy dates.")
                    start_date = pd.Timestamp('2023-01-01')
                    results.loc[p_mask, 'date'] = pd.date_range(start=start_date, periods=p_count)
        else:
            # Create dummy dates if dataset has no date column
            for p_id in results['participant_id'].unique():
                p_mask = results['participant_id'] == p_id
                p_count = p_mask.sum()
                start_date = pd.Timestamp('2023-01-01')
                results.loc[p_mask, 'date'] = pd.date_range(start=start_date, periods=p_count)
    except Exception as e:
        print(f"Error handling dates: {str(e)}. Using dummy dates.")
        for p_id in results['participant_id'].unique():
            p_mask = results['participant_id'] == p_id
            p_count = p_mask.sum()
            start_date = pd.Timestamp('2023-01-01')
            results.loc[p_mask, 'date'] = pd.date_range(start=start_date, periods=p_count)
    
    # Inverse transform predictions and targets if scalers are provided
    if target_scalers is not None:
        # Reshape for scaler
        pred_2d = all_predictions.reshape(-1, 1)
        target_2d = all_targets.reshape(-1, 1)
        
        # If we have per-participant scalers
        if isinstance(target_scalers, dict) and 'target' in target_scalers:
            if isinstance(target_scalers['target'], dict):
                # We have a scaler per participant
                inversed_preds = np.zeros_like(pred_2d)
                inversed_targets = np.zeros_like(target_2d)
                
                # Loop through each participant
                for p_id in np.unique(all_participant_ids):
                    if p_id in target_scalers['target']:
                        scaler = target_scalers['target'][p_id]
                        idx = all_participant_ids == p_id
                        inversed_preds[idx] = scaler.inverse_transform(pred_2d[idx])
                        inversed_targets[idx] = scaler.inverse_transform(target_2d[idx])
                
                results['predicted'] = inversed_preds.flatten()
                results['actual'] = inversed_targets.flatten()
            else:
                # We have a global scaler
                scaler = target_scalers['target']
                results['predicted'] = scaler.inverse_transform(pred_2d).flatten()
                results['actual'] = scaler.inverse_transform(target_2d).flatten()
        else:
            # We have a global scaler
            results['predicted'] = target_scalers.inverse_transform(pred_2d).flatten()
            results['actual'] = target_scalers.inverse_transform(target_2d).flatten()
    
    # Convert dates to datetime if they're not already
    if not pd.api.types.is_datetime64_any_dtype(results['date']):
        results['date'] = pd.to_datetime(results['date'])
    
    # Sort by participant ID and date
    results = results.sort_values(['participant_id', 'date']).reset_index(drop=True)
    
    # Add error column
    results['error'] = results['predicted'] - results['actual']
    results['abs_error'] = np.abs(results['error'])
    
    # Calculate metrics per participant
    participant_metrics = {}
    for p_id, group in results.groupby('participant_id'):
        mse = mean_squared_error(group['actual'], group['predicted'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(group['actual'], group['predicted'])
        
        # Only calculate R² if we have enough samples
        if len(group) >= 2:
            r2 = r2_score(group['actual'], group['predicted'])
        else:
            r2 = float('nan')
            
        participant_metrics[p_id] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_samples': len(group),
            'max_error': group['abs_error'].max(),
            'min_error': group['abs_error'].min(),
            'mean_error': group['abs_error'].mean()
        }
    
    # Overall metrics
    overall_mae = mean_absolute_error(results['actual'], results['predicted'])
    overall_mse = mean_squared_error(results['actual'], results['predicted'])
    overall_rmse = np.sqrt(overall_mse)
    
    if len(results) >= 2:
        overall_r2 = r2_score(results['actual'], results['predicted'])
    else:
        overall_r2 = float('nan')
    
    # Print overall metrics
    print(f"\n{split_name} Set Overall Metrics:")
    print(f"  MAE: {overall_mae:.4f}")
    print(f"  RMSE: {overall_rmse:.4f}")
    print(f"  R²: {overall_r2:.4f}")
    
    # Setup base directory for saving plots
    if save_path is None:
        base_dir = f"figures/participants/{split_name.lower()}"
    else:
        base_dir = os.path.join(os.path.dirname(save_path), f"participants/{split_name.lower()}")
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Save summary metrics to file
    summary_file = os.path.join(base_dir, "summary_metrics.txt")
    with open(summary_file, 'w') as f:
        f.write(f"{split_name} Set Overall Metrics:\n")
        f.write(f"  MAE: {overall_mae:.4f}\n")
        f.write(f"  RMSE: {overall_rmse:.4f}\n")
        f.write(f"  R²: {overall_r2:.4f}\n\n")
        
        f.write("Per-Participant Metrics:\n")
        for p_id, metrics in participant_metrics.items():
            f.write(f"  Participant {p_id}:\n")
            f.write(f"    MAE: {metrics['mae']:.4f}\n")
            f.write(f"    RMSE: {metrics['rmse']:.4f}\n")
            
            if not np.isnan(metrics['r2']):
                f.write(f"    R²: {metrics['r2']:.4f}\n")
            else:
                f.write(f"    R²: N/A\n")
                
            f.write(f"    Samples: {metrics['n_samples']}\n\n")
    
    print(f"Saved summary metrics to {summary_file}")
    
    # Get unique participants
    unique_participants = sorted(results['participant_id'].unique())
    
    # Create individual plots for each participant
    for p_id in unique_participants:
        # Get data for this participant
        p_data = results[results['participant_id'] == p_id].copy()
        
        # Skip if no data
        if len(p_data) == 0:
            print(f"No data for participant {p_id}, skipping.")
            continue
        
        # Create a new figure for this participant
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual values
        ax.plot(p_data['date'], p_data['actual'], 'o-', color='#1f77b4', label='Actual', markersize=8)
        
        # Plot predicted values
        ax.plot(p_data['date'], p_data['predicted'], 'x--', color='#d62728', label='Predicted', markersize=8)
        
        # Add confidence interval if requested
        if confidence_interval and len(p_data) > 2:
            # Calculate error
            errors = p_data['abs_error']
            std_error = errors.std()
            
            # Create confidence bounds
            upper_bound = p_data['predicted'] + 1.96 * std_error
            lower_bound = p_data['predicted'] - 1.96 * std_error
            
            # Plot confidence interval
            ax.fill_between(p_data['date'], lower_bound, upper_bound, 
                          color='#d62728', alpha=ci_alpha, label='95% Confidence')
        
        # Format x-axis for dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotate date labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add metrics to the title
        metrics = participant_metrics[p_id]
        title = (f"Participant {p_id} - "
                f"MAE: {metrics['mae']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}")
        
        if not np.isnan(metrics['r2']):
            title += f", R²: {metrics['r2']:.4f}"
            
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel(f"{target_col.capitalize()} Value", fontsize=11)
        
        # Enhance the legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Add grid but make it subtle
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add points to highlight actual data points
        ax.scatter(p_data['date'], p_data['actual'], color='#1f77b4', s=80, alpha=0.7, zorder=5)
        
        # Add a subtitle with the split name
        plt.suptitle(f'{split_name} Set: Actual vs Predicted {target_col.capitalize()} Values', 
                    fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
        
        # Create participant directory
        participant_dir = os.path.join(base_dir, f"participant_{p_id}")
        os.makedirs(participant_dir, exist_ok=True)
        
        # Save the figure
        save_file = os.path.join(participant_dir, f"{split_name.lower()}_participant_{p_id}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot for Participant {p_id} to {save_file}")
        
        # Also create a plot with error visualization
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.5), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Top subplot: predicted vs actual
        ax1.plot(p_data['date'], p_data['actual'], 'o-', color='#1f77b4', label='Actual', markersize=8)
        ax1.plot(p_data['date'], p_data['predicted'], 'x--', color='#d62728', label='Predicted', markersize=8)
        
        # Format top subplot
        ax1.set_title(title, fontsize=12)
        ax1.set_xlabel("")  # No x-label on top subplot
        ax1.set_ylabel(f"{target_col.capitalize()} Value", fontsize=11)
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Bottom subplot: error visualization
        error_bars = ax2.bar(p_data['date'], p_data['abs_error'], color='#9467bd', alpha=0.7)
        ax2.set_title("Absolute Prediction Error", fontsize=12)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylabel("Error", fontsize=11)
        
        # Add a horizontal line for mean error
        ax2.axhline(y=metrics['mean_error'], color='#000000', linestyle='--', 
                  alpha=0.8, label=f"Mean Error: {metrics['mean_error']:.4f}")
        ax2.legend()
        
        # Format x-axis for dates on both subplots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'{split_name} Set: Participant {p_id} - {target_col.capitalize()} Prediction', 
                   fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.3)  # Adjust spacing
        
        # Save the error plot
        error_file = os.path.join(participant_dir, f"{split_name.lower()}_participant_{p_id}_error.png")
        plt.savefig(error_file, dpi=300, bbox_inches='tight')
        print(f"Saved error plot for Participant {p_id} to {error_file}")
        
        # Close figures to save memory
        if not show_plot:
            plt.close(fig)
            plt.close(fig2)
    
    # Create a combined figure with all participants
    if len(unique_participants) > 0:
        # This combined plot helps for comparison but will be less detailed
        # Create multiple subplots, one per participant
        n_participants = len(unique_participants)
        fig, axes = plt.subplots(n_participants, 1, figsize=(figsize[0], figsize[1] * 0.8 * n_participants))
        
        # Handle single participant case
        if n_participants == 1:
            axes = [axes]
        
        # Plot each participant
        for i, p_id in enumerate(unique_participants):
            p_data = results[results['participant_id'] == p_id].copy()
            
            # Skip if no data
            if len(p_data) == 0:
                continue
                
            ax = axes[i]
            
            # Plot actual and predicted values
            ax.plot(p_data['date'], p_data['actual'], 'o-', color='#1f77b4', label='Actual', markersize=6)
            ax.plot(p_data['date'], p_data['predicted'], 'x--', color='#d62728', label='Predicted', markersize=6)
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add title with metrics
            metrics = participant_metrics[p_id]
            title = f"Participant {p_id} - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}"
            if not np.isnan(metrics['r2']):
                title += f", R²: {metrics['r2']:.4f}"
            
            ax.set_title(title)
            ax.set_ylabel(f"{target_col.capitalize()}")
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.suptitle(f'{split_name} Set: All Participants - {target_col.capitalize()} Prediction', 
                   fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4)
        
        # Save the combined plot
        combined_file = os.path.join(base_dir, f"{split_name.lower()}_all_participants.png")
        plt.savefig(combined_file, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to {combined_file}")
        
        if not show_plot:
            plt.close(fig)
    
    # Show plots if requested
    if show_plot:
        plt.show()
    
    return results, participant_metrics


def compare_split_metrics(train_metrics=None, val_metrics=None, test_metrics=None,
                      target_col='target', figsize=(12, 10), save_path=None):
    """
    Create plots comparing metrics across different data splits for each participant.
    
    Parameters:
        train_metrics: Dictionary with training metrics per participant
        val_metrics: Dictionary with validation metrics per participant
        test_metrics: Dictionary with test metrics per participant
        target_col: Name of the target column being predicted
        figsize: Size of the figure
        save_path: Path to save the comparison plot
        
    Returns:
        DataFrame with combined metrics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import seaborn as sns
    
    # Set style
    sns.set_style("whitegrid")
    
    # Get all unique participant IDs
    all_participants = set()
    
    if train_metrics:
        all_participants.update(train_metrics.keys())
    if val_metrics:
        all_participants.update(val_metrics.keys())
    if test_metrics:
        all_participants.update(test_metrics.keys())
    
    all_participants = sorted(list(all_participants))
    
    if not all_participants:
        print("No participants found.")
        return pd.DataFrame()
    
    # Create a DataFrame for visualization
    data = []
    
    for p_id in all_participants:
        # Training metrics
        if train_metrics and p_id in train_metrics:
            data.append({
                'participant_id': p_id,
                'split': 'Train',
                'MAE': train_metrics[p_id]['mae'],
                'RMSE': train_metrics[p_id]['rmse'],
                'R²': train_metrics[p_id]['r2'] if not np.isnan(train_metrics[p_id]['r2']) else 0,
                'has_r2': not np.isnan(train_metrics[p_id]['r2']),
                'samples': train_metrics[p_id]['n_samples']
            })
        
        # Validation metrics
        if val_metrics and p_id in val_metrics:
            data.append({
                'participant_id': p_id,
                'split': 'Validation',
                'MAE': val_metrics[p_id]['mae'],
                'RMSE': val_metrics[p_id]['rmse'],
                'R²': val_metrics[p_id]['r2'] if not np.isnan(val_metrics[p_id]['r2']) else 0,
                'has_r2': not np.isnan(val_metrics[p_id]['r2']),
                'samples': val_metrics[p_id]['n_samples']
            })
        
        # Test metrics
        if test_metrics and p_id in test_metrics:
            data.append({
                'participant_id': p_id,
                'split': 'Test',
                'MAE': test_metrics[p_id]['mae'],
                'RMSE': test_metrics[p_id]['rmse'],
                'R²': test_metrics[p_id]['r2'] if not np.isnan(test_metrics[p_id]['r2']) else 0,
                'has_r2': not np.isnan(test_metrics[p_id]['r2']),
                'samples': test_metrics[p_id]['n_samples']
            })
    
    df = pd.DataFrame(data)
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: MAE by participant and split
    sns.barplot(x='participant_id', y='MAE', hue='split', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Mean Absolute Error by Participant')
    axes[0, 0].set_xlabel('Participant ID')
    axes[0, 0].set_ylabel('MAE')
    
    # Plot 2: RMSE by participant and split
    sns.barplot(x='participant_id', y='RMSE', hue='split', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Root Mean Squared Error by Participant')
    axes[0, 1].set_xlabel('Participant ID')
    axes[0, 1].set_ylabel('RMSE')
    
    # Plot 3: R² by participant and split (only for valid R² values)
    r2_df = df[df['has_r2']]
    if len(r2_df) > 0:
        sns.barplot(x='participant_id', y='R²', hue='split', data=r2_df, ax=axes[1, 0])
        axes[1, 0].set_title('R² Score by Participant')
        axes[1, 0].set_xlabel('Participant ID')
        axes[1, 0].set_ylabel('R²')
    else:
        axes[1, 0].text(0.5, 0.5, 'No valid R² values available', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[1, 0].transAxes)
    
    # Plot 4: Sample count by participant and split
    sns.barplot(x='participant_id', y='samples', hue='split', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Number of Samples by Participant')
    axes[1, 1].set_xlabel('Participant ID')
    axes[1, 1].set_ylabel('Sample Count')
    
    plt.suptitle(f'Comparison of Model Performance Across Data Splits\n{target_col.capitalize()} Prediction', 
               fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure if requested
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        # Use default path
        save_dir = "figures/comparisons"
        os.makedirs(save_dir, exist_ok=True)
        default_path = os.path.join(save_dir, "split_comparison.png")
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {default_path}")
    
    plt.show()
    
    return df
