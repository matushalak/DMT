
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


def create_data_split(df, proportion_train=0.7, proportion_val=0.15, split_within_participants=True, seq_length=None):
    """
    Split data into train, validation, and test sets with simplified proportion handling
    
    Args:
        df: DataFrame containing the data
        proportion_train: Proportion of data for training
        proportion_val: Proportion of data for validation
        split_within_participants: Whether to split within participants or across participants
        seq_length: Sequence length parameter (no longer used for filtering)
        
    Returns:
        train_df, val_df, test_df: DataFrames for training, validation, and testing
    """

    proportion_test = 1 - proportion_train - proportion_val
    
    if split_within_participants:
        # Initialize empty lists for each split
        train_dfs, val_dfs, test_dfs = [], [], []
        split_dates = []
        
        # Split each participant's data
        for participant, group in df.groupby('id_num'):
            group = group.sort_values(by='day')
            group_size = len(group)
            
            # Skip participants with too few data points
            if group_size < 3:  # Absolute minimum to have at least 1 point in each split
                print(f"Skipping participant {participant} with only {group_size} data points")
                continue
            
            # Calculate split indices
            train_size = int(group_size * proportion_train)
            val_size = int(group_size * proportion_val)
            
            # Ensure at least 1 sample in each split if possible
            if train_size == 0 and group_size >= 3:
                train_size = 1
            if val_size == 0 and group_size >= 3:
                val_size = 1
                
            # Ensure test set gets at least 1 sample
            test_size = group_size - train_size - val_size
            if test_size == 0 and group_size >= 3:
                if val_size > 1:
                    val_size -= 1
                elif train_size > 1:
                    train_size -= 1
                test_size = 1
            
            # Split the data
            train_idx = train_size
            val_idx = train_size + val_size
            
            train_dfs.append(group.iloc[:train_idx])
            val_dfs.append(group.iloc[train_idx:val_idx])
            test_dfs.append(group.iloc[val_idx:])
            
            # Record split dates for this participant
            split_dates.append({
                "participant": participant,
                "train_start": group.iloc[0]['day'] if train_size > 0 else None,
                "train_end": group.iloc[train_idx-1]['day'] if train_size > 0 else None,
                "val_start": group.iloc[train_idx]['day'] if val_size > 0 else None,
                "val_end": group.iloc[val_idx-1]['day'] if val_size > 0 else None,
                "test_start": group.iloc[val_idx]['day'] if test_size > 0 else None,
                "test_end": group.iloc[-1]['day'] if test_size > 0 else None,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "train_prop": train_size / group_size,
                "val_prop": val_size / group_size,
                "test_prop": test_size / group_size
            })
        
        # Combine the splits
        train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
        val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
        test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()
        
        # Save split dates info
        dates_df = pd.DataFrame(split_dates)
        if not dates_df.empty:
            try:
                dates_df.to_csv("tables/training_dates_split.csv", index=False)
                
                # Print actual proportions achieved
                total_samples = len(train_df) + len(val_df) + len(test_df)
                actual_train_prop = len(train_df) / total_samples if total_samples > 0 else 0
                actual_val_prop = len(val_df) / total_samples if total_samples > 0 else 0
                actual_test_prop = len(test_df) / total_samples if total_samples > 0 else 0
                
                print(f"Actual proportions achieved:")
                print(f"Train: {actual_train_prop:.3f} (target: {proportion_train:.3f})")
                print(f"Validation: {actual_val_prop:.3f} (target: {proportion_val:.3f})")
                print(f"Test: {actual_test_prop:.3f} (target: {proportion_test:.3f})")
            except:
                print("Warning: Could not save split dates to CSV file")
    
    else:
        # Split across participants
        participant_ids = df['id_num'].unique()
        
        if len(participant_ids) < 3:
            print("Warning: Not enough participants for a proper split. Using fallback method.")
            # Fallback to within-participants split if we don't have enough participants
            return create_data_split(df, proportion_train, proportion_val, True, seq_length)
        
        # Get data count for each participant
        participant_weights = df.groupby('id_num').size()
        
        # Create a weighted split based on data volume
        participant_ids_with_counts = [(pid, participant_weights[pid]) for pid in participant_ids]
        total_points = sum(count for _, count in participant_ids_with_counts)
        
        # Sort by data count for better allocation
        participant_ids_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate participants to achieve target proportions
        train_ids, val_ids, test_ids = [], [], []
        train_points, val_points, test_points = 0, 0, 0
        
        for pid, count in participant_ids_with_counts:
            # Current proportions if we add to each set
            current_total = train_points + val_points + test_points + count
            
            train_ratio = (train_points + count) / current_total
            val_ratio = (val_points + count) / current_total
            test_ratio = (test_points + count) / current_total
            
            # Calculate deviation from target for each option
            train_dev = abs(train_ratio - proportion_train)
            val_dev = abs(val_ratio - proportion_val) 
            test_dev = abs(test_ratio - proportion_test)
            
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
        if not train_ids and participant_ids:
            pid = participant_ids[0]
            train_ids.append(pid)
            if pid in val_ids:
                val_ids.remove(pid)
            elif pid in test_ids:
                test_ids.remove(pid)
        
        if not val_ids and len(participant_ids) > 1:
            pid = participant_ids[1] if participant_ids[1] not in train_ids else participant_ids[0]
            val_ids.append(pid)
            if pid in test_ids:
                test_ids.remove(pid)
        
        if not test_ids and len(participant_ids) > 2:
            pid = next((p for p in participant_ids if p not in train_ids and p not in val_ids), None)
            if pid:
                test_ids.append(pid)
        
        print(f"Train IDs ({len(train_ids)}): {train_ids}")
        print(f"Validation IDs ({len(val_ids)}): {val_ids}")
        print(f"Test IDs ({len(test_ids)}): {test_ids}")
        
        # Filter the original DataFrame based on these IDs
        train_df = df[df['id_num'].isin(train_ids)].copy()
        val_df = df[df['id_num'].isin(val_ids)].copy()
        test_df = df[df['id_num'].isin(test_ids)].copy()
        
        # Sort by participant and day
        train_df.sort_values(by=['id_num', 'day'], inplace=True)
        val_df.sort_values(by=['id_num', 'day'], inplace=True)
        test_df.sort_values(by=['id_num', 'day'], inplace=True)
        
        # Print actual proportions achieved
        total_samples = len(train_df) + len(val_df) + len(test_df)
        actual_train_prop = len(train_df) / total_samples if total_samples > 0 else 0
        actual_val_prop = len(val_df) / total_samples if total_samples > 0 else 0
        actual_test_prop = len(test_df) / total_samples if total_samples > 0 else 0
        
        print(f"Actual proportions achieved:")
        print(f"Train: {actual_train_prop:.3f} (target: {proportion_train:.3f})")
        print(f"Validation: {actual_val_prop:.3f} (target: {proportion_val:.3f})")
        print(f"Test: {actual_test_prop:.3f} (target: {proportion_test:.3f})")

    return train_df, val_df, test_df


class MultiParticipantDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 seq_length: int, 
                 target_col: str = 'mood', 
                 id_col: str = 'id_num', 
                 include_target_in_features: bool = True,
                 normalize_data: bool = True,
                 transform_target: bool = False,
                 scaler_type: str = "StandardScaler",
                 per_participant_normalization: bool = False,
                 existing_scalers: Dict = None):
        """
        Custom dataset for multi-participant time series prediction with LSTM
        
        Args:
            df: pandas DataFrame sorted by time
            seq_length: number of time steps in each sample
            target_col: the column we want to predict
            id_col: column that identifies participants
            include_target_in_features: whether to include target variable as a feature
            normalize_data: whether to normalize the data
            transform_target: whether to normalize the target variable
            scaler_type: type of scaler to use ("StandardScaler" or "MinMaxScaler")
            per_participant_normalization: whether to normalize data per participant 
            existing_scalers: pre-trained scalers to use instead of fitting new ones
        """
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Drop unnecessary columns
        if "next_day" in df.columns and "next_day_mood" in df.columns:
            df = df.drop(columns=["next_day", "next_day_mood"])
        
        # Set instance variables
        self.seq_length = seq_length
        self.target_col = target_col
        self.id_col = id_col
        self.include_target_in_features = include_target_in_features
        self.per_participant_normalization = per_participant_normalization
        
        # Sort by participant and day
        df.sort_values(by=[id_col, 'day'], inplace=True)
        
        # Define feature columns
        if include_target_in_features:
            self.features = [col for col in df.columns if col not in [self.target_col, "day"]]
        else:
            self.features = [col for col in df.columns if col not in [self.target_col, self.id_col, "day"]]
        
        # Normalize data if requested
        self.scalers = {}
        if normalize_data:
            df, self.scalers = self._normalize_data(
                df, 
                transform_target=transform_target,
                scaler_type=scaler_type,
                existing_scalers=existing_scalers
            )
        
        # Store the processed data
        self.data = df.reset_index(drop=True)
        
        # Group data by participant
        grouped_data = self.data.groupby(self.id_col)
        
        # Create sequences with padding
        self.sequences = []
        self.participant_ids = []
        self.targets = []
        self.masks = []  # To track which positions are padded
        
        # Process each participant
        for participant_id, participant_data in grouped_data:
            # For each possible starting position in this participant's data
            max_idx = len(participant_data) - 1  # Need at least 1 target day
            
            for start_idx in range(max_idx):
                # Get available data for this sequence
                available_len = min(self.seq_length, max_idx - start_idx)
                end_idx = start_idx + available_len
                
                # Get features for available days
                x_seq = participant_data.iloc[start_idx:end_idx][self.features].values.astype(np.float32)
                
                # Create mask (1 for real data, 0 for padding)
                mask = np.ones(self.seq_length, dtype=np.float32)
                
                # If we need padding
                if available_len < self.seq_length:
                    # Create padded sequence with zeros
                    padded_seq = np.zeros((self.seq_length, len(self.features)), dtype=np.float32)
                    # Copy available data
                    padded_seq[:available_len] = x_seq
                    # Update mask to indicate padding
                    mask[available_len:] = 0
                    x_seq = padded_seq
                
                # Get target (next day's mood)
                if end_idx < len(participant_data):
                    target = participant_data.iloc[end_idx][self.target_col]
                    
                    # Store this sequence
                    self.sequences.append(x_seq)
                    self.participant_ids.append(participant_id)
                    self.targets.append(target)
                    self.masks.append(mask)
    
    def _normalize_data(self, 
                       df: pd.DataFrame, 
                       transform_target: bool = False,
                       scaler_type: str = "StandardScaler",
                       existing_scalers: Dict = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize data either globally or per participant
        
        Args:
            df: pandas DataFrame to normalize
            transform_target: whether to normalize target variable
            scaler_type: type of scaler to use
            existing_scalers: pre-trained scalers to use
            
        Returns:
            normalized DataFrame and dictionary of scalers
        """
        # Initialize empty scalers dictionary if not provided
        scalers = existing_scalers if existing_scalers is not None else {}
        
        # Choose scaler type
        def get_new_scaler():
            if scaler_type == "StandardScaler":
                return StandardScaler()
            elif scaler_type == "MinMaxScaler":
                return MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Normalize per participant if requested
        if self.per_participant_normalization:
            participant_ids = df[self.id_col].unique()
            
            # Initialize scalers for features if they don't exist
            if 'features' not in scalers:
                scalers['features'] = {p_id: get_new_scaler() for p_id in participant_ids}
                
            # Initialize scaler for target if requested and doesn't exist
            if transform_target and 'target' not in scalers:
                scalers['target'] = {p_id: get_new_scaler() for p_id in participant_ids}
            
            # Process each participant separately
            for p_id in participant_ids:
                mask = df[self.id_col] == p_id
                
                # Normalize features
                feature_cols = [col for col in self.features if col != self.id_col]
                if len(feature_cols) > 0:  # Only normalize if there are features
                    if p_id in scalers['features']:
                        df.loc[mask, feature_cols] = scalers['features'][p_id].fit_transform(df.loc[mask, feature_cols])
                    
                # Normalize target if requested
                if transform_target:
                    if p_id in scalers['target']:
                        df.loc[mask, self.target_col] = scalers['target'][p_id].fit_transform(df.loc[mask, [self.target_col]])
        
        # Global normalization across all participants
        else:
            # Initialize scalers if they don't exist
            if 'features' not in scalers:
                scalers['features'] = get_new_scaler()
                
            if transform_target and 'target' not in scalers:
                scalers['target'] = get_new_scaler()
            
            # Normalize features
            feature_cols = [col for col in self.features if col != self.id_col]
            if len(feature_cols) > 0:  # Only normalize if there are features
                df[feature_cols] = scalers['features'].fit_transform(df[feature_cols])
            
            # Normalize target if requested
            if transform_target:
                df[self.target_col] = scalers['target'].fit_transform(df[[self.target_col]])
        
        return df, scalers
    
    def get_scalers(self) -> Dict:
        """Return the scalers used for normalization"""
        return self.scalers
    
    def __len__(self) -> int:
        """Return the number of sequences"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target
        
        Args:
            idx: index of the sequence
            
        Returns:
            Tuple of (features, participant_id, target, mask)
        """
        # Get precomputed sequence data
        x_features = self.sequences[idx]
        participant_id = self.participant_ids[idx]
        y = self.targets[idx]
        mask = self.masks[idx]
        
        # Convert to tensors
        x_features_tensor = torch.tensor(x_features)
        x_id_tensor = torch.tensor([participant_id] * self.seq_length, dtype=torch.int64)
        y_tensor = torch.tensor(y).float()
        mask_tensor = torch.tensor(mask)
        
        return x_features_tensor, x_id_tensor, y_tensor, mask_tensor



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
    test_dataset's original data (which includes the 'day' and 'id_num' columns), and then plots
    real vs predicted values for all participants using matplotlib/seaborn.

    Parameters:
        model: Trained PyTorch model.
        data_loader: DataLoader for the dataset to predict on.
        test_dataset: The dataset instance (e.g., MultiParticipantDataset) used to create data_loader.
                      It must have a 'data' attribute containing the original DataFrame with a 'day' column.
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
        
        # Sort by day to ensure proper line plotting
        df_p = df_p.sort_values('day')
        
        # Plot real values
        sns.lineplot(x='day', y='Real', data=df_p, ax=axes[i], label='Real', marker='o')
        
        # Plot predicted values
        sns.lineplot(x='day', y='Predicted', data=df_p, ax=axes[i], label='Predicted', marker='x')
        
        # Customize plot
        axes[i].set_title(f'Participant {p}')
        axes[i].set_xlabel('Day')
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


def compare_train_val_predictions(train_model, train_loader, train_dataset, 
                                val_model, val_loader, val_dataset,
                                target_scaler=None, show_plot=True, 
                                save_fig=True, title="train_val_comparison",
                                scaler_type="StandardScaler"):
    """
    Runs predictions on both training and validation data, and plots them side by side
    for each participant across the whole duration.
    
    Parameters:
        train_model: Trained PyTorch model for training data.
        train_loader: DataLoader for the training dataset.
        train_dataset: The dataset instance for training data.
        val_model: Trained PyTorch model for validation data (often same as train_model).
        val_loader: DataLoader for the validation dataset.
        val_dataset: The dataset instance for validation data.
        target_scaler: (Optional) Scaler used to normalize the target data.
        show_plot: Whether to display the plot.
        save_fig: Whether to save the figure to disk.
        title: Title prefix for the plot.
        scaler_type: Type of scaler used ("StandardScaler" or "MinMaxScaler").
    """
    
    # Get predictions for training data
    train_results, train_mae, train_mse, train_rmse, train_r2 = predict_and_plot(
        train_model, train_loader, train_dataset, target_scaler, 
        show_plot=False, save_fig=False, title=f"{title}_train", scaler_type=scaler_type
    )
    
    # Get predictions for validation data
    val_results, val_mae, val_mse, val_rmse, val_r2 = predict_and_plot(
        val_model, val_loader, val_dataset, target_scaler,
        show_plot=False, save_fig=False, title=f"{title}_val", scaler_type=scaler_type
    )
    
    # Add a dataset column to identify train vs val
    train_results['Dataset'] = 'Train'
    val_results['Dataset'] = 'Validation'
    
    # Get participant column name
    participant_col = train_dataset.id_col  # e.g., 'id_num'
    
    # Get unique participants from both datasets
    train_participants = train_results[participant_col].unique()
    val_participants = val_results[participant_col].unique()
    all_participants = np.union1d(train_participants, val_participants)
    
    # Create a figure with subplots for each participant
    n_participants = len(all_participants)
    fig, axes = plt.subplots(n_participants, 2, figsize=(16, 4 * n_participants), constrained_layout=True)
    
    # If there's only one participant, axes won't be a 2D array, so reshape it
    if n_participants == 1:
        axes = axes.reshape(1, 2)
    
    # Plot train and validation results side by side for each participant
    for i, p in enumerate(all_participants):
        # Training data plot
        if p in train_participants:
            df_p_train = train_results[train_results[participant_col] == p]
            df_p_train = df_p_train.sort_values('day')
            
            # Plot real values
            sns.lineplot(x='day', y='Real', data=df_p_train, ax=axes[i, 0], 
                        label='Real', marker='o', color='blue')
            
            # Plot predicted values
            sns.lineplot(x='day', y='Predicted', data=df_p_train, ax=axes[i, 0], 
                        label='Predicted', marker='x', color='red')
            
            # Customize plot
            axes[i, 0].set_title(f'Training Data - Participant {p}')
            axes[i, 0].set_xlabel('Day')
            axes[i, 0].set_ylabel('Mood Value')
            axes[i, 0].legend()
            axes[i, 0].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[i, 0].text(0.5, 0.5, f'No training data for Participant {p}', 
                          ha='center', va='center', transform=axes[i, 0].transAxes)
            axes[i, 0].set_title(f'Training Data - Participant {p}')
            
        # Validation data plot
        if p in val_participants:
            df_p_val = val_results[val_results[participant_col] == p]
            df_p_val = df_p_val.sort_values('day')
            
            # Plot real values
            sns.lineplot(x='day', y='Real', data=df_p_val, ax=axes[i, 1], 
                        label='Real', marker='o', color='blue')
            
            # Plot predicted values
            sns.lineplot(x='day', y='Predicted', data=df_p_val, ax=axes[i, 1], 
                        label='Predicted', marker='x', color='red')
            
            # Customize plot
            axes[i, 1].set_title(f'Validation Data - Participant {p}')
            axes[i, 1].set_xlabel('Day')
            axes[i, 1].set_ylabel('Mood Value')
            axes[i, 1].legend()
            axes[i, 1].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[i, 1].text(0.5, 0.5, f'No validation data for Participant {p}', 
                          ha='center', va='center', transform=axes[i, 1].transAxes)
            axes[i, 1].set_title(f'Validation Data - Participant {p}')
    
    # Add an overall title to the figure
    fig.suptitle(f'Train vs Validation: Real and Predicted Mood Values\n{title}', fontsize=16)
    
    # Print metrics
    print("\nTraining Metrics:")
    print(f"MAE: {train_mae}, RMSE: {train_rmse}, R2: {train_r2}")
    print("\nValidation Metrics:")
    print(f"MAE: {val_mae}, RMSE: {val_rmse}, R2: {val_r2}")
    
    # Display the plot if requested
    if show_plot:
        plt.show()
    
    # Save the figure if requested
    if save_fig:
        outdir = "figures/matplotlib/train_val_comparison"
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"{title}.png"), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.join(outdir, f'{title}.png')}")
    
    return {
        'train_results': train_results,
        'train_metrics': {'mae': train_mae, 'mse': train_mse, 'rmse': train_rmse, 'r2': train_r2},
        'val_results': val_results,
        'val_metrics': {'mae': val_mae, 'mse': val_mse, 'rmse': val_rmse, 'r2': val_r2}
    }


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
                      dataset_name=None, imputation=None, dropped_vars=None, save_fig=True):
    """
    Train and evaluate a model with given hyperparameters.
    Updated to work with padding-enabled MultiParticipantDataset.
    
    Args:
        config: Dictionary of hyperparameters
        checkpoint_dir: Directory for checkpoints
        train_df, val_df, test_df: DataFrames for training, validation, and testing
        dataset_name, imputation, dropped_vars: Data information
    
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

    transform_target = config["transform_target"]
    per_participant_norm = config["per_participant_normalization"]
    scaler_type = config["scaler_type"]

    shuffle_data = config["shuffle_data"]

    
    # Normalize data if needed
    # Create datasets with normalization inside
    train_dataset = MultiParticipantDataset(
        df=train_df, 
        seq_length=seq_length,
        normalize_data=True,
        transform_target=transform_target,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_norm
    )
    
    # Get scalers from training set to reuse
    scalers = train_dataset.get_scalers()
    
    # Create validation and test datasets with same scalers
    val_dataset = MultiParticipantDataset(
        df=val_df, 
        seq_length=seq_length,
        normalize_data=True,
        transform_target=transform_target,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_norm,
        existing_scalers=scalers
    )
    
    test_dataset = MultiParticipantDataset(
        df=test_df, 
        seq_length=seq_length,
        normalize_data=True,
        transform_target=transform_target,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_norm,
        existing_scalers=scalers
    )
    
    # Get target scaler for later use
    if transform_target:
        if per_participant_norm:
            scaler_target = scalers.get('target', None)  # Dictionary of participant-specific scalers
        else:
            scaler_target = scalers.get('target', None)  # Single scaler
    else:
        scaler_target = None
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimensions from dataset
    sample_batch = next(iter(train_loader))
    if len(sample_batch) == 4:  # With mask
        x_features, x_id, _, _ = sample_batch
        input_dim = x_features.shape[2]  # [batch_size, seq_length, features]
    else:  # Without mask
        x_features, x_id, _ = sample_batch
        input_dim = x_features.shape[2]
    
    # Count number of unique participants for embedding
    num_participants = train_df[train_df.columns[0]].nunique()
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
    
    # Load checkpoint if available
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
            # Handle batch based on whether it includes masks
            if len(batch) == 4:  # With mask
                x_features, x_id, y, mask = [b.to(device) for b in batch]
                # Forward pass (with mask)
                outputs = model(x_features, x_id, mask)
            else:  # Without mask (for backward compatibility)
                x_features, x_id, y = [b.to(device) for b in batch]
                # Forward pass (without mask)
                outputs = model(x_features)
            

            # print(f"x_features shape: {x_features.shape}")
            # print(f"outputs shape: {outputs.shape}")
            # print(f"y shape: {y.shape}")

            # # Try to inspect the model structure:
            # print(f"Model structure: {model}")

            # # Right before loss calculation:
            # print(f"outputs.squeeze() shape: {outputs.squeeze().shape}")
            # Calculate loss
            loss = criterion(outputs.squeeze(), y)
            
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
                # Handle batch based on whether it includes masks
                if len(batch) == 4:  # With mask
                    x_features, x_id, y, mask = [b.to(device) for b in batch]
                    # Forward pass (with mask)
                    outputs = model(x_features, x_id, mask)
                else:  # Without mask (for backward compatibility)
                    x_features, x_id, y = [b.to(device) for b in batch]
                    # Forward pass (without mask)
                    outputs = model(x_features)
                
                # Calculate loss
                loss = criterion(outputs.squeeze(), y)
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
        # if patience_counter >= patience:
        #     print(f"Early stopping triggered after {epoch+1} epochs")
        #     break
    
    # Load best model for evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle batch based on whether it includes masks
            if len(batch) == 4:  # With mask
                x_features, x_id, y, mask = [b.to(device) for b in batch]
                # Forward pass (with mask)
                outputs = model(x_features, x_id, mask)
            else:  # Without mask (for backward compatibility)
                x_features, x_id, y = [b.to(device) for b in batch]
                # Forward pass (without mask)
                outputs = model(x_features)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), y)
            test_loss += loss.item()
            
            # Store predictions and actuals for metrics
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform if target was normalized
    if transform_target and scaler_target is not None:
        if isinstance(scaler_target, dict):
            # This is a simplified approach for per-participant normalization
            # For proper handling, would need to track participant IDs
            # Just using the first participant's scaler as an approximation
            first_participant = list(scaler_target.keys())[0]
            predictions_2d = predictions.reshape(-1, 1)
            actuals_2d = actuals.reshape(-1, 1)
            predictions = scaler_target[first_participant].inverse_transform(predictions_2d).flatten()
            actuals = scaler_target[first_participant].inverse_transform(actuals_2d).flatten()
        else:
            # Global scaler
            predictions_2d = predictions.reshape(-1, 1)
            actuals_2d = actuals.reshape(-1, 1)
            predictions = scaler_target.inverse_transform(predictions_2d).flatten()
            actuals = scaler_target.inverse_transform(actuals_2d).flatten()
    
    # Calculate metrics for test set
    mse_test = mean_squared_error(actuals, predictions)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(actuals, predictions)
    r2_test = r2_score(actuals, predictions)
    
    print(f"Test MSE: {mse_test:.4f}")
    print(f"Test RMSE: {rmse_test:.4f}")
    print(f"Test MAE: {mae_test:.4f}")
    print(f"Test R²: {r2_test:.4f}")
    
    # Create results dataframe for test set
    df_test_results = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
    })
    
    # Calculate metrics for train and validation sets similarly
    # (Simplified implementation compared to original - would need to adapt predict_and_plot function)
    
    # Training set metrics
    train_predictions = []
    train_actuals = []
    
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            if len(batch) == 4:
                x_features, x_id, y, mask = [b.to(device) for b in batch]
                outputs = model(x_features, x_id, mask)
            else:
                x_features, x_id, y = [b.to(device) for b in batch]
                outputs = model(x_features)
            
            train_predictions.extend(outputs.squeeze().cpu().numpy())
            train_actuals.extend(y.cpu().numpy())
    
    train_predictions = np.array(train_predictions)
    train_actuals = np.array(train_actuals)
    
    # Inverse transform if needed
    if transform_target and scaler_target is not None:
        if isinstance(scaler_target, dict):
            first_participant = list(scaler_target.keys())[0]
            train_predictions_2d = train_predictions.reshape(-1, 1)
            train_actuals_2d = train_actuals.reshape(-1, 1)
            train_predictions = scaler_target[first_participant].inverse_transform(train_predictions_2d).flatten()
            train_actuals = scaler_target[first_participant].inverse_transform(train_actuals_2d).flatten()
        else:
            train_predictions_2d = train_predictions.reshape(-1, 1)
            train_actuals_2d = train_actuals.reshape(-1, 1)
            train_predictions = scaler_target.inverse_transform(train_predictions_2d).flatten()
            train_actuals = scaler_target.inverse_transform(train_actuals_2d).flatten()
    
    mse_train = mean_squared_error(train_actuals, train_predictions)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(train_actuals, train_predictions)
    r2_train = r2_score(train_actuals, train_predictions)
    
    # Validation set metrics
    val_predictions = []
    val_actuals = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                x_features, x_id, y, mask = [b.to(device) for b in batch]
                outputs = model(x_features, x_id, mask)
            else:
                x_features, x_id, y = [b.to(device) for b in batch]
                outputs = model(x_features)
            
            val_predictions.extend(outputs.squeeze().cpu().numpy())
            val_actuals.extend(y.cpu().numpy())
    
    val_predictions = np.array(val_predictions)
    val_actuals = np.array(val_actuals)
    
    # Inverse transform if needed
    if transform_target and scaler_target is not None:
        if isinstance(scaler_target, dict):
            first_participant = list(scaler_target.keys())[0]
            val_predictions_2d = val_predictions.reshape(-1, 1)
            val_actuals_2d = val_actuals.reshape(-1, 1)
            val_predictions = scaler_target[first_participant].inverse_transform(val_predictions_2d).flatten()
            val_actuals = scaler_target[first_participant].inverse_transform(val_actuals_2d).flatten()
        else:
            val_predictions_2d = val_predictions.reshape(-1, 1)
            val_actuals_2d = val_actuals.reshape(-1, 1)
            val_predictions = scaler_target.inverse_transform(val_predictions_2d).flatten()
            val_actuals = scaler_target.inverse_transform(val_actuals_2d).flatten()
    
    mse_val = mean_squared_error(val_actuals, val_predictions)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(val_actuals, val_predictions)
    r2_val = r2_score(val_actuals, val_predictions)
    
    # Create results dataframes
    df_train_results = pd.DataFrame({
        'Actual': train_actuals,
        'Predicted': train_predictions,
    })
    
    df_val_results = pd.DataFrame({
        'Actual': val_actuals,
        'Predicted': val_predictions,
    })
    
    # Save plots if requested
    if save_fig:
        os.makedirs('figures', exist_ok=True)
        
        # Train set plot
        plt.figure(figsize=(10, 6))
        plt.scatter(train_actuals, train_predictions, alpha=0.5)
        plt.plot([min(train_actuals), max(train_actuals)], [min(train_actuals), max(train_actuals)], 'r--')
        plt.xlabel('Actual Mood')
        plt.ylabel('Predicted Mood')
        plt.title(f'Train Set: Actual vs Predicted (R² = {r2_train:.4f})')
        plt.savefig('figures/train_predictions.png')
        plt.close()
        
        # Validation set plot
        plt.figure(figsize=(10, 6))
        plt.scatter(val_actuals, val_predictions, alpha=0.5)
        plt.plot([min(val_actuals), max(val_actuals)], [min(val_actuals), max(val_actuals)], 'r--')
        plt.xlabel('Actual Mood')
        plt.ylabel('Predicted Mood')
        plt.title(f'Validation Set: Actual vs Predicted (R² = {r2_val:.4f})')
        plt.savefig('figures/val_predictions.png')
        plt.close()
        
        # Test set plot
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
        plt.xlabel('Actual Mood')
        plt.ylabel('Predicted Mood')
        plt.title(f'Test Set: Actual vs Predicted (R² = {r2_test:.4f})')
        plt.savefig('figures/test_predictions.png')
        plt.close()
        
        # Loss curves
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='o')
        plt.axhline(y=avg_test_loss, color='r', linestyle='-', label='Test Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training, Validation, and Test Loss")
        plt.legend()
        plt.savefig("figures/loss_curves.png")
        plt.close()
    
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
    
    # Collect hyperparameters (adapted for enhanced model)
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
        "num_epochs": epoch + 1,  # Actual number of epochs run
        "learning_rate": learning_rate,
        "patience": patience,
        "clip_gradients": config.get("clip_gradients", True),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        
        # Metadata
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device.type,
    }
    
    # Save model (assuming save_model function is defined elsewhere)
    if 'save_model' in globals():
        save_model(model, hyperparams, metrics)
    else:
        # Simple model saving if save_model function not available
        os.makedirs('models', exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparams': hyperparams,
            'metrics': metrics
        }, f'models/model_{model_type}_{timestamp}.pt')
    
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
    os.makedirs('tables/results', exist_ok=True)
    
    if not os.path.exists('tables/results/model_results.csv'):
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
                dropped_vars=dropped_vars,
                save_fig=False
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


