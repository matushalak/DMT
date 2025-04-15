import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime

import json
import sys
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
    target_col: str = 'categorical_target',
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
        
        # Initialize scalers for features only (we don't normalize the categorical target)
        if 'features' not in scalers:
            scalers['features'] = {p_id: create_scaler() for p_id in participant_ids}
        
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
    
    # Global normalization
    else:
        # Initialize global scalers
        if 'features' not in scalers:
            scalers['features'] = create_scaler()
        
        # Normalize features
        df[feature_cols] = scalers['features'].fit_transform(df[feature_cols])
    
    return df, scalers


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'categorical_target',
    seq_length: int = 7,
    id_col: str = 'id_num'
) -> Tuple[List[np.ndarray], List[int], List[int], List[np.ndarray]]:
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
            
            # Get target (next date's value) - store as integer class index
            if end_idx < len(participant_data):
                target = participant_data.iloc[end_idx][target_col]
                
                # Store the sequence data
                sequences.append(x_seq)
                participant_ids.append(participant_id)
                targets.append(target)
                masks.append(mask)
    
    return sequences, participant_ids, targets, masks


class CategoricalLSTMDataset(Dataset):
    """
    Dataset for multi-participant time series classification with LSTM.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_length: int,
        target_col: str = 'categorical_target',
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
        
        # Drop the original target to prevent data leakage
        if 'target' in df.columns:
            df = df.drop(columns=['target'])
        
        # Define feature columns (excluding target, date and id columns)
        exclude_cols = [target_col, 'date', id_col, "id_num", "time_of_day_encoded", "next_date"]
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Normalize data (features only, not the categorical target)
        df, self.scalers = normalize_data(
            df=df,
            feature_cols=self.feature_cols,
            target_col=target_col,
            per_participant_normalization=per_participant_normalization,
            scaler_type=scaler_type,
            existing_scalers=existing_scalers
        )
        
        # Map target classes to integers if they aren't already
        if df[target_col].dtype != int:
            self.class_mapping = {val: idx for idx, val in enumerate(sorted(df[target_col].unique()))}
            df[target_col] = df[target_col].map(self.class_mapping)
        else:
            self.class_mapping = None
        
        # Store the number of classes
        self.num_classes = len(df[target_col].unique())
        
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
        target_tensor = torch.tensor(target, dtype=torch.long)  # Use long for classification targets
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        return features_tensor, id_tensor, target_tensor, mask_tensor
    
    def get_scalers(self) -> Dict:
        """Return the scalers used for normalization."""
        return self.scalers
    
    def get_class_mapping(self) -> Dict:
        """Return the mapping from original classes to integers."""
        return self.class_mapping
    
    def get_num_classes(self) -> int:
        """Return the number of classes."""
        return self.num_classes


def process_lstm_data(
    df: pd.DataFrame,
    seq_length: int = 7,
    target_col: str = 'categorical_target',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    split_within_participants: bool = True,
    scaler_type: str = "MinMaxScaler",
    per_participant_normalization: bool = True
) -> Tuple[CategoricalLSTMDataset, CategoricalLSTMDataset, CategoricalLSTMDataset, Dict]:
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
    print("Target classes in train set:", train_df[target_col].value_counts())
    print("Target classes in validation set:", val_df[target_col].value_counts())
    print("Target classes in test set:", test_df[target_col].value_counts())
    
    # Create training dataset (with new scalers)
    train_dataset = CategoricalLSTMDataset(
        df=train_df,
        seq_length=seq_length,
        target_col=target_col,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_normalization
    )
    
    # Get trained scalers to use for validation and test sets
    trained_scalers = train_dataset.get_scalers()
    
    # Create validation dataset (using training scalers)
    val_dataset = CategoricalLSTMDataset(
        df=val_df,
        seq_length=seq_length,
        target_col=target_col,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_normalization,
        existing_scalers=trained_scalers
    )
    
    # Create test dataset (using training scalers)
    test_dataset = CategoricalLSTMDataset(
        df=test_df,
        seq_length=seq_length,
        target_col=target_col,
        scaler_type=scaler_type,
        per_participant_normalization=per_participant_normalization,
        existing_scalers=trained_scalers
    )
    
    return train_dataset, val_dataset, test_dataset, trained_scalers


class CategoricalLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super(CategoricalLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Two-layer LSTM with dropout applied to outputs of each layer (except the last)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Fully-connected layer to output the class logits
        self.fc = nn.Linear(hidden_dim, num_classes)
        
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
        logits = self.fc(last_out)  # Shape: [batch_size, num_classes]
        return logits


class CategoricalGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super(CategoricalGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Multi-layer GRU with dropout applied between layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout)
        
        # Fully-connected output layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
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
            idx = (seq_lengths - 1).view(-1, 1).unsqueeze(1).expand(-1, 1, self.hidden_dim).long()
            last_out = out.gather(1, idx).squeeze(1)
        else:
            # Standard forward pass without packing
            out, _ = self.gru(x, h0)
            # Use the last time step's output
            last_out = out[:, -1, :]
        
        # Final prediction
        logits = self.fc(last_out)
        return logits


class CategoricalRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super(CategoricalRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Multi-layer RNN with dropout
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout)
        
        # Fully-connected output layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
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
            idx = (seq_lengths - 1).view(-1, 1).unsqueeze(1).expand(-1, 1, self.hidden_dim).long()
            last_out = out.gather(1, idx).squeeze(1)
        else:
            # Standard forward pass without packing
            out, _ = self.rnn(x, h0)
            # Use the last time step's output
            last_out = out[:, -1, :]
        
        # Final prediction
        logits = self.fc(last_out)
        return logits
    

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
    target_col = config.get("target_col", "categorical_target")
    
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
    
    # Get number of classes from dataset
    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")
    
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
    
    # Initialize the model based on type
    if model_type == "LSTM":
        model = CategoricalLSTMModel(input_dim, hidden_dim, num_layers, num_classes, dropout)
    elif model_type == "SimpleRNN":
        model = CategoricalRNNModel(input_dim, hidden_dim, num_layers, num_classes, dropout)
    elif model_type == "GRU":
        model = CategoricalGRUModel(input_dim, hidden_dim, num_layers, num_classes, dropout)
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load checkpoint if available
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "model.pt")):
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
        print(f"Loaded model from {checkpoint_dir}")
    
    # Storage for losses and metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience = config.get("patience", 5)  # Early stopping patience
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            x_features, x_id, y, mask = [b.to(device) for b in batch]
            
            # Forward pass with mask
            outputs = model(x_features, x_id, mask)
            
            # Calculate loss
            loss = criterion(outputs, y)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            
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
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_features, x_id, y, mask = [b.to(device) for b in batch]
                outputs = model(x_features, x_id, mask)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate model on all datasets
    metrics = {}
    
    # Helper function for model evaluation
    def evaluate_dataset(model, data_loader, dataset_name):
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                x_features, x_id, y, mask = [b.to(device) for b in batch]
                outputs = model(x_features, x_id, mask)
                loss = criterion(outputs, y)
                total_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                # Store predictions and targets for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        # Get class mapping if available
        class_mapping = getattr(data_loader.dataset, 'class_mapping', None)
        
        # Convert predictions and targets to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Compute confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Print results
        print(f"\n{dataset_name} Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Print confusion matrix
        print(f"\n{dataset_name} Confusion Matrix:")
        print(cm)
        
        # Print classification report
        print(f"\n{dataset_name} Classification Report:")
        print(classification_report(targets, predictions, zero_division=0))
        
        # Save confusion matrix plot if requested
        if save_fig:
            os.makedirs('figures_classification', exist_ok=True)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{dataset_name} Confusion Matrix')
            plt.savefig(f'figures_classification/{dataset_name.lower()}_confusion_matrix.png')
            plt.close()
        
        # Return metrics
        return {
            f'loss_{dataset_name.lower()}': float(avg_loss),
            f'accuracy_{dataset_name.lower()}': float(accuracy),
            f'precision_{dataset_name.lower()}': float(precision),
            f'recall_{dataset_name.lower()}': float(recall),
            f'f1_{dataset_name.lower()}': float(f1),
            f'confusion_matrix_{dataset_name.lower()}': cm.tolist(),
            f'predictions_{dataset_name.lower()}': predictions.tolist(),
            f'targets_{dataset_name.lower()}': targets.tolist()
        }
    
    # Evaluate each dataset
    train_metrics = evaluate_dataset(model, train_loader, "Train")
    val_metrics = evaluate_dataset(model, val_loader, "Val")
    test_metrics = evaluate_dataset(model, test_loader, "Test")
    
    # Combine all metrics
    metrics = {}
    metrics.update(train_metrics)
    metrics.update(val_metrics)
    metrics.update(test_metrics)
    
    # Save loss and accuracy curves
    if save_fig:
        # Loss curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='o')
        plt.axhline(y=float(metrics['loss_test']), color='r', linestyle='-', label='Test Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Cross-Entropy)")
        plt.title("Training, Validation, and Test Loss")
        plt.legend()
        
        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy', marker='o')
        plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Val Accuracy', marker='o')
        plt.axhline(y=float(metrics['accuracy_test']), color='r', linestyle='-', label='Test Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training, Validation, and Test Accuracy")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("figures_classification/learning_curves.png")
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
        "num_classes": num_classes,
        "dropout": dropout,
        
        # Dataset sizes
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        
        # Data processing
        "sequence_length": seq_length,
        "scaler": scaler_type,
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
    
    # Create results summary DataFrame
    results_dict = {
        'Accuracy_train': float(metrics['accuracy_train']),
        'Accuracy_val': float(metrics['accuracy_val']),
        'Accuracy_test': float(metrics['accuracy_test']),
        'F1_train': float(metrics['f1_train']),
        'F1_val': float(metrics['f1_val']),
        'F1_test': float(metrics['f1_test']),
        'Precision_train': float(metrics['precision_train']),
        'Precision_val': float(metrics['precision_val']),
        'Precision_test': float(metrics['precision_test']),
        'Recall_train': float(metrics['recall_train']),
        'Recall_val': float(metrics['recall_val']),
        'Recall_test': float(metrics['recall_test']),
        'dataset': dataset_name
    }
    
    # Create DataFrame from the values
    results_df = pd.DataFrame({f"{model_type}_{timestamp}": results_dict}).T
    
    os.makedirs('tables/results', exist_ok=True)
    
    # Append to existing results or create new file
    results_path = 'tables/results/model_results.csv'
    if not os.path.exists(results_path):
        results_df.to_csv(results_path, index=True, header=True)
    else:
        results_df.to_csv(results_path, mode='a', header=False, index=True)
    
    # Generate visualization of predictions
    predict_and_plot_categorical(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        class_mapping=train_dataset.get_class_mapping()
    )
    
    return metrics

# ------------------------------------------------------

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
    num_classes = hyperparams['num_classes']
    dropout = hyperparams['dropout']
    model_type = hyperparams['model']
    
    if model_type == "LSTM":
        model = CategoricalLSTMModel(input_dim, hidden_dim, num_layers, num_classes, dropout)
    elif model_type == "SimpleRNN":
        model = CategoricalRNNModel(input_dim, hidden_dim, num_layers, num_classes, dropout)
    elif model_type == "GRU":
        model = CategoricalGRUModel(input_dim, hidden_dim, num_layers, num_classes, dropout)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Load model state
    model.load_state_dict(torch.load(os.path.join(model_path, "model_state.pt")))
    model = model.to(device)
    model.eval()
    
    return model, hyperparams, metrics


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
        # _create_parameter_plots(results_dir, all_results, key_param_names)
    else:
        print("No successful trials completed.")
        best_config = default_config
    
    return best_config

def predict_and_plot_categorical(model, train_loader, test_loader, num_classes, class_mapping, save_fig=True):
    """
    Generate predictions on training and test sets, plot confusion matrices,
    and optionally save the figures.

    Args:
        model: Trained PyTorch model.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
        num_classes: Total number of classes.
        class_mapping: Mapping from original class labels to integer indices.
                       If provided, the inverse mapping will be used for display.
        save_fig: Whether to save the generated plots.
    """
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import os

    # Determine the device from model parameters
    device = next(model.parameters()).device
    datasets = {"Train": train_loader, "Test": test_loader}

    for name, loader in datasets.items():
        all_targets = []
        all_preds = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                # Unpack batch elements (assuming order: features, id, target, mask)
                x_features, x_id, y, mask = batch
                x_features = x_features.to(device)
                y = y.to(device)
                mask = mask.to(device)
                outputs = model(x_features, x_id.to(device), mask)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

        # Create label names using inverse mapping if available, otherwise default labels
        if class_mapping:
            inv_mapping = {v: k for k, v in class_mapping.items()}
            labels = [inv_mapping[i] for i in range(num_classes)]
        else:
            labels = [f"Class {i}" for i in range(num_classes)]

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{name} Confusion Matrix')
        plt.tight_layout()

        # Save or show the figure
        if save_fig:
            os.makedirs('figures_classification', exist_ok=True)
            plt.savefig(os.path.join('figures_classification', f'{name.lower()}_confusion_matrix.png'))
            plt.close()
        else:
            plt.show()


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



    # combine all app.Categorical features into one
    df['appCat'] = df[df.columns[df.columns.str.contains('appCat')]].sum(axis=1)
    # Drop individual app.Categorical features
    df.drop(columns=df.columns[df.columns.str.contains('appCat.')], inplace=True)
    # print all appCat columns
    print("appCat columns:")
    print(df.columns[df.columns.str.contains('appCat')])
    
    # select features
    features = ["id_num", "date", "target",'mood_last_daily', 'circumplex.QUADRANT_daily_2', 'weekday_1', 'weekday_4', 'weekday_3', 'month_5', 'mva7_circumplex.valence_std.slide_daily', 'change_screen_sum_daily', 'circumplex.valence_std.slide_daily', 'activity_max_daily', 'circumplex.QUADRANT_daily_3', 'mood_std.slide_daily', 'circumplex.valence_mean_daily', 'mva7_change_screen_sum_daily', 'mood_first_daily', 'circumplex.valence_last_daily', 'mva7_mood_mean_daily']

    # Select only the relevant columns
    # df = df[features]

    do_hyperparameter_tuning = True  # Set to True to enable tuning
    
    
    # Default hyperparameters
    config = {
        # Data parameters
        "seq_length": 7,                        # Number of days to use for prediction
        "batch_size": 32,                       # Batch size for training
        
        # Model architecture
        "model_type": "SimpleRNN",                   # Model type (LSTM, GRU, or SimpleRNN)
        "hidden_dim": 16,                       # Size of LSTM hidden layer
        "num_layers": 2,                        # Number of LSTM layers
        "dropout": 0.3,                         # Dropout rate
        
        # Training parameters
        "learning_rate": 0.003,                 # Learning rate for Adam optimizer
        "num_epochs": 30,                       # Maximum number of training epochs
        "clip_gradients": False,                 # Whether to use gradient clipping
        "max_grad_norm": 1.0,                   # Maximum gradient norm if clipping
        
        # Data processing
        "transform_features": True,             # Whether to normalize features
        "transform_target": False,               # Whether to normalize target
        "scaler_type": "MinMaxScaler",        # Scaler type (StandardScaler or MinMaxScaler)
        "shuffle_data": True,                   # Whether to shuffle training data
        
        # Additional options for padding support
        "per_participant_normalization": True  # Whether to normalize per participant
    }
    
    # Create directories for outputs
    os.makedirs("tables/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures_classification", exist_ok=True)
    

    
    if do_hyperparameter_tuning:

        param_grid = {
            # Sequence parameters
            "seq_length": [5, 7],                 # Test shorter and longer sequence windows
            
            # Model architecture parameters
            "model_type": ["GRU", "SimpleRNN"],            # Focus on the two stronger model types
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

