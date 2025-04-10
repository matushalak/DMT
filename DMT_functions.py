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

print("DMT_functions.py loaded")

def create_daily_pivot(df, participant="all", return_dict=False, counts=True):
    """
    Create a daily pivot table for a given participant or all participants or list of participants.
    
    Each row corresponds to a day (from the earliest to the latest day the participant has data),
    each column corresponds to a variable, and if multiple datapoints occur on a given day,
    the value is aggregated as the mean. Days with no data for a variable are represented as NaN.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing at least the columns 'id_num', 'time', 
                           'variable', and 'value'. The 'time' column should be in datetime format.
        participant: Either "all" (to process all participants), a single participant id, or a list of participant ids.
        return_dict (bool): If True, returns a dictionary of pivot tables (keyed by participant).
                            Otherwise, returns a single DataFrame with 'id_num' and 'day' as columns.
        counts (bool): If True, includes a count of the number of values for each variable per day.
    
    Returns:
        Either a dict mapping participant IDs to their daily pivot table or a combined DataFrame.
    """
    # Determine the list of participant IDs to process
    if participant == "all":
        participants = df["id_num"].unique()
    elif isinstance(participant, list):
        participants = participant
    else:
        participants = [participant]
        
    # Ensure the time column is in datetime format
    df_copy = df.copy()
    if "time" not in df_copy.columns:
        pass
    elif not pd.api.types.is_datetime64_any_dtype(df_copy["time"]):
        df_copy["time"] = pd.to_datetime(df_copy["time"])
    
        # Create a column with just the day (flooring the datetime to day)
        df_copy["day"] = df_copy["time"].dt.floor("D")
    
    pivot_dict = {}
    pivot_list = []
    
    # Process each participant separately
    for part in participants:
        df_part = df_copy[df_copy["id_num"] == part].copy()

        # get a count of the number of values of variable per day
        # df_part["comprising_of"] = df_part.groupby(["day", "variable"])["value"].transform("count")

        # select the mean aggregation columns, time values should be aggregated by sum
        mean_agg = ["mood", "circumplex.valence", "circumplex.arousal"] # not sure about activity
        sum_agg = [col for col in df_part["variable"].unique() if col not in mean_agg + ["id_num", "time", "day"]]

        #TODO: add last first mood value of the day
        #TODO: add the min max mood of the day

        #TODO: add variance within the day for mood and over window

        #TODO: add the difference in mood in last 2 days


        
        # Create a new DataFrame with the selected columns
        df_part_sum = df_part[df_part["variable"].isin(sum_agg)].copy()
        pivot_sum = df_part_sum.pivot_table(index="day",
                                            columns="variable", 
                                            values="value", 
                                            aggfunc="sum")

        # only non-nan values are averaged
        df_part_mean = df_part[df_part["variable"].isin(mean_agg)].copy()
        pivot_mean = df_part_mean.pivot_table(index="day",
                                              columns="variable", 
                                              values="value", 
                                              aggfunc="mean")
    
        # get the count of the number of values comprising the variable per day
        pivot_count = df_part.pivot_table(index="day",
                                    columns="variable",
                                    values="value",
                                    aggfunc="count")

        # add maximum values in a day
        max_agg = ["activity", "circumplex.valence", "circumplex.arousal"]
        df_part_max = df_part[df_part["variable"].isin(max_agg)].copy()
        pivot_max = df_part_max.pivot_table(index="day",
                                        columns="variable",
                                        values="value",
                                        aggfunc="max")

        # add the minimum values in a day
        min_agg = ["circumplex.valence", "circumplex.arousal"]
        df_part_min = df_part[df_part["variable"].isin(min_agg)].copy()
        pivot_min = df_part_min.pivot_table(index="day",
                                        columns="variable",
                                        values="value",
                                        aggfunc="min")


        #nans are 0
        pivot_count = pivot_count.fillna(0) # NOT SURE ABOUT THIS
        pivot_count = pivot_count.add_suffix("_count")

        pivot_max = pivot_max.add_suffix("_max")
        pivot_min = pivot_min.add_suffix("_min")



        # Combine the two pivot tables
        pivot = pd.concat([pivot_sum, pivot_mean, pivot_max, pivot_min], axis=1)
        # Pandas aligns the data based on the index (which here represents the days).
        # The union of the indices is taken, and any missing data for a particular day in one of the tables is filled with NaN. 
        # The same applies when you concatenate the count pivot table. 
        # This means that even if the individual pivot tables have different numbers of rows, the concatenation will produce a DataFrame covering all days from the union of the indices.
        if counts:
            pivot = pd.concat([pivot, pivot_count], axis=1)
                                            
        # Create a complete date range from the earliest to the latest day for this participant
        full_range = pd.date_range(start=df_part["day"].min(), end=df_part["day"].max(), freq="D")
        pivot = pivot.reindex(full_range)
        pivot.index.name = "day"
        
        # Convert index to a column and add participant id
        pivot = pivot.reset_index()
        pivot["id_num"] = part

        # reorder the columns
        desired_order = ["id_num", "day", "mood", "screen", "activity", "circumplex.valence", "circumplex.arousal", "call", "sms"]
        other_columns = [p for p in pivot.columns if p not in desired_order and not p.endswith("_count") and not p.endswith("_max") and not p.endswith("_min")]
        new_order = desired_order + other_columns

        # add the counts to the new order
        final_order = []
        for col in new_order:
            final_order.append(col)
            if col not in ["id_num", "day"]:
                if counts:
                    final_order.append(f"{col}_count")
                if col in ["circumplex.valence", "circumplex.arousal"]:
                    final_order.append(f"{col}_min")
                if col in ["activity", "circumplex.valence", "circumplex.arousal"]:
                    final_order.append(f"{col}_max")

        pivot = pivot[final_order]
        
        # print("final columns", pivot.columns)

        # rearrange the columns to have the id_num, day, mood, screen, activity, circumplex.valence, circumplex.arousal then the rest

        if return_dict:
            pivot_dict[part] = pivot
        else:
            pivot_list.append(pivot)
    
    if return_dict:
        return pivot_dict
    else:
        # Concatenate the list of dataframes without setting a multi-index so that
        # both 'day' and 'id_num' remain as regular columns
        combined = pd.concat(pivot_list, ignore_index=True)
        # sort by id_num and day
        combined = combined.sort_values(by=["id_num", "day"])
        # save the combined dataframe to a csv file
        if not os.path.exists("tables/pivot_tables_daily"):
            os.makedirs("tables/pivot_tables_daily")
        combined.to_csv(f"tables/pivot_tables_daily/daily_pivot_table_{participant}.csv", index=False)
        return combined


def nan_exploration(df, create_pivot=False, title="nan_exploration"):
    """Create a pd dataframe with the percentage of NaN values for each variable per person, along with the count of unique values per variable
    Select pivot=True if the df is already pivoted
    also select a title to save the html
    """

    # create a pivot
    if create_pivot is True:
        pivot = create_daily_pivot(df, participant="all", return_dict=False)
    else:
        pivot = df.copy()

    # get the percentage of NaN values for each variable per person
    df_nans_list = []

    for participant in df["id_num"].unique():
        # get the pivot for the participant
        pivot_participant = pivot[pivot["id_num"] == participant]
        # get the percentage of NaN values for each variable
        nan_percentage = round(pivot_participant.isna().mean(), 3)

        nan_count = pivot_participant.isna().sum()
        # get the count of unique values per variable
        unique_values = pivot_participant.nunique()

        zero_count = pivot_participant.isin([0]).sum()
        zero_percent = round(zero_count / pivot_participant.shape[0], 3)
        # create a dataframe with the results
        df_nan = pd.DataFrame({"nan_percentage": nan_percentage,"nan_count": nan_count, "unique_values": unique_values, "zero_count": zero_count, "zero_percent": zero_percent})
        df_nan["participant"] = participant
        df_nan = df_nan.reset_index()
        df_nans_list.append(df_nan)

    # all participants
    nan_percentage_all = round(pivot.isna().mean(), 3)
    nan_count_all = pivot.isna().sum()
    unique_values_all = pivot.nunique()
    zero_count_all = pivot.isin([0]).sum()
    zero_percent_all = round(zero_count_all / pivot.shape[0], 3)
    df_nan_all = pd.DataFrame({"nan_percentage": nan_percentage_all,"nan_count": nan_count_all, "unique_values": unique_values_all, "zero_count": zero_count_all, "zero_percent": zero_percent_all})
    df_nan_all["participant"] = "all"
    df_nan_all = df_nan_all.reset_index()
    # get the count of zero values per variable
    df_nans_list.append(df_nan_all)


    # concatenate the dataframes
    df_nans = pd.concat(df_nans_list, ignore_index=True)

    df_nans = df_nans.reset_index().rename(columns={"index": "variable"})
    print("df_nan renamed")
    # get overall values for all participants
    # overall_nan_percentage = df_nans.groupby("variable")["nan_percentage"].mean()
    # overall_nan_count = df_nans.groupby("variable")["nan_count"].sum()

    # count how many values equal 0
    # overall_zero_count = df_nans.groupby("variable")["zero_count"].sum()
    # overall_zero_percent = df_nans.groupby("variable")["zero_percent"].mean()


    # save the dataframe
    if not os.path.exists("tables/nan_exploration"):
        os.makedirs("tables/nan_exploration")
    df_nans.to_csv(f"tables/nan_exploration/{title}.csv", index=False)

    df_nans = df_nans.sort_values(by=["nan_percentage"], ascending=False)

    return df_nans



def plotly_all_participants_timeseries(df_plot, save_html=True, show_plot=True, title="time_series_all_participants"):
    """
    Plot time series for all participants, one line per participant per variable,
    with a dropdown toggle to show specific participants.
    
    The function expects a daily pivot table DataFrame with a column "day" (date)
    and "id_num" (participant identifier) along with other numeric variable columns.
    
    Parameters:
        df_plot (pd.DataFrame): DataFrame that is already a pivot.
        save_html (bool): Whether to save the figure as an HTML file.
        show_plot (bool): Whether to display the plot.
    """
    # Get the daily pivot table for all participants.
    # This function should return a DataFrame with columns "day", "id_num", and variable columns.
    
    # df_plot = create_daily_pivot(df, participant="all", return_dict=False)
    
    # Make sure "day" is a column; if it's not, throw an error.
    if "day" not in df_plot.columns:
        raise ValueError("Expected a 'day' column in the DataFrame.")
    
    # Convert "day" to a proper date format and rename as "date" for clarity
    df_plot["date"] = pd.to_datetime(df_plot["day"]).dt.date

    # Check that 'id_num' is present.
    if "id_num" not in df_plot.columns:
        raise ValueError("Expected a column 'id_num' to identify participants.")

    if "next_day" in df_plot.columns:
        print("PLOTLY TIMESERIES FUNCTION will delete column next day")
        df_plot = df_plot.drop("next_day", axis=1)
    
    # Identify numeric columns (i.e. variables)
    numeric_cols = df_plot.select_dtypes(include="number").columns.tolist()
    # Remove group-by columns if present
    group_cols = ["id_num", "day", "date"]
    variables = [col for col in df_plot.columns if col not in group_cols]
    
    # Group by participant and date (if needed, here our pivot table should already be daily)
    # In case there are multiple entries per date for a participant (unlikely after pivoting),
    # we aggregate them by mean.
    grouped = df_plot.groupby(["id_num", "date"])[variables].mean(numeric_only=True).reset_index()
    
    participants = grouped["id_num"].unique()

    # Create a subplot for each variable, sharing the same x-axis.
    fig = make_subplots(
        rows=len(variables), cols=1, shared_xaxes=True,
        vertical_spacing=0.02, subplot_titles=variables
    )

    # This list will help map each trace to its participant for the dropdown.
    visibility_map = []

    # Create one trace per participant per variable.
    for row_idx, var in enumerate(variables, 1):
        for pid in participants:
            pid_data = grouped[grouped["id_num"] == pid]
            fig.add_trace(
                go.Scatter(
                    x=pid_data["date"],
                    y=pid_data[var],
                    mode="lines+markers",
                    name=f"Participant {pid}",
                    legendgroup=str(pid),
                    visible=True if row_idx == 1 else False,  # only show all for first subplot initially
                    showlegend=(row_idx == 1)
                ),
                row=row_idx, col=1
            )
            visibility_map.append((row_idx, pid))

    # Create dropdown buttons to toggle traces by participant.
    buttons = []
    for pid in participants:
        # Build visibility list: each trace is visible only if its participant matches pid.
        visible = []
        for (row, p) in visibility_map:
            visible.append(p == pid)
        buttons.append(dict(
            label=f"Participant {pid}",
            method="update",
            args=[{"visible": visible},
                  {"title": f"Time Series - Participant {pid}"}]
        ))

    # Add a button to show all participants.
    buttons.insert(0, dict(
        label="Show All",
        method="update",
        args=[{"visible": [True] * len(visibility_map)},
              {"title": "Time Series - All Participants"}]
    ))

    fig.update_layout(
        height=300 * len(variables),
        title="Time Series - All Participants",
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 1.02,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top"
        }],
        hovermode="x unified"
    )

    # Optionally save as HTML.
    if save_html:
        outdir = "figures/plotly/all_participants"
        os.makedirs(outdir, exist_ok=True)
        fig.write_html(os.path.join(outdir, f"{title}.html"))

    # Show the figure if requested.
    if show_plot:
        fig.show()



def plotly_all_participants_histograms(df_plot, save_html=True, show_plot=True, title="all_participants_histograms"):
    """
    Plot histograms for all participants, one histogram per variable,
    with a dropdown toggle to show specific participants.
    
    The function expects a daily pivot table DataFrame with a column "day" (date)
    and "id_num" (participant identifier) along with other numeric variable columns.
    
    Parameters:
        df_plot (pd.DataFrame): DataFrame that is already a pivot.
        save_html (bool): Whether to save the figure as an HTML file.
        show_plot (bool): Whether to display the plot.
    """
    # Get the daily pivot table for all participants.
    # This function should return a DataFrame with columns "day", "id_num", and variable columns.
    # df_plot = create_daily_pivot(df, participant="all", return_dict=False)
    
    # Make sure "day" is a column; if it's not, throw an error.
    if "day" not in df_plot.columns:
        raise ValueError("Expected a 'day' column in the DataFrame.")
    
    # Convert "day" to a proper date format and also create a 'date' column (for clarity)
    df_plot["date"] = pd.to_datetime(df_plot["day"]).dt.date

    # Check that 'id_num' is present.
    if "id_num" not in df_plot.columns:
        raise ValueError("Expected a column 'id_num' to identify participants.")

    if "next_day" in df_plot.columns:
        print("PLOTLY TIMESERIES FUNCTION will delete column next day")
        df_plot = df_plot.drop("next_day", axis=1)
    
    # Identify variable columns by excluding group-by columns.
    group_cols = ["id_num", "day", "date"]
    variables = [col for col in df_plot.columns if col not in group_cols]
    
    # Group by participant and date (this should be daily data, but we aggregate if needed)
    grouped = df_plot.groupby(["id_num", "date"])[variables].mean(numeric_only=True).reset_index()
    
    participants = grouped["id_num"].unique()

    # Create a subplot for each variable.
    fig = make_subplots(
        rows=len(variables), cols=1, shared_xaxes=False,
        vertical_spacing=0.02, subplot_titles=variables
    )

    # Map each trace to its participant for the dropdown toggle.
    visibility_map = []

    # Create one histogram trace per participant per variable.
    for row_idx, var in enumerate(variables, 1):
        for pid in participants:
            pid_data = grouped[grouped["id_num"] == pid]
            fig.add_trace(
                go.Histogram(
                    x=pid_data[var],
                    name=f"Participant {pid}",
                    legendgroup=str(pid),
                    visible=True if row_idx == 1 else False,  # show all traces for the first subplot initially
                    showlegend=(row_idx == 1)
                ),
                row=row_idx, col=1
            )
            visibility_map.append((row_idx, pid))
    
    # Create dropdown buttons to toggle traces by participant.
    buttons = []
    for pid in participants:
        # Build visibility list: each trace is visible only if its participant matches pid.
        visible = []
        for (row, p) in visibility_map:
            visible.append(p == pid)
        buttons.append(dict(
            label=f"Participant {pid}",
            method="update",
            args=[{"visible": visible},
                  {"title": f"Histograms - Participant {pid}"}]
        ))
    
    # Add a button to show all participants.
    buttons.insert(0, dict(
        label="Show All",
        method="update",
        args=[{"visible": [True] * len(visibility_map)},
              {"title": "Histograms - All Participants"}]
    ))
    
    fig.update_layout(
        height=300 * len(variables),
        title="Histograms - All Participants",
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 1.02,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top"
        }],
        barmode="overlay",
        hovermode="x unified"
    )
    
    # Optionally save the figure as HTML.
    if save_html:
        outdir = "figures/plotly/all_participants"
        os.makedirs(outdir, exist_ok=True)
        fig.write_html(os.path.join(outdir, f"{title}.html"))
    
    # Show the figure if requested.
    if show_plot:
        fig.show()


def plot_original_vs_transformed(data, column_name):
    # transform the data
    transformed_data = data.copy()
    transformed_data[column_name] = np.log1p(data[column_name])

    # count nans for that feature
    nans_count = data[column_name].isna().sum()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(data[column_name], bins=30, kde=True, ax=ax[0])
    ax[0].set_title(f'Original {column_name}')
    sns.histplot(transformed_data[column_name], bins=30, kde=True, ax=ax[1])
    ax[1].set_title(f'Log Transformed {column_name}, with {nans_count} nans')
    plt.show()




############### LSTM FUNCTIONS ######


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
    
    

