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


def plotly_all_participants_correlations(df, save_html=True, show_plot=True, title="all_participants_correlations"):
    """
    Create an interactive correlation analysis figure.
    
    For each participant option (either all participants combined or individual participants)
    and for each scatter plot variable pair (chosen from the numeric variables), this function creates:
    
    1. A heatmap correlation matrix (top row) computed on all numeric variables available for that participant.
    2. A scatter plot (bottom row) for the selected variable pair.
    
    A single combined update menu (dropdown) allows the user to select the (participant, variable pair)
    combination to display.
    
    Parameters:
      df (pd.DataFrame): DataFrame that will be used to create pivot dictionary.
                         Expected to contain at least 'id_num', 'day', and numeric variable columns.
      save_html (bool): Whether to save the figure as an HTML file.
      show_plot (bool): Whether to display the plot.
      
    Notes:
      - This function handles lots of NaNs by relying on pandas’ .corr() (which computes pairwise correlations).
      - It creates separate traces for each combination, and uses an update menu to toggle visibility.
    """
    # Create the daily pivot table for all participants.
    # Use return_dict=True to get separate DataFrames for each participant.
    pivot_dict = create_daily_pivot(df, participant="all", return_dict=True, counts=False)
    
    # Also create an "all" option by concatenating all participants' data.
    df_all = pd.concat(pivot_dict.values(), ignore_index=True)
    pivot_dict["all"] = df_all
    
    # Identify participant options. They will be the keys of pivot_dict.
    participant_options = list(pivot_dict.keys())
    
    # Assume the numeric variables are those not in the grouping columns:
    group_cols = ["id_num", "day"]
    sample_df = pivot_dict[participant_options[0]]
    all_vars = list(sample_df.columns)
    numeric_vars = [col for col in all_vars if col not in group_cols and col != "date"]
    
    # Optionally force a specific order for some variables.
    desired_order = ["mood", "screen", "activity", "circumplex.valence", "circumplex.arousal"]
    ordered_vars = [v for v in desired_order if v in numeric_vars] + [v for v in numeric_vars if v not in desired_order]
    numeric_vars = ordered_vars

    # Create a list of scatter plot pairs. Here we take all unordered pairs.
    scatter_pairs = list(itertools.combinations(numeric_vars, 2))
    if not scatter_pairs:
        raise ValueError("Not enough numeric variables to form scatter plot pairs.")

    # Build a figure with 2 rows: row 1 for the heatmap, row 2 for the scatter plot.
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.1,
        subplot_titles=("Correlation Matrix", "Scatter Plot")
    )
    
    # This list will be used to later update trace visibility.
    trace_visibility_defaults = []
    # Map (participant, scatter_pair index) to a "visible" vector.
    visibility_dict = {}
    
    # --- Create heatmap traces for each participant option.
    heatmap_traces = []
    for p_idx, p in enumerate(participant_options):
        df_p = pivot_dict[p]
        # Use only the numeric variables that are actually in df_p.
        available_vars = [col for col in numeric_vars if col in df_p.columns]
        if available_vars:
            df_corr = df_p[available_vars].corr()
            heat_trace = go.Heatmap(
                z = df_corr.values,
                x = df_corr.columns.tolist(),
                y = df_corr.index.tolist(),
                colorbar=dict(title="r"),
                visible=False  # update later
            )
        else:
            # If no variables exist for this participant, create an empty trace.
            heat_trace = go.Heatmap(z=[], x=[], y=[], visible=False)
        heatmap_traces.append(heat_trace)
        fig.add_trace(heat_trace, row=1, col=1)
        trace_visibility_defaults.append(False)
    
    # --- Create scatter plot traces for each participant and each scatter pair.
    scatter_traces = []
    for p_idx, p in enumerate(participant_options):
        df_p = pivot_dict[p]
        for sp_idx, (var_x, var_y) in enumerate(scatter_pairs):
            # Only use data if both variables exist for the participant.
            if var_x in df_p.columns and var_y in df_p.columns:
                scatter_trace = go.Scatter(
                    x = df_p[var_x],
                    y = df_p[var_y],
                    mode = "markers",
                    marker = dict(size=8, opacity=0.7),
                    name = f"{p} - {var_x} vs {var_y}",
                    visible = False  # update later
                )
            else:
                # Create an empty trace if one or both variables are missing.
                scatter_trace = go.Scatter(
                    x = [],
                    y = [],
                    mode = "markers",
                    marker = dict(size=8, opacity=0.7),
                    name = f"{p} - {var_x} vs {var_y}",
                    visible = False
                )
            scatter_traces.append(scatter_trace)
            fig.add_trace(scatter_trace, row=2, col=1)
            trace_visibility_defaults.append(False)
    
    total_traces = len(heatmap_traces) + len(scatter_traces)
    
    # Build mapping from (participant, scatter_pair index) to a full "visible" vector.
    for p_idx, p in enumerate(participant_options):
        for sp_idx in range(len(scatter_pairs)):
            # Create a boolean list (one per trace) initialized to False.
            visible = [False] * total_traces
            # For the heatmap: only the trace for participant p should be visible.
            visible[p_idx] = True
            # For scatter traces, they are arranged in blocks per participant.
            scatter_trace_index = len(heatmap_traces) + p_idx * len(scatter_pairs) + sp_idx
            visible[scatter_trace_index] = True
            visibility_dict[(p, sp_idx)] = visible

    # Set the default selection: use participant "all" and the first scatter pair.
    default_key = ("all", 0)
    default_visible = visibility_dict[default_key]
    for i, vis in enumerate(default_visible):
        fig.data[i].visible = vis

    # --- Create an update menu with one button per (participant, scatter pair) combination.
    menu_buttons = []
    for p in participant_options:
        for sp_idx, (var_x, var_y) in enumerate(scatter_pairs):
            label = f"{'All' if p=='all' else 'Participant '+str(p)}: {var_x} vs {var_y}"
            visible = visibility_dict[(p, sp_idx)]
            button = dict(
                label = label,
                method = "update",
                args = [
                    {"visible": visible},
                    {"title": f"Correlation Analysis - {'All' if p=='all' else 'Participant '+str(p)}: {var_x} vs {var_y}"}
                ]
            )
            menu_buttons.append(button)
    
    # Update the layout with the dropdown menu.
    fig.update_layout(
        updatemenus=[{
            "buttons": menu_buttons,
            "direction": "down",
            "showactive": True,
            "x": 1.05,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top"
        }],
        height=800,
        title="Correlation Analysis"
    )
    
    # Optionally save as HTML.
    if save_html:
        outdir = "figures/plotly/correlations"
        os.makedirs(outdir, exist_ok=True)
        fig.write_html(os.path.join(outdir, f"{title}.html"))
    
    if show_plot:
        fig.show()


def plot_original_vs_transformed(data, column_name):
    """
    plot the original and transformed data
    data: pd.DataFrame
    column_name: str, name of the column to transform
    """
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


def find_negative_values_OG_df(df, drop=False):
    """
    Check for negative values in the DataFrame.
    parameters:
    df: DataFrame to check
    drop: if True, drop the rows with negative values
    returns:
    df: DataFrame with negative values dropped if drop=True
    """

    exlude_variables = ["circumplex.valence", "circumplex.arousal"]

    for index, row in df.iterrows():
        if row["value"] < 0 and row["variable"] not in exlude_variables:
            # print(row)
            print(f"Negative value found: {row['value']} for variable {row['variable']} at time {row['time']}")
            print(f"Participant: {row['id_num']}, Day: {row['time']}")
            if drop:
                print("Dropping row...")
                df = df.drop(index=index)
    return df


# add next day values into next_day_mood column
def add_next_day_values(df):
    """
    Add next day values into next_day_mood column
    """
    # create a new column with the next day mood
    df["next_day_mood"] = df.groupby("id_num")["mood"].shift(-1)
    # create a new column with the next day date
    df["next_day"] = df.groupby("id_num")["day"].shift(-1)

    # get day column moved to the 3rd last position
    cols = df.columns.tolist()
    cols.insert(-2, cols.pop(cols.index("day")))
    cols.insert(-1, cols.pop(cols.index("next_day")))
    df = df[cols]
    return df


def correlate_with_next_day_mood(df):
    """
    Correlate all variables with next_day_mood
    """
    # get the columns to correlate
    cols = df.columns.tolist()
    cols.remove("id_num")
    cols.remove("day")
    cols.remove("next_day")
    cols.remove("next_day_mood")
    
    # create a new dataframe with the correlations
    df_corr = pd.DataFrame(columns=["variable", "correlation"])
    
    corrs = []
    for col in cols:
        corr = df[col].corr(df["next_day_mood"])
        corrs.append(corr)
    
    df_corr = pd.DataFrame({"variable": cols, "correlation": corrs})
    
    # sort the dataframe by correlation
    df_corr = df_corr.sort_values(by=["correlation"], ascending=False)

    df_corr.reset_index(drop=True, inplace=True)
    
    return df_corr



def remove_dates_without_mood(df, participant=None, start_date=None, end_date=None):
    """
    Remove dates from the dataframe. If start and end date aren't provided, 
    it will take the first and the last non-NaN mood value as the cutoffs.
    """
    
    # if no participant provided, then do it for all participants
    if participant is None:
        participant_ids = df["id_num"].unique().tolist()
    else:
        participant_ids = [participant]

    for specific_id in participant_ids:
        # df for an individual with mood values present
        mood_not_nan = df[(df['id_num'] == specific_id) & (df['mood'].notna())]

        # Compute local start and end dates for this participant
        local_start_date = pd.to_datetime(start_date) if start_date is not None else mood_not_nan['day'].min()
        local_end_date = pd.to_datetime(end_date) if end_date is not None else mood_not_nan["day"].max()
    
        # boolean mask: either not the specific id OR rows between local_start_date and local_end_date
        mask = (df['id_num'] != specific_id) | (
            (df['id_num'] == specific_id) & 
            (df['day'] >= local_start_date) & 
            (df['day'] <= local_end_date)
        )
        df = df[mask]
    
    return df


# plot histograms with sns in subplots for all columns
def plot_histograms(df, columns):
    """
    Plot histograms for selected columns in the dataframe
    """

    cols_with_nans = df[columns].isna().sum()
    cols_with_nans = cols_with_nans[cols_with_nans > 0].index.tolist()
    n = len(columns)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols_with_nans):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(col)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


# impute everything ending with min or max with mode
def impute_mode_groupped(df, columns):
    """
    Impute all columns ending with min or max with mode
    example usage: df = impute_mean_grouped(df, df.select_dtypes(include=[np.number]).columns.tolist())


    """
    for col in columns:
        df[col] = df.groupby(['id_num'])[col].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
    return df


def impute_mean_grouped(df, columns):
    """
    Impute with mean
    example usage: df = impute_mean_grouped(df, df.columns[df.columns.str.endswith('min') | df.columns.str.endswith('max')])
    """
    for col in columns:
        df[col] = df.groupby('id_num')[col].transform(lambda x: x.fillna(x.mean()))
    return df
