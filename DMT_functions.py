import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

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

