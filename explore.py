import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from preprocess import preprocess_df, preprocess_pipeline
import os


def main():
    # Check distribution of observations
    explore_hours()
    
    # Check hourly trends for each variable
    explore_hourly_trends()

    # After excluding flatlining variables, explore wake and bed times 
    # (no change w times of day, consistently values in middle of the night)
    explore_sleep()


def explore_correlations():
    df = preprocess_pipeline(load_from_file=True, method='date')
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the figure
    f, ax = plt.subplots(figsize=(10, 10))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, vmax=.3, center=0, cmap='rocket',
                square=True, linewidths=2, cbar_kws={"shrink": .5},
                robust=True, ax=ax)

    # Get the number of columns from the correlation matrix
    n_cols = len(corr.columns)
    # Create new labels as simple numbers (0, 1, 2, ..., n-1)
    new_labels = list(map(str, range(n_cols)))
    
    # Set the tick locations to match exactly the number of columns.
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_cols))
    
    # Replace tick labels with the new numeric labels.
    ax.set_xticklabels(new_labels, rotation=45, ha="right", fontsize = 6)
    ax.set_yticklabels(new_labels, rotation=0, fontsize = 8)
    
    # Instead of legend patches, add a text description mapping numbers to column names.
    mapping_text = "\n".join([f"{i}: {col}" for i, col in enumerate(corr.columns)])
    # Place the mapping text at the bottom right of the figure
    ax.text(
        0.5, 0.99, mapping_text,
        transform=ax.transAxes,
        fontsize = 5,
        bbox=dict(facecolor='white', alpha=0.7, pad=5),
        ha='left', va='top'  # adjust alignment
    )
    
    plt.tight_layout()
    plt.show()

def explore_hourly_trends():
    df = preprocess_df()
    df_long = df[df['variable'].notna()]

    # Group by the variable name and the hour, counting the number of occurrences.
    counts = df_long.groupby(['variable', 'hour']).size().reset_index(name='count')

    # Now, for each unique variable, create a scatterplot.
    variables = counts['variable'].unique()

    # Create a subplot for each variable (one column per variable)
    fig, ax = plt.subplots(figsize=(8, 8))
    # Create distinct colors for each variable using a colormap.
    cmap = cm.get_cmap('tab10')
    cmapAPPS = cm.get_cmap('Pastel2')
    
    for iv, var in enumerate(variables):
        colors = cmap(-iv % 10) if 'app' not in var else cmapAPPS(iv % 8)
        subset = counts[counts['variable'] == var]

        if 'app' in var and subset['count'].max() < 400:
            ax.scatter(subset['hour'], subset['count'], label = var, alpha = 0.4, color=colors)
            ax.plot(subset['hour'], subset['count'], alpha = 0.1, color=colors)
        else:
            ax.scatter(subset['hour'], subset['count'], label = var, alpha = 0.9, color=colors)
            ax.plot(subset['hour'], subset['count'], alpha = 0.25, color=colors)
        ax.set_ylabel("Count")
        ax.set_xticks(range(0, 24))
        
    ax.set_title(f"Distribution of variable datapoints by Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    plt.legend(loc =2)
    plt.tight_layout()
    plt.show()


def explore_sleep():
    df = preprocess_pipeline(load_from_file=True)
    # only keep ID and participant
    df = df[['id_num', 'wake_time_daily', 'bed_time_daily']]
    plt.figure()
    plt.hist(df['bed_time_daily'], bins = 23, color = 'mediumpurple', label = 'bed time')
    plt.hist(df['wake_time_daily'], bins = 23, color = 'gold', label= 'wake time')
    
    plt.title('Distribution of wake and bedtimes')
    plt.xlim(0, 23)
    plt.xlabel('Hour')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()


def explore_hours():
    df = preprocess_df() # for hours
    # only keep ID and participant
    df = df[['id_num', 'hour']]
    plt.figure()
    plt.hist(df['hour'], bins = np.arange(0, 24), color = 'k')
   
    plt.title('Distribution of all datapoints by Hour')
    plt.xlim(0, 23)
    plt.xlabel('Hour')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Cross corrrelations of features
    explore_correlations()
    
    # main()
