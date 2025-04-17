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


def explore_correlations(method='date'):
    df = preprocess_pipeline(load_from_file=True, method=method)
    df = df.drop(columns=["id_num", "date", "next_date"], errors="ignore")
    corr = df.corr()
    # 3) Rank features by |corr| with 'target'
    #    (Assumes 'target' is one of the remaining columns.)
    ranking = corr["target"].abs().sort_values(ascending=False)
    sorted_cols = ranking.index.tolist()

    # 4) Reorder the correlation matrix
    corr = corr.loc[sorted_cols, sorted_cols]

    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create a figure manually, specifying where the main Axes (for the heatmap)
    # and the colorbar Axes should go.
    fig = plt.figure(figsize=(11, 9))

    # main_axes: the main area for the heatmap
    # [left, bottom, width, height] in figure fraction (0..1)
    main_axes = fig.add_axes([0.05, 0.1, 0.8, 0.85])

    # cbar_axes: a smaller area for the color bar, at the top, oriented horizontally
    cbar_axes = fig.add_axes([0.2, 0.92, 0.5, 0.05])

    # Draw the heatmap on main_axes, telling Seaborn to put the colorbar on cbar_axes.
    hm = sns.heatmap(
        corr, mask=mask, center=0, cmap='rocket',
        square=True, linewidths=1, ax=main_axes,robust=True,
        cbar_ax=cbar_axes, cbar_kws={"orientation": "horizontal", 'shrink':.5}
    )

    # 3) label the colorbar
    cbar = hm.collections[0].colorbar
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label("Correlation", fontsize=10, labelpad=5)
    cbar.ax.tick_params(axis='x', labelsize=6)
    
    # Tweak main_axes ticks if desired (just an example)
    n_cols = len(corr.columns)
    main_axes.set_xticks(np.arange(n_cols))
    main_axes.set_yticks(np.arange(n_cols))
    new_labels = list(map(str, range(n_cols)))
    main_axes.set_xticklabels(new_labels, rotation=60, ha="right", fontsize = 6 if method == 'date' else 5)
    main_axes.set_yticklabels(new_labels, rotation=0, fontsize = 6 if method == 'date' else 5)

    # Example: place mapping text on the right side of the figure
    mapping_text = "\n".join([f"{i}: {col}" for i, col in enumerate(sorted_cols)])
    # fig.text(x, y, text, ...) places text in figure coordinates
    fig.text(
        0.8, 0.525, mapping_text,  # shift x,y as needed
        va='center', ha='left',fontsize = 6 if method == 'date' else 4.5,
        bbox=dict(facecolor='white')
    )
    # 6) Tighten everything up
    fig.subplots_adjust(
        left=0.02, right=0.95,
        top=0.98, bottom=0.05,
        wspace=0.02, hspace=0.02
    )
    plt.savefig(f'correlations_{method}.png', dpi = 400)
    plt.show()

def pairplot_by_substrings(substrings, method='both'):
    """
    For each string in `substrings`, find the feature whose name contains that string
    and has the largest |corr(target, feature)|, then draw a pairplot of those features
    (plus target), with hue='id_num'.

    substrings: List[str] e.g. ['mood', 'arousal', 'circumplex']
    """
    from matplotlib.ticker import MaxNLocator, ScalarFormatter, FormatStrFormatter
    # 1) Preprocess once, keep id_num around
    full = preprocess_pipeline(load_from_file=True, method=method)

    # 2) For ranking, drop the metadata columns
    corr_df = full.drop(columns=["id_num", "date", "next_date", "categorical_target"], errors="ignore")
    corr = corr_df.corr()

    # 3) Rank by absolute correlation with 'target'
    ranking = corr["target"].abs().sort_values(ascending=False)
    ranking = ranking.drop("target", errors="ignore")
    
    # 4) For each entry in substrings, pick the best‐correlated feature
    selected = []
    for entry in substrings:
        if isinstance(entry, (list, tuple)):
            # match any of the substrings in the tuple
            matches = [feat for feat in ranking.index 
                       if any(sub in feat for sub in entry)]
        else:
            # match the single substring
            matches = [feat for feat in ranking.index if entry in feat]

        if not matches:
            print(f"⚠️  no features contain {entry!r}")
        else:
            selected.append(matches[0])  # first is highest‐corr

    if not selected:
        raise ValueError("No features found for any of the provided substrings!")

    # breakpoint()
    # 4) Prepare the plotting DF with id_num + those top20 features
    plot_df = full[["id_num", "target"] + selected].copy()
    # cast id_num to string so seaborn treats it as categorical
    plot_df["id_num"] = plot_df["id_num"].astype(str)

    # 5) Draw the pairplot
    pp = sns.pairplot(
        plot_df,
        vars=["target"] + selected,
        hue="id_num",
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.4, "s": 15},
        diag_kws={"alpha": 0.6},
        height=2.0,      # each facet will be 2×2 inches
        aspect=1.0       # square facets
    )
    pp.figure.set_size_inches(12, 10)
    # 1) Force integer ticks everywhere
    for ax in pp.axes.flatten():
        if ax is None:
            continue
        # 1) Pick a locator that allows floats (here up to 5 ticks)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

        # 2) Format tick labels to one decimal place
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='x', labelsize=6, rotation=45)
        ax.tick_params(axis='y', labelsize=6)
        # y axis scientific notation
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(5)
        # x axis scientific notation
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(4)

    for ax in pp.axes[:, 0]:  # left column: y‑labels
        if ax is None:
            continue
        ax.set_ylabel(ax.get_ylabel(), fontsize=5.5)
    for ax in pp.axes[-1, :]:  # bottom row: x‑labels
        if ax is None:
            continue
        ax.set_xlabel(ax.get_xlabel(), fontsize=6)
    # 6) Tidy up
    # pp.figure.suptitle(f"Pairplot of Top‑{N} Features by |corr with target| ({method})", fontsize=16)
    # plt.subplots_adjust(top=0.93)  # make room for the suptitle
    plt.savefig('features_pairplot', dpi = 300)
    plt.show()

def fluctuations():
    df = preprocess_df()
    df = df[df['variable'].notna()]
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday

    plot_3x3_individual_vs_overall(df)
    # # mood
    # plot_individual_and_overall(df, 'time_of_day',    measure='mood')
    # plot_individual_and_overall(df, 'weekday', measure='mood')
    # plot_individual_and_overall(df, 'month',   measure='mood')
    # # arousal
    # plot_individual_and_overall(df, 'time_of_day',    measure='circumplex.arousal')
    # plot_individual_and_overall(df, 'weekday', measure='circumplex.arousal')
    # plot_individual_and_overall(df, 'month',   measure='circumplex.arousal')
    # # valence
    # plot_individual_and_overall(df, 'time_of_day',    measure='circumplex.valence')
    # plot_individual_and_overall(df, 'weekday', measure='circumplex.valence')
    # plot_individual_and_overall(df, 'month',   measure='circumplex.valence')

def plot_individual_and_overall(df, group_var, measure="mood"):
    """
    group_var: one of 'hour', 'month', or 'weekday'
    value_var: the column you want to average (here 'mood')
    """
    # 1) Filter to your measure
    sub = df[df['variable'] == measure]
    
    # 2) Compute the overall mean trace
    overall = sub.groupby(group_var)['value'].mean()
    
    # 3) Prepare the figure & axes
    fig, ax = plt.subplots(figsize=(6,6))
    
    # 4) Build a color map with one distinct color per participant
    participants = sub['id_num'].unique()
    N = len(participants)
    cmap = plt.get_cmap('tab20', N)  # up to 20 distinct colors; for more use e.g. 'hsv'
    
    # 5) Plot each participant
    for idx, pid in enumerate(participants):
        g = sub[sub['id_num'] == pid]
        # compute their mean per bin, then reindex to align with overall.index
        ind = g.groupby(group_var)['value'].mean().reindex(overall.index)
        color = cmap(idx)
        ax.scatter(overall.index, ind.values,
                   color=color, s=20, alpha=0.6)
        ax.plot(overall.index, ind.values,
                color=color, linestyle='--', linewidth=1, alpha=0.6)
    
    # 6) Overlay the overall mean
    ax.plot(overall.index, overall.values,
            color='k', linewidth=2, label='Overall mean')
    
    # 7) Styling
    ax.set_xlabel(group_var.capitalize())
    ax.set_ylabel(f"Mean {measure}")
    ax.set_title(f"{measure.capitalize()} by {group_var}")
    ax.set_xticks(overall.index)
    ax.legend(loc='upper right')
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig('mood_exploration_{group_var}')
    plt.show()

def plot_3x3_individual_vs_overall(df):
    measures  = ['mood', 'circumplex.arousal', 'circumplex.valence']
    group_vars = ['time_of_day', 'weekday', 'month']

    # Create a 3×3 grid of subplots, sharing Y within each row
    fig, axes = plt.subplots(
        nrows = 3, ncols = 3,
        figsize = (15, 15),
        sharey = 'row',
        # sharex='col'
    )

    for i, measure in enumerate(measures):
        # Filter once per measure
        sub = df[df['variable'] == measure]
        participants = sub['id_num'].unique()
        cmap = plt.get_cmap('tab20', len(participants))

        for j, group_var in enumerate(group_vars):
            ax = axes[i, j]

            # 1) overall mean
            overall = sub.groupby(group_var)['value'].mean()

            # 2) each participant
            for k, pid in enumerate(participants):
                g = sub[sub['id_num'] == pid]
                ind = (
                    g.groupby(group_var)['value']
                     .mean()
                     .reindex(overall.index)
                )
                color = cmap(k)
                ax.scatter(
                    overall.index, ind.values,
                    color=color, s=15, alpha=0.5
                )
                ax.plot(
                    overall.index, ind.values,
                    color=color, linestyle='--', linewidth=1, alpha=0.5
                )

            # 3) overall mean line
            ax.plot(
                overall.index, overall.values,
                color='k', linewidth=2, label='Overall mean'
            )

            # 4) styling
            ax.set_title(f"{measure} by {group_var}", fontsize=10)
            ax.set_xticks(overall.index)
            ax.set_xticklabels(overall.index, fontsize=8)
            ax.legend(loc='upper right', fontsize=8)
            if j == 0:
                ax.set_ylabel(f"Mean {measure}", fontsize=9)
            else:
                ax.set_ylabel("")
            if i == 2:
                ax.set_xlabel(group_var, fontsize=9)
            else:
                ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig('mood_arousal_valence_exploration.png', dpi = 300)
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
    plt.savefig('hourly_counts.png', dpi = 300)
    plt.show()


def explore_sleep():
    df = preprocess_pipeline(load_from_file=True)
    # only keep ID and participant
    df = df[['id_num', 'wake_time_daily', 'bed_time_daily']]
    plt.figure(figsize=(8,8))
    bedtimes = df['bed_time_daily'].copy()
    bedtimes.loc[bedtimes > 23] -= 24
    plt.hist(bedtimes, bins = np.arange(-0.5,24,1), color = 'mediumpurple', label = 'bed time')
    plt.hist(df['wake_time_daily'], bins = np.arange(-0.5,24,1), color = 'gold', label= 'wake time')
    
    plt.title('Distribution of wake and bedtimes')
    plt.xlim(0, 23.5)
    plt.xlabel('Hour')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('wake_and_bed_times.png', dpi = 200)
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
    explore_sleep()
    # pairplot_by_substrings(substrings=['mood', 'circumplex', 
    #                                    'activity', 'screen', 
    #                                    ('wake', 'bed', 'sleep'), 
    #                                    'change'])
    # fluctuations()
    
    # main()
    
    # Cross corrrelations of features
    # explore_correlations('date')
    # explore_correlations('both')
    
