# @matushalak
import pandas as pd
import numpy as np
import os
import re
from typing import Literal

# --------- Pipeline -----------
def preprocess_pipeline(filename: str = 'dataset_mood_smartphone.csv'):
    # add weekday, hour, month
    data: pd.DataFrame = preprocess_df(filename)
    
    # daily / 3 times of day / both pivot and remove days without mood
    data = create_pivot(data, method='both', remove_no_mood= True)
    breakpoint()

# -------------- Generally useful functions ------------------
def preprocess_df(filename: str
                  ) -> pd.DataFrame:
    '''
    Very first preprocessing function
    '''
    # load in data from csv
    data: pd.DataFrame = pd.read_csv(filename, index_col=0)
    data["id_num"] = data["id"].apply(lambda x: int(x.split(".")[1]))
    data["time"] = pd.to_datetime(data["time"])
    # extract useful information about date time
    data['month'] = data['time'].dt.month
    data['date'] = data['time'].dt.date
    data['weekday'] = data['time'].dt.weekday
    data['hour'] = data['time'].dt.hour
    times_of_day =  np.digitize(data['hour'], [7, 17]) # 0 night / 1 day / 2 evening
    data['time_of_day'] = times_of_day

    return data


def create_pivot(df: pd.DataFrame, method: Literal['date', 'time_of_day', 'both'] = 'both',
                 remove_no_mood: bool = True):
    """
    adapted from @urbansirca
    Create a pivot table with daily and/or time-of-day features.
    If method is "both", time-of-day features (that vary per time) are merged with daily
    features (that are constant for all times in a day).

    Parameters:
        df (pd.DataFrame): Expected to contain at least the columns 'id_num', 'time', 
                           'variable', and 'value'. The 'time' must be in datetime format.
        method (str): One of 'day', 'time_of_day', or 'both'.
    
    Returns:
        Combined pivot table (or list/dict per participant if further processing is desired).
    """
    # for all participants
    pivot_list = []
    participants = df['id_num'].unique()

    # Process each participant separately
    for part in participants:
        df_part = df[df["id_num"] == part].copy()

        # select the mean aggregation columns, time values should be aggregated by sum
        # on daily AND time of day level
        mean_agg = ["mood", "circumplex.valence", "circumplex.arousal"] # not sure about activity
        # only DAILY
        # TODO: sliding window!!!
        std_agg = ["mood", "circumplex.valence", "circumplex.arousal"] # variance 
        # only DAILY
        sum_agg = [col for col in df_part["variable"].unique() if col not in mean_agg + ["id_num", "time", "date"]]
        # also select min, max - on day level AND time of day level
        max_agg = ["activity", "circumplex.valence", "circumplex.arousal", "mood"]
        min_agg = ["circumplex.valence", "circumplex.arousal", "mood"]
        # first, last - only DAILY
        first_agg = ["mood", "circumplex.valence", "circumplex.arousal"]
        last_agg = ["mood", "circumplex.valence", "circumplex.arousal"]

        # Create a new DataFrame with the selected columns for each possible aggregation
        aggs = ['mean', 'std', 'std.slide', 'sum', 'max', 'min', 'first', 'last']
        var_lists = [mean_agg, std_agg, std_agg, sum_agg, max_agg, min_agg, first_agg, last_agg]
        dailyOnly = ('std', 'std.slide', 'sum', 'first', 'last')

        # pivots for different aggregations
        agg_pivots = []
        # Get the complete list of dates for this participant
        full_range = pd.date_range(start=df_part["date"].min(), end=df_part["date"].max(), freq="D")
        # Create a full multi-index grid of (date, time_of_day) combinations:
        full_index = pd.MultiIndex.from_product([full_range, [0, 1, 2]], names=['date', 'time_of_day'])
        # Also create a DataFrame version of the full grid
        all_df = pd.DataFrame(list(full_index), columns=['date','time_of_day'])
        
        for aggregation, var_combs in zip(aggs, var_lists):
            df_agg = df_part[df_part["variable"].isin(var_combs)].copy()
            # sliding window std
            if aggregation == 'std.slide':                
                # Use sliding_std_df as your daily pivot
                pivot_part_daily = sliding_std(full_range, aggregation, var_combs, df_agg)
            else:
                pivot_part_daily = df_agg.pivot_table(index= 'date', 
                                                        columns="variable", 
                                                        values="value", aggfunc= aggregation)
                
                pivot_part_daily = pivot_part_daily.add_suffix(f'_{aggregation}_daily')
    
            # Make sure the daily pivot index is datetime and reset_index for merging:
            pivot_part_daily = pivot_part_daily.reindex(full_range)
            pivot_part_daily.index.name = "date"
            daily_reset = pivot_part_daily.reset_index()
    
            if method == 'both':
                if aggregation in dailyOnly:
                    # Reindex the daily pivot so that each day is represented by 3 time_of_day rows.
                    full_index = pd.MultiIndex.from_product([full_range, [0, 1, 2]],
                                                            names=['date', 'time_of_day'])
                    pivot_agg = pivot_part_daily.reindex(full_index)

                else:
                    # Otherwise, compute the time-of-day pivot normally.
                    tod_pivot = df_agg.pivot_table(index=["date", "time_of_day"],
                                                columns="variable",
                                                values="value",
                                                aggfunc=aggregation)
                    tod_pivot = tod_pivot.add_suffix(f'_{aggregation}_tod')
                    
                    # Merge the daily pivot onto each time-of-day row (after resetting index)
                    tod_pivot = tod_pivot.reset_index().merge(
                        pivot_part_daily.reset_index(), on="date", how="left"
                    ).set_index(["date", "time_of_day"])
                    
                    full_index = pd.MultiIndex.from_product([full_range, [0, 1, 2]],
                                                            names=['date', 'time_of_day'])
                    tod_pivot = tod_pivot.reindex(full_index)
                    pivot_agg = tod_pivot
            
            elif method == 'date':
                pivot_agg = pivot_part_daily
                # Create a complete date range from the earliest to the latest day for this participant
                pivot_agg = pivot_agg.reindex(full_range)
                pivot_agg.index.name = "date"
            
            elif method == 'time_of_day':
                pivot_agg = df_agg.pivot_table(
                                    index=["date", "time_of_day"],
                                    columns="variable",
                                    values="value",
                                    aggfunc=aggregation)
                # time of day
                pivot_agg = pivot_agg.add_suffix(f'{aggregation}')
                # Build a complete multi-index: each date paired with time_of_day 0, 1, and 2
                full_index = pd.MultiIndex.from_product([full_range, [0, 1, 2]], names=['date', 'time_of_day'])
                # Reindex the pivot table so that missing (date, time_of_day) combinations become NaNs
                pivot_agg = pivot_agg.reindex(full_index)

            agg_pivots.append(pivot_agg)

        # Combine the pivot tables
        pivot = pd.concat(agg_pivots, axis=1)
        # Pandas aligns the data based on the index (which here represents the days | time_of_day).
        # The union of the indices is taken, and any missing data for a particular day in one of the tables is filled with NaN. 
        # The same applies when you concatenate the count pivot table. 
        # This means that even if the individual pivot tables have different numbers of rows, the concatenation will produce a DataFrame covering all days from the union of the indices.
        
        # Convert index to a column and add participant id
        pivot = pivot.reset_index()
        pivot["id_num"] = part

        # sort columns
        sorted_cols = sort_pivot_columns(pivot.columns)
        pivot = pivot[sorted_cols]
        pivot_list.append(pivot)
    
    # Concatenate the list of dataframes without setting a multi-index so that
    # both 'day' and 'id_num' remain as regular columns
    combined = pd.concat(pivot_list, ignore_index=True)
    # sort by id_num, (time of day) and day
    sortby = ["id_num", "date"] if method == "date" else ["id_num", "date", "time_of_day"]
    combined = combined.sort_values(by=sortby)

    # remove data without mood
    if remove_no_mood:
        combined = remove_dates_without_VAR(combined, VAR = 'mood_mean_daily')

    # save the combined dataframe to a csv file
    if not os.path.exists("tables/pivot_tables_daily"):
        os.makedirs("tables/pivot_tables_daily")
    combined.to_csv(f"tables/pivot_tables_daily/daily_pivot_table_{method}.csv", index=False)
    return combined


def remove_dates_without_VAR(df: pd.DataFrame, VAR: str, start_date=None, end_date=None):
    """
    @urbansirca
    Remove dates from the dataframe. If start and end date aren't provided, 
    it will take the first and the last non-NaN mood value as the cutoffs.
    """
    participant_ids = df["id_num"].unique().tolist()

    for specific_id in participant_ids:
        # df for an individual with mood values present
        mood_not_nan = df[(df['id_num'] == specific_id) & (df[VAR].notna())]

        # Compute local start and end dates for this participant
        local_start_date = pd.to_datetime(start_date) if start_date is not None else mood_not_nan['date'].min()
        local_end_date = pd.to_datetime(end_date) if end_date is not None else mood_not_nan["date"].max()
    
        # boolean mask: either not the specific id OR rows between local_start_date and local_end_date
        mask = (df['id_num'] != specific_id) | (
            (df['id_num'] == specific_id) & 
            (df['date'] >= local_start_date) & 
            (df['date'] <= local_end_date)
        )
        df = df[mask]
    
    return df


def sliding_std(full_range, aggregation, var_combs, df_agg) -> pd.DataFrame:
    # For a sliding window std, create an empty DataFrame indexed by the full date range.
    sliding_std_df = pd.DataFrame(index=full_range)
    
    # Compute the sliding-window standard deviation on the raw values for each variable.
    for var in var_combs:
        # Filter raw data for the current variable.
        var_data = df_agg[df_agg['variable'] == var].copy()
        # Ensure that the 'date' column is of datetime type.
        var_data['date'] = pd.to_datetime(var_data['date'])
        
        # Prepare a list to hold the computed std for each date in the window.
        std_values = []
        # needs to be like this because each day can have different number of measurements
        for current_date in full_range:
            # Define window: from current_date - 2 days to current_date.
            start_date = current_date - pd.Timedelta(days=2)
            # Select raw values in this time window.
            window_values = var_data.loc[(var_data['date'] >= start_date) & (var_data['date'] <= current_date), 'value']
            # Compute the standard deviation (will be NaN if there are no measurements).
            # TODO: impute if no measurements?
            std_val = window_values.std()
            std_values.append(std_val)
        
        # Save the computed sliding standard deviation for this variable.
        # The column name will be like "mood_std_daily" (or similar).
        col_name = f"{var}_{aggregation}_daily"
        sliding_std_df[col_name] = std_values
    
    return sliding_std_df
    


def sort_pivot_columns(cols):
    """
    Reorders a list of pivot table columns so that:
      - 'id_num', 'date', 'time_of_day' come first.
      - Then all _daily columns appear in a custom order based on a defined base variable order
        and a custom order of aggregations (mean first unless the variable is in the sum group).
      - Then the _tod columns follow, ordered similarly.
    
    Parameters:
        cols (List[str]): List of column names.
    
    Returns:
        List[str]: The sorted list of columns.
    """
    # Fixed columns always come first.
    fixed = ["id_num", "date", "time_of_day"] if "time_of_day" in cols else ["id_num", "date"]
    
    # Separate daily and tod columns.
    daily_cols = [c for c in cols if c.endswith("_daily")]
    if "time_of_day" in cols:
        tod_cols = [c for c in cols if c.endswith("_tod")]
    
    # Define your desired base variable order. For apps, we group any variable
    # that starts with "appCat" into the "appCat" group.
    base_order = ["mood", "screen", "activity", "circumplex.valence", "circumplex.arousal", "call", "sms", "appCat"]
    
    # Define aggregation orders.
    non_sum_order = ["mean", "std", "std.slide", "max", "min", "first", "last"]
    sum_order = ["sum", "max", "min", "first", "last"]
    
    # Define which base variables are to be considered sum variables.
    sum_vars = ["activity", "call", "screen", "sms"]
    
    def get_sort_key(col, suffix):
        """
        Given a column name ending with suffix (e.g. "_daily" or "_tod"),
        extract the base and the aggregation, and return a tuple used for sorting.
        """
        # Remove the suffix (e.g. "_daily").
        base_agg = col[:-len(suffix)]
        # Split from the right to separate the aggregation from the base.
        try:
            base, agg = base_agg.rsplit('_', 1)
        except ValueError:
            base = base_agg
            agg = ""
        # For any variable that starts with "appCat", assign its group as "appCat".
        if base.startswith("appCat"):
            base_group = "appCat"
        else:
            base_group = base
        
        # Determine the order of the base variable.
        try:
            base_index = base_order.index(base_group)
        except ValueError:
            base_index = len(base_order)
        
        # Determine which aggregation order to use.
        if base in sum_vars or base_group == "appCat":
            try:
                agg_index = sum_order.index(agg)
            except ValueError:
                agg_index = len(sum_order)
        else:
            try:
                agg_index = non_sum_order.index(agg)
            except ValueError:
                agg_index = len(non_sum_order)
        # Return a tuple sort key: first sort by the base order, then by aggregation order,
        # then lexically by the base and agg (as tie-breakers), then the full column name.
        return (base_index, base, agg_index, agg, col)
    
    sorted_daily = sorted(daily_cols, key=lambda c: get_sort_key(c, "_daily"))
    
    if "time_of_day" in cols:
        sorted_tod = sorted(tod_cols, key=lambda c: get_sort_key(c, "_tod"))
    else:
        sorted_tod = []
    
    # Return the final sorted order.
    return fixed + sorted_daily + sorted_tod


if __name__ == '__main__':
    preprocess_pipeline()