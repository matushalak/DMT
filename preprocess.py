# @matushalak
import pandas as pd
import numpy as np
import os
from typing import Literal
import impute as IMP

# --------- Preprocess pipeline -----------
def preprocess_pipeline(filename: str = 'dataset_mood_smartphone.csv',
                        load_from_file: bool = False,
                        method: Literal['date', 'time_of_day', 'both'] = 'both',
                        remove_no_mood: bool = True) -> pd.DataFrame:
    '''
    Main preprocessing function to be used elsewhere
    '''
    ### 1) BASIC PREPROCESSING
    # add weekday, hour, month
    data: pd.DataFrame = preprocess_df(filename)

    # drop negative values for screentime variables
    data = find_negative_values_OG_FAST(data, drop = True)
    
    # daily / 3 times of day / both pivot and remove days without mood 
    # also saves the dataframe
    data = create_pivot(data, method=method, remove_no_mood= remove_no_mood, 
                        load_from_file=load_from_file)
    print('Basic preprocessing done!')

    ### 2) IMPUTATIONS
    # get categories of variables for different imputations
    imput_categories = IMP.categories(data.columns)
    
    # perform all imputations
    data = IMP.imputations(imput_categories, data)
    
    # add target and crop last day for each participant without target
    data = add_next_day_values(data, -3 if method != 'date' else -1)
    print('Imputations complete!')
    
    ### 3) FEATURE ENGINEERING
    # data = engineer_features(data)

    breakpoint()
    

    
    return data

# -------------- Preprocessing functions ------------------
def preprocess_df(filename: str = 'dataset_mood_smartphone.csv'
                  ) -> pd.DataFrame:
    '''
    Very first preprocessing function
    '''
    # load in data from csv
    data: pd.DataFrame = pd.read_csv(filename, index_col=0)
    data["id_num"] = data["id"].apply(lambda x: int(x.split(".")[1]))
    data["time"] = pd.to_datetime(data["time"])
    # extract useful information about date time
    data['date'] = data['time'].dt.date
    data['hour'] = data['time'].dt.hour
    times_of_day =  np.digitize(data['hour'], 
                                [10, 17]) # 0 night & morning / 1 work day / 2 evening
    data['time_of_day'] = times_of_day

    return data

# can be quite easily modified to add day or even measurement-level features now
def create_pivot(df: pd.DataFrame, 
                 method: Literal['date', 'time_of_day', 'both'] = 'both',
                 remove_no_mood: bool = True, 
                 load_from_file: bool = False) -> pd.DataFrame:
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
    # makes it faster
    if load_from_file:
        return pd.read_csv(f"tables/pivot_tables_daily/daily_pivot_table_{method}.csv")
    
    # for all participants
    pivot_list = []
    participants = df['id_num'].unique()

    # for wake and bed times, see 1 day as between 5AM and 5AM the next day
    df["time"] = pd.to_datetime(df["time"])
    df["shifted_time"] = df["time"] - pd.Timedelta(hours=5)
    df["shifted_date"] = df["shifted_time"].dt.date

    # Process each participant separately
    for part in participants:
        df_part = df[df["id_num"] == part].copy()

        # select the mean aggregation columns, time values should be aggregated by sum
        # on daily AND time of day level
        mean_agg = ["mood", "circumplex.valence", "circumplex.arousal"] # not sure about activity
        # only DAILY
        std_agg = ["mood", "circumplex.valence", "circumplex.arousal"] # variance 
        # only DAILY
        sum_agg = [col for col in df_part["variable"].unique() if col not in mean_agg + ["id_num", "time", "date", "month", "weekday"]]
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
        
        # get wakeup and sleep times
        # exclude variables which do not fluctuate throughout the day (activity)
        exclude = ['activity']
        df_TIMES = df_part.copy()
        df_TIMES.drop(df_TIMES[df_TIMES['variable'].isin(exclude)].index, inplace = True)
        daily_times = df_TIMES.groupby("shifted_date").agg(
                        bed_time_last_daily=("hour", get_bed_time),
                        wakeup_time_first_daily=("hour", get_wakeup_time)
                    ).reset_index()      
        
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

            # NOTE: For sum aggregations NaN = 0
            if aggregation == 'sum':
                pivot_part_daily = pivot_part_daily.fillna(0)
            
            # Make sure the daily pivot index is datetime and reset_index for merging:
            pivot_part_daily = pivot_part_daily.reindex(full_range)
            pivot_part_daily.index.name = "date"
            daily_reset = pivot_part_daily.reset_index()

            # only need to do this once
            if aggregation == 'last':
                # Merge in the daily meta features (bed_time and wakeup_time)
                daily_reset["date"] = pd.to_datetime(daily_reset["date"])
                daily_times["date"] = pd.to_datetime(daily_times["shifted_date"])
                daily_reset = daily_reset.merge(daily_times, on="date", how="left")
    
            if method == 'both':
                if aggregation in dailyOnly:
                    # Merge daily pivot values onto every row in our full grid.
                    # The daily pivot has one row per date; we want that repeated for each time_of_day.
                    pivot_agg = all_df.merge(daily_reset, on="date", how="left")
                    pivot_agg = pivot_agg.set_index(["date", "time_of_day"])

                else:
                    tod_pivot = df_agg.pivot_table(index=["date", "time_of_day"],
                                           columns="variable",
                                           values="value",
                                           aggfunc=aggregation)
                    
                    tod_pivot = tod_pivot.add_suffix(f'_{aggregation}_tod').reset_index()
                    tod_pivot['date'] = pd.to_datetime(tod_pivot['date'])
                    # Merge the time-of-day pivot with the full grid so that every (date, time_of_day) exists:
                    pivot_tod = all_df.merge(tod_pivot, on=['date', 'time_of_day'], how="left")
                    # Then merge in the daily pivot values (which will fill in for dates missing in the tod data).
                    pivot_agg = pivot_tod.merge(daily_reset, on="date", how="left")
                    pivot_agg = pivot_agg.set_index(["date", "time_of_day"])
            
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

    # rename columns
    combined.rename(columns={
        'wakeup_time_first_daily':'wake_time',
        'bed_time_last_daily': 'bed_time',
        'month_first_daily' : 'month',
        'weekday_first_daily' : 'weekday'}, inplace=True)
    
    # fill nans for month and weekday
    combined['date'] = pd.to_datetime(combined['date'])
    combined['month'] = combined['date'].dt.month
    combined['weekday'] = combined['date'].dt.weekday
    combined.insert(2, "month", combined.pop('month'))
    combined.insert(3, "weekday", combined.pop('weekday'))
    
    # save the combined dataframe to a csv file
    if not os.path.exists("tables/pivot_tables_daily"):
        os.makedirs("tables/pivot_tables_daily")
    combined.to_csv(f"tables/pivot_tables_daily/daily_pivot_table_{method}.csv", index=False)
    return combined


# works for any variable specified now
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


def find_negative_values_OG_FAST(df, drop=False):
    '''
    removes negative values
    '''
    exclude_variables = ["circumplex.valence", "circumplex.arousal"]
    # Create a boolean mask for rows with negative 'value' that are NOT in exclude_variables.
    mask = (df["value"] < 0) & (~df["variable"].isin(exclude_variables))
    if drop:
        df = df.loc[~mask]
        return df
    
    else: # for exploration
        negatives = df.loc[mask]
        for row in negatives.itertuples():
            print(f"Negative value found: {row.value} for variable {row.variable} at time {row.time}")
            print(f"Participant: {row.id_num}, Day: {row.time}")


def add_next_day_values(df, shift: int = -3):
    """
    modified from @urbansirca
    Add next day values into next_day_mood column
    """
    # create a new column with the next day mood
    df["next_day_mood"] = df.groupby("id_num")["mood_mean_daily"].shift(shift)
    # create a new column with the next day date
    df["next_day"] = df.groupby("id_num")["date"].shift(shift)

    # target right before mood_mean_daily position
    df.insert(5, "target", df.pop("next_day_mood"))
    df.insert(2, "next_date", df.pop("next_day"))
    # drop last day without target for each participant
    df = df.dropna(subset=['target', 'next_date'])
    return df

# ------------- Early feature engineering functions -----------÷÷
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


# Define functions for bedtime and wakeup time within specific windows.
def get_bed_time(h):
    # - times >= 19 are kept as is (e.g. 19,20,...,23)
    h_late = h[h >= 18]                  # late evening values (unchanged)
    # - times < 5 are adjusted by adding 24 (so 3 becomes 27)
    h_early = h[h < 5].copy()              # early morning values
    h_early_adjusted = h_early + 24       # shift these by 24
    # Combine both series
    combined = h_late._append(h_early_adjusted)
    if not combined.empty:
        max_val = combined.max()
        # Convert back: if max_val is 24 or more, subtract 24.
        if max_val >= 24:
            return max_val - 24
        else:
            return max_val
    else:
        return np.nan
    
def get_wakeup_time(h):
    # Consider only times in the wakeup window (e.g., 4:00 to 15:00)
    h = h[(h >= 5) & (h < 13)]
    return h.min() if not h.empty else np.nan


# ------------ utils ------------
# NOTE: feel free to change ordering
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
    base_order = ["mood", "screen", "activity", "circumplex.valence", "circumplex.arousal", "wakeup_time", "bed_time", "call", "sms", "appCat"]
    
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