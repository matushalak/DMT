import pandas as pd
import numpy as np
import os
from typing import Literal

# --------- Pipeline -----------
def preprocess_pipeline(filename: str = 'dataset_mood_smartphone.csv'):
    # add weekday, hour, month
    data: pd.DataFrame = preprocess_df(filename)
    breakpoint()
    # 3 times of day pivot
    data = create_pivot(data)

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
    data['month'] = data['time'].apply(lambda x: x.month)
    data['date'] = data['time'].apply(lambda x: x.date())
    data['weekday'] = data['time'].apply(lambda x: x.weekday())
    data['hour'] = data['time'].apply(lambda x: x.hour)
    times_of_day =  np.digitize(data['hour'], [6, 12, 18])
    times_of_day[np.where(times_of_day == 0 | times_of_day == 3)] = 0 # night if between 18 - 06
    data['time_of_day'] = times_of_day

    return data


def create_pivot(df: pd.DataFrame, method: Literal['day', 'time_of_day', 'both'] = 'both'):
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
    # for all participants
    pivot_list = []
    participants = df['id_num'].unique()

    # Process each participant separately
    for part in participants:
        df_part = df[df["id_num"] == part].copy()

        # select the mean aggregation columns, time values should be aggregated by sum
        # on daily AND time of day level
        mean_agg = ["mood", "circumplex.valence", "circumplex.arousal"] # not sure about activity
        # rolling window on nans (same as moving average), also DAILY
        std_agg = ["mood", "circumplex.valence", "circumplex.arousal"] # variance 
        # both daily AND time of day level
        sum_agg = [col for col in df_part["variable"].unique() if col not in mean_agg + ["id_num", "time", "day"]]
        # also select min, max - on day level AND time of day level
        max_agg = ["activity", "circumplex.valence", "circumplex.arousal", "mood"]
        min_agg = ["circumplex.valence", "circumplex.arousal", "mood"]
        # first, last - on day level
        first_agg = ["mood", "circumplex.valence", "circumplex.arousal"]
        last_agg = ["mood", "circumplex.valence", "circumplex.arousal"]
        # difference - on day AND time of day level
        diff_agg = ["mood", "circumplex.valence", "circumplex.arousal"]

        # Create a new DataFrame with the selected columns for each possible aggregation
        aggs = ['mean', 'std', 'sum', 'max', 'min', 'first', 'last']
        var_lists = [mean_agg, std_agg, sum_agg, max_agg, min_agg, first_agg, last_agg]
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
        df_part_max = df_part[df_part["variable"].isin(max_agg)].copy()
        pivot_max = df_part_max.pivot_table(index="day",
                                        columns="variable",
                                        values="value",
                                        aggfunc="max")

        # add the minimum values in a day
        df_part_min = df_part[df_part["variable"].isin(min_agg)].copy()
        pivot_min = df_part_min.pivot_table(index="day",
                                        columns="variable",
                                        values="value",
                                        aggfunc="min")
        
        # add the first values in a day
        df_part_first = df_part[df_part["variable"].isin(first_agg)].copy()
        pivot_first = df_part_first.pivot_table(index="day",
                                        columns="variable",
                                        values="value",
                                        aggfunc="first")
        
        # add the last values in a day
        df_part_first = df_part[df_part["variable"].isin(first_agg)].copy()
        pivot_first = df_part_first.pivot_table(index="day",
                                        columns="variable",
                                        values="value",
                                        aggfunc="first")


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
                if col in ["circumplex.valence", "circumplex.arousal"]:
                    final_order.append(f"{col}_min")
                if col in ["activity", "circumplex.valence", "circumplex.arousal"]:
                    final_order.append(f"{col}_max")

        pivot = pivot[final_order]
        
        # print("final columns", pivot.columns)

        # rearrange the columns to have the id_num, day, mood, screen, activity, circumplex.valence, circumplex.arousal then the rest

        pivot_list.append(pivot)
    
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
    

if __name__ == '__main__':
    preprocess_pipeline()