#@matushalak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_pipeline
from collections import deque

def engineer_features(preprocessed_df: pd.DataFrame, method: str,
                      mvaN = 7, sleep_debt_days = 3) -> pd.DataFrame:
    if method == 'both':
        dailyDF, todDF = separate_date_and_tod(preprocessed_df)
    else:
        dailyDF = preprocessed_df
    # 1) TODO: Differences - on _daily AND _tod level
    vars_differences = ('mood', 'circumplex.valence', 'circumplex.arousal', 
                        'wake', 'bed', 'screen', 'activity')
    breakpoint()
    # df = mood_change(preprocessed_df, vars_differences)
    
    # TODO: valence / arousal quadrant
    # > 0, < 0 combinations

    # TODO: shift from one quadrant to another?
    # bool
    
    # TODO: add sleep (bed_time[i] - wake_time[i+1]) 
    # if bed_time[i] <= 23: (23-bed_time[i]) + wake_time[i+1] 
    # else: wake_time[i+1] - bed_time[i]
    
    # TODO: exponentially weighed MVA
    # exponentially weighed moving average windowsize N, emphasizes recent events
    # mva = pdata['mood'].ewm(span = mvaN, min_periods=1).mean()
    vars_mva = ('mood', 'circumplex.valence', 'circumplex.arousal', 'activity', 'screen', 'sleep')

    # TODO: add screentime x activity interaction

def mood_change(data: pd.DataFrame, vars: tuple[str]) -> pd.DataFrame:
    exclude_features = ('min', 'max', 'std', 'first', 'last')
    for ipart in data['id_num'].unique():
        part = (data['id_num'] == ipart) 
        pdata = data[part].copy()  
        for var in vars:
            varname_is = [(var in vname) and all(s not in vname for s in exclude_features) 
                            for vname in pdata.columns]
            varname_is = np.where(varname_is)[0]
            assert len(varname_is) == 1, 'Should be one variable that meets this conditon'
            breakpoint()
            var_change =  pdata[pdata.columns[varname_i]][1:] - pdata[pdata.columns[varname_i]][0:-1]
            var_change = np.append([0], np.array(var_change)) # first day no change

def separate_date_and_tod(tod_data:pd.DataFrame
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_df = tod_data.drop_duplicates(subset=['id_num', 'date']).copy()
    daily_df = daily_df.drop(columns=[col for col in daily_df.columns if col.endswith("_tod")])
    tod_df = tod_data.drop(columns=[col for col in tod_data.columns if col.endswith("_daily")])
    assert all(tod_df['date'].unique() == daily_df['date'].unique()), 'Need same dates for both daily and time of day dfs'
    return daily_df, tod_df


if __name__ == '__main__':
    # explore_hours()
    data = preprocess_pipeline(load_from_file=True)
    engineer_features(data, method='both')
