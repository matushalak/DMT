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

    # 1) Mood according to circumplex model: valence / arousal quadrant
    dailyDF = circumplex(dailyDF)
    if method == 'both':
        todDF = circumplex(todDF)

    # 2) Lag feature - Differences - on _daily AND _tod level
    vars_differencesD = ('mood', 'circumplex.valence', 'circumplex.arousal', 
                        'wake', 'bed', 'screen', 'activity')
    vars_differencesTOD = ('mood', 'circumplex.valence', 'circumplex.arousal')
    
    dailyDF = mood_change(dailyDF, vars_differencesD)
    if method == 'both':
        todDF = mood_change(todDF, vars_differencesTOD)
    
    breakpoint()

    # TODO: add sleep (bed_time[i] - wake_time[i+1]) 
    # if bed_time[i] <= 23: (23-bed_time[i]) + wake_time[i+1] 
    # else: wake_time[i+1] - bed_time[i]
    
    # TODO: exponentially weighed MVA
    # exponentially weighed moving average windowsize N, emphasizes recent events
    # mva = pdata['mood'].ewm(span = mvaN, min_periods=1).mean()
    vars_mva = ('mood', 'circumplex.valence', 'circumplex.arousal', 'activity', 'screen', 'sleep')

    # TODO: add screentime x activity interaction


def mood_change(data: pd.DataFrame, vars: tuple[str]) -> pd.DataFrame:
    exclude_features = ('min', 'max', 'std', 'first', 'last', 'change')
    for var in vars:
        varname_is = [(var in vname) and all(s not in vname for s in exclude_features) 
                        for vname in data.columns]
        varname_is = np.where(varname_is)[0]
        assert len(varname_is) == 1, f'Should be one variable that meets this conditon, Var: {var}, {[data.columns[v] for v in varname_is]}'
        varname = data.columns[varname_is[0]]
        new_col_name = 'change_' + varname
        data[new_col_name] = data.groupby("id_num")[varname].transform(lambda s: s.diff().fillna(0))
        
        assert data[new_col_name].isna().sum() == 0, f'There are {data[new_col_name].isna().sum()} unexpected NaNs!'
    return data


def circumplex(df: pd.DataFrame)-> pd.DataFrame:
    valence_i = np.where(['valence_mean' in vname for vname in df.columns])[0]
    valence = df.columns[valence_i]
    arousal_i = np.where(['arousal_mean' in vname for vname in df.columns])[0]
    arousal = df.columns[arousal_i]
    conditions = [
    np.array((df[valence] > 0) & (df[arousal] > 0)),   # Quadrant 1
    np.array((df[valence] < 0) & (df[arousal] > 0)),   # Quadrant 2
    np.array((df[valence] < 0) & (df[arousal] < 0)),   # Quadrant 3
    np.array((df[valence] > 0) & (df[arousal] < 0))]   # Quadrant 4
    
    quadrants = [0,1,2,3]
    breakpoint()
    df['circumplex'] = np.select(conditions, quadrants)
    assert df['circumplex'].isna().sum() == 0, 'Should not have any NaNs'
    return df


# ------- utils --------
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
