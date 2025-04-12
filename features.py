#@matushalak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def engineer_features(preprocessed_df: pd.DataFrame, method: str,
                      mvaN = 7) -> pd.DataFrame:
    if method == 'both':
        dailyDF, todDF = separate_date_and_tod(preprocessed_df)
    else:
        dailyDF = preprocessed_df

    # 1) Mood according to circumplex model: valence / arousal quadrant
    dailyDF = circumplex(dailyDF, 'daily')
    if method == 'both':
        todDF = circumplex(todDF, 'tod')

    # 2) Sleep (bed_time[i] - wake_time[i+1]) [DAILY]
    dailyDF = sleep_per_participant(dailyDF)
    
    # 3) Lag feature - Differences - on _daily AND _tod level
    vars_differencesD = ('mood', 'circumplex.valence', 'circumplex.arousal', 
                        'wake', 'bed', 'screen', 'activity')
    vars_differencesTOD = ('mood', 'circumplex.valence', 'circumplex.arousal')
    
    dailyDF = mood_change(dailyDF, vars_differencesD)
    if method == 'both':
        todDF = mood_change(todDF, vars_differencesTOD)
    
    # 4) Exponentially weighed MVA
    # exponentially weighed moving average windowsize N, emphasizes recent events
    vars_mvaD = ('mood', 'circumplex.valence', 'circumplex.arousal', 'activity', 'screen', 'sleep', 'wake','bed')
    vars_mvaTOD = ('mood', 'circumplex.valence', 'circumplex.arousal')
    dailyDF = mva(dailyDF, vars_mvaD, mvaN=mvaN)
    if method == 'both':
        todDF = mva(todDF, vars_mvaTOD, mvaN=mvaN)
    
    # 4.5 Add categorical target
    dailyDF = categorical_target(dailyDF, x_daily_SD=.5)

    # 5) Combine daily and time of day dataframes
    if method == 'both':
        dfCombined = todDF.merge(dailyDF,
                                on=['id_num', 'date', 'month', 'weekday'],
                                how = 'left')
        # for c, s in zip(dfCombined.columns,  dfCombined.isna().sum()): print(c,s)
        dfCombined.drop(columns='time_of_day_y', inplace=True)
        dfCombined.rename(columns={'time_of_day_x':'time_of_day'}, inplace=True)

        featuresDF = dfCombined
    else:
        featuresDF = dailyDF

    return featuresDF

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


def circumplex(df: pd.DataFrame,
               method: str)-> pd.DataFrame:
    valence_i = np.where(['valence_mean' in vname for vname in df.columns])[0]
    valence = df.columns[valence_i]
    arousal_i = np.where(['arousal_mean' in vname for vname in df.columns])[0]
    arousal = df.columns[arousal_i]
    val = np.array(df[valence].values)[:,0]
    ar = np.array(df[arousal].values)[:,0]
    
    # different quadrants
    q0 = np.where((val == 0) & (ar == 0))[0] #both 0
    q1 = np.where((val >= 0) & (ar > 0))[0]
    q2 = np.where((val < 0) & (ar >= 0))[0]
    q3 = np.where((val <= 0) & (ar < 0))[0]
    q4 = np.where((val > 0) & (ar <= 0))[0]
    qs = [q0, q1, q2, q3, q4]
    quadrants = [-10, # if both 0
                 1,2,3,4]

    circumplex = np.full_like(val, fill_value=-1, dtype=int)
    for qi, qv in zip(qs, quadrants):
        circumplex[qi] = qv
    df[f'circumplex.QUADRANT_{method}'] = circumplex
    unvals = df[f'circumplex.QUADRANT_{method}'].nunique()
    assert df[f'circumplex.QUADRANT_{method}'].isna().sum() == 0 and unvals == len(quadrants), f'Should not have any NaNs and only have {len(quadrants)} values, not {unvals}'
    # boolean dummy encoding
    df = pd.get_dummies(df, columns=[f'circumplex.QUADRANT_{method}'], drop_first=True) # 4 for 4 nonzero categories

    return df


def sleep(df: pd.DataFrame)->pd.DataFrame:
    '''
    For each day gives number of hours of sleep last night
    '''
    bts = np.array(df['bed_time_daily'])
    wts = np.array(df['wake_time_daily'])
    bts_prev = bts[:-1]
    bts_prev = np.append(np.nan, bts_prev)
    sleep = np.empty_like(bts)
    
    # if bed_time[i] <= 23: (23-bed_time[i]) + wake_time[i+1] 
    # else: wake_time[i+1] - bed_time[i]
    sleep_duration = np.where((bts_prev <= 23) & (bts_prev >= 15), (24 - bts_prev) + wts, wts - bts_prev)
    sleep_duration[0] = (avSleep := sleep_duration[1:].mean()) # first day is mean sleep duration
    
    # fix interpolation problem :=> slept average number of hours
    prob = np.where(sleep_duration <= 0)[0]
    bts[prob-1] = wts[prob] - avSleep
    sleep_duration[prob] = avSleep

    df['bed_time_daily'] = bts
    df['sleep_daily'] = sleep_duration
    return df


def sleep_per_participant(df: pd.DataFrame) -> pd.DataFrame:
    # Compute sleep duration per participant:
    df_list = []
    for pid, sub_df in df.groupby("id_num"):
        sub_df = sub_df.sort_values("date").copy()
        sub_df = sleep(sub_df)
        df_list.append(sub_df)
    return pd.concat(df_list, ignore_index=True)


def mva(df:pd.DataFrame, vars: tuple[str], mvaN: int) -> pd.DataFrame:
    exclude_features = ('min', 'max', 'first', 'last')
    for var in vars:
        varname_is = [(var in vname) and all(s not in vname for s in exclude_features) 
                        for vname in df.columns]
        varname_is = np.where(varname_is)[0]
        for vi in varname_is:
            varname = df.columns[vi]
            new_col_name = f'mva{mvaN}_' + varname
            df[new_col_name] = df.groupby("id_num"
                                          )[varname].transform(
                                              lambda x: x.ewm(span=mvaN, min_periods=1).mean())
    
    assert all(df.isna().sum()) == 0, 'Still some NaNs!!!'
    return df


def categorical_target(df:pd.DataFrame, x_daily_SD: float = 0.7) -> pd.DataFrame:
    same = (df['change_mood_mean_daily'] <= x_daily_SD * df['mva7_mood_std_daily']) & (df['change_mood_mean_daily'] >= -x_daily_SD * df['mva7_mood_std_daily'])
    higher = (df['change_mood_mean_daily'] > x_daily_SD * df['mva7_mood_std_daily'])
    lower = (df['change_mood_mean_daily'] < -x_daily_SD * df['mva7_mood_std_daily'])

    cat_target = np.zeros(df.shape[0])
    cat_target[same] = 0
    cat_target[higher] = 1
    cat_target[lower] = -1

    df['categorical_target'] = cat_target 
    return df



# ------- utils --------
def separate_date_and_tod(tod_data:pd.DataFrame
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclude_varsD = ('time_of_day')
    exclude_varsTOD = ('month', 'weekday')
    daily_df = tod_data.drop_duplicates(subset=['id_num', 'date']).copy()
    daily_df = daily_df.drop(columns=[col for col in daily_df.columns if col.endswith("_tod") and col not in exclude_varsTOD])
    tod_df = tod_data.drop(columns=[col for col in tod_data.columns if col.endswith("_daily") and col not in exclude_varsD])
    assert all(tod_df['date'].unique() == daily_df['date'].unique()), 'Need same dates for both daily and time of day dfs'
    return daily_df, tod_df
