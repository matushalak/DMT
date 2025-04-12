#@matushalak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_pipeline
from collections import deque

def engineer_features(preprocessed_df: pd.DataFrame,
                      mvaN = 7, sleep_debt_days = 3) -> pd.DataFrame:
    # TODO: differences - on _daily AND _tod level
    # mood_change =  np.array(pdata['mood'][1:]) - np.array(pdata['mood'][0:-1])
    vars_differences = ('mood', 'circumplex.valence', 'circumplex.arousal')
    
    # TODO: add sleep (bed_time[i] - wake_time[i+1]) 
    # if bed_time[i] <= 23: (23-bed_time[i]) + wake_time[i+1] 
    # else: wake_time[i+1] - bed_time[i]
    
    # TODO: exponentially weighed MVA
    # exponentially weighed moving average windowsize N, emphasizes recent events
    # mva = pdata['mood'].ewm(span = mvaN, min_periods=1).mean()
    vars_mva = ('mood', 'circumplex.valence', 'circumplex.arousal', 'activity', 'screen', 'sleep')

    # TODO: add screentime x activity interaction


if __name__ == '__main__':
    # explore_hours()
    data = preprocess_pipeline(load_from_file=True)
    engineer_features(data)
