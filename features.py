import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_pipeline
from collections import deque

def engineer_features(preprocessed_df: pd.DataFrame,
                      mvaN = 7, sleep_debt_days = 3) -> pd.DataFrame:
    # TODO: difference - on day AND time of day level
    # TODO: diff_agg = ["mood", "circumplex.valence", "circumplex.arousal"]
    # difference feature from previous day
    mood_changes = []
    mvaNs = []
    hsleep = []
    sleep_debt = []
    for participant, pdata in preprocessed_df.groupby('id_num'):
        mood_change =  np.array(pdata['mood'][1:]) - np.array(pdata['mood'][0:-1])
        # add no mood change for first day
        mood_change = np.append([0], mood_change)
        assert mood_change.size == pdata['mood'].size, 'each row should have a mood change variable (first row = 0)'
        
        # exponentially weighed moving average windowsize N, emphasizes recent events
        mva = pdata['mood'].ewm(span = mvaN, min_periods=1).mean()
        assert mva.size == pdata['mood'].size, 'each row should have a moving average variable (first row = 0)'
    pass



if __name__ == '__main__':
    # explore_hours()
    data = preprocess_pipeline(load_from_file=True)
    engineer_features(data)
