import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main(plot = False, mvaN = 7):
    data = pd.read_csv('tables/preprocessed/df_interpolated.csv',
                       index_col=False)
    # extract useful information about date time
    data["date"] = pd.to_datetime(data["date"])
    data['month'] = data['date'].apply(lambda x: x.month)
    data['weekday'] = data['date'].apply(lambda x: x.weekday())

    mood_changes = []
    mvaNs = []
    for participant, pdata in data.groupby('id_num'):
        if plot:
        # plot relationship between month, weekday and mood
            # monthly overview
            monthly_stats = pdata.groupby('month')['mood'].agg(['mean', 'std']
                                                            ).reset_index()
            
            plt.figure(figsize=(8, 5))
            plt.errorbar(monthly_stats['month'], monthly_stats['mean'], yerr=monthly_stats['std'], fmt='-o', capsize=5)
            plt.title(f'Average Mood by Month p{participant}')
            plt.xlabel('Month')
            plt.ylabel('Mood')
            plt.xticks(monthly_stats['month'])
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.close()
            
            # weekday overview
            weekday_stats = pdata.groupby('weekday')['mood'].agg(['mean', 'std']
                                                            ).reset_index()
            nextweekday_stats = pdata.groupby('weekday')['next_day_mood'].agg(['mean', 'std']
                                                            ).reset_index()
            
            plt.figure(figsize=(8, 5))
            plt.errorbar(weekday_stats['weekday'], weekday_stats['mean'], yerr=weekday_stats['std'], fmt='-o', capsize=5, label = 'current day mood')
            plt.errorbar(nextweekday_stats['weekday'], nextweekday_stats['mean'], yerr=nextweekday_stats['std'], fmt='-o', capsize=5, label = 'next day mood')
            plt.title(f'Average Mood by Weekday p{participant}')
            plt.xlabel('Weekday')
            plt.ylabel('Mood')
            plt.grid(True)
            plt.legend(loc = 2)
            plt.tight_layout()
            plt.show()
            plt.close()
    

        # difference feature from previous day
        mood_change =  np.array(pdata['mood'][1:]) - np.array(pdata['mood'][0:-1])
        # add no mood change for first day
        mood_change = np.append([0], mood_change)
        assert mood_change.size == pdata['mood'].size, 'each row should have a mood change variable (first row = 0)'
        
        # exponentially weighed moving average windowsize N, emphasizes recent events
        mva = pdata['mood'].ewm(span = mvaN, min_periods=1).mean()
        assert mva.size == pdata['mood'].size, 'each row should have a moving average variable (first row = 0)'
        
        # Create daily pivot function
##

if __name__ == '__main__':
    main(True)
