import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl

def main():
    # load in data and add useful features
    data = preprocess_df()
    
    # split by participant
    data_by_partic : dict[int : pd.DataFrame] = split_by_X(data, col = 'id')

    for ip, pdata in data_by_partic.items():
        # split data by variables
        data_by_var : dict[str : pd.DataFrame] = split_by_X(pdata, col = 'variable')
        
        # see how much data we have for each variable
        print(f'\n  Participant {ip}:')
        for var in data_by_var.keys():
            print(var, data_by_var[var].shape[0], 'datapoints ', 
                  ndays := (data_by_var[var]['date'].unique().size), f"days between {data_by_var[var]['date'].iloc[0]} and {data_by_var[var]['date'].iloc[ndays - 1]} ",
                  # TODO: add average number of hour / minute measurements per day
                  data_by_var[var]['value'].unique().size, f"unique variable values between {np.nanmin(data_by_var[var]['value'].unique())} and {np.nanmax(data_by_var[var]['value'].unique())}"
                  )

#-------------- Generally useful functions ------------------
def preprocess_df(filename:str = 'dataset_mood_smartphone.csv') -> pd.DataFrame:
    # load in data from csv
    data : pd.DataFrame = pd.read_csv(filename, index_col=0)
    data["id"] = data["id"].apply(lambda x: int(x.split(".")[1]))
    data["time"] = pd.to_datetime(data["time"])
    # extract useful information about date time
    data['month'] = data['time'].apply(lambda x: x.month)
    data['date'] = data['time'].apply(lambda x: x.date())
    data['weekday'] = data['time'].apply(lambda x: x.weekday())
    data['hour'] = data['time'].apply(lambda x: x.hour)
    data['minute'] = data['time'].apply(lambda x: x.minute)

    return data

def split_by_X(data : pd.DataFrame, col: str) -> dict[int | str : pd.DataFrame]:
    assert col in data.columns, '{} is not a column of the chosen dataframe!'.format(col)
    return {var : data.loc[data[col] == var]
            for var in data[col].unique()}

if __name__ == '__main__':
    main()