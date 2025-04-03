import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import sklearn as skl
import os


def main():
    # load in data and add useful features
    data = preprocess_df()

    # split by participant
    data_by_partic: dict[int: pd.DataFrame] = split_by_X(data, col='id')

    # overview of variables
    if not os.path.exists('overview.csv'):
        vars_overview: pd.DataFrame = overview_table(data_by_partic)
    else:
        vars_overview = pd.read_csv('overview.csv', index_col=0)

    # TODO: plot with boxplot overview for each variable
    # TODO: cross correlations (heatmap) & distributions
    vars_overview
    breakpoint()


def overview_table(data_by_participant: dict[int, pd.DataFrame],
                   verbose: bool = False) -> pd.DataFrame:
    rows = []
    all_vars: np.ndarray[str] = data_by_participant[1]['variable'].unique()
    for pid, pdata in data_by_participant.items():
        # split data by variables
        data_by_var: dict[str, pd.DataFrame] = split_by_X(pdata,
                                                          col='variable')

        # for var, df in data_by_var.items():
        for var in all_vars:
            if var in data_by_var.keys():
                df = data_by_var[var]
                datapoints = df.shape[0]
                ndays = df['date'].nunique()
                dates_range = (df['date'].iloc[0], df['date'].iloc[ndays - 1])
                unique_vals = df['value'].nunique()
                vals_range = (np.nanmin(df['value'].unique()),
                              np.nanmax(df['value'].unique()))
            else:
                NANs = np.nan, np.nan, (np.nan, np.nan), np.nan, (np.nan,
                                                                  np.nan)
                datapoints, ndays, dates_range, unique_vals, vals_range = NANs

            rows.append({
                'id': pid,
                'variable': var,
                'n_datapoints': datapoints,
                'n_days': ndays,
                'first_date': dates_range[0],
                'last_date': dates_range[1],
                'n_unique_values': unique_vals,
                'val_min': vals_range[0],
                'val_max': vals_range[1],
            })

            if verbose:
                print(f'\n  Participant {pid}:')
                print(var, f'{datapoints} datapoints ',
                      f'''{ndays} days between {dates_range[0]}
                       and {dates_range[1]} ''',
                      f'''{unique_vals} unique values between {vals_range[0]}
                       and {vals_range[1]}''')

    overview = pd.DataFrame(rows)
    # save res
    overview.to_csv('overview.csv')
    return overview


# -------------- Generally useful functions ------------------
def preprocess_df(filename: str = 'dataset_mood_smartphone.csv'
                  ) -> pd.DataFrame:
    # load in data from csv
    data: pd.DataFrame = pd.read_csv(filename, index_col=0)
    data["id"] = data["id"].apply(lambda x: int(x.split(".")[1]))
    data["time"] = pd.to_datetime(data["time"])
    # extract useful information about date time
    data['month'] = data['time'].apply(lambda x: x.month)
    data['date'] = data['time'].apply(lambda x: x.date())
    data['weekday'] = data['time'].apply(lambda x: x.weekday())
    data['hour'] = data['time'].apply(lambda x: x.hour)
    data['minute'] = data['time'].apply(lambda x: x.minute)

    return data


def split_by_X(data: pd.DataFrame, col: str) -> dict[int | str: pd.DataFrame]:
    assert col in data.columns, f'{col} is not a column of the dataframe!'
    return {var: data.loc[data[col] == var]
            for var in data[col].unique()}


if __name__ == '__main__':
    main()
