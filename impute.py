#@matushalak
import pandas as pd
import os
# -------------- imputation functions from urban
# impute everything ending with min or max with mode
# per participant
def impute_mode_grouped(df, columns):
    """
    Impute all columns ending with min or max with mode
    example usage: df = impute_mean_grouped(df, df.select_dtypes(include=[np.number]).columns.tolist())
    """
    for col in columns:
        df[col] = df.groupby('id_num')[col].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
    return df

# per participant
def impute_mean_grouped(df, columns):
    """
    Impute with mean
    example usage: df = impute_mean_grouped(df, df.columns[df.columns.str.endswith('min') | df.columns.str.endswith('max')])
    """
    for col in columns:
        df[col] = df.groupby('id_num')[col].transform(lambda x: x.fillna(x.mean()))
    return df

# per participant
def interpolate(df: pd.DataFrame, columns, method= 'linear', order = 2):
    """
    Impute with interpolation of chosen method
    """
    for ipart in df['id_num'].unique():
        part = (df['id_num'] == ipart) 
        dfpart = df[part].copy()   

        for col in columns:
            # want daily interpolation but
            if col.endswith('_daily') or col.endswith('_time'):
                # Ensure the "date" column is a datetime type:
                dfpart["date"] = pd.to_datetime(df["date"])
                
                # Create a daily-level DataFrame; one row per date.
                daily_df = dfpart.groupby("date")[col].first().sort_index()
                
                # Perform linear interpolation along date (the index)
                daily_df_interpolated = daily_df.interpolate(method=method, limit_direction = 'both')
                
                # Now merge these daily (interpolated) values back onto the original data, using the "date" column.
                # This merge will broadcast each day's interpolated value to all rows with that date.
                dfpart = dfpart.drop(columns=col)
                dfpart = dfpart.merge(daily_df_interpolated, on="date", how="left")
                df.loc[part, col] = dfpart[col]
                
            else:
                df.loc[part, col] = dfpart[col].interpolate(method = method, limit_direction = 'both')
    return df

# all at once
def impute_not_used(df, columns):
    for col in df.columns:
        if any(col.startswith(sumcol) for sumcol in columns):
            df[col] = df[col].fillna(0)
    return df


# ----------- functions to find and run imputations on different variable categories -------
def categories(columns):
    sum_imp, mean_imp, mode_imp, interp_imp = [], [], [], []
    for col in columns:
        # sum columns
        if any(sumcol in col for sumcol in ('appCat', 'sms', 'call')):
            sum_imp.append(col)
        # mean columns
        # elif any(meancol in col for meancol in ('wake', 'bed')):
        #     mean_imp.append(col)
        # mode columns
        elif any(modecol in col for modecol in ('min', 'max')):
            mode_imp.append(col)
        # interpolation columns
        elif any(interp in col for interp in ('mood', 'circumplex',
                                              'activity', 'screen',
                                              'wake', 'bed'
                                              )):
            interp_imp.append(col)

    return sum_imp, mean_imp, mode_imp, interp_imp

def imputations(column_categories, data):
    sum_imp, mean_imp, mode_imp, interp_imp = column_categories
    # for sum categories, NaNs = 0
    data = impute_not_used(data, sum_imp)
    # mean imputations
    data = impute_mean_grouped(data, mean_imp)
    # mode imputations
    data = impute_mode_grouped(data, mode_imp)
    # iterpolations
    data = interpolate(data, interp_imp)

    assert all(data.notna()), 'Still some non-interpolated values'

    return data


if __name__ == '__main__':
    pass