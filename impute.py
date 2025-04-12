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
def interpolate(df: pd.DataFrame, columns, method='linear', order=2):
    """
    Impute with interpolation of chosen method with improved handling of NaN values
    fixed by Caude
    """
    print("Shape before interpolation:", df.shape)
    # print("NaNs in interpolation columns before:", df[columns].isna().sum())
    
    # Create a copy of the original dataframe to avoid modifying the input
    df_result = df.copy()
    
    # Ensure date column is datetime type for the entire dataframe
    if 'date' in df_result.columns:
        df_result['date'] = pd.to_datetime(df_result['date'])
    
    for ipart in df_result['id_num'].unique():
        part = (df_result['id_num'] == ipart)
        dfpart = df_result.loc[part].copy()
        
        if len(dfpart) == 0:
            continue  # Skip if no data for this participant
        
        for col in columns:
            if col not in dfpart.columns:
                print(f"Warning: Column {col} not found for participant {ipart}")
                continue
                
            # For daily or time columns, interpolate at the daily level
            if col.endswith('_daily') or col.endswith('_time'):
                if 'date' not in dfpart.columns:
                    print(f"Warning: Date column not found for participant {ipart}")
                    continue
                    
                # Group by date, checking if there's any non-null data first
                if dfpart[col].notna().any():
                    # Create a complete date range to ensure no gaps
                    date_range = pd.date_range(dfpart['date'].min(), dfpart['date'].max())
                    
                    # Create daily level dataframe with complete date range
                    daily_df = dfpart.groupby('date')[col].first()
                    daily_df = daily_df.reindex(date_range)
                    
                    # Interpolate with fallback to ffill/bfill for values that can't be interpolated
                    daily_df_interpolated = daily_df.interpolate(method=method, limit_direction='both')
                    
                    # # For any remaining NaNs, use forward/backward fill
                    # if daily_df_interpolated.isna().any():
                    #     daily_df_interpolated = daily_df_interpolated.fillna(method='ffill').fillna(method='bfill')
                    
                    # Create a lookup dictionary for faster assignment
                    date_to_value = daily_df_interpolated.to_dict()
                    
                    # Update values in the original dataframe directly
                    for date, rows in dfpart.groupby('date').groups.items():
                        if date in date_to_value:
                            df_result.loc[rows, col] = date_to_value[date]
                
            else:
                # Standard interpolation for non-daily columns
                if dfpart[col].isna().any() and dfpart[col].notna().any():
                    interpolated = dfpart[col].interpolate(method=method, limit_direction='both')
                    
                    # Handle any remaining NaNs with forward/backward fill
                    if interpolated.isna().any():
                        interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')
                        
                    df_result.loc[part, col] = interpolated
    
    print("Shape after interpolation:", df_result.shape)
    # print("NaNs in interpolation columns after:", df_result[columns].isna().sum())
    return df_result

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
    assert all(data.isna().sum()) == 0, 'Still some non-interpolated values'

    return data


if __name__ == '__main__':
    pass