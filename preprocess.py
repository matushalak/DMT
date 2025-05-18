import pandas as pd
import numpy as np

def preprocess_df(df:pd.DataFrame,
                  booking_val5:bool = True,
                  TEST:bool = False)->pd.DataFrame:
    if not TEST:
        # booking * 4 if you want loss we will be tested on (worse performance)
        df['relevance'] = df['click_bool'] + (df['booking_bool'] * 4 
                                            if booking_val5 
                                            else df['booking_bool'])
        
        df['gross_bookings_usd'] = df['gross_bookings_usd'].clip(10, 2000)
        df = df.loc[(df['gross_bookings_usd'].isna()) | ((df['gross_bookings_usd'] >= 10) & (df['gross_bookings_usd'] <= 2000))]
        
        for col in df.columns:
            if 'rate_percent_diff' in col:
                df = df.loc[(df['gross_bookings_usd'].isna()) | (df[col] <= 50).sum()]
                # print(f'{col} n_rows <= 50', (df[col] <= 50).sum())




    # add relevant information from date-time 
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df['weekday'] = df['date_time'].dt.weekday
    df['dayOFyear'] = df['date_time'].dt.day_of_year
    df.drop(columns=['date_time'], inplace=True)

    # impute missing values
    df = impute(df)

    return df


def impute(df:pd.DataFrame
           )->pd.DataFrame:
    df = impute_orig_dest_distance(df)
    
    df['price_usd'].clip(10, 1000)

    return df


def impute_orig_dest_distance(df: pd.DataFrame
                              ) -> pd.DataFrame:
    """
    Impute NaNs in `orig_destination_distance` (ODD) with a two–level back‑off:
        1. mean ODD for (prop_id, visitor_location_country_id)
        2. mean ODD for (prop_id, site_id)
    Returns a new DataFrame; the original is left unchanged.
    """
    out = df.copy()

    # --- level 1 -------------------------------------------------------------
    lvl1_mean = (out.groupby(['prop_id', 'visitor_location_country_id'])['orig_destination_distance']
                .transform('mean'))
    # --- level 2 -------------------------------------------------------------
    lvl2_mean = (out.groupby(['prop_id', 'site_id'])['orig_destination_distance']
                .transform('mean'))
    # --- apply the hierarchy -----------------------------------------------
    out['orig_destination_distance'] = (out['orig_destination_distance']
                                        .fillna(lvl1_mean)
                                        .fillna(lvl2_mean))

    return out


def location_clustering(df):
    pass


def relative_in_SID(df:pd.DataFrame, 
                    columns:list[str]):
    
    relative = (df
                .groupby('srch_id')[columns]
                .transform(lambda query: query.rank(pct=True)))
    relative.columns = [f"{c}_relative" for c in relative.columns]
    df = df.join(relative)
    return df

def add_features(df:pd.DataFrame
                 )->pd.DataFrame:
    # # 1) Add distance to other properties in query
    # df = mean_distance_to_other_props(df)

    # location_clustering()

    df['prop_desirability'] = (df['prop_location_score1'] + 
                            df['prop_location_score2'] + 
                            df['prop_review_score'] + 
                            df['prop_starrating']) / df['price_usd']
    
    df['price_desirability'] = df['price_usd'] / df['prop_log_historical_price']

    df['price_surprise'] = df['price_usd'] / df['visitor_hist_adr_usd']

    df['international'] = df['visitor_location_country_id'] != df['prop_country_id']

    # relative features
    df = relative_in_SID(df, columns=[
        'price_usd',
        'prop_location_score1', 
        'prop_location_score2',
        'prop_review_score',
        'prop_starrating',
        'prop_desirability', 
        'price_desirability',
        'price_surprise'])
    
    # df.drop(columns=['prop_desirability', 'price_desirability', 'price_surprise', 
    #                  'prop_location_score1', 'prop_location_score2','prop_review_score', 'prop_starrating'],
    #         inplace=True)
    
    return df


def mean_distance_to_other_props(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (srch_id, prop_id) row, return the mean distance of
    all *other* properties in the same srch_id group.
    If a group contains only one property the result is NaN.
    """

    # 1.  Replace NaNs within each search‑query group by the group mean
    df['orig_destination_distance'] = (
        df.groupby('srch_id')['orig_destination_distance']
           .transform(lambda x: x.fillna(x.mean()))
    )

    # 2.  Pre‑compute per‑group sums and sizes (vectorised, so very fast)
    grp_sum  = df.groupby('srch_id')['orig_destination_distance'].transform('sum')
    grp_size = df.groupby('srch_id')['orig_destination_distance'].transform('size')

    # 3.  Mean of "all others"  =  (sum of group − distance of this row) / (size − 1)
    df['dist_query_props'] = np.where(
        grp_size > 1,
        (grp_sum - df['orig_destination_distance']) / (grp_size - 1),
        np.nan  # undefined when the group has a single property
    )
    return df


# prop location_score2 sometimes has missing values -> mean impute

# price USD varies sometimes per night, other times for whole stay -> URBAN disentangle

# orig destination distance also has missing values 
# -> impute with mean of orig_destination distance for 
    # the same hotel from different queries from the same visitor_locatino_country_id
