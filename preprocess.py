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
        
        # df['gross_bookings_usd'] = df['gross_bookings_usd'].clip(10, 2000)
        df['gross_bookings_usd'] = df['gross_bookings_usd'].fillna(0)

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
    
    df['price_usd'].clip(10, 2000)

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
    # 1) Add distance to other properties in query
    df = mean_distance_to_other_props(df)

    # location_clustering()

    df['prop_desirability'] = (df['prop_location_score1'] + 
                            df['prop_location_score2'] + 
                            df['prop_review_score'] + 
                            df['prop_starrating']) / df['price_usd']
    
    df['price_desirability'] = df['price_usd'] / df['prop_log_historical_price']

    df['price_surprise'] = df['price_usd'] / df['visitor_hist_adr_usd']

    df['international'] = df['visitor_location_country_id'] == df['prop_country_id']

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


def add_prop_rates(df: pd.DataFrame,
                   arbitrary_low: float = 1e-6
                  ) -> pd.DataFrame:
    """
    Add two new columns for each row, representing the booking‐rate
    and click‐rate of that prop _up to_ the moment of this query.
    
    Uses only the columns you already have:
      - year
      - dayOFyear
      - visitor_location_country_id
      - prop_country_id
      - prop_id
      - booking_bool (0/1)
      - click_bool   (0/1)
    
    Output cols:
      - booking_rate
      - click_rate
    
    Fallback order for each row:
      1) prior history of this prop_id
      2) prior history of (visitor_location_country_id, prop_country_id)
      3) prior history of prop_country_id
      4) prior history of visitor_location_country_id
      5) global mean of prop‐level rates
      6) if this is the very first row, an arbitrary low number
    """
    # 1) sort by time so that .cumsum/.cumcount only sees the past
    tmp = (
        df
        .reset_index(drop=False)                # preserve original index
        .sort_values(['year','dayOFyear'])      # your time order
        .reset_index(drop=True)
    )

    # helper to shift out current row
    def prior_cumsum(s):  return s.cumsum() - s
    def prior_count(g):   return g.cumcount()

    # 2) per‐prop
    tmp['p_cum_book']  = prior_cumsum(tmp.groupby('prop_id')['booking_bool'].transform('sum'))
    tmp['p_cum_click'] = prior_cumsum(tmp.groupby('prop_id')['click_bool'].transform('sum'))
    tmp['p_count']     =   tmp.groupby('prop_id').cumcount()
    tmp['p_rate_book'] = tmp['p_cum_book']  / tmp['p_count'].replace(0, np.nan)
    tmp['p_rate_click']= tmp['p_cum_click'] / tmp['p_count'].replace(0, np.nan)

    # 3) visitor–prop_country combo
    key_combo = ['visitor_location_country_id','prop_country_id']
    tmp['c_cum_book']  = prior_cumsum(tmp.groupby(key_combo)['booking_bool'].transform('sum'))
    tmp['c_cum_click'] = prior_cumsum(tmp.groupby(key_combo)['click_bool'].transform('sum'))
    tmp['c_count']     = tmp.groupby(key_combo).cumcount()
    tmp['c_rate_book'] = tmp['c_cum_book']  / tmp['c_count'].replace(0, np.nan)
    tmp['c_rate_click']= tmp['c_cum_click'] / tmp['c_count'].replace(0, np.nan)

    # 4) prop_country
    tmp['pc_cum_book']  = prior_cumsum(tmp.groupby('prop_country_id')['booking_bool'].transform('sum'))
    tmp['pc_cum_click'] = prior_cumsum(tmp.groupby('prop_country_id')['click_bool'].transform('sum'))
    tmp['pc_count']     = tmp.groupby('prop_country_id').cumcount()
    tmp['pc_rate_book'] = tmp['pc_cum_book']  / tmp['pc_count'].replace(0, np.nan)
    tmp['pc_rate_click']= tmp['pc_cum_click'] / tmp['pc_count'].replace(0, np.nan)

    # 5) visitor_country
    tmp['vc_cum_book']  = prior_cumsum(tmp.groupby('visitor_location_country_id')['booking_bool'].transform('sum'))
    tmp['vc_cum_click'] = prior_cumsum(tmp.groupby('visitor_location_country_id')['click_bool'].transform('sum'))
    tmp['vc_count']     = tmp.groupby('visitor_location_country_id').cumcount()
    tmp['vc_rate_book'] = tmp['vc_cum_book']  / tmp['vc_count'].replace(0, np.nan)
    tmp['vc_rate_click']= tmp['vc_cum_click'] / tmp['vc_count'].replace(0, np.nan)

    # 6) chain the fallbacks
    tmp['prop_booking_rate'] = (
        tmp['p_rate_book']
        .fillna(tmp['c_rate_book'])
        .fillna(tmp['pc_rate_book'])
        .fillna(tmp['vc_rate_book'])
    )
    tmp['prop_click_rate'] = (
        tmp['p_rate_click']
        .fillna(tmp['c_rate_click'])
        .fillna(tmp['pc_rate_click'])
        .fillna(tmp['vc_rate_click'])
    )

    # 7) final global‐mean fallback
    tmp['prop_booking_rate'] = tmp['prop_booking_rate'].fillna(tmp['p_rate_book'].mean())
    tmp['prop_click_rate']   = tmp['prop_click_rate'].fillna(tmp['p_rate_click'].mean())

    # 8) for the very first row in time, force your low default
    tmp.loc[0, ['prop_booking_rate','prop_click_rate']] = arbitrary_low

    # restore original order & drop helpers
    out = (
        tmp
        .sort_values('index')
        .set_index('index')[df.columns.tolist() + ['prop_booking_rate','prop_click_rate']]
    )
    return out


def project_prop_rates_with_fallback(train_df: pd.DataFrame,
                                     new_df: pd.DataFrame
                                    ) -> pd.DataFrame:
    """
    Given train_df with columns [prop_id, visitor_location_country_id,
    prop_country_id, click_bool, booking_bool], compute rates on train_df
    and merge them onto new_df using the fallback hierarchy:
      1) prop_id rate
      2) (visitor_location_country_id, prop_country_id) rate
      3) prop_country_id rate
      4) visitor_location_country_id rate

    Returns a copy of new_df with two new columns:
      - prop_booking_rate
      - prop_click_rate
    """
    # 1) base prop_id rates
    prop_rates = (
        train_df
        .groupby('prop_id')[['click_bool','booking_bool']]
        .mean()
        .rename(columns={
            'click_bool':   'prop_click_rate',
            'booking_bool': 'prop_booking_rate'
        })
        .reset_index()
    )

    # 2) (visitor_country, prop_country) combo rates
    combo_rates = (
        train_df
        .groupby(['visitor_location_country_id','prop_country_id'])
        [['click_bool','booking_bool']]
        .mean()
        .rename(columns={
            'click_bool':   'combo_click_rate',
            'booking_bool': 'combo_booking_rate'
        })
        .reset_index()
    )

    # 3) prop_country_id rates
    prop_country_rates = (
        train_df
        .groupby('prop_country_id')[['click_bool','booking_bool']]
        .mean()
        .rename(columns={
            'click_bool':   'pc_click_rate',
            'booking_bool': 'pc_booking_rate'
        })
        .reset_index()
    )

    # 4) visitor_location_country_id rates
    visitor_country_rates = (
        train_df
        .groupby('visitor_location_country_id')[['click_bool','booking_bool']]
        .mean()
        .rename(columns={
            'click_bool':   'vc_click_rate',
            'booking_bool': 'vc_booking_rate'
        })
        .reset_index()
    )

    # Start merging onto new_df
    out = new_df.copy()

    out = (
        out
        .merge(prop_rates,
               on='prop_id', how='left')
        .merge(combo_rates,
               on=['visitor_location_country_id','prop_country_id'],
               how='left')
        .merge(prop_country_rates,
               on='prop_country_id', how='left')
        .merge(visitor_country_rates,
               on='visitor_location_country_id', how='left')
    )

    # Now apply the fallback hierarchy for each rate
    new_df['prop_booking_rate'] = (
        out['prop_booking_rate']
        .fillna(out['combo_booking_rate'])
        .fillna(out['pc_booking_rate'])
        .fillna(out['vc_booking_rate'])
        .fillna(out['prop_booking_rate'].mean())
    )

    new_df['prop_click_rate'] = (
        out['prop_click_rate']
        .fillna(out['combo_click_rate'])
        .fillna(out['pc_click_rate'])
        .fillna(out['vc_click_rate'])
        .fillna(out['prop_click_rate'].mean())
    )

    return new_df

# prop location_score2 sometimes has missing values -> mean impute

# price USD varies sometimes per night, other times for whole stay -> URBAN disentangle

# orig destination distance also has missing values 
# -> impute with mean of orig_destination distance for 
    # the same hotel from different queries from the same visitor_locatino_country_id

#  