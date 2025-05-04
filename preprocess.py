import pandas as pd
import numpy as np

def preprocess_df(df:pd.DataFrame,
                  booking_val5:bool = False,
                  TEST:bool = False)->pd.DataFrame:
    if not TEST:
        # booking * 4 if you want loss we will be tested on (worse performance)
        df['relevance'] = df['click_bool'] + (df['booking_bool'] * 4 
                                            if booking_val5 
                                            else df['booking_bool'])
        df['gross_bookings_usd'] = df['gross_bookings_usd'].fillna(0)

    # add relevant information from date-time 
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['month'] = df['date_time'].dt.month
    df['weekday'] = df['date_time'].dt.weekday
    df.drop(columns=['date_time'], inplace=True)

    return df


def add_features(df:pd.DataFrame
                 )->pd.DataFrame:
    
    return df

# prop location_score2 sometimes has missing values -> mean impute

# price USD varies sometimes per night, other times for whole stay -> URBAN disentangle

# orig destination distance also has missing values 
# -> impute with mean of orig_destination distance for 
    # the same hotel from different queries from the same visitor_locatino_country_id

#  