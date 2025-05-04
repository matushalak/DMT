import sklearn as skl
import lightgbm as lgb
import numpy as np
import pandas as pd

# Training dataset with outcome variables
DF = pd.read_csv('training_set_VU_DM.csv', index_col=False)
DF['relevance'] = DF['click_bool'] + DF['booking_bool']
DF['gross_bookings_usd'].fillna(0)
DF.drop(columns=['date_time'], inplace=True)

all_searchIDs = DF['srch_id'].unique()
train_SIDS = np.random.choice(all_searchIDs, size=round(0.7*len(all_searchIDs)))
val_SIDS = all_searchIDs[np.isin(all_searchIDs, train_SIDS, invert=True)]

TRAIN = DF.iloc[train_SIDS, :]
VAL = DF.iloc[val_SIDS, :]

# Prepare TRAIN dataset
sids_train = TRAIN['srch_id']
X_train = TRAIN.drop(columns=['srch_id', 'relevance'])
y_train = TRAIN['relevance']

# Prepare VAL dataset
sids_val = VAL['srch_id']
X_val = VAL.drop(columns=['srch_id', 'relevance'])
y_val = VAL['relevance']

breakpoint()

# Lambda Rank
model = lgb.LGBMRanker(objective='lambdarank',
                       metric = 'ndcg')

# Train
model.fit(X = X_train, y = y_train, group=sids_train,
          eval_set=[(X_val, y_val)], eval_group=[sids_val],
          eval_at=5)


breakpoint()
