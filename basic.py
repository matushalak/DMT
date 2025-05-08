import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load & preprocess
DF = pd.read_csv('data/training_set_VU_DM.csv')
DF['relevance'] = DF['click_bool'] + 4 * DF['booking_bool']
DF['gross_bookings_usd'] = DF['gross_bookings_usd'].fillna(0)

# add relevant information from date-time 
DF['date_time'] = pd.to_datetime(DF['date_time'])
DF['month'] = DF['date_time'].dt.month
DF['weekday'] = DF['date_time'].dt.weekday
DF.drop(columns=['date_time'], inplace=True)

# Train/val split by srch_id
all_searchIDs = DF['srch_id'].unique()
train_sids = np.random.choice(all_searchIDs, size=int(0.7 * len(all_searchIDs)), replace=False)
val_sids   = np.setdiff1d(all_searchIDs, train_sids)

TRAIN = DF[DF['srch_id'].isin(train_sids)].copy()
VAL   = DF[DF['srch_id'].isin(val_sids)].copy()

# Compute group sizes
group_train = TRAIN.groupby('srch_id').size().to_numpy()
group_val   = VAL.groupby('srch_id').size().to_numpy()

# Features and labels
X_train = TRAIN.drop(columns=['srch_id', 'relevance', 
                              'position', 'click_bool', 'booking_bool', 'gross_bookings_usd'])
y_train = TRAIN['relevance']

X_val = VAL.drop(columns=['srch_id', 'relevance', 
                          'position', 'click_bool', 'booking_bool', 'gross_bookings_usd'])
y_val = VAL['relevance']

# LambdaMART model
model = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', 
                       importance_type='gain',
                    #    n_estimators=500 # number of boosting rounds (iterations)
                       n_jobs=-1, # parallel processing
                       )

# Train
model.fit(
    X=X_train,
    y=y_train,
    group=group_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_group=[group_train, group_val],
    eval_at=[5, 5],
    eval_names=['train', 'val'],
)

print(model.best_score_)

# plot results
# loss curves
lgb.plot_metric(model)
plt.show()
plt.close()

# importance
# overall
lgb.plot_importance(model, max_num_features=12)
plt.show()
plt.close()
# gain
lgb.plot_importance(model, max_num_features=12,
                    importance_type='gain')
plt.show()