import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from preprocess import preprocess_df, add_features, add_prop_rates, project_prop_rates_with_fallback

def train_val_split(DF:pd.DataFrame, 
                    train_prop:float = 0.7
                    )->tuple[
                        # Train (Query sizes, X_train, y_train)
                        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                        # Validate (Query sizes, X_val, y_val)
                        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    # Train/val split by srch_id
    all_searchIDs = DF['srch_id'].unique()
    train_sids = np.random.choice(all_searchIDs, size=int(0.7 * len(all_searchIDs)), replace=False)
    val_sids   = np.setdiff1d(all_searchIDs, train_sids)

    TRAIN = DF[DF['srch_id'].isin(train_sids)].copy()
    # TRAIN = add_prop_rates(TRAIN)


    VAL = DF[DF['srch_id'].isin(val_sids)].copy()
    # VAL = project_prop_rates_with_fallback(TRAIN, VAL)


    # Compute search id group sizes
    query_size_train = TRAIN.groupby('srch_id').size().to_numpy()
    query_size_val   = VAL.groupby('srch_id').size().to_numpy()

    # Features and labels
    X_train = TRAIN.drop(columns=['srch_id', 'relevance', 
                                'position', 'click_bool', 'booking_bool', 'gross_bookings_usd'])
    y_train = TRAIN['relevance']

    X_val = VAL.drop(columns=['srch_id', 'relevance', 
                            'position', 'click_bool', 'booking_bool', 'gross_bookings_usd'])
    y_val = VAL['relevance']

    return ((query_size_train, X_train, y_train),
            (query_size_val, X_val, y_val))


def train_model(plot:bool = True
                )->lgb.LGBMRanker:
    # Load & preprocess
    DF = pd.read_csv('training_set_VU_DM.csv')

    # add relevance and relevant monthly information
    DF = preprocess_df(DF)

    # feature engineering
    DF = add_features(DF)

    # Test / Val split
    T_V_split = train_val_split(DF, train_prop=0.7)
    query_size_train, X_train, y_train = T_V_split[0]
    query_size_val, X_val, y_val = T_V_split[1]

    # LambdaMART model
    model = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', 
                        importance_type='gain',
                        learning_rate=0.05,
                        n_estimators=500, # number of boosting rounds (iterations)
                        n_jobs=-1, # parallel processing
                        )

    # Train & Validate
    model.fit(
        X=X_train,
        y=y_train,
        group=query_size_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_group=[query_size_train, query_size_val],
        eval_at=[5, 5],
        eval_names=['train', 'val'],
    )

    print(model.best_score_)

    if plot:
        # plot results
        # loss curves
        lgb.plot_metric(model)
        plt.show()
        plt.close()

        # importance
        # overall
        lgb.plot_importance(model, max_num_features=15)
        plt.show()
        plt.close()
        
        # gain
        lgb.plot_importance(model, max_num_features=15,
                            importance_type='gain')
        plt.show()
    
    return model


def get_predictions(model:lgb.LGBMRanker)-> pd.DataFrame:
    # Test output
    # LamdaMART predicts real-valued 'RELEVANCE' independent of grouping
    #   given X_test (without search_id) it predicts relevance for each row
    #   each query can then be sorted by this predicted relevance
    TESTDF = pd.read_csv('test_set_VU_DM.csv')
    # add relevant information from date-time 
    TESTDF = preprocess_df(TESTDF, TEST=True)
    TESTDF = add_features(TESTDF)
    
    # Produce relevance predictions for each row
    X_test = TESTDF.drop(columns=['srch_id'])
    predicted_relevance = model.predict(X_test)

    # Add predictions to dataset
    TESTDF['predicted_relevance'] = predicted_relevance

    # Produce ranking predictions for each search ID
    RANKED = TESTDF.sort_values(['srch_id', 'predicted_relevance'], 
                                # from search id 0, ...; within each search id FROM highest relevance)
                                ascending=[True, False]).reset_index(drop=True)

    return RANKED[['srch_id', 'prop_id']] 


if __name__ == '__main__':
    LambdaMART = train_model(plot=True)
    TEST_pred = get_predictions(LambdaMART)
    TEST_pred.to_csv('VU-DM-2025-Group-100.csv', index=False)