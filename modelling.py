 #%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


# ------------------------------------------------------
# Flag to control hyperparameter tuning
tune_hyperparameters = True  # set to False to disable tuning

# dataset and label details
dataset_name = "mean_mode_imputation_combinedAppCat"
label = 'combined_app_cat_mean_impute'
df = pd.read_csv(f'tables/preprocessed/{dataset_name}.csv')
print(df.head())

# For this example, we are only using the 'mood' as input and 'next_day_mood' as target.


# X = df["mood"].values.reshape(-1, 1) # if you want to use mood as a single feature
X = df.drop(columns=['next_day_mood', 'id_num', 'next_day', "day"])  # drop non-feature columns
y = df['next_day_mood']

print("X features:", X.columns)
print("y target:", y.name)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base models
models = {
    'Linear Regression': LinearRegression(), 
    'Ridge Regression': Ridge(), 
    'Lasso Regression': Lasso(), 
    'Elastic Net': ElasticNet(), 
    'Decision Tree': DecisionTreeRegressor(), 
    'Random Forest': RandomForestRegressor(), 
    'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
}

# Define parameter grids for hyperparameter tuning
param_grids = {
    'Linear Regression': {},  # No hyperparameters to tune for LinearRegression in most cases
    'Ridge Regression': {
        'alpha': [0.1, 1.0, 10.0, 50.0]
    },
    'Lasso Regression': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
    },
    'Elastic Net': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    },
    'Decision Tree': {
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    },
    'XGBoost Regressor': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]

    }
}

# If hyperparameter tuning is enabled, replace the models with tuned versions
if tune_hyperparameters:
    tuned_models = {}
    for name, model in models.items():
        print(f"Tuning hyperparameters for {name}...")
        grid = param_grids.get(name, {})
        if grid:  # if there's a non-empty grid
            gs = GridSearchCV(model, grid, cv=3, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)
            tuned_models[name] = gs.best_estimator_
            print(f"Best parameters for {name}: {gs.best_params_}")
        else:
            # If no grid is provided, use the base model
            model.fit(X_train, y_train)
            tuned_models[name] = model
    models = tuned_models

# Evaluate each model and collect results
results = {}
for name, model in models.items():
    # Fit the model if not already fitted by GridSearchCV
    if not hasattr(model, 'predict'):  # safety check, though GridSearchCV returns a fitted estimator
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)

    results[name] = {
        'R2_test': r2,
        'R2_train': r2_train,
        'MAE_test': mae,
        'MAE_train': mae_train,
        'MSE_test': mse,
        'MSE_train': mse_train,
        'RMSE_test': rmse,
        'RMSE_train': rmse_train,
        'dataset': dataset_name
    }

# Create a DataFrame for the results and sort by test R2 score
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='R2_test', ascending=False)
print(results_df)

# ------------------------------------------------------
# Plotting results

x = np.arange(len(results_df.index))  # label locations
width = 0.35  # width of the bars

# R2 Scores
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results_df['R2_train'], width, label='Train R2')
plt.bar(x + width/2, results_df['R2_test'], width, label='Test R2')
plt.title('Model R2 Scores')
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.xticks(x, results_df.index, rotation=45)
plt.legend()
plt.tight_layout()
# plt.show()

# MAE Scores
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results_df['MAE_train'], width, label='Train MAE')
plt.bar(x + width/2, results_df['MAE_test'], width, label='Test MAE')
plt.title('Model MAE Scores')
plt.xlabel('Model')
plt.ylabel('MAE Score')
plt.xticks(x, results_df.index, rotation=45)
plt.legend()
plt.tight_layout()
# plt.show()

# MSE Scores
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results_df['MSE_train'], width, label='Train MSE')
plt.bar(x + width/2, results_df['MSE_test'], width, label='Test MSE')
plt.title('Model MSE Scores')
plt.xlabel('Model')
plt.ylabel('MSE Score')
plt.xticks(x, results_df.index, rotation=45)
plt.legend()
plt.tight_layout()
# plt.show()

# RMSE Scores
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results_df['RMSE_train'], width, label='Train RMSE')
plt.bar(x + width/2, results_df['RMSE_test'], width, label='Test RMSE')
plt.title('Model RMSE Scores')
plt.xlabel('Model')
plt.ylabel('RMSE Score')
plt.xticks(x, results_df.index, rotation=45)
plt.legend()
plt.tight_layout()
# plt.show()

# Plot predictions scatter plots for each model
for name, model in models.items():
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.title(f'{name} Predictions')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlim(y.min(), y.max())
    plt.ylim(y.min(), y.max())
    plt.tight_layout()
    # plt.show()Đ

# ------------------------------------------------------
# Feature Importance

feature_names = X.columns  # assuming X is a DataFrame

importances = {}
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances[name] = model.coef_
    else:
        importances[name] = None

# For models with valid feature importance, plot them.
# (Use the original DataFrame columns where applicable.)

for name, importance in importances.items():
    if importance is not None:
        imp = np.array(importance).flatten()
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, imp)  # use the actual feature names
        plt.title(f'{name} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    else:
        print(f'{name} does not have feature importances.')

# ------------------------------------------------------
# Save results to CSV
results_path = 'tables/results'
if not os.path.exists(results_path):
    os.makedirs(results_path)
results_df.to_csv(os.path.join(results_path, 'model_results.csv'), index=True, header=True)
# %%
