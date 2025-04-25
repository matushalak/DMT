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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

# import SVR and gradient boosting
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


from ML_utils import normalize_data_and_split, plot_predictions
# ------------------------------------------------------
# Define the updated normalize_data_and_split function with metadata for timeseries plotting

# ------------------------------------------------------
# Flag to control hyperparameter tuning
tune_hyperparameters = False  # set to False to disable tuning

# dataset and label details
dataset_name = "df_ready_date"
label = ''
df = pd.read_csv(f'tables/imputed/{dataset_name}.csv')
# print(df.head())
print("df columns", df.columns)



# combine all app.Categorical features into one
df['appCat'] = df[df.columns[df.columns.str.contains('appCat')]].sum(axis=1)
# Drop individual app.Categorical features
df.drop(columns=df.columns[df.columns.str.contains('appCat.')], inplace=True)

df_numeric = df.select_dtypes(include=[np.number])

# # check correlations between features and target
correlations = df_numeric.corr()
correlation_with_target = correlations["target"].sort_values(ascending=False)
correlation_with_CATtarget = correlations["categorical_target"].abs().sort_values(ascending=False)
print("Correlation with REGRESSION target:")
print(correlation_with_target)
print("Correlation with CLASSIFICATION target:")
print(correlation_with_CATtarget)



# Remove features that should not be used in the model
features = correlation_with_target.index.tolist()[:37]
features.remove('id_num')
features.remove('target')  # Make sure this matches your actual target column name
features.remove('categorical_target')  # Make sure this matches your actual target column name
# features.remove('date')  # Remove date if present
# features.remove("next_date")

if "time_of_day_non_encoded" in features:
    features.remove("time_of_day_non_encoded")

# features = ["id_num", "mood_mean_daily"]  # for baseline model

# select 15 most correlated features in absolute value

print("features", features, "count", len(features))


# Update the function call to include timestamp_col parameter
X_train, X_val, X_test, y_train, y_val, y_test, scalers, metadata = normalize_data_and_split(
    df,
    features=features,
    target_col="target",  # Make sure this matches your actual target column name
    # target_col="target",  # Make sure this matches your actual target column name
    id_col='id_num',
    timestamp_col='date',  # Add this parameter for timeseries plotting,
    per_participant_normalization=True,
    scaler_type="StandardScaler",
    test_size=0.1,
    val_size=0.00001,
    random_state=42
)
# Store feature names before fitting models
feature_names = X_train.columns.tolist()  # Use X_train instead of X

# Define base models
models = {
    'Linear Regression': LinearRegression(), 
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(), 
    'Elastic Net': ElasticNet(), 
    'Decision Tree': DecisionTreeRegressor(), 
    'Random Forest': RandomForestRegressor(), 
    'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', eval_metric='rmse'),
    'Gradient Boosting': GradientBoostingRegressor(),
    # 'SVR': SVR()
}

param_grids = {
    'Linear Regression': {},  # No hyperparameters to tune
    
    'Ridge Regression': {
        'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    
    'Lasso Regression': {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        'max_iter': [10000],
        'tol': [1e-4],
        'selection': ['cyclic', 'random']
    },
    
    'Elastic Net': {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [10000],
        'tol': [1e-4],
        'selection': ['cyclic', 'random']
    },
    
    # Given that linear models are performing better, we can simplify tree-based models
    'Decision Tree': {
        'max_depth': [2, 3, 4, 5],  # Limiting depth to prevent overfitting
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],  # More limited depth to control overfitting
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    
    'XGBoost Regressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4],  # More restricted depth
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0.1, 1.0, 10.0]
    },
    
    # Adding Gradient Boosting which can sometimes outperform XGBoost
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 0.9, 1.0]
    },
    
    # Adding SVR which might work well with your data structure
    'SVR': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'epsilon': [0.05, 0.1, 0.2]
    }
}

param_grids['Gradient Boosting'] = {
    # moderate number of trees with early‑stopping can be added via fit_params
    'n_estimators': [100, 200],

    # slower learning to require more trees to fit residuals
    'learning_rate': [0.01, 0.03],

    # very shallow trees
    'max_depth': [1, 2, 3],

    # require more samples to split and to form a leaf
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],

    # only use a fraction of samples and features per tree
    'subsample': [0.5, 0.7],
    'max_features': ['sqrt', 0.5]
}

param_grids['Random Forest'] = {
    # moderate number of trees but each built very simply
    'n_estimators': [100, 200],
    'max_depth': [3, 5],                 # shallow trees
    'min_samples_split': [20, 50],       # require many samples to split
    'min_samples_leaf': [10, 20],        # leaves cover many samples
    'max_features': ['sqrt', 0.3, 0.5],  # only a small subset of features per split
    'bootstrap': [True],
    'max_samples': [0.6, 0.8],           # each tree sees 60–80% of data
    'ccp_alpha': [0.001, 0.01]           # cost‑complexity pruning
}

param_grids['XGBoost Regressor'] = {
    # fewer trees (since we’ll use very small learning rates)
    'n_estimators': [100, 200],

    # exceptionally shallow trees
    'max_depth': [1, 2],

    # ultra‑slow learning
    'learning_rate': [0.001, 0.005],

    # heavier row/column subsampling
    'subsample': [0.3, 0.5],
    'colsample_bytree': [0.3, 0.5],

    # require large loss reduction for any split
    'gamma': [0.5, 1.0],

    # ensure each leaf has many observations
    'min_child_weight': [10, 20],

    # very strong regularization
    'reg_alpha': [5.0, 10.0],
    'reg_lambda': [50.0, 100.0],

    # constrain weight updates per leaf
    'max_delta_step': [1, 2]
}

param_grids['Gradient Boosting'] = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05],       # slower learning
    'max_depth': [2, 3],                 # shallow trees
    'min_samples_split': [10, 20],       # require more samples to split
    'min_samples_leaf': [5, 10],         # larger leaves
    'subsample': [0.6, 0.8],             # row sampling per tree
    'max_features': [0.5, 'sqrt']        # feature subsetting per split
}


# If hyperparameter tuning is enabled, replace the models with tuned versions
if tune_hyperparameters:
    tuned_models = {}
    for name, model in models.items():
        print(f"Tuning hyperparameters for {name}...")
        grid = param_grids.get(name, {})
        if grid:  # if there's a non-empty grid
            gs = GridSearchCV(model, grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
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
    if not tune_hyperparameters or not hasattr(model, 'predict'):
        model.fit(X_train, y_train)
    
    # Predictions on test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Predictions on training set
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
results_df = results_df.sort_values(by='MSE_test', ascending=True)
print(results_df)

# ------------------------------------------------------
# Create required directories for plots
plots_path = 'plots'
feature_importances_path = os.path.join(plots_path, 'feature_importances')
predictions_path = os.path.join(plots_path, 'predictions')
metrics_path = os.path.join(plots_path, 'metrics')
visualizations_path = os.path.join(plots_path, 'visualizations')
timeseries_path = os.path.join(plots_path, 'timeseries')  # New directory for timeseries plots

for directory in [plots_path, feature_importances_path, predictions_path, metrics_path, visualizations_path, timeseries_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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
plt.savefig(os.path.join(metrics_path, 'r2_scores.png'))
plt.close()

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
plt.savefig(os.path.join(metrics_path, 'mae_scores.png'))
plt.close()

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
plt.savefig(os.path.join(metrics_path, 'mse_scores.png'))
plt.close()

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
plt.savefig(os.path.join(metrics_path, 'rmse_scores.png'))
plt.close()

# ------------------------------------------------------
# Plot predictions scatter plots for each model with train and validation data
for name, model in models.items():
    # Get predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Create a figure with 2 subplots: one for predictions, one for metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Scatter plot of predictions
    ax1.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label='Train')
    # ax1.scatter(y_val, y_pred_val, color='green', alpha=0.5, label='Validation')
    ax1.scatter(y_test, y_pred_test, color='red', alpha=0.5, label='Test')
    
    ax1.set_title(f'{name} Predictions')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    
    # Add the perfect prediction line
    min_val = min(min(y_test), min(y_pred_test), min(y_train), min(y_pred_train), min(y_val), min(y_pred_val))
    max_val = max(max(y_test), max(y_pred_test), max(y_train), max(y_pred_train), max(y_val), max(y_pred_val))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.legend()
    
    # Plot 2: Metrics comparison between train, validation, and test
    # Calculate metrics for validation set
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mse_val)
    
    # Get metrics from results dictionary for train and test
    r2_train = results[name]['R2_train']
    mae_train = results[name]['MAE_train']
    mse_train = results[name]['MSE_train']
    rmse_train = results[name]['RMSE_train']
    
    r2_test = results[name]['R2_test']
    mae_test = results[name]['MAE_test']
    mse_test = results[name]['MSE_test']
    rmse_test = results[name]['RMSE_test']
    
    # Set up bar chart data
    metrics = ['R2', 'MAE', 'MSE', 'RMSE']
    train_values = [r2_train, mae_train, mse_train, rmse_train]
    val_values = [r2_val, mae_val, mse_val, rmse_val]
    test_values = [r2_test, mae_test, mse_test, rmse_test]
    
    # Bar positions
    bar_width = 0.25
    r1 = np.arange(len(metrics))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    ax2.bar(r1, train_values, width=bar_width, label='Train', color='blue')
    ax2.bar(r2, val_values, width=bar_width, label='Validation', color='green')
    ax2.bar(r3, test_values, width=bar_width, label='Test', color='red')
    
    # Add labels and legend
    ax2.set_title(f'{name} Performance Metrics')
    ax2.set_xticks([r + bar_width for r in range(len(metrics))])
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Value')
    ax2.legend()
    
    # Add a log scale for better visualization if needed
    if any(val > 10 * min(filter(lambda x: x > 0, train_values + val_values + test_values)) for val in train_values + val_values + test_values):
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(predictions_path, f'{name}_predictions_and_metrics.png'))
    plt.close()

    # NEW: Generate timeseries plots for each model
    # Plot timeseries predictions
    save_path = os.path.join(timeseries_path, f'{name}_timeseries')

    plot_predictions(
        y_train, 
        y_pred_train, 
        metadata, 
        dataset='train', 
        title=f'{name} Train Predictions (All Participants)',
        save_path=f"{save_path}_train_combined",
        show_plot=False
    )
    
    # Plot validation set predictions
    plot_predictions(
        y_test, 
        y_pred_test, 
        metadata, 
        dataset='test', 
        title=f'{name} Test Predictions (All Participants)',
        save_path=f"{save_path}_test_combined",
        show_plot=False
    )

# ------------------------------------------------------
# Feature Importance analysis
importances = {}
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances[name] = model.coef_
    else:
        importances[name] = None

# For models with valid feature importance, plot them.
for name, importance in importances.items():
    if importance is not None:
        imp = np.array(importance).flatten()
        plt.figure(figsize=(12, 8))
        # Sort features by importance
        indices = np.argsort(imp)
        plt.barh(range(len(indices)), imp[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(f'{name} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(feature_importances_path, f'{name}_feature_importance.png'))
        plt.close()
    else:
        print(f'{name} does not have feature importances.')



# ------------------------------------------------------
# Helper function to format small numbers for better readability
def format_value(value):
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"

# Create a combined visualization with all model metrics
# First, set up the figure based on number of models
num_models = len(models)
# Calculate grid dimensions
cols = min(3, num_models)  # Maximum 3 columns
rows = (num_models + cols - 1) // cols  # Ceiling division

plt.figure(figsize=(6*cols, 5*rows))

# Plot all models' metrics in a grid
for i, (name, model) in enumerate(models.items()):
    ax = plt.subplot(rows, cols, i+1)
    
    # Get predictions for all datasets
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    metrics = {
        'Train': {
            'R²': r2_score(y_train, y_pred_train),
            'MAE': mean_absolute_error(y_train, y_pred_train),
            'MSE': mean_squared_error(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train))
        },
        'Val': {
            'R²': r2_score(y_val, y_pred_val),
            'MAE': mean_absolute_error(y_val, y_pred_val),
            'MSE': mean_squared_error(y_val, y_pred_val),
            'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val))
        },
        'Test': {
            'R²': r2_score(y_test, y_pred_test),
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'MSE': mean_squared_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
    }



# display most important features for each model
for name, importance in importances.items():
    if importance is not None:
        imp = np.array(importance).flatten()
        indices = np.argsort(imp)[::-1]  # Sort in descending order
        print(f"Top features for {name}:")
        for i in range(min(10, len(indices))):  # Display top 10 features
            print(f"{feature_names[indices[i]]}: {format_value(imp[indices[i]])}")
        # print also the list of top features
        print("\n")
        try:
            print(f"Feature list: {[feature_names[i] for i in indices]}")
        except IndexError:
            print("Error: Feature names do not match the indices of feature importances.")
        print("\n")
    else:
        print(f'{name} does not have feature importances.')