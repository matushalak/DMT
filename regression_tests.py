import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import json
import shutil

from ML_utils import normalize_data_and_split, plot_predictions

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
# Flag to control hyperparameter tuning
tune_hyperparameters = False  # set to False to disable tuning

# Datasets to analyze
dataset_names = ["df_ready_date"]

# Feature selection approaches
approaches = {
    "baseline": {"description": "Single feature baseline model"},
    "top_correlated": {"description": "Top correlated features", "n_features": 35},
    "all_features": {"description": "All features"}
}

# Base directory for results
results_base_dir = "results/regression"

# ------------------------------------------------------
# Create required directories for plots and results
# ------------------------------------------------------
def create_directories(dataset_name, approach):
    """Create required directories for plots and results"""
    base_dir = os.path.join(results_base_dir, dataset_name, approach)
    
    directories = {
        'plots': os.path.join(base_dir, 'plots'),
        'feature_importances': os.path.join(base_dir, 'plots', 'feature_importances'),
        'predictions': os.path.join(base_dir, 'plots', 'predictions'),
        'metrics': os.path.join(base_dir, 'plots', 'metrics'),
        'visualizations': os.path.join(base_dir, 'plots', 'visualizations'),
        'timeseries': os.path.join(base_dir, 'plots', 'timeseries')
    }
    
    for directory in directories.values():
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    return directories

# ------------------------------------------------------
# Define models and parameter grids
# ------------------------------------------------------
def get_models_and_param_grids():
    """Define regression models and their parameter grids"""
    models = {
        'Linear Regression': LinearRegression(), 
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(), 
        'Elastic Net': ElasticNet(), 
        'Decision Tree': DecisionTreeRegressor(), 
        'Random Forest': RandomForestRegressor(), 
        'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', eval_metric='rmse'),
        'Gradient Boosting': GradientBoostingRegressor(),
        'SVR': SVR()
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
        
        'Decision Tree': {
            'max_depth': [2, 3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        
        'XGBoost Regressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 3, 4],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0.1, 1.0, 10.0]
        },
        
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [2, 3, 4],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 0.9, 1.0]
        },
        
        'SVR': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'epsilon': [0.05, 0.1, 0.2]
        }
    }
    
    return models, param_grids

# ------------------------------------------------------
# Helper function to format small numbers for better readability
# ------------------------------------------------------
def format_value(value):
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"

# ------------------------------------------------------
# Function to tune model hyperparameters
# ------------------------------------------------------
def tune_model_hyperparameters(models, param_grids, X_train, y_train):
    """Tune hyperparameters for each model if enabled"""
    if not tune_hyperparameters:
        return models
        
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
    
    return tuned_models

# ------------------------------------------------------
# Function to train and evaluate models
# ------------------------------------------------------
def evaluate_models(models, X_train, X_val, X_test, y_train, y_val, y_test, dataset_name):
    """Train and evaluate models, return results"""
    results = {}
    
    for name, model in models.items():
        # Fit the model if not already fitted by GridSearchCV
        if not hasattr(model, 'predict'):
            model.fit(X_train, y_train)
        
        # Predictions on all sets
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'R2_train': r2_score(y_train, y_pred_train),
            'R2_val': r2_score(y_val, y_pred_val),
            'R2_test': r2_score(y_test, y_pred_test),
            
            'MAE_train': mean_absolute_error(y_train, y_pred_train),
            'MAE_val': mean_absolute_error(y_val, y_pred_val),
            'MAE_test': mean_absolute_error(y_test, y_pred_test),
            
            'MSE_train': mean_squared_error(y_train, y_pred_train),
            'MSE_val': mean_squared_error(y_val, y_pred_val),
            'MSE_test': mean_squared_error(y_test, y_pred_test),
            
            'RMSE_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'RMSE_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'RMSE_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            
            'dataset': dataset_name
        }
    
    return results

# ------------------------------------------------------
# Function to plot metrics
# ------------------------------------------------------
def plot_metrics(results_df, metrics_path):
    """Plot various metrics comparison charts"""
    x = np.arange(len(results_df.index))  # label locations
    width = 0.35  # width of the bars

    # R2 Scores
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/3, results_df['R2_train'], width/3, label='Train R2')
    plt.bar(x, results_df['R2_val'], width/3, label='Val R2')
    plt.bar(x + width/3, results_df['R2_test'], width/3, label='Test R2')
    plt.title('Model R2 Scores')
    plt.xlabel('Model')
    plt.ylabel('R2 Score')
    plt.xticks(x, results_df.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_path, 'r2_scores.png'))
    plt.close()

    # MAE Scores
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/3, results_df['MAE_train'], width/3, label='Train MAE')
    plt.bar(x, results_df['MAE_val'], width/3, label='Val MAE')
    plt.bar(x + width/3, results_df['MAE_test'], width/3, label='Test MAE')
    plt.title('Model MAE Scores')
    plt.xlabel('Model')
    plt.ylabel('MAE Score')
    plt.xticks(x, results_df.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_path, 'mae_scores.png'))
    plt.close()

    # MSE Scores
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/3, results_df['MSE_train'], width/3, label='Train MSE')
    plt.bar(x, results_df['MSE_val'], width/3, label='Val MSE')
    plt.bar(x + width/3, results_df['MSE_test'], width/3, label='Test MSE')
    plt.title('Model MSE Scores')
    plt.xlabel('Model')
    plt.ylabel('MSE Score')
    plt.xticks(x, results_df.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_path, 'mse_scores.png'))
    plt.close()

    # RMSE Scores
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/3, results_df['RMSE_train'], width/3, label='Train RMSE')
    plt.bar(x, results_df['RMSE_val'], width/3, label='Val RMSE')
    plt.bar(x + width/3, results_df['RMSE_test'], width/3, label='Test RMSE')
    plt.title('Model RMSE Scores')
    plt.xlabel('Model')
    plt.ylabel('RMSE Score')
    plt.xticks(x, results_df.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_path, 'rmse_scores.png'))
    plt.close()

# ------------------------------------------------------
# Function to plot model predictions
# ------------------------------------------------------
def plot_model_predictions(
    models, X_train, X_val, X_test, y_train, y_val, y_test, 
    results, predictions_path, timeseries_path, metadata=None
):
    """Plot prediction scatter plots and metrics for each model"""
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
        # Get metrics from results dictionary
        r2_train = results[name]['R2_train']
        r2_val = results[name]['R2_val']
        r2_test = results[name]['R2_test']
        
        mae_train = results[name]['MAE_train']
        mae_val = results[name]['MAE_val']
        mae_test = results[name]['MAE_test']
        
        mse_train = results[name]['MSE_train']
        mse_val = results[name]['MSE_val']
        mse_test = results[name]['MSE_test']
        
        rmse_train = results[name]['RMSE_train']
        rmse_val = results[name]['RMSE_val']
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
        # ax2.bar(r2, val_values, width=bar_width, label='Validation', color='green')
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

        # Generate timeseries plots if metadata is available
        if metadata is not None:
            save_path = os.path.join(timeseries_path, f'{name}_timeseries')

            # plot_predictions(
            #     y_train, 
            #     y_pred_train, 
            #     metadata, 
            #     dataset='train', 
            #     title=f'{name} Train Predictions (All Participants)',
            #     save_path=f"{save_path}_train_combined",
            #     show_plot=False
            # )
            
            # plot_predictions(
            #     y_test, 
            #     y_pred_test, 
            #     metadata, 
            #     dataset='test', 
            #     title=f'{name} Test Predictions (All Participants)',
            #     save_path=f"{save_path}_test_combined",
            #     show_plot=False
            # )

# ------------------------------------------------------
# Function to analyze feature importance
# ------------------------------------------------------
def analyze_feature_importance(models, feature_names, feature_importances_path):
    """Analyze and plot feature importance for models that support it"""
    importances = {}
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances[name] = model.feature_importances_[:10]
        elif hasattr(model, 'coef_'):
            importances[name] = model.coef_
        else:
            importances[name] = None
        
        # select top 10 features
        

    # For models with valid feature importance, plot them
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
            
            # Print top features
            indices = np.argsort(imp)[::-1]  # Sort in descending order
            print(f"Top features for {name}:")
            for i in range(min(10, len(indices))):  # Display top 10 features
                print(f"{feature_names[indices[i]]}: {format_value(imp[indices[i]])}")
            
            # Print also the list of top features
            print("\nFeature list by importance:")
            try:
                print(f"{[feature_names[i] for i in indices]}")
            except IndexError:
                print("Error: Feature names do not match the indices of feature importances.")
            print("\n")
        else:
            print(f'{name} does not have feature importances.')
    
    return importances

# ------------------------------------------------------
# Function to save results to JSON
# ------------------------------------------------------
def save_results_to_json(results, dataset_name, approach, directories):
    """Save results dictionary to JSON file"""
    # Convert numpy values to Python native types for JSON serialization
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {
            k: float(v) if isinstance(v, (np.float32, np.float64)) else v
            for k, v in metrics.items()
        }
    
    # Save to JSON file
    results_file = os.path.join(directories['plots'], f'results_{dataset_name}_{approach}.json')
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    return results_file

# ------------------------------------------------------
# Function to run full analysis for one dataset with specific feature selection
# ------------------------------------------------------
def run_analysis(dataset_name, approach, n_correlated=15):
    """Run complete analysis for a dataset with specified feature selection approach"""
    print(f"\n{'='*80}")
    print(f"ANALYZING DATASET: {dataset_name} with approach: {approach}")
    print(f"{'='*80}\n")
    
    # Load data
    df = pd.read_csv(f'tables/imputed/{dataset_name}.csv')
    print(f"Loaded dataset {dataset_name} with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Create directories
    directories = create_directories(dataset_name, approach)
    
    # Select features based on approach
    features = None
    df_numeric = df.select_dtypes(include=[np.number])
    
    if approach == "baseline":
        # Use a single feature for baseline (mood_mean_daily if available, otherwise the most correlated with target)
        correlations = df_numeric.corr()
        target_correlations = correlations["target"].sort_values(ascending=False)
        
        if "mood_mean_daily" in df.columns:
            features = ["mood_mean_daily"]
            print("Using mood_mean_daily as baseline feature")
        else:
            # Use the highest correlated feature (excluding target itself)
            top_feature = target_correlations.index[1]  # Index 0 is the target itself
            features = [top_feature]
            print(f"Using highest correlated feature as baseline: {top_feature}")

    
    elif approach == "top_correlated":
        # Use top N absolutely correlated features
        correlations = df_numeric.corr()
        target_correlations = correlations["target"].abs().sort_values(ascending=False)
        
        # Remove id_num and categorical_target
        target_correlations = target_correlations.drop(['id_num', 'categorical_target'], errors='ignore')
        
        # Skip the target itself (it's the first item)
        top_features = target_correlations.index[1:n_correlated+1].tolist()
        features = top_features
        print(f"Using top {n_correlated} absolutely correlated features: {features}")
    
    elif approach == "all_features":
        # Use all features except those that should be excluded
        features = df.columns.to_list()
        
        # Remove features that should not be used in the model
        exclude_cols = ['target', 'categorical_target', 'date', 'next_date', "id_num", "categorical_target"]
        
        # Add any time-related non-encoded columns to exclude
        if "time_of_day_non_encoded" in features:
            exclude_cols.append("time_of_day_non_encoded")
        
        # Process appCat features if present
        appcat_cols = [col for col in features if 'appCat.' in col]
        if appcat_cols:
            print("Processing appCat features...")
            df['appCat'] = df[appcat_cols].sum(axis=1)
            exclude_cols.extend(appcat_cols)
        
        # Remove excluded columns from features
        features = [f for f in features if f not in exclude_cols]
        print(f"Using all {len(features)} features after excluding {exclude_cols}")
    
    # Normalize data and split
    X_train, X_val, X_test, y_train, y_val, y_test, scalers, metadata = normalize_data_and_split(
        df,
        features=features,
        target_col="target",
        id_col='id_num',
        timestamp_col='date',  # For timeseries plotting
        per_participant_normalization=True,
        scaler_type="StandardScaler",
        test_size=0.1,
        val_size=0.0001,
        random_state=42
    )
    
    # Store feature names
    feature_names = X_train.columns.tolist()
    
    # Get models and parameter grids
    models, param_grids = get_models_and_param_grids()
    
    # Tune hyperparameters if enabled
    if tune_hyperparameters:
        models = tune_model_hyperparameters(models, param_grids, X_train, y_train)
    else:
        # Fit models
        for name, model in models.items():
            print(f"Fitting {name}...")
            model.fit(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_train, X_val, X_test, y_train, y_val, y_test, dataset_name)
    
    # Create DataFrame for results and sort by test R2 score
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by='R2_test', ascending=False)
    print("\nModel performance summary:")
    print(results_df[['R2_train', 'R2_val', 'R2_test', 'MAE_test', 'RMSE_test']])
    
    # Plot metrics
    plot_metrics(results_df, directories['metrics'])
    
    # Plot model predictions
    plot_model_predictions(
        models, X_train, X_val, X_test, y_train, y_val, y_test, 
        results, directories['predictions'], directories['timeseries'], metadata
    )
    
    # Analyze feature importance
    importances = analyze_feature_importance(models, feature_names, directories['feature_importances'])
    
    # Save results to JSON
    results_file = save_results_to_json(results, dataset_name, approach, directories)
    
    # Save a copy of results DataFrame
    results_df_file = os.path.join(directories['plots'], f'results_df_{dataset_name}_{approach}.csv')
    results_df.to_csv(results_df_file)
    
    return {
        'results_df': results_df,
        'results_file': results_file,
        'approach': approach,
        'dataset_name': dataset_name
    }

# ------------------------------------------------------
# Function to compare and summarize all approaches
# ------------------------------------------------------
def compare_approaches(all_results):
    """Compare and summarize results across all datasets and approaches"""
    # Create comparison directory
    comparison_dir = os.path.join(results_base_dir, 'comparison')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Group results by dataset
    datasets = {}
    for result in all_results:
        dataset_name = result['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(result)
    
    # For each dataset, create a comparison of approaches
    for dataset_name, results_list in datasets.items():
        print(f"\n{'='*80}")
        print(f"COMPARING APPROACHES FOR DATASET: {dataset_name}")
        print(f"{'='*80}\n")
        
        # Create a combined DataFrame with the best model from each approach
        best_models = []
        for result in results_list:
            approach = result['approach']
            results_df = result['results_df']
            
            # Get the best model for this approach
            best_model_row = results_df.iloc[0].copy()  # Top row is best model by R2_test
            best_model_name = results_df.index[0]
            
            # Add approach information
            best_model_row['approach'] = approach
            best_model_row['model_name'] = best_model_name
            
            best_models.append(best_model_row)
        
        # Create DataFrame of best models
        best_models_df = pd.DataFrame(best_models)
        best_models_df = best_models_df.sort_values(by='R2_test', ascending=False)
        
        # Save to CSV
        comparison_file = os.path.join(comparison_dir, f'{dataset_name}_approach_comparison.csv')
        best_models_df.to_csv(comparison_file)
        
        # Print the comparison
        print(f"\nBest model from each approach for {dataset_name}:")
        comparison_columns = ['approach', 'model_name', 'R2_test', 'MAE_test', 'RMSE_test']
        print(best_models_df[comparison_columns])
        
        # Plot comparison of approaches
        plt.figure(figsize=(14, 8))
        
        # Metrics to compare
        metrics = ['R2_test', 'MAE_test', 'RMSE_test']
        
        # Set up the plot
        barWidth = 0.25
        r1 = np.arange(len(best_models_df))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        
        # Create bars
        plt.bar(r1, best_models_df['R2_test'], width=barWidth, label='R2 (higher is better)', color='green')
        plt.bar(r2, best_models_df['MAE_test'], width=barWidth, label='MAE (lower is better)', color='red')
        plt.bar(r3, best_models_df['RMSE_test'], width=barWidth, label='RMSE (lower is better)', color='orange')
        
        # Add labels
        plt.xlabel('Approach', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.title(f'Comparison of Best Models Across Approaches - {dataset_name}')
        


        # Add x-axis labels using approach and model name
        labels = [f"{row['approach']}\n({row['model_name']})" for _, row in best_models_df.iterrows()]
        plt.xticks([r + barWidth for r in range(len(best_models_df))], labels, rotation=45, ha='right')
        
        # Add legend
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(comparison_dir, f'{dataset_name}_approach_comparison.png'))
        plt.close()
    
    # Compare best models across all datasets
    all_best_models = []
    for result in all_results:
        dataset_name = result['dataset_name']
        approach = result['approach']
        results_df = result['results_df']
        
        # Get the best model for this dataset and approach
        best_model_row = results_df.iloc[0].copy()
        best_model_name = results_df.index[0]
        
        # Add dataset and approach information
        best_model_row['dataset'] = dataset_name
        best_model_row['approach'] = approach
        best_model_row['model_name'] = best_model_name
        
        all_best_models.append(best_model_row)
    
    # Create DataFrame of all best models
    all_best_df = pd.DataFrame(all_best_models)
    all_best_df = all_best_df.sort_values(by=['dataset', 'R2_test'], ascending=[True, False])
    
    # Save to CSV
    all_comparison_file = os.path.join(comparison_dir, 'all_datasets_comparison.csv')
    all_best_df.to_csv(all_comparison_file)
    
    # Print the comparison
    print(f"\nBest models across all datasets and approaches:")
    all_comparison_columns = ['dataset', 'approach', 'model_name', 'R2_test', 'MAE_test', 'RMSE_test']
    print(all_best_df[all_comparison_columns])
    
    # Create overall comparison plot
    plt.figure(figsize=(16, 10))
    
    # Pivot data for grouped bar chart
    datasets_list = sorted(all_best_df['dataset'].unique())
    approaches_list = sorted(all_best_df['approach'].unique())
    
    # Set up the plot - we'll use R2 score for comparison
    barWidth = 0.25
    x_ticks = []
    x_labels = []
    
    for i, dataset in enumerate(datasets_list):
        dataset_data = all_best_df[all_best_df['dataset'] == dataset]
        
        # Starting position for this dataset group
        r = np.arange(len(approaches_list)) + (i * (len(approaches_list) + 1))
        x_ticks.extend(r + barWidth)
        
        # Add values for each approach
        for j, approach in enumerate(approaches_list):
            approach_data = dataset_data[dataset_data['approach'] == approach]
            if not approach_data.empty:
                plt.bar(r[j], approach_data['R2_test'].values[0], width=barWidth, 
                        label=f"{approach}" if i == 0 else "", 
                        color=plt.cm.tab10(j), alpha=0.7)
                
                # Add model name annotation above the bar
                model_name = approach_data['model_name'].values[0]
                plt.text(r[j], approach_data['R2_test'].values[0] + 0.01, 
                        model_name, ha='center', va='bottom', rotation=45, 
                        fontsize=8)
        
        # Add dataset label
        x_labels.append(dataset)
    
    # Add labels and title
    plt.xlabel('Dataset and Approach', fontweight='bold')
    plt.ylabel('R2 Score (higher is better)', fontweight='bold')
    plt.title('Comparison of R2 Scores Across All Datasets and Approaches')
    
    # Add x-axis labels
    dataset_positions = []
    for i, dataset in enumerate(datasets_list):
        pos = np.mean(np.arange(len(approaches_list)) + (i * (len(approaches_list) + 1)))
        dataset_positions.append(pos)
    
    plt.xticks(dataset_positions, datasets_list)
    
    # Add approach legend
    plt.legend(title="Approach")
    plt.tight_layout()
    
    # Add grid for easier reading
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(os.path.join(comparison_dir, 'all_datasets_r2_comparison.png'))
    plt.close()
    
    return comparison_dir, all_best_df

# ------------------------------------------------------
# Main function to run all analyses
# ------------------------------------------------------
def main():
    """Main function to run all analyses"""
    print(f"\n{'#'*80}")
    print(f"# REGRESSION ANALYSIS ACROSS MULTIPLE DATASETS AND FEATURE SELECTION APPROACHES")
    print(f"{'#'*80}\n")
    
    # Clear previous results if desired
    if os.path.exists(results_base_dir):
        print(f"Clearing previous results directory: {results_base_dir}")
        shutil.rmtree(results_base_dir)
    
    # Store all results
    all_results = []
    
    # Process each dataset
    for dataset_name in dataset_names:
        # Process each feature selection approach
        for approach_name, approach_config in approaches.items():
            # Get number of correlated features if applicable
            n_correlated = approach_config.get('n_features', 5) if approach_name == 'top_correlated' else None
            
            # Run analysis
            result = run_analysis(dataset_name, approach_name, n_correlated)
            all_results.append(result)
    
    # Compare all approaches
    comparison_dir, comparison_df = compare_approaches(all_results)
    
    print(f"\n{'#'*80}")
    print(f"# ANALYSIS COMPLETE")
    print(f"# Results saved to {results_base_dir}")
    print(f"# Comparison saved to {comparison_dir}")
    print(f"{'#'*80}\n")
    
    # Print overall best model
    best_overall = comparison_df.sort_values(by='R2_test', ascending=False).iloc[0]
    print(f"Best overall model:\n")
    print(f"Dataset: {best_overall['dataset']}")
    print(f"Approach: {best_overall['approach']}")
    print(f"Model: {best_overall['model_name']}")
    print(f"R2 Score: {best_overall['R2_test']:.4f}")
    print(f"MAE: {best_overall['MAE_test']:.4f}")
    print(f"RMSE: {best_overall['RMSE_test']:.4f}")

# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    main()