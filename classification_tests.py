import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import os
import shutil
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

# Import SVC and gradient boosting
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from ML_utils import normalize_data_and_split, plot_predictions

# ------------------------------------------------------
# Flag to control hyperparameter tuning
tune_hyperparameters = False  # Set to False to disable tuning

# Datasets to process
datasets = ["df_ready_date"]

# Feature selection approaches
feature_selections = {
    "baseline": ["mood_mean_daily"],  # Baseline with single feature
    "top_features": None,  # Will be set dynamically based on correlations
    "all_features": None,  # Will be set dynamically based on all available features
}

# Number of top correlated features to use
n_top_features = 35

# Base directory for results
results_base_dir = 'results/classification'
if not os.path.exists(results_base_dir):
    os.makedirs(results_base_dir)

# Create results DataFrame to store all experiment results
all_results_df = pd.DataFrame()

# Function to create directory structure for each experiment
def create_experiment_dirs(dataset_name, feature_selection_name):
    experiment_dir = os.path.join(results_base_dir, dataset_name, feature_selection_name)
    
    # Create main experiment directory
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # Create subdirectories for plots and other outputs
    plots_path = os.path.join(experiment_dir, 'plots')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    
    subdirs = ['feature_importances', 'predictions', 'metrics', 'visualizations', 
               'timeseries', 'confusion_matrices']
    
    for subdir in subdirs:
        path = os.path.join(plots_path, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
    
    return experiment_dir, plots_path

# Function for feature importance plotting
def plot_feature_importance(model, feature_names, model_name, save_path):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For multi-class classification, average across classes
        if len(model.coef_.shape) > 1:
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            importance = model.coef_
    else:
        print(f'{model_name} does not have feature importances.')
        return None
    
    # Sort features by importance
    importance = np.array(importance).flatten()
    indices = np.argsort(importance)[::-1]  # Sort in descending order
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(f'{model_name} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return importance, indices

# Helper function to format small numbers for better readability
def format_value(value):
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"

# Process each dataset
for dataset_name in datasets:
    print(f"\n{'='*50}\nProcessing dataset: {dataset_name}\n{'='*50}")
    
    # Load dataset
    df = pd.read_csv(f'tables/imputed/{dataset_name}.csv')
    
    # Transform the categorical target from [-1, 0, 1] to [0, 1, 2]
    target_mapping = {-1: 0, 0: 1, 1: 2}
    df['categorical_target'] = df['categorical_target'].map(target_mapping)

    df['appCat'] = df[df.columns[df.columns.str.contains('appCat')]].sum(axis=1)
    # Drop individual app.Categorical features
    df.drop(columns=df.columns[df.columns.str.contains('appCat.')], inplace=True)
            
    # Get numeric columns for correlation analysis
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Check correlations between features and target
    correlations = df_numeric.corr()
    correlation_with_target = correlations["target"].sort_values(ascending=False)
    correlation_with_CATtarget = correlations["categorical_target"].sort_values(ascending=False)
    
    print("Correlation with REGRESSION target:")
    print(correlation_with_target)
    print("\nCorrelation with CLASSIFICATION target:")
    print(correlation_with_CATtarget)
        
    # Prepare all potential features
    all_features = df.columns.to_list()
    
    # Remove features that should naot be used in models
    for col in ['id_num', 'target', 'categorical_target', 'date', 'next_date', 'time_of_day_non_encoded']:
        if col in all_features:
            all_features.remove(col)
    
    # Update feature selection options with all available features
    feature_selections["all_features"] = all_features
    
    # Get top N correlated features with categorical target
    top_features = correlation_with_CATtarget.drop(['target', 'categorical_target'], errors='ignore')
    top_features = top_features.abs().sort_values(ascending=False).index[:n_top_features].tolist()
    feature_selections["top_features"] = top_features
    
    print(f"\nTop {n_top_features} correlated features:")
    print(top_features)
    
    # Process each feature selection approach
    for selection_name, features in feature_selections.items():
        print(f"\n{'-'*40}\nFeature selection: {selection_name}\nFeatures: {features}\n{'-'*40}")
        
        # Create directory structure for this experiment
        experiment_dir, plots_path = create_experiment_dirs(dataset_name, selection_name)
        
        # Normalize and split data using selected features
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, metadata = normalize_data_and_split(
            df,
            features=features,
            target_col="categorical_target",
            id_col='id_num',
            timestamp_col='date',
            per_participant_normalization=True,
            scaler_type="MinMaxScaler",
            test_size=0.1,
            val_size=0.000000000001,
            random_state=42
        )
        
        # Store feature names
        feature_names = X_train.columns.tolist()
        
        # Define classification models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=10000, multi_class='multinomial'), 
            'Decision Tree': DecisionTreeClassifier(), 
            'Random Forest': RandomForestClassifier(), 
            'XGBoost Classifier': XGBClassifier(objective='multi:softmax', eval_metric='mlogloss'),
            'Gradient Boosting': GradientBoostingClassifier(),
            'SVC': SVC(probability=True)
        }
        
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'saga'],
                'penalty': ['l2', 'None']
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
            
            'XGBoost Classifier': {
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
            
            'SVC': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01],
            }
        }
        
        # If hyperparameter tuning is enabled, tune models
        if tune_hyperparameters:
            tuned_models = {}
            for name, model in models.items():
                print(f"Tuning hyperparameters for {name}...")
                grid = param_grids.get(name, {})
                if grid:
                    gs = GridSearchCV(model, grid, cv=3, scoring='accuracy', n_jobs=-1)
                    gs.fit(X_train, y_train)
                    tuned_models[name] = gs.best_estimator_
                    print(f"Best parameters for {name}: {gs.best_params_}")
                else:
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
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Predictions on training set
            y_pred_train = model.predict(X_train)
            accuracy_train = accuracy_score(y_train, y_pred_train)
            precision_train = precision_score(y_train, y_pred_train, average='weighted')
            recall_train = recall_score(y_train, y_pred_train, average='weighted')
            f1_train = f1_score(y_train, y_pred_train, average='weighted')
            
            results[name] = {
                'Dataset': dataset_name,
                'Feature_Selection': selection_name,
                'Accuracy_test': accuracy,
                'Accuracy_train': accuracy_train,
                'Precision_test': precision,
                'Precision_train': precision_train,
                'Recall_test': recall,
                'Recall_train': recall_train,
                'F1_test': f1,
                'F1_train': f1_train
            }
        
        # Create a DataFrame for the results and sort by test accuracy
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values(by='Accuracy_test', ascending=False)
        
        # Append to the overall results
        all_results_df = pd.concat([all_results_df, results_df])
        
        # Save results to CSV
        results_df.to_csv(os.path.join(experiment_dir, 'model_performance.csv'))
        
        print(f"\nPerformance metrics for {dataset_name} with {selection_name} features:")
        print(results_df)
        
        # ------------------------------------------------------
        # Plotting results
        
        # Ensure directories exist
        for subdir in ['metrics', 'predictions', 'confusion_matrices', 'feature_importances']:
            path = os.path.join(plots_path, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        # Plot model performance metrics
        metrics_to_plot = [
            ('Accuracy', ['Accuracy_train', 'Accuracy_test']),
            ('Precision', ['Precision_train', 'Precision_test']),
            ('Recall', ['Recall_train', 'Recall_test']),
            ('F1', ['F1_train', 'F1_test'])
        ]
        
        for metric_name, columns in metrics_to_plot:
            x = np.arange(len(results_df.index))  # Label locations
            width = 0.35  # Width of the bars
            
            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, results_df[columns[0]], width, label=f'Train {metric_name}')
            plt.bar(x + width/2, results_df[columns[1]], width, label=f'Test {metric_name}')
            plt.title(f'Model {metric_name} Scores ({dataset_name}, {selection_name})')
            plt.xlabel('Model')
            plt.ylabel(f'{metric_name} Score')
            plt.xticks(x, results_df.index, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_path, 'metrics', f'{metric_name.lower()}_scores.png'))
            plt.close()
        
        # Plot confusion matrices for each model
        for name, model in models.items():
            # Get predictions
            y_pred_test = model.predict(X_test)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {name}')
            plt.colorbar()
            tick_marks = np.arange(3)  # Assuming 3 classes (0, 1, 2)
            plt.xticks(tick_marks, ['Class 0', 'Class 1', 'Class 2'])
            plt.yticks(tick_marks, ['Class 0', 'Class 1', 'Class 2'])
            
            # Add text annotations in the confusion matrix
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]),
                           horizontalalignment="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(plots_path, 'confusion_matrices', f'{name}_confusion_matrix.png'))
            plt.close()
        
        # Plot classification reports for each model
        for name, model in models.items():
            # Get predictions for all datasets
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Create a figure with 2 subplots: one for confusion matrix, one for metrics
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Plot 1: Confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax1.set_title(f'{name} Test Confusion Matrix')
            fig.colorbar(im, ax=ax1)
            tick_marks = np.arange(3)  # Assuming 3 classes (0, 1, 2)
            ax1.set_xticks(tick_marks)
            ax1.set_yticks(tick_marks)
            ax1.set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
            ax1.set_yticklabels(['Class 0', 'Class 1', 'Class 2'])
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax1.text(j, i, str(cm[i, j]),
                           horizontalalignment="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            ax1.set_ylabel('True label')
            ax1.set_xlabel('Predicted label')
            
            # Plot 2: Metrics comparison between train, validation, and test
            # Calculate metrics for validation set
            accuracy_val = accuracy_score(y_val, y_pred_val)
            precision_val = precision_score(y_val, y_pred_val, average='weighted')
            recall_val = recall_score(y_val, y_pred_val, average='weighted')
            f1_val = f1_score(y_val, y_pred_val, average='weighted')
            
            # Get metrics from results dictionary for train and test
            accuracy_train = results[name]['Accuracy_train']
            precision_train = results[name]['Precision_train']
            recall_train = results[name]['Recall_train']
            f1_train = results[name]['F1_train']
            
            accuracy_test = results[name]['Accuracy_test']
            precision_test = results[name]['Precision_test']
            recall_test = results[name]['Recall_test']
            f1_test = results[name]['F1_test']
            
            # Set up bar chart data
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
            train_values = [accuracy_train, precision_train, recall_train, f1_train]
            val_values = [accuracy_val, precision_val, recall_val, f1_val]
            test_values = [accuracy_test, precision_test, recall_test, f1_test]
            
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
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_path, 'predictions', f'{name}_classification_report.png'))
            plt.close()
            
            # Print classification report for test set
            print(f"\nClassification Report for {name} (Test Set):")
            print(classification_report(y_test, y_pred_test, target_names=['Class 0', 'Class 1', 'Class 2']))
        
        # ------------------------------------------------------
        # Feature Importance analysis
        if len(feature_names) > 1:  # Only relevant if we have multiple features
            for name, model in models.items():
                importance_path = os.path.join(plots_path, 'feature_importances', f'{name}_feature_importance.png')
                importance_result = plot_feature_importance(model, feature_names, name, importance_path)
                
                if importance_result:
                    importance, indices = importance_result
                    print(f"Top features for {name}:")
                    for i in range(min(10, len(indices))):  # Display top 10 features
                        print(f"{feature_names[indices[i]]}: {format_value(importance[indices[i]])}")
                    print("\n")

# ------------------------------------------------------
# Generate overall comparison report
print("\n\n" + "="*80)
print("                   OVERALL COMPARISON OF ALL EXPERIMENTS")
print("="*80)

# Save all results to CSV
all_results_df.to_csv(os.path.join(results_base_dir, 'all_experiments_results.csv'))

# Group by dataset and feature selection approach, taking the best model for each group
best_models = all_results_df.groupby(['Dataset', 'Feature_Selection'])['Accuracy_test'].max().reset_index()
best_models = best_models.merge(all_results_df, on=['Dataset', 'Feature_Selection', 'Accuracy_test'])

print("\nBest model for each experiment:")
print(best_models[['Dataset', 'Feature_Selection', 'Accuracy_test', 'F1_test']].sort_values(
    by=['Dataset', 'Accuracy_test'], ascending=[True, False]))

# Create comparison plots
plt.figure(figsize=(14, 8))
for dataset in datasets:
    dataset_results = best_models[best_models['Dataset'] == dataset]
    
    # Set up bar positions
    bar_positions = np.arange(len(dataset_results))
    plt.bar(bar_positions, dataset_results['Accuracy_test'], 
            label=f'{dataset} - Accuracy', alpha=0.7)
    
    # Add model names as annotations
    for i, row in enumerate(dataset_results.itertuples()):
        model_name = str(row.Index)  # Convert Index to string
        plt.text(i, row.Accuracy_test + 0.01, 
                model_name[:10] if len(model_name) > 10 else model_name, 
                ha='center', rotation=45, fontsize=8)
plt.xlabel('Feature Selection Approach')
plt.ylabel('Test Accuracy')
plt.title('Comparison of Best Models Across Datasets and Feature Selection Approaches')
plt.xticks(np.arange(len(best_models)), best_models['Feature_Selection'], rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_base_dir, 'overall_comparison.png'))
plt.close()

# Print summary of findings
print("\nSummary of all experiments:")
summary = all_results_df.groupby(['Dataset', 'Feature_Selection']).agg(
    {'Accuracy_test': ['mean', 'max', 'min', 'std'],
     'F1_test': ['mean', 'max']}).reset_index()
print(summary)

print("\nDone! All results saved to:", results_base_dir)