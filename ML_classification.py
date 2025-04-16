#%%

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
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

# import SVC and gradient boosting
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


from ML_utils import normalize_data_and_split, plot_predictions
# ------------------------------------------------------
# Flag to control hyperparameter tuning
tune_hyperparameters = False  # set to False to disable tuning

# dataset and label details
dataset_name = "df_ready_date"
label = ''
df = pd.read_csv(f'tables/imputed/{dataset_name}.csv')
# print(df.head())

# Transform the categorical target from [-1, 0, 1] to [0, 1, 2]
# Create a mapping dictionary
target_mapping = {-1: 0, 0: 1, 1: 2}
df['categorical_target'] = df['categorical_target'].map(target_mapping)

df_numeric = df.select_dtypes(include=[np.number])

# # check correlations between features and target
correlations = df_numeric.corr()
correlation_with_target = correlations["target"].sort_values(ascending=False)
correlation_with_CATtarget = correlations["categorical_target"].sort_values(ascending=False)
print("Correlation with REGRESSION target:")
print(correlation_with_target)
print("Correlation with CLASSIFICATION target:")
print(correlation_with_CATtarget)

# combine all app.Categorical features into one
df['appCat'] = df[df.columns[df.columns.str.contains('appCat')]].sum(axis=1)
# Drop individual app.Categorical features
df.drop(columns=df.columns[df.columns.str.contains('appCat.')], inplace=True)

# Remove features that should not be used in the model
features = df.columns.to_list()
# features.remove('id_num')
features.remove('target')  # Make sure this matches your actual target column name
features.remove('categorical_target')  # Make sure this matches your actual target column name
features.remove('date')  # Remove date if present
features.remove("next_date")

# features = ["id_num", "mood_mean_daily"] # for baseline model

if "time_of_day_non_encoded" in features:
    features.remove("time_of_day_non_encoded")

print("features", features)


# Update the function call to include timestamp_col parameter
X_train, X_val, X_test, y_train, y_val, y_test, scalers, metadata = normalize_data_and_split(
    df,
    features=features,
    target_col="categorical_target",  # Changed to categorical_target
    id_col='id_num',
    timestamp_col='date',  # Add this parameter for timeseries plotting,
    per_participant_normalization=True,
    scaler_type="StandardScaler",
    test_size=0.1,
    val_size=0.000000000001,
    random_state=42
)
# Store feature names before fitting models
feature_names = X_train.columns.tolist()  # Use X_train instead of X

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
        'penalty': ['l2', 'none']
    },
    
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
    
    'XGBoost Classifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4],  # More restricted depth
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

# If hyperparameter tuning is enabled, replace the models with tuned versions
if tune_hyperparameters:
    tuned_models = {}
    for name, model in models.items():
        print(f"Tuning hyperparameters for {name}...")
        grid = param_grids.get(name, {})
        if grid:  # if there's a non-empty grid
            gs = GridSearchCV(model, grid, cv=3, scoring='accuracy', n_jobs=-1)
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
        'Accuracy_test': accuracy,
        'Accuracy_train': accuracy_train,
        'Precision_test': precision,
        'Precision_train': precision_train,
        'Recall_test': recall,
        'Recall_train': recall_train,
        'F1_test': f1,
        'F1_train': f1_train,
        'dataset': dataset_name
    }

# Create a DataFrame for the results and sort by test Accuracy score
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='Accuracy_test', ascending=False)
print(results_df)

# ------------------------------------------------------
# Create required directories for plots
plots_path = 'plots_classification'
if not os.path.exists(plots_path):
    os.makedirs(plots_path)
feature_importances_path = os.path.join(plots_path, 'feature_importances')
predictions_path = os.path.join(plots_path, 'predictions')
metrics_path = os.path.join(plots_path, 'metrics')
visualizations_path = os.path.join(plots_path, 'visualizations')
timeseries_path = os.path.join(plots_path, 'timeseries')  # New directory for timeseries plots
confusion_matrices_path = os.path.join(plots_path, 'confusion_matrices')  # New directory for confusion matrices

for directory in [plots_path, feature_importances_path, predictions_path, metrics_path, visualizations_path, timeseries_path, confusion_matrices_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ------------------------------------------------------
# Plotting results

x = np.arange(len(results_df.index))  # label locations
width = 0.35  # width of the bars

# Accuracy Scores
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results_df['Accuracy_train'], width, label='Train Accuracy')
plt.bar(x + width/2, results_df['Accuracy_test'], width, label='Test Accuracy')
plt.title('Model Accuracy Scores')
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.xticks(x, results_df.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(metrics_path, 'accuracy_scores.png'))
plt.close()

# Precision Scores
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results_df['Precision_train'], width, label='Train Precision')
plt.bar(x + width/2, results_df['Precision_test'], width, label='Test Precision')
plt.title('Model Precision Scores')
plt.xlabel('Model')
plt.ylabel('Precision Score')
plt.xticks(x, results_df.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(metrics_path, 'precision_scores.png'))
plt.close()

# Recall Scores
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results_df['Recall_train'], width, label='Train Recall')
plt.bar(x + width/2, results_df['Recall_test'], width, label='Test Recall')
plt.title('Model Recall Scores')
plt.xlabel('Model')
plt.ylabel('Recall Score')
plt.xticks(x, results_df.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(metrics_path, 'recall_scores.png'))
plt.close()

# F1 Scores
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results_df['F1_train'], width, label='Train F1')
plt.bar(x + width/2, results_df['F1_test'], width, label='Test F1')
plt.title('Model F1 Scores')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.xticks(x, results_df.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(metrics_path, 'f1_scores.png'))
plt.close()

# ------------------------------------------------------
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
    plt.savefig(os.path.join(confusion_matrices_path, f'{name}_confusion_matrix.png'))
    plt.close()

# ------------------------------------------------------
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
    y_pred_val = model.predict(X_val)
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
    plt.savefig(os.path.join(predictions_path, f'{name}_classification_report.png'))
    plt.close()

    # Print classification report for test set
    print(f"\nClassification Report for {name} (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Class 0', 'Class 1', 'Class 2']))

# ------------------------------------------------------
# Feature Importance analysis
importances = {}
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For multi-class classification, we need to average across classes
        if len(model.coef_.shape) > 1:
            importances[name] = np.abs(model.coef_).mean(axis=0)
        else:
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
            pass
            # print(f"Feature list: {[feature_names[i] for i in indices]}")
        except IndexError:
            print("Error: Feature names do not match the indices of feature importances.")
        print("\n")
    else:
        print(f'{name} does not have feature importances.')