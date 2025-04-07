# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from scipy.stats import skew

import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve



def load_data(file):
    """
    Function to load data from a CSV file and save it as a binary file for faster loading in the future.
    :param file:
    :return:
    """
    # Check if the preprocessed data is already saved as a binary file
    if os.path.exists(f'{file}.pkl'):
        df = joblib.load(f'{file}.pkl')
        print("Data loaded from pre-saved binary file.")
    else:
        df = pd.read_csv(f'{file}.csv')
        joblib.dump(df, f'{file}.pkl')
        print("Data loaded from CSV and saved as binary.")
    return df

def plot_numerical(numerical):
    """
    Function to plot the distribution of numerical features.
    :param numerical:
    :return:
    """
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 10))
    for ax, column in zip(axes.flatten(), numerical.columns):
        sns.distplot(numerical[column].dropna(), ax=ax, color='darkred')
        ax.set_title(column, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.set_xlabel('')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_categorical_vs_sales_price(df, categorical_columns, n, title):
    """
    Plots box plots for the specified categorical columns against 'Sales Price'.
    If there are more than `n` categories, only the top `n` categories are plotted.
    Each plot is displayed separately with its own title.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - categorical_columns (list of str): List of column names to plot against 'Sales Price'.
    - n (int): Maximum number of categories to plot for each categorical column.
    - title (str): Title prefix for the plots.
    """
    for col in categorical_columns:
        # Create a new figure for each categorical column
        plt.figure(figsize=(10, 6))

        # Get the top n categories by count
        top_categories = df[col].value_counts().nlargest(n).index
        # Filter the dataframe to only include the top categories
        filtered_df = df[df[col].isin(top_categories)]

        # Plot the box plot
        sns.boxplot(data=filtered_df, x=col, y="Sales Price")
        plt.title(f'{title}: {col} vs Sales Price', fontsize=16)
        plt.ylabel('Sales Price', fontsize=12)
        plt.xlabel(col, fontsize=12)
        plt.xticks(rotation=45)

        # Display the plot
        plt.tight_layout()
        plt.show()



############################################################################################################
# PREPROCESSING



def find_duplicate_columns(df):
    """
    Function to find duplicate columns in a DataFrame by factorizing them first.
    :param df:
    :return:
    """
    factorized_cols = {}  # key is a tuple of factorized column values, value is the column name
    duplicates = []  # list of tuples of duplicate columns

    for col in df.columns:
        # Factorize the column values
        f = pd.factorize(df[col])[
            0]  # Factorizes each column, returning an array of integers that correspond to distinct values.

        # Check if this factorized column already exists
        if tuple(f) in factorized_cols:  # cast to a touple to make it hashable
            duplicates.append((col, factorized_cols[tuple(f)]))
        else:
            factorized_cols[tuple(f)] = col  # only add do dictionary if it is not already there

    # Print out duplicate columns
    if duplicates:
        print("Duplicate columns found:")

        for dup in duplicates:
            print(f"Column '{dup[0]}' is a duplicate of column '{dup[1]}'")
    else:
        print("No duplicate columns found.")



    return duplicates

def corr(df, colname):
    """
    Function to compute the correlation matrix of a DataFrame and display the top 5 positively correlated and bottom 5 negatively correlated features with 'colname'.
    :param df:
    :param colname:
    :return:
    """
    # Create a copy of the dataframe to avoid modifying the original one
    df_copy = df.copy()
    # Apply LabelEncoder only to categorical columns
    for column in df_copy.select_dtypes(include=['object', 'category']).columns:
        # Ensure consistency in the categorical column by filling NaNs and converting all values to strings
        df_copy[column] = df_copy[column].astype(str).fillna('missing')
        df_copy[column] = LabelEncoder().fit_transform(df_copy[column])

    # Handle missing values in numerical columns (if desired, e.g., using median imputation)
    df_copy.fillna(df_copy.median(), inplace=True)

    # Compute the correlation matrix
    corr_matrix = df_copy.corr()
    # print(corr_matrix)

    # Display the top 5 positively correlated and bottom 5 negatively correlated features with 'colname'
    print(f"Top 5 correlations with {colname}:")
    print(corr_matrix[colname].sort_values(ascending=False)[:5])  # Top 5 correlations

    print(f"Bottom 5 correlations with {colname}:")
    print(corr_matrix[colname].sort_values(ascending=False)[-5:])  # Bottom 5 correlations


def corr_names(df, colname):
    """
    Function to compute the correlation matrix of a DataFrame and return the top 5 positively correlated and bottom 5 negatively correlated features with 'colname'.
    :param df:
    :param colname:
    :return:
    """
    # Create a copy of the dataframe to avoid modifying the original one
    df_copy = df.copy()

    # Apply LabelEncoder only to categorical columns
    for column in df_copy.select_dtypes(include=['object', 'category']).columns:
        # Ensure consistency in the categorical column by filling NaNs and converting all values to strings
        df_copy[column] = df_copy[column].astype(str).fillna('missing')
        df_copy[column] = LabelEncoder().fit_transform(df_copy[column])

    # Handle missing values in numerical columns (e.g., median imputation)
    df_copy.fillna(df_copy.median(), inplace=True)

    # Compute the correlation matrix
    corr_matrix = df_copy.corr()

    # Sort the correlations with respect to the given 'colname'
    sorted_corr = corr_matrix[colname].sort_values(ascending=False)

    # Get top 5 biggest correlations (excluding the column itself)
    top_5_biggest = sorted_corr[:5].index.tolist()  #
    top_5_smallest = sorted_corr[-5:].index.tolist()  # Bottom 5 correlations

    return corr_matrix, top_5_biggest, top_5_smallest

def split_spec(spec):
    """
    Function to split a specification string into the numeric part and the unit part.
    :param spec:
    :return:
    """
    # Handle "Unidentified" cases
    if 'Unidentified (Compact Construction)' in spec:
        return None, 'Unidentified (Compact Construction)'
    elif 'Unidentified' in spec:
        return None, 'Unidentified'

    # Strip any leading/trailing spaces
    spec = spec.strip()

    # Check for "to" for range-based values (e.g., "150.0 to 175.0 Horsepower")
    if ' to ' in spec:
        # Split into the range part and unit part
        range_part, unit_part = spec.split(' to ')
        # Capture the second part of the range (after "to") as part of the range
        numeric_part = f"{range_part.strip()} to {unit_part.split(' ')[0].strip()}"
        unit_part = ' '.join(unit_part.split(' ')[1:]).strip()  # Get the rest as the unit

    # Check for "+" directly after numbers (e.g., "2701.0+ Lb Operating Capacity")
    elif '+' in spec:
        # Split using the "+" symbol, even if there's no space
        numeric_part = spec.split('+')[0].strip() + "+"
        unit_part = ' '.join(spec.split('+')[1:]).strip()  # Extract the unit part

    # If no range or "+" is found, just return the spec as the unit
    else:
        print("empty ", spec)
        numeric_part = None
        unit_part = spec

    return numeric_part, unit_part



def plot_skewness_pyplot(numerical_df, title, xaxis_label, yaxis_label, color_palette='Blues_d'):
    """
    Creates a bar plot for skewness of numerical columns in a DataFrame using Matplotlib/Seaborn.

    Parameters:
    numerical_df (pd.DataFrame): The DataFrame containing numerical columns to analyze.
    title (str): Title for the plot.
    xaxis_label (str): Label for the x-axis.
    yaxis_label (str): Label for the y-axis.
    color_palette (str): Color palette to use for the bars (default is 'Blues_d').

    Returns:
    None: The function displays the plot using plt.show().
    """
    # Calculate skewness for numerical columns
    skew_merged = pd.DataFrame(data=numerical_df.select_dtypes(include=['int64', 'float64']).skew(),
                               columns=['Skewness'])

    # Sort by skewness in descending order
    skew_merged_sorted = skew_merged.sort_values(ascending=False, by='Skewness')

    # Set up the plot size and style
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Create the bar plot using Seaborn (you can replace this with plt.bar if you prefer pure Matplotlib)
    sns.barplot(
        x=skew_merged_sorted.index,
        y=skew_merged_sorted['Skewness'],
        palette=color_palette
    )

    # Set titles and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xaxis_label, fontsize=12)
    plt.ylabel(yaxis_label, fontsize=12)

    # Rotate x labels for better readability
    plt.xticks(rotation=90)

    # Show the plot
    plt.tight_layout()
    plt.show()

def log_transform(df, column, skew_threshold=0.5, exclude=None):
    """Log-transform a column if it is right-skewed."""

    df_transformed = df.copy()
    transformed = []
    for column in df_transformed.columns:
        # Apply the log transformation
        if skew(df_transformed[column]) > skew_threshold and column != exclude:
            df_transformed[column] = np.log1p(df_transformed[column])
            transformed.append(column)
    print(f"Log-transformed {len(column)} columns: {transformed}")
    return df_transformed


def frequency_encoding(df, column, drop_original=True):
    """
    Perform frequency encoding on a categorical column.
    :param df:
    :param column:
    :param drop_original:
    :return:
    """
    freq_encoding = df[column].value_counts() / len(df)
    df[f"{column}_freq"] = df[column].map(freq_encoding)  # Keep both the original and new column
    if drop_original:
        df.drop(column, axis=1, inplace=True) # if you want to drop OG column
    return df


def explore_missing_cat(categorical):
    """
    Function to explore missing values in categorical columns.
    :param categorical:
    :return:
    """
    # Check the number of missing values in each column
    missingCat = categorical.isnull().sum().sort_values(ascending=False)
    uniqueCat = categorical.nunique().sort_values(ascending=False)

    # add most frequent value
    most_frequent = categorical.mode().iloc[0]

    # Calculate the percentage of unique values and missing values
    uniquePercent = uniqueCat / len(categorical)
    missingPercent = missingCat / len(categorical)

    # Rename for clarity
    uniquePercent = uniquePercent.rename('unique_percent')
    missingPercent = missingPercent.rename('missing_percent')
    uniqueCat = uniqueCat.rename('unique')
    missingCat = missingCat.rename('missing')
    most_frequent = most_frequent.rename('most frequent value')

    # Create DataFrames
    uniqueCat = pd.DataFrame(uniqueCat)
    missingCat = pd.DataFrame(missingCat)

    # Join the missing and unique DataFrames with their percentages
    uniqueCat = uniqueCat.join(missingCat)
    uniqueCat = uniqueCat.join(uniquePercent)
    uniqueCat = uniqueCat.join(missingPercent)
    uniqueCat = uniqueCat.join(most_frequent)

    # Add a new column for the list of unique elements in each column
    uniqueCat['unique_elements'] = categorical.apply(lambda col: col.dropna().unique().tolist())

    # Display the final DataFrame
    return uniqueCat


# Function to convert feet and inches to total inches
def feet_inches_to_inches(length):
    """
    Function to convert a string representation of length in feet and inches to total inches.
    :param length:
    :return:
    """
    if pd.isnull(length) or length == "None or Unspecified":
        return 0  # If value is NaN, return 0
    # Split the string by feet (') and inches (")
    parts = length.split("'")
    feet = int(parts[0].strip())  # Get feet part and convert to integer
    inches = int(parts[1].replace('"', '').strip())  # Get inches part, remove " and convert to integer
    return feet * 12 + inches  # Convert feet to inches and add the inches




############################################################################################################
# MODELING

def plot_feature_importance(model, df_train_final, title, top_n=None):
    """
    Function to plot feature importance using Seaborn with an option to limit the number of features displayed.

    Parameters:
    model: Trained model (should have feature_importances_ attribute)
    df_train_final: DataFrame of the training features
    title: Title of the plot
    top_n: Number of top features to display (default is None, which plots all features)
    """
    # Get feature importance from the model
    importance = pd.DataFrame({'Features': df_train_final.columns, 'Importance': model.feature_importances_})
    importance = importance.set_index('Features')

    # Sort importance values in descending order
    importance = importance.sort_values(by='Importance', ascending=False)

    # Limit the number of features to plot
    if top_n is not None:
        importance = importance.head(top_n)

    # Plot feature importance using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance['Importance'], y=importance.index, palette='viridis')

    # Add plot labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Show plot
    plt.tight_layout()
    plt.show()



def metrics(model, X_train, y_train, X_test, y_test, log_transformed=False):
    """
    Fit a model and calculate R^2 and RMSE for train and test sets.
    :param model:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param log_transformed:
    :return:
    """
    model_fit = model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate R^2 for train and test sets
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    if log_transformed:
        y_train = np.expm1(y_train)
        y_test = np.expm1(y_test)
        y_pred_train = np.expm1(y_pred_train)
        y_pred_test = np.expm1(y_pred_test)

    # calculate MAE
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Calculate RMSE on original scale
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return train_r2, test_r2, rmse_train, rmse_test, y_pred_train, y_pred_test, model_fit, mae_train, mae_test
def bar_train_test_seaborn(models, train_scores, test_scores, title, xaxis, yaxis, palette='muted'):
    """
    Create a grouped bar plot for training and testing scores using Seaborn.

    Parameters:
    models (list): List of model names.
    train_scores (list): Training scores corresponding to the models.
    test_scores (list): Testing scores corresponding to the models.
    title (str): The title of the plot.
    xaxis (str): The label for the x-axis.
    yaxis (str): The label for the y-axis.
    palette (str): Color palette for the bars.

    Returns:
    None: Displays the bar plot using plt.show().
    """
    # Create a DataFrame to store the data for Seaborn
    data = pd.DataFrame({
        'Model': models,
        'Training Score': train_scores,
        'Testing Score': test_scores
    })

    # Melt the DataFrame to have a format suitable for Seaborn
    data_melted = pd.melt(data, id_vars='Model', value_vars=['Training Score', 'Testing Score'],
                          var_name='Dataset', value_name='Score')

    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Create a bar plot
    sns.barplot(x='Model', y='Score', hue='Dataset', data=data_melted, palette=palette)

    # Set plot title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xaxis, fontsize=12)
    plt.ylabel(yaxis, fontsize=12)

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_residuals(model, X_test, y_test, log_transformed=False):
    """
    Plot the residuals for a model.
    :param model:
    :param X_test:
    :param y_test:
    :param log_transformed:
    :return:
    """
    # Predict the target
    y_pred = model.predict(X_test)

    # If log-transformed, inverse the log1p transformation
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Create a subplot for both scatter plot and histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot of residuals
    axes[0].scatter(y_test, residuals, color='red', alpha=0.6)
    axes[0].set_title(f'Residuals for {model.__class__.__name__} (Scatter)', fontsize=16)
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].axhline(y=0, color='black', linestyle='--')

    # Histogram of residuals
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color='skyblue')
    axes[1].set_title(f'Residual Distribution for {model.__class__.__name__} (Histogram)', fontsize=16)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_xlim(-50000, 50000)

    plt.tight_layout()
    plt.show()


def plot_residuals_with_stats_and_colors(model, X_test, y_test, log_transformed=False):
    """
    Plot the residuals for a model, with mean and median lines. Color-code scatterplot and histogram bins to indicate whether residuals correspond to high, medium, or low actual values.

    :param model:
    :param X_test:
    :param y_test:
    :param log_transformed:
    :return:
    """
    # Predict the target
    y_pred = model.predict(X_test)

    # If log-transformed, inverse the log1p transformation
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Compute mean and median of residuals
    mean_residual = residuals.mean()
    median_residual = residuals.median()

    # Categorize `y_test` into three groups: Low, Medium, High
    # Categorize `y_test` into three groups: Low, Medium, High using equal ranges
    y_min, y_max = y_test.min(), y_test.max()
    y_categories = pd.cut(y_test, bins=[y_min, (y_min + y_max) / 3, 2 * (y_min + y_max) / 3, y_max],
                          labels=['Low', 'Medium', 'High'])
    # count the number of residuals in each category
    print("y_categories value counts",y_categories.value_counts())
    hist_data = pd.DataFrame({'Actual Values': y_test, 'Residuals': residuals, 'Category': y_categories})

    # Define the color palette
    palette = {'Low': 'lightcoral', 'Medium': 'skyblue', 'High': 'seagreen'}

    # Create a subplot for both scatter plot and histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot of residuals with colors based on `y_categories`
    for category, color in palette.items():
        subset = hist_data[hist_data['Category'] == category]
        axes[0].scatter(subset['Actual Values'], subset['Residuals'], color=color, alpha=0.6,
                        label=f"{category}")

    axes[0].set_title(f'Residuals for {model.__class__.__name__} (Scatter)', fontsize=16)
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].axhline(y=0, color='black', linestyle='--')

    # Add mean and median lines
    axes[0].axhline(y=mean_residual, color='purple', linestyle='-', label=f'Mean')
    axes[0].axhline(y=median_residual, color='orange', linestyle='--', label=f'Median')
    axes[0].legend()

    # Histogram of residuals with color-coded bins
    sns.histplot(data=hist_data, x='Residuals', hue='Category', multiple='stack', bins=50, kde=True, ax=axes[1],
                 palette=palette)
    axes[1].set_title(f'Residual Distribution for {model.__class__.__name__} (Histogram)', fontsize=16)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_xlim(-50000, 50000)

    # Add mean and median lines to the histogram
    axes[1].axvline(x=mean_residual, color='purple', linestyle='-', label=f'Mean')
    axes[1].axvline(x=median_residual, color='orange', linestyle='--', label=f'Median')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_residuals_with_category_means(model, X_test, y_test, log_transformed=False):
    """
    Plot the residuals for a model, with mean and median lines, and category-specific mean lines.
    Color-code scatterplot and histogram bins to indicate whether residuals correspond to high, medium, or low actual values.

    :param model:
    :param X_test:
    :param y_test:
    :param log_transformed:
    :return:
    """
    # Predict the target
    y_pred = model.predict(X_test)

    # If log-transformed, inverse the log1p transformation
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Compute overall mean and median of residuals
    mean_residual = residuals.mean()
    median_residual = residuals.median()

    # Categorize `y_test` into three groups: Low, Medium, High
    # Categorize `y_test` into three groups: Low, Medium, High using equal ranges
    y_min, y_max = y_test.min(), y_test.max()
    y_categories = pd.cut(y_test, bins=[y_min, (y_min + y_max) / 3, 2 * (y_min + y_max) / 3, y_max],
                          labels=['Low', 'Medium', 'High'])
    hist_data = pd.DataFrame({'Actual Values': y_test, 'Residuals': residuals, 'Category': y_categories})

    # Define the color palette
    palette = {'Low': 'lightcoral', 'Medium': 'skyblue', 'High': 'seagreen'}

    # Create a subplot for both scatter plot and histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot of residuals with colors based on `y_categories`
    for category, color in palette.items():
        subset = hist_data[hist_data['Category'] == category]
        category_mean = subset['Residuals'].mean()
        axes[0].scatter(subset['Actual Values'], subset['Residuals'], color=color, alpha=0.6,
                        label=f"{category} ({subset['Actual Values'].min():.2f} - {subset['Actual Values'].max():.2f})")
        # Add mean line for each category
        axes[0].axhline(y=category_mean, color=color, linestyle='-', linewidth=1.5,
                        label=f'{category} Mean: {category_mean:.2f}')

    axes[0].set_title(f'Residuals for {model.__class__.__name__} (Scatter)', fontsize=16)
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].axhline(y=0, color='black', linestyle='--')

    # Add overall mean and median lines
    axes[0].axhline(y=mean_residual, color='blue', linestyle='-', label=f'Overall Mean: {mean_residual:.2f}')
    axes[0].axhline(y=median_residual, color='green', linestyle='--', label=f'Overall Median: {median_residual:.2f}')
    axes[0].legend()

    # Histogram of residuals with color-coded bins
    sns.histplot(data=hist_data, x='Residuals', hue='Category', multiple='stack', bins=50, kde=True, ax=axes[1],
                 palette=palette)
    axes[1].set_title(f'Residual Distribution for {model.__class__.__name__} (Histogram)', fontsize=16)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_xlim(-50000, 50000)

    # Add mean and median lines to the histogram
    axes[1].axvline(x=mean_residual, color='blue', linestyle='-', label=f'Overall Mean: {mean_residual:.2f}')
    axes[1].axvline(x=median_residual, color='green', linestyle='--', label=f'Overall Median: {median_residual:.2f}')

    # Add mean lines for each category to the histogram
    for category, color in palette.items():
        category_mean = hist_data[hist_data['Category'] == category]['Residuals'].mean()
        axes[1].axvline(x=category_mean, color=color, linestyle='-', linewidth=1.5,
                        label=f'{category} Mean: {category_mean:.2f}')

    axes[1].legend()

    plt.tight_layout()
    plt.show()


def categorize_values(y_values, y_min, y_max):
    """
    Categorize `y_values` into three groups: Low, Medium, High using equal ranges based on `y_min` and `y_max`.

    :param y_values: Series of values to categorize
    :param y_min: Minimum value across the dataset
    :param y_max: Maximum value across the dataset
    :return: Categorical Series with labels 'Low', 'Medium', 'High'
    """
    bins = [y_min, (y_min + y_max) / 3, 2 * (y_min + y_max) / 3, y_max]
    labels = ['Low', 'Medium', 'High']
    return pd.cut(y_values, bins=bins, labels=labels)

def plot_residuals_with_colored_background(model, X_test, y_test, log_transformed=False):
    """
    Plot the residuals for a model, with mean and median lines. Color-code the background of the scatter plot based on
    whether residuals correspond to high, medium, or low actual values, and adjust dot size and color.

    :param model:
    :param X_test:
    :param y_test:
    :param log_transformed:
    :return:
    """
    # Predict the target
    y_pred = model.predict(X_test)

    # If log-transformed, inverse the log1p transformation
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Compute mean and median of residuals
    mean_residual = residuals.mean()
    median_residual = residuals.median()

    print("mean_residual", mean_residual)
    print("median_residual", median_residual)

    # Use the consistent categorization function
    y_min, y_max = y_test.min(), y_test.max()
    y_categories = categorize_values(y_test, y_min, y_max)

    print("y_categories value counts", y_categories.value_counts())
    print("total counts", sum(y_categories.value_counts()))
    hist_data = pd.DataFrame({'Actual Values': y_test, 'Residuals': residuals, 'Category': y_categories})

    # Define the color palette for background shading
    palette = {'Low': 'lightcoral', 'Medium': 'skyblue', 'High': 'seagreen'}

    # Create a subplot for both scatter plot and histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define the x-range for each category to shade the background
    ranges = {
        'Low': (y_min, (y_min + y_max) / 3),
        'Medium': ((y_min + y_max) / 3, 2 * (y_min + y_max) / 3),
        'High': (2 * (y_min + y_max) / 3, y_max)
    }

    # Scatter plot of residuals with background colors based on `y_categories`
    for category, (start, end) in ranges.items():
        axes[0].axvspan(start, end, color=palette[category], alpha=0.2, label=f"{category}")

    # Scatter the residual points with smaller size and grey color
    axes[0].scatter(y_test, residuals, color='grey', alpha=0.6, s=10)

    axes[0].set_title(f'Residuals for {model.__class__.__name__} (Scatter)', fontsize=16)
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].axhline(y=0, color='black', linestyle='--')

    # Add mean and median lines
    axes[0].axhline(y=mean_residual, color='purple', linestyle='-', label=f'Mean')
    axes[0].axhline(y=median_residual, color='orange', linestyle='--', label=f'Median')
    axes[0].legend()

    # Histogram of residuals with color-coded bins
    sns.histplot(data=hist_data, x='Residuals', hue='Category', multiple='stack', bins=50, kde=True, ax=axes[1],
                 palette=palette)
    axes[1].set_title(f'Residual Distribution for {model.__class__.__name__} (Histogram)', fontsize=16)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_xlim(-50000, 50000)

    # Add mean and median lines to the histogram
    axes[1].axvline(x=mean_residual, color='purple', linestyle='-', label=f'Mean')
    axes[1].axvline(x=median_residual, color='orange', linestyle='--', label=f'Median')
    axes[1].legend()

    plt.tight_layout()
    plt.show()



def plot_residuals_and_confusion_matrix(model, X_test, y_test, log_transformed=False):
    """
    Plot the residuals for a model, with mean and median lines. Color-code the background of the scatter plot based on
    whether residuals correspond to high, medium, or low actual values, and create a confusion matrix to evaluate classification.

    :param model:
    :param X_test:
    :param y_test:
    :param log_transformed:
    :return:
    """
    # Predict the target
    y_pred = model.predict(X_test)

    # If log-transformed, inverse the log1p transformation
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Compute mean and median of residuals
    mean_residual = residuals.mean()
    median_residual = residuals.median()

    # Use the consistent categorization function
    y_min, y_max = y_test.min(), y_test.max()
    y_categories = categorize_values(y_test, y_min, y_max)
    y_pred_categories = categorize_values(y_pred, y_min, y_max)

    # Convert categorical labels to numerical codes
    labels = ['Low', 'Medium', 'High']
    y_true_encoded = pd.Categorical(y_categories, categories=labels).codes
    y_pred_encoded = pd.Categorical(y_pred_categories, categories=labels).codes

    # Create the confusion matrix
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)

    # Display the confusion matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted Categories')
    ax.set_ylabel('Actual Categories')
    ax.set_title('Confusion Matrix')

    plt.show()
def plot_predicted_vs_actual(model, X_test, y_test, log_transformed=False):
    # Predict the target
    y_pred = model.predict(X_test)

    # If log-transformed, inverse the log1p transformation
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.title(f'Predicted vs Actual for {model.__class__.__name__}', fontsize=16)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.show()

def plot_learning_curve_rmse(model, df_train_final, y_train, cv, seed, model_name, log_transformed=False):
    """
    Function to train the model and plot the learning curve for a model using RMSE as the scoring metric.
    :param model:
    :param df_train_final:
    :param y_train:
    :param cv:
    :param seed:
    :param model_name:
    :param log_transformed: If True, transform RMSE back to original scale using exponential transformation.
    :return:
    """

    # Generate CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, df_train_final, np.expm1(y_train) if log_transformed else y_train,
                                                            train_sizes=np.linspace(0.01, 1.0, 20), cv=cv,
                                                            scoring='neg_mean_squared_error',
                                                            n_jobs=-1, random_state=seed)

    # Convert negative mean squared error to RMSE by taking square root of absolute values
    train_rmse = np.sqrt(-train_scores)
    test_rmse = np.sqrt(-test_scores)



    # Create means and standard deviations of training set RMSE
    train_mean = np.mean(train_rmse, axis=1)
    train_std = np.std(train_rmse, axis=1)

    # Create means and standard deviations of test set RMSE
    test_mean = np.mean(test_rmse, axis=1)
    test_std = np.std(test_rmse, axis=1)

    # Plot lines
    plt.plot(train_sizes, train_mean, 'o-', color='red', label='Training RMSE')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation RMSE')

    # Plot bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    # Create plot
    font_size = 12
    plt.title(f'Learning Curve (RMSE) for {model_name}', fontsize=font_size)
    plt.xlabel('Training Set Size', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(loc='best')
    plt.grid()
    plt.show()




def plot_predicted_vs_actual(model, X_test, y_test, log_transformed=False):
    # Predict the target
    y_pred = model.predict(X_test)

    # If log-transformed, inverse the log1p transformation
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, s=10)
    plt.title(f'Predicted vs Actual for {model.__class__.__name__}', fontsize=16)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.show()

def plot_learning_curve_rmse(model, df_train_final, y_train, cv, seed, model_name, log_transformed=False):
    """
    Function to train the model and plot the learning curve for a model using RMSE as the scoring metric.
    :param model:
    :param df_train_final:
    :param y_train:
    :param cv:
    :param seed:
    :param model_name:
    :param log_transformed: If True, transform RMSE back to original scale using exponential transformation.
    :return:
    """

    # Generate CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, df_train_final, np.expm1(y_train) if log_transformed else y_train,
                                                            train_sizes=np.linspace(0.01, 1.0, 20), cv=cv,
                                                            scoring='neg_mean_squared_error',
                                                            n_jobs=-1, random_state=seed)

    # Convert negative mean squared error to RMSE by taking square root of absolute values
    train_rmse = np.sqrt(-train_scores)
    test_rmse = np.sqrt(-test_scores)



    # Create means and standard deviations of training set RMSE
    train_mean = np.mean(train_rmse, axis=1)
    train_std = np.std(train_rmse, axis=1)

    # Create means and standard deviations of test set RMSE
    test_mean = np.mean(test_rmse, axis=1)
    test_std = np.std(test_rmse, axis=1)

    # Plot lines
    plt.plot(train_sizes, train_mean, 'o-', color='red', label='Training RMSE')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation RMSE')

    # Plot bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    # Create plot
    font_size = 12
    plt.title(f'Learning Curve (RMSE) for {model_name}', fontsize=font_size)
    plt.xlabel('Training Set Size', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(loc='best')
    plt.grid()
    plt.show()



def plot_learning_curve_rmse(model, df_train_final, y_train, cv, seed, model_name, log_transformed=False):
    """
    Function to train the model and plot the learning curve for a model using RMSE as the scoring metric.
    :param model:
    :param df_train_final:
    :param y_train:
    :param cv:
    :param seed:
    :param model_name:
    :param log_transformed: If True, transform RMSE back to original scale using exponential transformation.
    :return:
    """

    # Generate CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, df_train_final, np.expm1(y_train) if log_transformed else y_train,
                                                            train_sizes=np.linspace(0.01, 1.0, 20), cv=cv,
                                                            scoring='neg_mean_squared_error',
                                                            n_jobs=-1, random_state=seed)

    # Convert negative mean squared error to RMSE by taking square root of absolute values
    train_rmse = np.sqrt(-train_scores)
    test_rmse = np.sqrt(-test_scores)



    # Create means and standard deviations of training set RMSE
    train_mean = np.mean(train_rmse, axis=1)
    train_std = np.std(train_rmse, axis=1)

    # Create means and standard deviations of test set RMSE
    test_mean = np.mean(test_rmse, axis=1)
    test_std = np.std(test_rmse, axis=1)

    # Plot lines
    plt.plot(train_sizes, train_mean, 'o-', color='red', label='Training RMSE')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation RMSE')

    # Plot bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    # Create plot
    font_size = 12
    plt.title(f'Learning Curve (RMSE) for {model_name}', fontsize=font_size)
    plt.xlabel('Training Set Size', fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def plot_grouped_feature_importance(model, df_train_final, title, high_cardinality_columns, one_hot_encoded_columns, top_n=None):
    """
    Function to plot aggregated feature importance using Seaborn, grouping together features based on frequency and one-hot encoding.

    Parameters:
    model: Trained model (should have feature_importances_ attribute)
    df_train_final: DataFrame of the training features
    title: Title of the plot
    high_cardinality_columns: List of columns that were frequency encoded.
    one_hot_encoded_columns: List of columns that were one-hot encoded.
    top_n: Number of top features to display (default is None, which plots all features)
    """
    # Get feature importance from the model
    importance = pd.DataFrame({'Features': df_train_final.columns, 'Importance': model.feature_importances_})

    # Prepare aggregation based on encoding
    aggregated_importance = {}

    # Handle frequency encoded columns
    for column in high_cardinality_columns:
        encoded_feature = f"{column}_freq"
        if encoded_feature in importance['Features'].values:
            aggregated_importance[column] = importance[importance['Features'] == encoded_feature]['Importance'].sum()

    # Handle one-hot encoded columns
    for column in one_hot_encoded_columns:
        # Collect all encoded versions of this column
        encoded_features = [feat for feat in importance['Features'] if feat.startswith(f"{column}_")]
        if encoded_features:
            total_importance = importance[importance['Features'].isin(encoded_features)]['Importance'].sum()
            aggregated_importance[column] = total_importance

    # Add other unencoded columns directly if they weren't aggregated
    for feature in importance['Features']:
        if feature not in aggregated_importance and not any(feature.startswith(f"{col}_") for col in high_cardinality_columns + one_hot_encoded_columns):
            aggregated_importance[feature] = importance[importance['Features'] == feature]['Importance'].sum()

    # Convert to DataFrame
    importance_agg_df = pd.DataFrame(list(aggregated_importance.items()), columns=['Features', 'Importance'])

    # Sort importance values in descending order
    importance_agg_df = importance_agg_df.sort_values(by='Importance', ascending=False)

    # Limit the number of features to plot
    if top_n is not None:
        importance_agg_df = importance_agg_df.head(top_n)

    # Plot feature importance using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance_agg_df['Importance'], y=importance_agg_df['Features'], palette='viridis')

    # Add plot labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Show plot
    plt.tight_layout()
    plt.show()

def plot_grouped_feature_importance_with_pie(model, df_train_final, title, high_cardinality_columns, one_hot_encoded_columns, top_n=None):
    """
    Function to plot aggregated feature importance using Seaborn, grouping together features based on frequency and one-hot encoding.
    Includes a pie chart that groups less important features into "Other".

    Parameters:
    model: Trained model (should have feature_importances_ attribute)
    df_train_final: DataFrame of the training features
    title: Title of the plot
    high_cardinality_columns: List of columns that were frequency encoded.
    one_hot_encoded_columns: List of columns that were one-hot encoded.
    top_n: Number of top features to display (default is None, which plots all features)
    """
    # Get feature importance from the model
    importance = pd.DataFrame({'Features': df_train_final.columns, 'Importance': model.feature_importances_})

    # Prepare aggregation based on encoding
    aggregated_importance = {}

    # Handle frequency encoded columns
    for column in high_cardinality_columns:
        encoded_feature = f"{column}_freq"
        if encoded_feature in importance['Features'].values:
            aggregated_importance[column] = importance[importance['Features'] == encoded_feature]['Importance'].sum()

    # Handle one-hot encoded columns
    for column in one_hot_encoded_columns:
        # Collect all encoded versions of this column
        encoded_features = [feat for feat in importance['Features'] if feat.startswith(f"{column}_")]
        if encoded_features:
            total_importance = importance[importance['Features'].isin(encoded_features)]['Importance'].sum()
            aggregated_importance[column] = total_importance

    # Add other unencoded columns directly if they weren't aggregated
    for feature in importance['Features']:
        if feature not in aggregated_importance and not any(feature.startswith(f"{col}_") for col in high_cardinality_columns + one_hot_encoded_columns):
            aggregated_importance[feature] = importance[importance['Features'] == feature]['Importance'].sum()

    # Convert to DataFrame
    importance_agg_df = pd.DataFrame(list(aggregated_importance.items()), columns=['Features', 'Importance'])

    # Sort importance values in descending order
    importance_agg_df = importance_agg_df.sort_values(by='Importance', ascending=False)

    # Limit the number of features to plot and group others into "Other"
    if top_n is not None and len(importance_agg_df) > top_n:
        top_features = importance_agg_df.head(top_n)
        other_importance = importance_agg_df.iloc[top_n:]['Importance'].sum()
        other_row = pd.DataFrame({'Features': ['Other'], 'Importance': [other_importance]})
        top_features = pd.concat([top_features, other_row], ignore_index=True)
    else:
        top_features = importance_agg_df

    # Plot feature importance using Seaborn (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features['Importance'], y=top_features['Features'], palette='viridis')

    # Add plot labels and title
    plt.title(f"{title} - Bar Plot", fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Show bar plot
    plt.tight_layout()
    plt.show()

    # Pie Chart for visualizing proportions
    plt.figure(figsize=(8, 8))
    plt.pie(top_features['Importance'], labels=top_features['Features'], autopct='%1.1f%%', colors=sns.color_palette("viridis", len(top_features)))
    plt.title(f"{title} - Pie Chart", fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

    # Show pie chart
    plt.show()
    # Pie Chart for visualizing proportions
    plt.figure(figsize=(8, 8))
    plt.pie(top_features['Importance'], labels=top_features['Features'], autopct='%1.1f%%', colors=sns.color_palette("viridis", len(top_features)))
    plt.title(f"{title} - Pie Chart", fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

    # Show pie chart
    plt.show()