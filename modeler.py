
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import optuna
from typing import List, Dict, Tuple, Union, Optional
import logging


# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Formatter: add timestamp, logger name, function, line number, message
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(funcName)s - line %(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# File handler
file_handler = logging.FileHandler("logs/modeler.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class Modeler:
    """
    A class for training, evaluating, and using ranking models.
    """

    def __init__(self, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None, model_params: Dict = None):
        """
        Initialize the Modeler with optional training and validation dataframes and model parameters.

        Parameters:
        -----------
        train_df : pd.DataFrame, optional
            The training dataframe to use for modeling
        val_df : pd.DataFrame, optional
            The validation dataframe to use for modeling
        model_params : Dict, optional
            Parameters for the LGBMRanker model
        """
        self.train_df = train_df.copy() if train_df is not None else None
        self.val_df = val_df.copy() if val_df is not None else None
        self.model = None
        self.feature_importances = None
        self.evals_result = None

        # Default model parameters
        self.model_params = model_params or {
            "objective": "lambdarank",
            "metric": "ndcg",
            "importance_type": "gain",
            "n_estimators": 100,
            "num_leaves": 80,
            "learning_rate": 0.1,
            "min_child_samples": 10,
            "lambda_l1": 1.0,
            "lambda_l2": 2.0,
            "random_state": 42,
            "n_jobs": -1
        }
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        logger.info("Initialized Modeler (plots directory created if it didn't exist)")

        # Target columns that should be excluded from features
        self.target_cols = ['srch_id', 'prop_id', 'position', 'click_bool', 'booking_bool',
                           'gross_bookings_usd', 'target', 'label']

    def set_train_data(self, train_df: pd.DataFrame) -> None:
        """
        Set the training dataframe to use for modeling.

        Parameters:
        -----------
        train_df : pd.DataFrame
            The training dataframe to use for modeling
        """
        self.train_df = train_df.copy()
        logger.info("Set new training dataframe for modeling")

    def set_val_data(self, val_df: pd.DataFrame) -> None:
        """
        Set the validation dataframe to use for modeling.

        Parameters:
        -----------
        val_df : pd.DataFrame
            The validation dataframe to use for modeling
        """
        self.val_df = val_df.copy()
        logger.info("Set new validation dataframe for modeling")

    def train_model(self, eval_at: List[int] = [5], target_col: str = 'target') -> lgb.LGBMRanker:
        """
        Train a LightGBM ranker model with separate training and validation sets.

        Parameters:
        -----------
        eval_at : List[int], optional
            Positions at which to evaluate NDCG
        target_col : str, optional
            Column name for the target variable

        Returns:
        --------
        lgb.LGBMRanker
            The trained model
        """
        if self.train_df is None:
            raise ValueError("No training dataframe. Please set a training dataframe first.")

        if self.val_df is None:
            raise ValueError("No validation dataframe. Please set a validation dataframe first.")

        logger.info("Training model with provided train/validation sets")

        # Sort by search ID for consistency
        train_df = self.train_df.sort_values(by="srch_id")
        val_df = self.val_df.sort_values(by="srch_id")

        # Create group sizes (number of items per search query)
        group_sizes_train = train_df.groupby("srch_id").size().tolist()
        group_sizes_val = val_df.groupby("srch_id").size().tolist()

        # Drop target columns for features
        to_drop = [col for col in self.target_cols if col in train_df.columns]

        # Prepare train/val sets
        train_x = train_df.drop(columns=to_drop)
        val_x = val_df.drop(columns=to_drop)
        train_y = train_df[target_col]
        val_y = val_df[target_col]

        logger.info(f"Training set shape: {train_x.shape}, Validation set shape: {val_x.shape}")

        # Initialize and train model
        model = lgb.LGBMRanker(**self.model_params)


        model.fit(
            X=train_x,
            y=train_y,
            group=group_sizes_train,
            eval_set=[(train_x, train_y), (val_x, val_y)],
            eval_group=[group_sizes_train, group_sizes_val],
            eval_at=eval_at,
            eval_metric="ndcg",
            eval_names=["train", "val"],
        )

        # Store model and evaluation results
        self.model = model
        self.evals_result = model.evals_result_

        # Calculate final NDCG score
        final_val_score = self.evals_result["val"][f"ndcg@{eval_at[0]}"][-1]
        logger.info(f"Final NDCG@{eval_at[0]} on validation set: {final_val_score:.5f}")

        return model

    def train_full_model(self, full_df: pd.DataFrame = None, target_col: str = 'target') -> lgb.LGBMRanker:
        """
        Train a LightGBM ranker model on the full dataset (training + validation).

        Parameters:
        -----------
        full_df : pd.DataFrame, optional
            Full dataframe to train on (if not provided, combines train_df and val_df)
        target_col : str, optional
            Column name for the target variable

        Returns:
        --------
        lgb.LGBMRanker
            The trained model
        """
        if full_df is not None:
            # Use provided full dataframe
            df = full_df.copy()
        elif self.train_df is not None and self.val_df is not None:
            # Combine training and validation dataframes
            df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        else:
            raise ValueError("No dataframe to train on. Please provide a full dataframe or set train_df and val_df.")

        logger.info("Training model on full dataset")

        # Drop target columns for features
        to_drop = [col for col in self.target_cols if col in df.columns]

        # Prepare training set
        train_x = df.drop(columns=to_drop)
        train_y = df[target_col]

        # Group sizes (number of items per search query)
        group_sizes = df.groupby("srch_id").size().tolist()

        logger.info(f"Full training set shape: {train_x.shape}")

        # Initialize and train model
        model = lgb.LGBMRanker(**self.model_params)

        model.fit(
            X=train_x,
            y=train_y,
            group=group_sizes,
            verbose=True
        )

        # Store model
        self.model = model

        return model

    def plot_training_metrics(self, eval_at: int = 5, figsize: Tuple[int, int] = (10, 5)) -> None:
        """
        Plot training and validation NDCG scores over boosting rounds.

        Parameters:
        -----------
        eval_at : int, optional
            Position at which NDCG was evaluated
        figsize : Tuple[int, int], optional
            Figure size for the plot
        """
        if self.evals_result is None:
            raise ValueError("No evaluation results available. Train a model with validation first.")

        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)

        # Create figure and plot
        plt.figure(figsize=figsize)
        plt.plot(self.evals_result["train"][f"ndcg@{eval_at}"], label=f"Train NDCG@{eval_at}")
        plt.plot(self.evals_result["val"][f"ndcg@{eval_at}"], label=f"Validation NDCG@{eval_at}")
        plt.xlabel("Boosting Round")
        plt.ylabel(f"NDCG@{eval_at}")
        plt.title(f"NDCG@{eval_at} over Boosting Rounds")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the figure first
        save_path = f"plots/ndcg@{eval_at}_over_boosting_rounds.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training metrics plot to {save_path}")

        # Then display the figure
        plt.show()

        # Close the figure to free memory
        plt.close()

        logger.info("Plotted training and validation NDCG scores")

    def get_feature_importance(self, top_n: int = 40, plot: bool = True,
                              figsize: Tuple[int, int] = (10, 6)) -> List[str]:
        """
        Get and optionally plot feature importance from the trained model.

        Parameters:
        -----------
        top_n : int, optional
            Number of top features to return
        plot : bool, optional
            Whether to plot feature importance
        figsize : Tuple[int, int], optional
            Figure size for the plot

        Returns:
        --------
        List[str]
            List of top feature names
        """
        if self.model is None:
            raise ValueError("No model available. Train a model first.")

        # Get feature importance
        importances = self.model.booster_.feature_importance(importance_type='gain')
        feature_names = self.model.booster_.feature_name()

        # Create DataFrame for easier handling
        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).head(top_n)

        # Store feature importances
        self.feature_importances = feat_imp

        # Plot if requested
        if plot:
            # Create plots directory if it doesn't exist
            os.makedirs("plots", exist_ok=True)

            plt.figure(figsize=figsize)
            plt.barh(feat_imp["feature"], feat_imp["importance"], color="skyblue")
            plt.xlabel("Feature importance (gain)")
            plt.title(f"Top {top_n} Feature Importances")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            # Save the figure first
            save_path = f"plots/top_{top_n}_feature_importances.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")

            # Then display the figure
            plt.show()

            # Close the figure to free memory
            plt.close()

        # Get list of top features
        top_features_list = feat_imp["feature"].tolist()
        logger.info(f"Extracted top {top_n} features by importance")

        return top_features_list

    def select_important_features(self, top_n: int = 40) -> List[str]:
        """
        Select the most important features from the trained model.

        Parameters:
        -----------
        top_n : int, optional
            Number of top features to return

        Returns:
        --------
        List[str]
            List of top feature names
        """
        return self.get_feature_importance(top_n=top_n, plot=False)

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on test data.

        Parameters:
        -----------
        test_df : pd.DataFrame
            Test dataframe to predict on

        Returns:
        --------
        pd.DataFrame
            Dataframe with predictions
        """
        if self.model is None:
            raise ValueError("No model available. Train a model first.")

        logger.info("Making predictions on test data")

        # Make a copy to avoid modifying the original
        test_df = test_df.copy()

        # Drop target columns if they exist
        drop_cols = [col for col in self.target_cols if col in test_df.columns and col != 'srch_id' and col != 'prop_id']
        X_test = test_df.drop(columns=drop_cols, errors='ignore')

        # Predict relevance scores
        predicted_relevance = self.model.predict(X_test)
        test_df["predicted_relevance"] = predicted_relevance

        logger.info(f"Made predictions for {len(test_df)} rows")

        return test_df

    def sort_and_save_predictions(self, test_df: pd.DataFrame, output_file: str = 'predictions.csv') -> pd.DataFrame:
        """
        Sort predictions by search ID and predicted relevance, and save to CSV.

        Parameters:
        -----------
        test_df : pd.DataFrame
            Test dataframe with predictions
        output_file : str, optional
            Path to save the sorted predictions

        Returns:
        --------
        pd.DataFrame
            Sorted dataframe with predictions
        """
        if 'predicted_relevance' not in test_df.columns:
            test_df = self.predict(test_df)

        logger.info("Sorting predictions and saving to CSV")

        # Sort by search ID and predicted relevance (descending)
        ranked = test_df.sort_values(
            by=["srch_id", "predicted_relevance"],
            ascending=[True, False]
        ).reset_index(drop=True)

        # Save to CSV
        submission = ranked[["srch_id", "prop_id"]]
        submission.to_csv(output_file, index=False)

        logger.info(f"Saved sorted predictions to {output_file}")

        return ranked

    def hyperparameter_tuning(self, n_trials: int = 20, target_col: str = 'target', 
                             visualize: bool = True, dashboard: bool = False) -> Dict:
        """
        Tune hyperparameters using Optuna with separate training and validation sets.

        Parameters:
        -----------
        n_trials : int, optional
            Number of Optuna trials
        target_col : str, optional
            Column name for the target variable
        visualize : bool, optional
            Whether to create visualization plots
        dashboard : bool, optional
            Whether to launch Optuna Dashboard

        Returns:
        --------
        Dict
            Best hyperparameters
        """
        if self.train_df is None:
            raise ValueError("No training dataframe. Please set a training dataframe first.")

        if self.val_df is None:
            raise ValueError("No validation dataframe. Please set a validation dataframe first.")

        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")

        # Sort by search ID for consistency
        train_df = self.train_df.sort_values(by="srch_id")
        val_df = self.val_df.sort_values(by="srch_id")

        # Create group sizes
        group_sizes_train = train_df.groupby("srch_id").size().tolist()
        group_sizes_val = val_df.groupby("srch_id").size().tolist()

        # Drop target columns for features
        to_drop = [col for col in self.target_cols if col in train_df.columns]

        # Prepare train/val sets
        train_x = train_df.drop(columns=to_drop)
        val_x = val_df.drop(columns=to_drop)
        train_y = train_df[target_col]
        val_y = val_df[target_col]

        # Create study with a SQLite storage for persistence
        storage_name = "sqlite:///optuna_study.db"
        study = optuna.create_study(direction="maximize", study_name="lgbm_ranker_tuning", 
                                   storage=storage_name, load_if_exists=True)
        
        # Launch Optuna Dashboard in a separate process if requested
        dashboard_process = None
        if dashboard:
            try:
                import subprocess
                import sys
                import time
                
                logger.info("Launching Optuna Dashboard. Access it at http://127.0.0.1:8080")
                dashboard_process = subprocess.Popen(
                    [sys.executable, "-m", "optuna", "dashboard", "--storage", storage_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # Give the dashboard a moment to start
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Failed to launch Optuna Dashboard: {e}")
                dashboard_process = None

        def objective(trial):
            # Define hyperparameters to tune
            param = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "importance_type": "gain",
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
                "random_state": 42,
                "n_jobs": -1
            }

            # Initialize and train model
            model = lgb.LGBMRanker(**param)

            # Create early stopping callback
            early_stopping_callback = lgb.early_stopping(50, verbose=False)

            model.fit(
                X=train_x,
                y=train_y,
                group=group_sizes_train,
                eval_set=[(val_x, val_y)],
                eval_group=[group_sizes_val],
                eval_at=[5],
                eval_metric="ndcg",
                callbacks=[early_stopping_callback]
                    )

            # Return validation NDCG score
            return model.best_score_['valid_0']['ndcg@5']

        try:
            # Run optimization
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value

            logger.info(f"Best NDCG@5: {best_score:.5f}")
            logger.info(f"Best parameters: {best_params}")

            # Update model parameters
            tuned_params = self.model_params.copy()
            tuned_params.update(best_params)
            self.model_params = tuned_params

            # Create visualization plots
            if visualize and len(study.trials) > 1:  # Only create plots if we have multiple trials
                try:
                    import matplotlib.pyplot as plt
                    
                    # Create plots directory if it doesn't exist
                    os.makedirs("plots/optuna", exist_ok=True)
                    
                    # Plot optimization history
                    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
                    if fig:
                        plt.figure(fig.figure)
                        plt.tight_layout()
                        plt.savefig("plots/optuna/optimization_history.png", dpi=300)
                        plt.close()
                    
                    # Plot parameter importances (requires at least 2 completed trials)
                    if len(study.trials) >= 2:
                        fig = optuna.visualization.matplotlib.plot_param_importances(study)
                        if fig:
                            plt.figure(fig.figure)
                            plt.tight_layout()
                            plt.savefig("plots/optuna/param_importances.png", dpi=300)
                            plt.close()
                    
                    # Plot parallel coordinate (requires at least 2 completed trials)
                    if len(study.trials) >= 2:
                        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
                        if fig:
                            plt.figure(fig.figure)
                            plt.tight_layout()
                            plt.savefig("plots/optuna/parallel_coordinate.png", dpi=300)
                            plt.close()
                    
                    logger.info("Saved Optuna visualization plots to plots/optuna/")
                except Exception as e:
                    logger.warning(f"Failed to create visualization plots: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
            
            return best_params
        
        finally:
            # Clean up dashboard process if it was started
            if dashboard_process is not None:
                logger.info("Stopping Optuna Dashboard...")
                dashboard_process.terminate()
                dashboard_process.wait()
                logger.info("Optuna Dashboard stopped.")

    def calculate_ndcg(self, test_df: pd.DataFrame, k: int = 5, target_col: str = 'target') -> float:
        """
        Calculate NDCG score for predictions.

        Parameters:
        -----------
        test_df : pd.DataFrame
            Test dataframe with predictions and true labels
        k : int, optional
            Position at which to calculate NDCG
        target_col : str, optional
            Column name for the target variable

        Returns:
        --------
        float
            NDCG score
        """
        if 'predicted_relevance' not in test_df.columns:
            test_df = self.predict(test_df)

        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test dataframe")

        logger.info(f"Calculating NDCG@{k} score")

        # Calculate NDCG per search query and average
        ndcg_scores = []

        for srch_id, group in test_df.groupby('srch_id'):
            true_relevance = group[target_col].values
            pred_relevance = group['predicted_relevance'].values

            # Skip if all true relevance values are the same (NDCG not defined)
            if len(set(true_relevance)) <= 1:
                continue

            # Calculate NDCG for this search query
            try:
                score = ndcg_score([true_relevance], [pred_relevance], k=min(k, len(true_relevance)))
                ndcg_scores.append(score)
            except Exception as e:
                logger.warning(f"Error calculating NDCG for search ID {srch_id}: {e}")

        # Average NDCG across all search queries
        avg_ndcg = np.mean(ndcg_scores)
        logger.info(f"Average NDCG@{k}: {avg_ndcg:.5f}")

        return avg_ndcg

    def save_model(self, filepath: str = 'model.pkl') -> None:
        """
        Save the trained model to a file.

        Parameters:
        -----------
        filepath : str, optional
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        import pickle

        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Saved model to {filepath}")

    def load_model(self, filepath: str = 'model.pkl') -> None:
        """
        Load a trained model from a file.

        Parameters:
        -----------
        filepath : str, optional
            Path to load the model from
        """
        import pickle

        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f"Loaded model from {filepath}")

    def run_pipeline(self, test_df: pd.DataFrame = None, output_file: str = 'predictions.csv',
                    use_full_model: bool = False, full_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run the full modeling pipeline: train, evaluate, and predict.

        Parameters:
        -----------
        test_df : pd.DataFrame, optional
            Test dataframe to predict on
        output_file : str, optional
            Path to save the predictions
        use_full_model : bool, optional
            Whether to train on the full dataset (train+val) for final predictions
        full_df : pd.DataFrame, optional
            Full dataframe to train on if use_full_model is True

        Returns:
        --------
        pd.DataFrame
            Dataframe with predictions
        """
        if self.train_df is None:
            raise ValueError("No training dataframe. Please set a training dataframe first.")

        if not use_full_model and self.val_df is None:
            raise ValueError("No validation dataframe. Please set a validation dataframe first.")

        if use_full_model:
            # Train on full dataset for final predictions
            logger.info("Training on full dataset for final predictions")
            self.train_full_model(full_df=full_df)
        else:
            # Train model with validation split
            logger.info("Training model with validation split")
            self.train_model()

            # Plot training metrics
            self.plot_training_metrics()

        # Get feature importance
        self.get_feature_importance()

        # If test data is provided, make predictions
        if test_df is not None:
            predictions = self.predict(test_df)
            ranked = self.sort_and_save_predictions(predictions, output_file)
            return ranked

        return None

    def plot_learning_curve(self, n_estimators_list: List[int] = None,
                           target_col: str = 'target', figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot learning curve showing model performance with different numbers of estimators.

        Parameters:
        -----------
        n_estimators_list : List[int], optional
            List of n_estimators values to try (default: [10, 50, 100, 200, 300])
        target_col : str, optional
            Column name for the target variable
        figsize : Tuple[int, int], optional
            Figure size for the plot
        """
        if self.train_df is None or self.val_df is None:
            raise ValueError("Both training and validation dataframes are required.")

        # Default n_estimators values if not provided
        if n_estimators_list is None:
            n_estimators_list = [10, 50, 100, 200, 300]

        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)

        # Sort by search ID for consistency
        train_df = self.train_df.sort_values(by="srch_id")
        val_df = self.val_df.sort_values(by="srch_id")

        # Create group sizes
        group_sizes_train = train_df.groupby("srch_id").size().tolist()
        group_sizes_val = val_df.groupby("srch_id").size().tolist()

        # Drop target columns for features
        to_drop = [col for col in self.target_cols if col in train_df.columns]

        # Prepare train/val sets
        train_x = train_df.drop(columns=to_drop)
        val_x = val_df.drop(columns=to_drop)
        train_y = train_df[target_col]
        val_y = val_df[target_col]

        # Lists to store results
        train_scores = []
        val_scores = []

        # Train models with different n_estimators
        for n_est in n_estimators_list:
            logger.info(f"Training model with {n_est} estimators")

            # Update model parameters
            params = self.model_params.copy()
            params['n_estimators'] = n_est

            # Initialize and train model
            model = lgb.LGBMRanker(**params)

            model.fit(
                X=train_x,
                y=train_y,
                group=group_sizes_train,
                eval_set=[(train_x, train_y), (val_x, val_y)],
                eval_group=[group_sizes_train, group_sizes_val],
                eval_at=[5],
                eval_metric="ndcg",
                eval_names=["train", "val"],
                verbose=False
            )

            # Get final scores
            train_score = model.evals_result_["train"]["ndcg@5"][-1]
            val_score = model.evals_result_["val"]["ndcg@5"][-1]

            train_scores.append(train_score)
            val_scores.append(val_score)

            logger.info(f"n_estimators={n_est}, Train NDCG@5={train_score:.5f}, Val NDCG@5={val_score:.5f}")

        # Plot learning curve
        plt.figure(figsize=figsize)
        plt.plot(n_estimators_list, train_scores, 'o-', label='Training NDCG@5')
        plt.plot(n_estimators_list, val_scores, 'o-', label='Validation NDCG@5')
        plt.xlabel('Number of Estimators')
        plt.ylabel('NDCG@5')
        plt.title('Learning Curve: NDCG@5 vs Number of Estimators')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the figure
        save_path = "plots/learning_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning curve plot to {save_path}")

        # Display the figure
        plt.show()

        # Close the figure to free memory
        plt.close()
