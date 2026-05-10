"""
model_trainer.py
----------------
Trains, evaluates, and compares multiple ML models.
Uses cross-validation and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains multiple regression models, evaluates with cross-validation,
    and selects the best-performing one.
    """

    def __init__(self, target_column: str, cv_folds: int = 5, random_state: int = 42):
        self.target_column = target_column
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(
                max_depth=8,
                min_samples_split=10,
                random_state=random_state
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1
            )
        }

        self.results = {}
        self.best_model_name = None
        self.best_model = None

    def train_and_evaluate(self, df: pd.DataFrame) -> dict:
        """
        Train all models and evaluate with K-Fold cross-validation.

        Returns:
            Dict of results for each model.
        """
        X, y = self._split_features_target(df)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        logger.info(f"Training {len(self.models)} models with {self.cv_folds}-fold cross-validation...")

        for name, model in self.models.items():
            logger.info(f"Training: {name}")

            # Cross-validated scores
            r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
            mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
            rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))

            # Fit full model for final evaluation
            model.fit(X, y)
            y_pred = model.predict(X)

            self.results[name] = {
                "cv_r2_mean": round(r2_scores.mean(), 4),
                "cv_r2_std": round(r2_scores.std(), 4),
                "cv_mae_mean": round(mae_scores.mean(), 4),
                "cv_rmse_mean": round(rmse_scores.mean(), 4),
                "train_r2": round(r2_score(y, y_pred), 4),
                "train_mae": round(mean_absolute_error(y, y_pred), 4),
                "train_rmse": round(np.sqrt(mean_squared_error(y, y_pred)), 4),
            }

            logger.info(f"{name} → CV R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

        self._select_best_model(X, y)
        return self.results

    def _select_best_model(self, X, y):
        """Select model with highest CV R² score."""
        best_name = max(self.results, key=lambda k: self.results[k]["cv_r2_mean"])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        self.best_model.fit(X, y)
        logger.info(f"Best model selected: {best_name} (CV R²={self.results[best_name]['cv_r2_mean']})")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using best model."""
        if self.best_model is None:
            raise RuntimeError("No model trained yet. Call train_and_evaluate() first.")
        X = df.drop(columns=[self.target_column], errors="ignore")
        return self.best_model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances if best model supports it."""
        model = self.best_model
        X_cols = None  # populated externally
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            return pd.Series(np.abs(model.coef_)).sort_values(ascending=False)
        return pd.Series()

    def save_results(self, filepath: str = "outputs/model_results.json") -> None:
        """Save evaluation results to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        output = {
            "best_model": self.best_model_name,
            "results": self.results
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {filepath}")

    def print_summary(self) -> None:
        """Print a formatted comparison table."""
        print("\n" + "="*70)
        print(f"{'Model':<20} {'CV R²':>10} {'CV MAE':>10} {'CV RMSE':>10}")
        print("="*70)
        for name, res in self.results.items():
            marker = " ← BEST" if name == self.best_model_name else ""
            print(f"{name:<20} {res['cv_r2_mean']:>10.4f} {res['cv_mae_mean']:>10.4f} {res['cv_rmse_mean']:>10.4f}{marker}")
        print("="*70 + "\n")

    def _split_features_target(self, df: pd.DataFrame):
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y
