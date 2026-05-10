"""
main.py
-------
Main pipeline entry point.
Runs the full flow: Load → Parallel Process → Preprocess → Train → Evaluate → Visualize
"""

import logging
import sys
import os
import pandas as pd
import numpy as np

from data_loader import load_csv, profile_data, save_csv
from parallel_processor import ParallelProcessor, clean_and_engineer_features
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from visualizer import (
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_distribution,
    plot_benchmark
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


def generate_sample_dataset(n_rows: int = 5000, filepath: str = "data/sample_data.csv") -> str:
    """Generate a synthetic dataset if no real CSV is provided."""
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)

    df = pd.DataFrame({
        "feature_1": np.random.normal(50, 15, n_rows),
        "feature_2": np.random.normal(30, 10, n_rows),
        "feature_3": np.random.exponential(5, n_rows),
        "feature_4": np.random.uniform(0, 100, n_rows),
        "feature_5": np.random.normal(200, 50, n_rows),
        "category_a": np.random.choice(["TypeA", "TypeB", "TypeC"], n_rows),
        "category_b": np.random.choice(["Low", "Medium", "High"], n_rows),
        "target": None
    })

    # Target is a noisy linear combination of features
    df["target"] = (
        2.5 * df["feature_1"] +
        1.8 * df["feature_2"] -
        0.5 * df["feature_3"] +
        0.3 * df["feature_4"] +
        np.random.normal(0, 10, n_rows)
    )

    # Inject some missing values and duplicates for realism
    for col in ["feature_2", "feature_4", "category_a"]:
        mask = np.random.rand(n_rows) < 0.05
        df.loc[mask, col] = np.nan

    df = pd.concat([df, df.sample(50, random_state=42)], ignore_index=True)  # add duplicates

    df.to_csv(filepath, index=False)
    logger.info(f"Generated sample dataset: {filepath} ({len(df)} rows)")
    return filepath


def run_pipeline(csv_path: str = None, target_column: str = "target"):
    logger.info("=" * 60)
    logger.info("PARALLEL DATA PROCESSING & PREDICTION PIPELINE")
    logger.info("=" * 60)

    os.makedirs("outputs", exist_ok=True)

    # STEP 1: Load Data
    if csv_path is None or not os.path.exists(csv_path):
        logger.info("No CSV provided. Generating sample dataset...")
        csv_path = generate_sample_dataset()

    df = load_csv(csv_path)
    profile = profile_data(df)
    logger.info(f"Dataset shape: {profile['shape']} | Duplicates: {profile['duplicates']}")

    # STEP 2: Parallel Feature Engineering
    processor = ParallelProcessor()
    benchmark = processor.benchmark(df.drop(columns=[target_column]), clean_and_engineer_features)
    df_parallel = processor.process(df, clean_and_engineer_features)
    plot_benchmark(benchmark)
    logger.info(f"Speedup achieved: {benchmark['speedup']}x over sequential")

    # Restore target column if lost
    if target_column not in df_parallel.columns and target_column in df.columns:
        df_parallel[target_column] = df[target_column].values[:len(df_parallel)]

    # STEP 3: Preprocessing
    preprocessor = DataPreprocessor(target_column=target_column)
    df_clean = preprocessor.fit_transform(df_parallel)
    save_csv(df_clean, "outputs/processed_data.csv")
    plot_feature_distribution(df_clean)

    # STEP 4: Train & Evaluate Models
    trainer = ModelTrainer(target_column=target_column, cv_folds=5)
    results = trainer.train_and_evaluate(df_clean)
    trainer.print_summary()
    trainer.save_results("outputs/model_results.json")

    # STEP 5: Visualize Results
    plot_model_comparison(results, trainer.best_model_name)

    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    y_pred = trainer.best_model.predict(X)
    plot_actual_vs_predicted(y, y_pred, trainer.best_model_name)
    plot_residuals(y, y_pred, trainer.best_model_name)

    logger.info("Pipeline complete! Check outputs/ folder for results and charts.")
    return results, trainer.best_model_name


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    target = sys.argv[2] if len(sys.argv) > 2 else "target"
    run_pipeline(csv_path=csv_file, target_column=target)
