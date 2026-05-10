"""
visualizer.py
-------------
Generates performance metric plots for model comparison
and data distribution analysis.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
OUTPUT_DIR = "outputs"


def _save(fig, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart: {path}")


def plot_model_comparison(results: dict, best_model: str) -> None:
    """Bar chart comparing CV R², MAE, RMSE across all models."""
    models = list(results.keys())
    r2 = [results[m]["cv_r2_mean"] for m in models]
    mae = [results[m]["cv_mae_mean"] for m in models]
    rmse = [results[m]["cv_rmse_mean"] for m in models]

    x = np.arange(len(models))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, r2, width, label="CV R²", color=colors[0])
    bars2 = ax.bar(x, mae, width, label="CV MAE", color=colors[1])
    bars3 = ax.bar(x + width, rmse, width, label="CV RMSE", color=colors[2])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison (5-Fold Cross-Validation)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Highlight best model
    best_idx = models.index(best_model)
    ax.get_xticklabels()[best_idx].set_color("#E91E63")
    ax.get_xticklabels()[best_idx].set_fontweight("bold")
    ax.annotate("Best", xy=(best_idx - width, r2[best_idx]),
                xytext=(best_idx - width, r2[best_idx] + 0.02),
                ha="center", fontsize=9, color="#E91E63", fontweight="bold")

    _save(fig, "model_comparison.png")


def plot_actual_vs_predicted(y_true, y_pred, model_name: str) -> None:
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#2196F3", edgecolors="white", linewidth=0.3, s=40)

    # Perfect prediction line
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")

    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, "actual_vs_predicted.png")


def plot_residuals(y_true, y_pred, model_name: str) -> None:
    """Residual plot to check for patterns/bias."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, color="#FF5722", s=30)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted Values", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title(f"Residual Plot — {model_name}", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)
    _save(fig, "residuals.png")


def plot_feature_distribution(df: pd.DataFrame, max_cols: int = 6) -> None:
    """Histograms for numeric feature distributions."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    n = len(numeric_cols)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=30, color="#4CAF50", edgecolor="white", alpha=0.8)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "feature_distributions.png")


def plot_benchmark(benchmark_result: dict) -> None:
    """Bar chart comparing sequential vs parallel processing time."""
    labels = ["Sequential", "Parallel"]
    times = [benchmark_result["sequential_time_s"], benchmark_result["parallel_time_s"]]
    colors = ["#FF7043", "#42A5F5"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, times, color=colors, width=0.4, edgecolor="white")
    ax.bar_label(bars, fmt="%.3fs", fontsize=11, padding=4)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(
        f"Parallel vs Sequential Processing\n({benchmark_result['workers']} workers | {benchmark_result['rows']:,} rows | {benchmark_result['speedup']}x speedup)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "benchmark_comparison.png")
