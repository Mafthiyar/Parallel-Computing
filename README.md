# Parallel Data Processing & Prediction Pipeline

An end-to-end machine learning pipeline built in **Python** that processes large datasets in parallel using multiprocessing, cleans and transforms data, and trains/evaluates multiple prediction models.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Parallelism | Python Multiprocessing |
| Visualization | Matplotlib |

---

## Features

- **Parallel Processing** — Splits large DataFrames into chunks, processes each chunk in a separate CPU process using `multiprocessing.Pool`
- **Benchmark Comparison** — Measures and visualizes sequential vs parallel runtime with speedup ratio
- **Smart Preprocessing** — Handles missing values (median/mode imputation), outlier capping (IQR Winsorization), categorical encoding, and feature scaling
- **Multi-Model Training** — Trains Linear Regression, Decision Tree, and Random Forest with 5-fold cross-validation
- **Automated Model Selection** — Selects best model based on CV R² score
- **Visualization Suite** — Generates charts for model comparison, actual vs predicted, residuals, and feature distributions

---

## Project Structure

```
├── main.py                    # Pipeline entry point (run this)
├── data_loader.py             # CSV ingestion and data profiling
├── parallel_processor.py      # Parallel chunk processing engine
├── preprocessor.py            # Data cleaning and feature engineering
├── model_trainer.py           # ML model training and evaluation
├── visualizer.py              # Chart generation
├── requirements.txt
├── data/
│   └── sample_data.csv        # Auto-generated if no CSV provided
└── outputs/
    ├── processed_data.csv
    ├── model_results.json
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    ├── residuals.png
    ├── feature_distributions.png
    ├── benchmark_comparison.png
    └── pipeline.log
```

---

## Pipeline Flow

```
CSV Input
   ↓
Data Loading & Profiling      (data_loader.py)
   ↓
Parallel Feature Engineering  (parallel_processor.py)  ← multiprocessing here
   ↓
Data Cleaning & Preprocessing (preprocessor.py)
   ↓
Model Training & Evaluation   (model_trainer.py)        ← cross-validation
   ↓
Visualization & Reports       (visualizer.py)
```

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Mafthiyar/Parallel-Computing.git
cd Parallel-Computing

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run with your own CSV (must have a 'target' column)
python main.py path/to/your/data.csv target_column_name

# 3b. Run with auto-generated sample data (no CSV needed)
python main.py
```

Results and charts saved to the `outputs/` folder.

---

## Sample Output

```
[INFO] Loaded 5050 rows, 8 columns.
[INFO] ParallelProcessor initialized with 8 workers.
[INFO] Benchmark: Sequential=1.24s | Parallel=0.31s | Speedup=4.00x
[INFO] Preprocessing complete. Final shape: (5000, 10)

======================================================================
Model                   CV R²     CV MAE    CV RMSE
======================================================================
LinearRegression        0.9823     3.2411     4.1823
DecisionTree            0.9541     5.8932     7.2341
RandomForest            0.9891     2.7823     3.6712  ← BEST
======================================================================

[INFO] Pipeline complete! Check outputs/ folder.
```

---

## Key Concepts Demonstrated

- **Concurrent processing** — Python `multiprocessing.Pool` for CPU-bound parallelism
- **Stateful preprocessing** — Fit on train, transform on test (no data leakage)
- **Cross-validation** — K-Fold CV to prevent overfitting
- **Outlier handling** — IQR-based Winsorization
- **Model selection** — Automated selection by CV performance

---

## Author

**Shaik Mofthiyar**
[LinkedIn](https://www.linkedin.com/in/mofthiyar-shaik-44869b28b) | [GitHub](https://github.com/Mafthiyar)
