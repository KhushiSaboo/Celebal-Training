"""
Utility functions shared across the project.
"""
import numpy as np
import pandas as pd


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).

    This is the metric used by the Kaggle "Store Item Demand Forecasting
    Challenge" leaderboard, so it is used here for both validation and
    reporting.

    SMAPE = 100/n * sum( |F - A| / ((|A| + |F|) / 2) )
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    # avoid division by zero when both true and pred are 0
    ratio = np.where(denom == 0, 0.0, diff / denom)
    return 100.0 * np.mean(ratio)


def smape_lgb(y_pred, dataset):
    """LightGBM-compatible custom eval metric wrapper."""
    y_true = dataset.get_label()
    return "smape", smape(y_true, y_pred), False


def load_raw_data(data_dir="data"):
    """Load train/test/sample_submission with parsed dates."""
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["date"])
    sample_sub = pd.read_csv(f"{data_dir}/sample_submission.csv")
    return train, test, sample_sub
