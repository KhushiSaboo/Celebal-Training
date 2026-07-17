"""
Train a global LightGBM model for the 500 store-item series.

Validation strategy
--------------------
The competition's test period is Jan-Mar 2018 (a 90-day horizon that is
NOT contiguous with the end of the training data in a "just forecast the
next 90 days" sense from a modelling perspective -- it IS immediately
after 2017-12-31, but it is also the same calendar months as the START of
every year in the training set). To get a validation score that is
representative of true forecasting performance, we hold out Jan-Mar 2017
as the validation fold and train on everything before it. This:
  * gives a validation window with the same length (90 days) and the same
    seasonal position (Jan-Mar) as the real test period,
  * avoids the common mistake of validating on a random/shuffled split,
    which would leak future information through lag features.

Once the validation score + best iteration count are established, we
retrain on the FULL training history (2013-2017) before generating the
actual submission (see predict.py).
"""
import json
import os

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import build_pivot, compute_feature_frame, attach_target, FEATURE_COLUMNS, CATEGORICAL_COLUMNS
from utils import smape, load_raw_data

DATA_DIR = "data"
MODEL_DIR = "models"
REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figures")

VAL_START = pd.Timestamp("2017-01-01")
VAL_END = pd.Timestamp("2017-03-31")
FEATURE_START = pd.Timestamp("2014-01-01")  # first date with a full 364-day lag available


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading raw data ...")
    train, _test, _sub = load_raw_data(DATA_DIR)

    full_range = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    print(f"Building pivot table: {len(full_range)} dates x "
          f"{train.groupby(['store', 'item']).ngroups} series ...")
    pivot = build_pivot(train, full_range)

    feature_dates = full_range[(full_range >= FEATURE_START)]
    print(f"Computing features for {len(feature_dates)} dates ...")
    feat = compute_feature_frame(pivot, feature_dates)
    feat = attach_target(feat, train)
    feat = feat.dropna(subset=["sales"])  # safety net

    for c in CATEGORICAL_COLUMNS:
        feat[c] = feat[c].astype("category")

    train_mask = feat["date"] < VAL_START
    valid_mask = (feat["date"] >= VAL_START) & (feat["date"] <= VAL_END)

    X_train, y_train = feat.loc[train_mask, FEATURE_COLUMNS], feat.loc[train_mask, "sales"]
    X_valid, y_valid = feat.loc[valid_mask, FEATURE_COLUMNS], feat.loc[valid_mask, "sales"]
    print(f"Train rows: {len(X_train):,} | Validation rows (Jan-Mar 2017): {len(X_valid):,}")

    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLUMNS)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=CATEGORICAL_COLUMNS, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "None",  # we score with our own SMAPE below
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 0.5,
        "verbose": -1,
        "seed": 42,
    }

    def smape_eval(preds, dataset):
        labels = dataset.get_label()
        preds = np.clip(preds, 0, None)
        return "smape", smape(labels, preds), False

    print("Training LightGBM with early stopping ...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_valid],
        valid_names=["valid"],
        feval=smape_eval,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True),
                   lgb.log_evaluation(period=100)],
    )

    best_iter = model.best_iteration
    val_preds = np.clip(model.predict(X_valid, num_iteration=best_iter), 0, None)
    val_smape = smape(y_valid.values, val_preds)
    print(f"Best iteration: {best_iter} | Validation SMAPE: {val_smape:.4f}")

    model.save_model(os.path.join(MODEL_DIR, "lgbm_model.txt"), num_iteration=best_iter)
    with open(os.path.join(MODEL_DIR, "best_iteration.json"), "w") as f:
        json.dump({"best_iteration": int(best_iter)}, f)

    # Feature importance plot
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).head(20)
    plt.figure(figsize=(8, 7))
    plt.barh(imp["feature"][::-1], imp["gain"][::-1], color="#3b6fa0")
    plt.xlabel("Total gain")
    plt.title("Top 20 LightGBM feature importances (gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=130)
    plt.close()

    # Validation report
    with open(os.path.join(REPORT_DIR, "model_report.md"), "w") as f:
        f.write("# Model training report\n\n")
        f.write(f"- Training rows: {len(X_train):,}\n")
        f.write(f"- Validation window: {VAL_START.date()} to {VAL_END.date()} "
                f"({len(X_valid):,} rows)\n")
        f.write(f"- Best boosting iteration (early stopping): **{best_iter}**\n")
        f.write(f"- Validation SMAPE: **{val_smape:.4f}**\n\n")
        f.write("## Top features by gain\n\n")
        f.write(imp.to_markdown(index=False))
        f.write("\n\n![feature importance](figures/feature_importance.png)\n")

    print("Saved model, best_iteration.json and model_report.md")


if __name__ == "__main__":
    main()
