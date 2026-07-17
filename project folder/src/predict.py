"""
Generate the final Jan-Mar 2018 forecast (test.csv) and submission.csv.

Because lag/rolling features for a given day depend on the (unknown)
sales of previous days in the forecast horizon, the 90-day test period
cannot be predicted in a single batch. Instead this script performs
walk-forward (recursive) forecasting:

    for each date d in the 90-day test horizon (in order):
        1. compute lag/rolling/calendar features for d using all actual
           sales history PLUS whatever days of the horizon have already
           been predicted
        2. predict sales for every (store, item) pair on date d
        3. write the prediction back into the sales pivot table so it can
           be used as history for later dates (e.g. lag_7 needs the
           prediction from day d-7 once d-7 is itself inside the horizon)

The final model is retrained on the FULL 2013-2017 training history (using
the optimal number of boosting rounds found in train.py's validation) so
that no information is thrown away before producing the submission.
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import build_pivot, compute_feature_frame, attach_target, FEATURE_COLUMNS, CATEGORICAL_COLUMNS
from utils import smape, load_raw_data

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
FEATURE_START = pd.Timestamp("2014-01-01")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading raw data ...")
    train, test, sample_sub = load_raw_data(DATA_DIR)

    with open(os.path.join(MODEL_DIR, "best_iteration.json")) as f:
        best_iteration = json.load(f)["best_iteration"]
    print(f"Using best_iteration = {best_iteration} from validation run")

    full_range = pd.date_range(train["date"].min(), test["date"].max(), freq="D")
    pivot = build_pivot(train, full_range)  # test dates present as all-NaN rows

    # ---- retrain final model on ALL of 2013-2017 -------------------------
    print("Building training features on full history (2014-01-01 .. 2017-12-31) ...")
    train_feature_dates = full_range[(full_range >= FEATURE_START) & (full_range <= train["date"].max())]
    feat = compute_feature_frame(pivot, train_feature_dates)
    feat = attach_target(feat, train)
    feat = feat.dropna(subset=["sales"])
    for c in CATEGORICAL_COLUMNS:
        feat[c] = feat[c].astype("category")

    X_full, y_full = feat[FEATURE_COLUMNS], feat["sales"]

    params = {
        "objective": "regression",
        "metric": "None",
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

    print(f"Training final model on {len(X_full):,} rows for {best_iteration} rounds ...")
    lgb_full = lgb.Dataset(X_full, label=y_full, categorical_feature=CATEGORICAL_COLUMNS)
    final_model = lgb.train(params, lgb_full, num_boost_round=best_iteration)
    final_model.save_model(os.path.join(MODEL_DIR, "lgbm_model_final.txt"))

    # ---- walk-forward forecast over the 90-day test horizon --------------
    test_dates = pd.DatetimeIndex(sorted(test["date"].unique()))
    print(f"Forecasting {len(test_dates)} days recursively ...")
    for i, d in enumerate(test_dates):
        feat_d = compute_feature_frame(pivot, pd.DatetimeIndex([d]))
        for c in CATEGORICAL_COLUMNS:
            feat_d[c] = feat_d[c].astype("category").cat.set_categories(
                X_full[c].cat.categories
            )
        preds = final_model.predict(feat_d[FEATURE_COLUMNS])
        preds = np.clip(preds, 0, None)

        # write predictions back into pivot so future lag/rolling features
        # can see them
        col_index = pd.MultiIndex.from_frame(feat_d[["store", "item"]])
        pivot.loc[d, col_index] = preds
        if (i + 1) % 30 == 0 or i == 0:
            print(f"  forecasted {i + 1}/{len(test_dates)} days (last date {d.date()})")

    # ---- assemble submission ----------------------------------------------
    forecast_long = (
        pivot.loc[test_dates]
        .rename_axis("date")
        .stack(["store", "item"], future_stack=True)
        .rename("sales")
        .reset_index()
    )
    submission = test.merge(forecast_long, on=["date", "store", "item"], how="left")
    submission["sales"] = submission["sales"].round().astype(int)
    submission = submission[["id", "sales"]].sort_values("id")

    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(submission):,} rows)")
    print(submission.head())

    assert list(submission["id"]) == list(sample_sub["id"]), "id column does not match sample_submission order"


if __name__ == "__main__":
    main()
