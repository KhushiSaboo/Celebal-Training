"""
Feature engineering for the store-item demand forecasting problem.

Approach
--------
The 500 store-item series are modelled with a single "global" gradient
boosted tree model (LightGBM) rather than 500 independent models. This:
  * lets the model share statistical strength across series (a new item
    launched in one store can borrow seasonal patterns from similar items),
  * scales trivially to many more series,
  * still lets the tree model recover per-series behaviour through the
    (store, item) categorical features combined with lag/rolling features.

All lag and rolling statistics are computed from a wide "pivot" table
(rows = calendar date, columns = (store, item) pair, values = sales).
Working in this wide format lets us compute every lag/rolling feature for
every series with a single vectorised `.shift()` / `.rolling()` call
instead of looping over 500 individual series.
"""
import numpy as np
import pandas as pd

LAGS = [7, 14, 21, 28, 35, 60, 90, 364]
ROLL_WINDOWS = [7, 14, 28, 90, 365]


def build_pivot(train_df: pd.DataFrame, full_date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build a (date x store_item) pivot table of sales, reindexed onto the
    full date range (so future test dates exist as all-NaN rows ready to be
    filled in during recursive forecasting).
    """
    pivot = train_df.pivot(index="date", columns=["store", "item"], values="sales")
    pivot = pivot.reindex(full_date_range)
    pivot.index.name = "date"
    return pivot


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df[date_col]
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dayofweek"] = d.dt.dayofweek
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def compute_feature_frame(pivot: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute lag + rolling + calendar features for every (store, item) pair
    on the requested `dates`, using only information available strictly
    before each date (no leakage).

    Returns a long dataframe with one row per (date, store, item).
    """
    feature_frames = []

    lag_frames = {}
    for lag in LAGS:
        lag_frames[f"lag_{lag}"] = pivot.shift(lag).loc[dates]

    shifted = pivot.shift(1)
    roll_mean_frames = {}
    roll_std_frames = {}
    for w in ROLL_WINDOWS:
        roll_mean_frames[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=max(3, w // 4)).mean().loc[dates]
        roll_std_frames[f"roll_std_{w}"] = shifted.rolling(w, min_periods=max(3, w // 4)).std().loc[dates]

    trend = (roll_mean_frames["roll_mean_7"] / roll_mean_frames["roll_mean_28"].replace(0, np.nan))

    store_item_cols = pivot.columns  # MultiIndex (store, item)

    for d in dates:
        rows = pd.DataFrame(index=store_item_cols)
        rows["date"] = d
        for name, frame in lag_frames.items():
            rows[name] = frame.loc[d].values
        for name, frame in roll_mean_frames.items():
            rows[name] = frame.loc[d].values
        for name, frame in roll_std_frames.items():
            rows[name] = frame.loc[d].values
        rows["trend_ratio"] = trend.loc[d].values
        rows = rows.reset_index()  # brings back store, item columns
        feature_frames.append(rows)

    out = pd.concat(feature_frames, ignore_index=True)
    out = add_calendar_features(out)
    return out


def attach_target(feature_df: pd.DataFrame, train_long: pd.DataFrame) -> pd.DataFrame:
    """Left-join the actual sales value onto the feature frame (for training)."""
    return feature_df.merge(train_long[["date", "store", "item", "sales"]],
                             on=["date", "store", "item"], how="left")


FEATURE_COLUMNS = (
    ["store", "item", "year", "month", "day", "dayofweek", "dayofyear",
     "weekofyear", "quarter", "is_weekend", "is_month_start", "is_month_end",
     "month_sin", "month_cos", "dow_sin", "dow_cos", "trend_ratio"]
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
)

CATEGORICAL_COLUMNS = ["store", "item"]
