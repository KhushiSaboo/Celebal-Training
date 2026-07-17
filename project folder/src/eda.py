"""
Exploratory data analysis for the store-item demand dataset.

Generates a handful of plots into reports/figures/ that motivate the
feature-engineering choices made in features.py / train.py:
  * overall_trend.png     -> multi-year upward trend + yearly seasonality
  * weekly_seasonality.png-> strong day-of-week effect (weekend uplift)
  * monthly_seasonality.png -> summer peak / winter trough
  * store_item_heatmap.png -> average demand varies a lot store x item
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_raw_data

FIG_DIR = "reports/figures"


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    train, test, _sub = load_raw_data("data")

    # 1. overall trend: total daily sales across all 500 series
    daily = train.groupby("date")["sales"].sum()
    plt.figure(figsize=(11, 4))
    plt.plot(daily.index, daily.values, linewidth=0.7, color="#2b6cb0")
    plt.title("Total daily sales across all 500 store-item series (2013-2017)")
    plt.xlabel("Date")
    plt.ylabel("Total sales")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/overall_trend.png", dpi=130)
    plt.close()

    # 2. weekly seasonality
    dow = train.copy()
    dow["dayofweek"] = dow["date"].dt.dayofweek
    dow_avg = dow.groupby("dayofweek")["sales"].mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, dow_avg.values, color="#3b6fa0")
    plt.title("Average sales by day of week")
    plt.ylabel("Average sales (per store-item)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/weekly_seasonality.png", dpi=130)
    plt.close()

    # 3. monthly seasonality
    mon = train.copy()
    mon["month"] = mon["date"].dt.month
    mon_avg = mon.groupby("month")["sales"].mean()
    plt.figure(figsize=(7, 4))
    plt.plot(mon_avg.index, mon_avg.values, marker="o", color="#c05621")
    plt.xticks(range(1, 13))
    plt.title("Average sales by month (seasonality)")
    plt.xlabel("Month")
    plt.ylabel("Average sales (per store-item)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/monthly_seasonality.png", dpi=130)
    plt.close()

    # 4. store x item heatmap of mean sales
    pivot_mean = train.pivot_table(index="store", columns="item", values="sales", aggfunc="mean")
    plt.figure(figsize=(12, 4))
    im = plt.imshow(pivot_mean.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Mean daily sales")
    plt.yticks(range(len(pivot_mean.index)), pivot_mean.index)
    plt.xlabel("Item")
    plt.ylabel("Store")
    plt.title("Mean daily sales by store x item")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/store_item_heatmap.png", dpi=130)
    plt.close()

    # print a short summary to stdout for the README / report
    print("Date range:", train["date"].min().date(), "to", train["date"].max().date())
    print("Stores:", train["store"].nunique(), "| Items:", train["item"].nunique(),
          "| Series:", train.groupby(["store", "item"]).ngroups)
    print("Sales stats:\n", train["sales"].describe())
    print("Missing values:\n", train.isnull().sum())
    print(f"Saved plots to {FIG_DIR}/")


if __name__ == "__main__":
    main()
