"""Exploratory analysis (Task 2).

Produces exactly the outputs the assignment asks for, plus two self-chosen
analyses on the busiest square that are chosen to feed directly into the
modelling decisions:

* **Autocorrelation (ACF/PACF)** - tells us how long an input window has to be.
  If the ACF still shows a large spike at lag 144 (24 h) and 1008 (7 days), a
  model whose receptive field is shorter than a day cannot see the dominant
  structure.
* **Stationarity + STL decomposition** - tells us whether the series needs
  differencing, and how much of its variance is deterministic seasonality
  versus the residual that actually has to be learned.

Every figure is saved to ``outputs/figures`` and every number to
``outputs/tables`` so the report quotes computed values, never eyeballed ones.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import dataio

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --------------------------------------------------------------------------
# 2.1 Distribution across the 10,000 areas
# --------------------------------------------------------------------------
def plot_traffic_distribution(grid: dataio.Grid) -> pd.DataFrame:
    totals = grid.totals().sort_values(ascending=False)
    positive = totals[totals > 0]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    axes[0].hist(totals.to_numpy(), bins=80, color="#3b6ea5", edgecolor="white", linewidth=0.3)
    axes[0].set(xlabel="Total Internet activity", ylabel="Number of areas",
                title="Linear scale")

    axes[1].hist(np.log10(positive.to_numpy()), bins=80, color="#3b6ea5",
                 edgecolor="white", linewidth=0.3)
    axes[1].set(xlabel="log10(total Internet activity)", ylabel="Number of areas",
                title="Log scale")

    # Lorenz curve: how concentrated is the traffic?
    ordered = np.sort(totals.to_numpy())
    cum = np.cumsum(ordered) / ordered.sum()
    frac = np.arange(1, len(ordered) + 1) / len(ordered)
    axes[2].plot(frac, cum, color="#b5482a", linewidth=1.6)
    axes[2].plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.9)
    axes[2].set(xlabel="Cumulative fraction of areas", ylabel="Cumulative fraction of traffic",
                title="Concentration (Lorenz)")

    fig.suptitle("Distribution of total Internet traffic across the 10,000 Milan grid areas",
                 y=1.03)
    fig.savefig(C.FIGURE_DIR / "fig1_traffic_distribution.png")
    plt.close(fig)

    gini = float(1 - 2 * np.trapezoid(cum, frac))
    share_top1 = float(totals.head(100).sum() / totals.sum())
    share_top10 = float(totals.head(1000).sum() / totals.sum())

    stats = pd.DataFrame(
        [
            {"statistic": "areas", "value": int(len(totals))},
            {"statistic": "areas with zero traffic", "value": int((totals == 0).sum())},
            {"statistic": "mean", "value": float(totals.mean())},
            {"statistic": "median", "value": float(totals.median())},
            {"statistic": "std", "value": float(totals.std())},
            {"statistic": "skewness", "value": float(totals.skew())},
            {"statistic": "kurtosis", "value": float(totals.kurtosis())},
            {
                "statistic": "max / median ratio",
                "value": float(totals.max() / totals.median()) if totals.median() > 0 else np.nan,
            },
            {
                "statistic": "max / median ratio (non-zero areas)",
                "value": float(positive.max() / positive.median()) if len(positive) else np.nan,
            },
            {"statistic": "Gini coefficient", "value": gini},
            {"statistic": "share of traffic in busiest 1% of areas", "value": share_top1},
            {"statistic": "share of traffic in busiest 10% of areas", "value": share_top10},
        ]
    )
    stats.to_csv(C.TABLE_DIR / "table1_distribution_stats.csv", index=False)
    return stats


# --------------------------------------------------------------------------
# 2.2 Time series for the five named areas
# --------------------------------------------------------------------------
def plot_five_series(grid: dataio.Grid, top_k: int = 3) -> list[int]:
    top = grid.top_squares(top_k)
    squares = top + C.REFERENCE_SQUARES
    labels = ([f"Rank {i+1}: square {s}" for i, s in enumerate(top)]
              + [f"Square {s} (reference)" for s in C.REFERENCE_SQUARES])

    fig, axes = plt.subplots(len(squares), 1, figsize=(12, 1.9 * len(squares)), sharex=True)
    for ax, sq, label in zip(np.atleast_1d(axes), squares, labels):
        s = dataio.slice_window(grid.series(sq), *C.EDA_WINDOW)
        ax.plot(s.index, s.to_numpy(), linewidth=0.7, color="#2c5f8a")
        ax.set_ylabel("Activity")
        ax.set_title(label, loc="left", fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel(f"{C.EDA_WINDOW[0]} to {C.EDA_WINDOW[1]} (Europe/Rome)")
    fig.suptitle("Internet traffic, first two weeks of the observation period", y=1.005)
    fig.savefig(C.FIGURE_DIR / "fig2_five_series_two_weeks.png")
    plt.close(fig)

    # Comparable summary statistics so the discussion is quantitative.
    rows = []
    for sq, label in zip(squares, labels):
        s = dataio.slice_window(grid.series(sq), *C.EDA_WINDOW)
        daily = s.groupby(s.index.dayofweek).mean()
        weekday = daily.loc[daily.index <= 4].mean()
        weekend = daily.loc[daily.index >= 5].mean()
        rows.append(
            {
                "square_id": sq,
                "label": label,
                "mean": s.mean(),
                "std": s.std(),
                "cv": s.std() / s.mean() if s.mean() else np.nan,
                "min": s.min(),
                "max": s.max(),
                "peak_to_mean": s.max() / s.mean() if s.mean() else np.nan,
                "zero_fraction": float((s == 0).mean()),
                "peak_hour": int(s.groupby(s.index.hour).mean().idxmax()),
                "weekend_over_weekday": weekend / weekday if weekday else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(C.TABLE_DIR / "table2_five_series_stats.csv", index=False)
    return top


# --------------------------------------------------------------------------
# 2.3a Autocorrelation structure
# --------------------------------------------------------------------------
def analyse_autocorrelation(series: pd.Series, square_id: int, max_lag: int = 3 * C.SLOTS_PER_DAY) -> dict:
    from statsmodels.tsa.stattools import acf, pacf

    values = series.to_numpy(dtype=float)
    acf_vals = acf(values, nlags=max_lag, fft=True)
    pacf_vals = pacf(values, nlags=min(max_lag, len(values) // 2 - 1), method="ywm")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2))
    conf = 1.96 / np.sqrt(len(values))

    axes[0].stem(np.arange(len(acf_vals)), acf_vals, markerfmt=" ", basefmt=" ",
                 linefmt="#2c5f8a")
    axes[0].axhspan(-conf, conf, color="grey", alpha=0.2)
    for lag in (C.SLOTS_PER_DAY, 2 * C.SLOTS_PER_DAY, 3 * C.SLOTS_PER_DAY):
        if lag < len(acf_vals):
            axes[0].axvline(lag, color="#b5482a", linestyle="--", linewidth=0.9)
            axes[0].text(lag, axes[0].get_ylim()[1] * 0.85, f"{lag // C.SLOTS_PER_DAY}d",
                         color="#b5482a", fontsize=8, ha="center")
    axes[0].set(title=f"ACF - square {square_id}", xlabel="Lag (10-minute steps)", ylabel="ACF")

    axes[1].stem(np.arange(len(pacf_vals)), pacf_vals, markerfmt=" ", basefmt=" ",
                 linefmt="#2c5f8a")
    axes[1].axhspan(-conf, conf, color="grey", alpha=0.2)
    axes[1].set(title="PACF", xlabel="Lag (10-minute steps)", ylabel="PACF")
    fig.tight_layout()
    fig.savefig(C.FIGURE_DIR / "fig3_acf_pacf.png")
    plt.close(fig)

    # Where does the ACF first drop below the significance band, and how strong
    # are the seasonal lags? These two numbers justify the input window length.
    below = np.where(np.abs(acf_vals[1:]) < conf)[0]
    first_insignificant = int(below[0] + 1) if below.size else None

    weekly_lag = C.SLOTS_PER_WEEK
    weekly_acf = None
    if weekly_lag < len(values) // 2:
        from statsmodels.tsa.stattools import acf as _acf
        weekly_acf = float(_acf(values, nlags=weekly_lag, fft=True)[weekly_lag])

    result = {
        "square_id": int(square_id),
        "acf_lag_1": float(acf_vals[1]),
        "acf_lag_6_1h": float(acf_vals[6]) if len(acf_vals) > 6 else None,
        "acf_lag_144_1d": float(acf_vals[C.SLOTS_PER_DAY]) if len(acf_vals) > C.SLOTS_PER_DAY else None,
        "acf_lag_288_2d": float(acf_vals[2 * C.SLOTS_PER_DAY]) if len(acf_vals) > 2 * C.SLOTS_PER_DAY else None,
        "acf_lag_1008_7d": weekly_acf,
        "first_insignificant_lag": first_insignificant,
        "significance_band": float(conf),
    }
    pd.Series(result).to_json(C.TABLE_DIR / "table3_autocorrelation.json", indent=2)
    return result


# --------------------------------------------------------------------------
# 2.3b Stationarity and seasonal decomposition
# --------------------------------------------------------------------------
def analyse_stationarity(series: pd.Series, square_id: int) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import STL

    values = series.to_numpy(dtype=float)

    adf_stat, adf_p, *_ = adfuller(values, autolag="AIC")
    diff = np.diff(values, n=1)
    adf_d_stat, adf_d_p, *_ = adfuller(diff, autolag="AIC")
    # KPSS has the opposite null (stationary), so the pair is more informative
    # than either test alone.
    kpss_stat, kpss_p, *_ = kpss(values, regression="c", nlags="auto")

    stl = STL(pd.Series(values, index=series.index), period=C.SLOTS_PER_DAY, robust=True).fit()
    var_total = float(np.var(values))
    strength_seasonal = float(max(0.0, 1 - np.var(stl.resid) / max(np.var(stl.seasonal + stl.resid), 1e-12)))
    strength_trend = float(max(0.0, 1 - np.var(stl.resid) / max(np.var(stl.trend + stl.resid), 1e-12)))

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for ax, comp, name in zip(
        axes,
        [pd.Series(values, index=series.index), stl.trend, stl.seasonal, stl.resid],
        ["Observed", "Trend", "Seasonal (daily)", "Residual"],
    ):
        ax.plot(series.index, np.asarray(comp), linewidth=0.6, color="#2c5f8a")
        ax.set_ylabel(name, fontsize=8)
    axes[-1].set_xlabel("Time (Europe/Rome)")
    fig.suptitle(f"STL decomposition, daily period - square {square_id}", y=1.005)
    fig.savefig(C.FIGURE_DIR / "fig4_stl_decomposition.png")
    plt.close(fig)

    result = {
        "square_id": int(square_id),
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "adf_statistic_first_diff": float(adf_d_stat),
        "adf_pvalue_first_diff": float(adf_d_p),
        "kpss_statistic": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
        "variance_total": var_total,
        "seasonal_strength": strength_seasonal,
        "trend_strength": strength_trend,
        "residual_share_of_variance": float(np.var(stl.resid) / max(var_total, 1e-12)),
    }
    pd.Series(result).to_json(C.TABLE_DIR / "table4_stationarity.json", indent=2)
    return result


# --------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the exploratory analysis.")
    ap.add_argument("--processed-dir", type=str, default=None)
    args = ap.parse_args(argv)

    grid = dataio.load_grid(args.processed_dir)
    print(f"Grid loaded: {grid.shape[0]} timesteps x {grid.shape[1]} squares")

    coverage = dataio.describe_coverage(grid)
    coverage.to_csv(C.TABLE_DIR / "table0_coverage.csv", index=False)
    print(coverage.to_string(index=False))

    print("\n[2.1] Traffic distribution across areas")
    print(plot_traffic_distribution(grid).to_string(index=False))

    print("\n[2.2] Five-area time series")
    top = plot_five_series(grid)
    print(f"  top three squares by total traffic: {top}")

    busiest = top[0]
    full = grid.series(busiest)
    train_part = full.loc[: pd.Timestamp(C.TEST_WINDOW[0], tz=full.index.tz)]

    print(f"\n[2.3a] Autocorrelation, square {busiest}")
    print(json.dumps(analyse_autocorrelation(train_part, busiest), indent=2))

    print(f"\n[2.3b] Stationarity and STL, square {busiest}")
    print(json.dumps(analyse_stationarity(train_part, busiest), indent=2))

    (C.OUTPUT_DIR / "top_squares.json").write_text(json.dumps(top))
    print(f"\nFigures -> {C.FIGURE_DIR}\nTables  -> {C.TABLE_DIR}")


if __name__ == "__main__":
    main()
