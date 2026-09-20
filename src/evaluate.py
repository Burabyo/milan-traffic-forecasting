"""Metrics, plots and timing for the forecasting experiments (Task 4)."""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat, eps: float = 1e-8) -> float:
    """Mean absolute percentage error, computed on strictly positive actuals.

    MAPE is undefined at y = 0 and explodes for y near 0. Traffic series contain
    near-zero night-time values, so the denominators that would dominate the
    average are exactly the ones carrying least information. Observations with
    y <= eps are therefore excluded and the excluded fraction is reported
    alongside, so the number is not quietly misleading.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = y > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def smape(y, yhat, eps: float = 1e-8) -> float:
    """Symmetric MAPE - bounded at 200% and defined at zero, unlike MAPE."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    mask = denom > eps
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100)


def r2(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((y - np.asarray(yhat, dtype=float)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def all_metrics(y, yhat, eps: float = 1e-8) -> dict:
    y = np.asarray(y, dtype=float)
    return {
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "MAPE_%": mape(y, yhat, eps),
        "sMAPE_%": smape(y, yhat, eps),
        "R2": r2(y, yhat),
        "n": int(len(y)),
        "mape_excluded_%": float((y <= eps).mean() * 100),
    }


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def time_inference(model, history: pd.Series, test: pd.Series, repeats: int = 5) -> dict:
    """Median wall-clock time of a full rolling pass over the test week.

    The median of several repeats is reported rather than a single run, because
    a single measurement on a shared machine is dominated by scheduling noise.
    The first (warm-up) run is discarded.
    """
    model.predict_rolling(history, test)  # warm-up: caches, lazy CUDA init
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_rolling(history, test)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return {
        "inference_total_s_median": float(np.median(times)),
        "inference_total_s_min": float(times.min()),
        "inference_total_s_max": float(times.max()),
        "inference_ms_per_step": float(np.median(times) / len(test) * 1000),
        "repeats": repeats,
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_actual_vs_predicted(actual: pd.Series, predicted: np.ndarray,
                             model_name: str, square_id: int, out_path=None):
    """One of the nine required superposed actual-vs-predicted plots."""
    resid = actual.to_numpy() - np.asarray(predicted)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(actual.index, actual.to_numpy(), linewidth=1.0,
                 color="#2c3e50", label="Observed")
    axes[0].plot(actual.index, predicted, linewidth=1.0, color="#c0392b",
                 alpha=0.85, label="Predicted")
    axes[0].set_ylabel("Internet activity")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].set_title(f"{model_name} - square {square_id} - "
                      f"{C.TEST_WINDOW[0]} to {C.TEST_WINDOW[1]}", loc="left")

    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].plot(actual.index, resid, linewidth=0.7, color="#7f8c8d")
    axes[1].set(ylabel="Residual", xlabel="Time (Europe/Rome)")

    out_path = out_path or (C.FIGURE_DIR / f"pred_{model_name.lower()}_sq{square_id}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_model_comparison(actual: pd.Series, predictions: dict[str, np.ndarray],
                          square_id: int, out_path=None):
    """All models on one axis - easier to read than three separate panels."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(actual.index, actual.to_numpy(), linewidth=1.3, color="#2c3e50",
            label="Observed", zorder=3)
    for (name, pred), colour in zip(predictions.items(),
                                    ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]):
        ax.plot(actual.index, pred, linewidth=0.9, alpha=0.8, color=colour, label=name)
    ax.set(ylabel="Internet activity", xlabel="Time (Europe/Rome)")
    ax.set_title(f"All models - square {square_id}", loc="left")
    ax.legend(loc="upper right", ncol=4, frameon=False)
    out_path = out_path or (C.FIGURE_DIR / f"comparison_sq{square_id}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_error_by_hour(actual: pd.Series, predictions: dict[str, np.ndarray],
                       square_id: int, out_path=None):
    """Where in the day does each model lose? Feeds the failure-case discussion."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    hours = actual.index.hour
    for name, pred in predictions.items():
        err = np.abs(actual.to_numpy() - np.asarray(pred))
        by_hour = pd.Series(err, index=hours).groupby(level=0).mean()
        ax.plot(by_hour.index, by_hour.to_numpy(), marker="o", markersize=3, label=name)
    ax.set(xlabel="Hour of day (Europe/Rome)", ylabel="Mean absolute error",
           xticks=range(0, 24, 2))
    ax.set_title(f"Error profile across the day - square {square_id}", loc="left")
    ax.legend(frameon=False)
    out_path = out_path or (C.FIGURE_DIR / f"error_by_hour_sq{square_id}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def worst_windows(actual: pd.Series, predictions: dict[str, np.ndarray],
                  window: int = 6, top: int = 5) -> pd.DataFrame:
    """Rank the hours of the test week by mean absolute error, per model.

    This is the mechanical half of the required failure analysis: it finds the
    periods to look at. The explanation of *why* they fail has to come from
    inspecting them against the EDA.
    """
    rows = []
    for name, pred in predictions.items():
        err = pd.Series(np.abs(actual.to_numpy() - np.asarray(pred)), index=actual.index)
        rolled = err.rolling(window).mean()
        for ts, value in rolled.nlargest(top).items():
            rows.append({
                "model": name,
                "window_end": ts,
                "weekday": ts.day_name(),
                "hour": ts.hour,
                "mean_abs_error": float(value),
                "mean_actual": float(
                    actual.loc[ts - pd.Timedelta(minutes=C.SLOT_MINUTES * (window - 1)):ts].mean()
                ),
            })
    return pd.DataFrame(rows).sort_values(["model", "mean_abs_error"], ascending=[True, False])
