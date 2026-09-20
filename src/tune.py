"""Hyper-parameter search with a persistent experiment log.

The brief asks for an *iterative* process: after each run, look at what happened
and justify the next change. This module supports that rather than replacing it.
Every run appends a row to ``outputs/logs/experiment_log.csv`` with the exact
parameters, the validation score, and an empty ``reasoning`` column that you
fill in yourself - that column is the part a grid search cannot write for you,
and it is the part the report and the viva are actually about.

Selection is always on the **validation** split. The test week (16-22 December)
is never read here; if it were, the reported test metrics would be optimistic.

    # one deliberate run, with your reasoning recorded
    python -m src.tune --model lstm --params '{"hidden": 128, "layers": 2}' \
        --note "64 units underfit: train and val loss plateaued together"

    # or a grid, when there is no strong prior to reason from
    python -m src.tune --model tcn --grid '{"channels": [16, 32], "levels": [4, 5, 6]}'

    # SARIMA order selection by AIC on the differenced training series
    python -m src.tune --model sarima --sarima-search
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from . import dataio, evaluate, models

warnings.filterwarnings("ignore", category=FutureWarning)

LOG_PATH = C.LOG_DIR / "experiment_log.csv"


def _append_log(row: dict) -> None:
    df = pd.DataFrame([row])
    header = not LOG_PATH.exists()
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)


def evaluate_on_validation(model, train: pd.Series, val: pd.Series) -> dict:
    """Fit on train, score one-step-ahead predictions over the validation split."""
    model.fit(train, val)
    pred = model.predict_rolling(train, val)
    return evaluate.all_metrics(val.to_numpy(), pred)


def run_one(model_key: str, params: dict, square_id: int, note: str) -> dict:
    grid = dataio.load_grid()
    series = grid.series(square_id)
    train, val, _ = dataio.train_val_test_split(series)

    if model_key == "sarima":
        model = models.SarimaForecaster(**params)
    elif model_key == "seasonal_naive":
        model = models.SeasonalNaive(**params)
    else:
        cfg = models.NeuralConfig(**params)
        model = (models.LSTMForecaster(cfg) if model_key == "lstm"
                 else models.TCNForecaster(cfg))

    t0 = time.perf_counter()
    metrics = evaluate_on_validation(model, train, val)
    elapsed = time.perf_counter() - t0

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model.name,
        "square_id": square_id,
        "params": json.dumps(params, sort_keys=True),
        "val_MAE": round(metrics["MAE"], 4),
        "val_RMSE": round(metrics["RMSE"], 4),
        "val_MAPE_%": round(metrics["MAPE_%"], 3),
        "train_seconds": round(float(model.train_seconds), 2),
        "total_seconds": round(elapsed, 2),
        "epochs_run": model.describe().get("epochs_run"),
        "n_parameters": model.describe().get("n_parameters") or model.describe().get("n_params"),
        "reasoning": note,
    }
    _append_log(row)
    print(f"  {model.name} {params} -> val MAE {row['val_MAE']:.3f} "
          f"RMSE {row['val_RMSE']:.3f} ({row['train_seconds']}s)")
    return row


def run_grid(model_key: str, grid_spec: dict, square_id: int, base: dict) -> pd.DataFrame:
    keys = list(grid_spec)
    rows = []
    for combo in itertools.product(*(grid_spec[k] for k in keys)):
        params = {**base, **dict(zip(keys, combo))}
        rows.append(run_one(model_key, params, square_id,
                            note="grid search (automated)"))
    table = pd.DataFrame(rows).sort_values("val_MAE")
    print("\nGrid results (best first):")
    print(table[["model", "params", "val_MAE", "val_RMSE", "train_seconds"]].to_string(index=False))
    return table


def sarima_order_search(square_id: int, max_p: int = 3, max_q: int = 3) -> pd.DataFrame:
    """Rank low-order ARMA terms by AIC on the seasonally differenced series.

    AIC is used for the order search because it is cheap and does not consume
    the validation split; the shortlist it produces is then scored on validation
    like every other candidate.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    grid = dataio.load_grid()
    train, val, _ = dataio.train_val_test_split(grid.series(square_id))
    y = np.log1p(pd.concat([train, val]).to_numpy())
    s = C.SLOTS_PER_DAY
    z = y[s:] - y[:-s]

    rows = []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            t0 = time.perf_counter()
            res = SARIMAX(z, order=(p, 0, q), enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            rows.append({"p": p, "d": 0, "q": q, "aic": float(res.aic),
                         "bic": float(res.bic), "fit_seconds": round(time.perf_counter() - t0, 2)})
            print(f"  ARMA({p},{q}) AIC={res.aic:.1f}")
        except Exception as exc:  # pragma: no cover
            print(f"  ARMA({p},{q}) failed: {exc}")
    table = pd.DataFrame(rows).sort_values("aic")
    table.to_csv(C.TABLE_DIR / "sarima_order_search.csv", index=False)
    print("\nBest by AIC:")
    print(table.head(5).to_string(index=False))
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Hyper-parameter experiments.")
    ap.add_argument("--model", required=True, choices=list(models.MODEL_REGISTRY))
    ap.add_argument("--square", type=int, default=None,
                    help="default: busiest square (tuning is done on one area)")
    ap.add_argument("--params", type=str, default="{}", help="JSON dict for a single run")
    ap.add_argument("--grid", type=str, default=None, help="JSON dict of lists")
    ap.add_argument("--note", type=str, default="",
                    help="why you are trying this configuration")
    ap.add_argument("--sarima-search", action="store_true")
    args = ap.parse_args(argv)

    square = args.square or dataio.load_grid().top_squares(1)[0]
    print(f"Tuning on square {square} (validation split only)")

    if args.sarima_search:
        sarima_order_search(square)
        return
    if args.grid:
        run_grid(args.model, json.loads(args.grid), square, json.loads(args.params))
        return
    run_one(args.model, json.loads(args.params), square, args.note)
    print(f"\nLog -> {LOG_PATH}")


if __name__ == "__main__":
    main()
