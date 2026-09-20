"""Task 4: train, forecast, evaluate, time, and write every required artefact.

    python -m src.run_experiments                  # all models, all three squares
    python -m src.run_experiments --models sarima  # one model, for iteration
    python -m src.run_experiments --squares 5060   # one square

Outputs (all under ``outputs/``):
  figures/pred_<model>_sq<id>.png     9 superposed actual-vs-predicted plots
  figures/comparison_sq<id>.png       all models on one axis, per square
  figures/error_by_hour_sq<id>.png    where in the day each model loses
  tables/metrics_sq<id>.csv           one metrics table per square
  tables/metrics_all.csv              combined
  tables/timing.csv                   training and inference time + hardware
  tables/failure_windows.csv          worst 1-hour windows per model
  tables/model_configs.json           exact hyper-parameters used
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import dataio, evaluate, models

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")


def build_model(key: str, overrides: dict | None = None):
    overrides = overrides or {}
    if key == "seasonal_naive":
        return models.SeasonalNaive()
    if key == "sarima":
        return models.SarimaForecaster(**overrides)
    cfg = models.NeuralConfig(**overrides)
    return models.LSTMForecaster(cfg) if key == "lstm" else models.TCNForecaster(cfg)


def run_square(grid: dataio.Grid, square_id: int, model_keys: list[str],
               overrides: dict, repeats: int) -> tuple[pd.DataFrame, dict, list[dict]]:
    series = grid.series(square_id)
    train, val, test = dataio.train_val_test_split(series)
    history = pd.concat([train, val])

    print(f"\n=== square {square_id} ===")
    print(f"  train {len(train):5d} | val {len(val):4d} | test {len(test):4d} steps")
    print(f"  test window {test.index[0]} -> {test.index[-1]}")

    rows, predictions, timing = [], {}, []
    for key in model_keys:
        model = build_model(key, overrides.get(key))
        print(f"  fitting {model.name} ...", end="", flush=True)
        model.fit(train, val)
        pred = model.predict_rolling(history, test)
        m = evaluate.all_metrics(test.to_numpy(), pred)
        print(f" MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} "
              f"MAPE={m['MAPE_%']:.1f}% ({model.train_seconds:.1f}s train)")

        predictions[model.name] = pred
        rows.append({"square_id": square_id, "model": model.name, **m})
        timing.append({
            "square_id": square_id,
            "model": model.name,
            "training_s": float(model.train_seconds),
            **evaluate.time_inference(model, history, test, repeats=repeats),
            **{k: v for k, v in model.describe().items()
               if k in ("n_parameters", "epochs_run", "device", "n_params")},
        })
        evaluate.plot_actual_vs_predicted(test, pred, model.name, square_id)
        (C.TABLE_DIR / f"config_{model.name.lower()}_sq{square_id}.json").write_text(
            json.dumps(model.describe(), indent=2, default=str)
        )
        if getattr(model, "history_", None):
            pd.DataFrame(model.history_).to_csv(
                C.LOG_DIR / f"trainlog_{model.name.lower()}_sq{square_id}.csv", index=False
            )

    evaluate.plot_model_comparison(test, predictions, square_id)
    evaluate.plot_error_by_hour(test, predictions, square_id)

    table = pd.DataFrame(rows)
    table.to_csv(C.TABLE_DIR / f"metrics_sq{square_id}.csv", index=False)
    failures = evaluate.worst_windows(test, predictions)
    return table, {"square_id": square_id, "failures": failures}, timing


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the forecasting experiments.")
    ap.add_argument("--models", nargs="+",
                    default=["seasonal_naive", "sarima", "lstm", "tcn"],
                    choices=list(models.MODEL_REGISTRY))
    ap.add_argument("--squares", nargs="+", type=int, default=None,
                    help="default: the top three by total traffic")
    ap.add_argument("--overrides", type=str, default=None,
                    help='JSON, e.g. \'{"lstm": {"hidden": 128, "epochs": 40}}\'')
    ap.add_argument("--timing-repeats", type=int, default=5)
    args = ap.parse_args(argv)

    overrides = json.loads(args.overrides) if args.overrides else {}
    grid = dataio.load_grid()
    squares = args.squares or grid.top_squares(3)
    print(f"Squares under study: {squares}")

    hw = models.hardware_info()
    print("Hardware: " + json.dumps(hw))

    all_metrics, all_timing, all_failures = [], [], []
    for sq in squares:
        table, fail, timing = run_square(grid, sq, args.models, overrides,
                                         args.timing_repeats)
        all_metrics.append(table)
        all_timing.extend(timing)
        all_failures.append(fail["failures"])

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(C.TABLE_DIR / "metrics_all.csv", index=False)

    timing_df = pd.DataFrame(all_timing)
    # Times are reported as the mean across the three areas; the per-area rows
    # are kept so the spread is visible rather than hidden by the average.
    summary = (timing_df.groupby("model")[["training_s", "inference_total_s_median",
                                           "inference_ms_per_step"]]
               .agg(["mean", "std"]).round(4))
    timing_df.to_csv(C.TABLE_DIR / "timing.csv", index=False)
    summary.to_csv(C.TABLE_DIR / "timing_summary.csv")
    (C.TABLE_DIR / "hardware.json").write_text(json.dumps(hw, indent=2, default=str))

    pd.concat(all_failures, ignore_index=True).to_csv(
        C.TABLE_DIR / "failure_windows.csv", index=False)

    print("\n" + "=" * 70)
    print("METRICS (all squares)")
    print(metrics.round(3).to_string(index=False))
    print("\nTIMING (mean +/- sd across areas)")
    print(summary.to_string())

    pivot = metrics.pivot(index="model", columns="square_id", values="MAE")
    print("\nMAE by square")
    print(pivot.round(2).to_string())
    print(f"\nAll artefacts written under {C.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
