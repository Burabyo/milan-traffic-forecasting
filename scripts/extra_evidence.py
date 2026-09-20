"""Generate the evidence gaps: weekly baseline, calendar anomalies, RF comparison.

    python -m scripts.extra_evidence --all

Writes to outputs/tables/ and outputs/figures/. Each block is independent; run
with --weekly, --anomalies or --receptive-field to do one at a time.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src import dataio, evaluate, models

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------
def weekly_baseline(squares: list[int]) -> pd.DataFrame:
    """Daily-period vs weekly-period naive.

    The daily naive fails at weekday/weekend boundaries because it assumes today
    resembles yesterday. If a weekly-period variant recovers most of that error,
    the failure is specifically the weekly cycle rather than general noise - which
    turns the observation into a diagnosis.
    """
    grid = dataio.load_grid()
    rows, per_day = [], []
    for sq in squares:
        tr, va, te = dataio.train_val_test_split(grid.series(sq))
        hist = pd.concat([tr, va])
        for period, label in [(C.SLOTS_PER_DAY, "Naive-daily"),
                              (C.SLOTS_PER_WEEK, "Naive-weekly")]:
            pred = models.SeasonalNaive(period=period).predict_rolling(hist, te)
            rows.append({"square_id": sq, "model": label,
                         **evaluate.all_metrics(te.to_numpy(), pred)})
            err = pd.Series(np.abs(te.to_numpy() - pred), index=te.index)
            for day, grp in err.groupby(err.index.date):
                per_day.append({"square_id": sq, "model": label, "date": str(day),
                                "weekday": pd.Timestamp(day).day_name(),
                                "MAE": round(float(grp.mean()), 2)})

    table = pd.DataFrame(rows)
    daily = pd.DataFrame(per_day)
    table.to_csv(C.TABLE_DIR / "naive_baselines.csv", index=False)
    daily.to_csv(C.TABLE_DIR / "naive_baselines_by_day.csv", index=False)

    print("\n=== Weekly vs daily naive ===")
    print(table[["square_id", "model", "MAE", "RMSE", "MAPE_%", "R2"]]
          .round(2).to_string(index=False))
    print("\nPer-day MAE (the boundary effect):")
    print(daily.pivot_table(index=["square_id", "date", "weekday"],
                            columns="model", values="MAE").round(1).to_string())
    return table


# --------------------------------------------------------------------------
def calendar_anomalies(square_id: int | None = None) -> pd.DataFrame:
    """Find days that deviate from their own weekday's normal level.

    Comparing a Saturday to the overall mean conflates the weekly cycle with
    genuine anomaly. Each day is therefore scored against the distribution of
    *the same weekday* across the observation period, so what surfaces is
    deviation the weekly pattern does not already explain.
    """
    grid = dataio.load_grid()
    sq = square_id or grid.top_squares(1)[0]
    s = grid.series(sq)

    daily = s.resample("1D").sum()
    frame = pd.DataFrame({"total": daily.to_numpy()}, index=daily.index)
    frame["weekday"] = frame.index.day_name()
    frame["dow"] = frame.index.dayofweek

    # Robust z-score within weekday: median and MAD resist the very outliers
    # being searched for, which a mean and standard deviation would absorb.
    stats = frame.groupby("dow")["total"].agg(
        med="median", mad=lambda x: float(np.median(np.abs(x - np.median(x)))))
    frame = frame.join(stats, on="dow")
    frame["robust_z"] = (frame["total"] - frame["med"]) / (1.4826 * frame["mad"]).replace(0, np.nan)

    known = {
        "2013-11-01": "All Saints' Day (national holiday)",
        "2013-12-07": "Sant'Ambrogio (Milan patron saint, local holiday)",
        "2013-12-08": "Immaculate Conception (national holiday)",
        "2013-12-24": "Christmas Eve",
        "2013-12-25": "Christmas Day",
        "2013-12-26": "St Stephen's Day (national holiday)",
        "2013-12-31": "New Year's Eve",
        "2014-01-01": "New Year's Day",
    }
    frame["note"] = [known.get(str(d.date()), "") for d in frame.index]
    frame["date"] = [str(d.date()) for d in frame.index]

    out = frame[["date", "weekday", "total", "robust_z", "note"]].copy()
    out["total"] = out["total"].round(0)
    out["robust_z"] = out["robust_z"].round(2)
    out.to_csv(C.TABLE_DIR / "calendar_anomalies.csv", index=False)

    flagged = out.loc[out["robust_z"].abs() >= 3].sort_values("robust_z")
    print(f"\n=== Calendar anomalies, square {sq} ===")
    print("Days deviating >=3 robust MADs from their own weekday's median:")
    print(flagged.to_string(index=False) if len(flagged) else "  none")
    print("\nKnown holidays in the period:")
    print(out.loc[out["note"] != ""].to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 3.6))
    colours = ["#c0392b" if abs(z) >= 3 else "#2c5f8a"
               for z in frame["robust_z"].fillna(0)]
    ax.bar(frame.index, frame["total"], color=colours, width=0.8)
    for d, row in frame.loc[frame["note"] != ""].iterrows():
        ax.annotate(row["note"].split(" (")[0], (d, row["total"]),
                    textcoords="offset points", xytext=(0, 6),
                    fontsize=6.5, rotation=45, ha="left")
    ax.set(ylabel="Daily total activity", xlabel="Date (Europe/Rome)")
    ax.set_title(f"Daily totals and calendar anomalies - square {sq} "
                 "(red = |robust z| >= 3 within weekday)", loc="left")
    fig.savefig(C.FIGURE_DIR / "fig5_calendar_anomalies.png")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
def receptive_field_table() -> pd.DataFrame:
    """What each TCN depth can actually see, against the 144-step daily cycle."""
    rows = []
    for levels in range(3, 8):
        rf = 1 + 2 * (3 - 1) * (2 ** levels - 1)
        rows.append({"levels": levels, "receptive_field": rf,
                     "hours_visible": round(rf * C.SLOT_MINUTES / 60, 1),
                     "covers_daily_cycle": rf >= C.SLOTS_PER_DAY})
    table = pd.DataFrame(rows)
    table.to_csv(C.TABLE_DIR / "tcn_receptive_field.csv", index=False)
    print("\n=== TCN receptive field by depth (kernel=3) ===")
    print(table.to_string(index=False))
    print(f"\nDaily cycle = {C.SLOTS_PER_DAY} steps. The reported run used "
          "levels=5 (125 steps), which does NOT cover a full day.")
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--anomalies", action="store_true")
    ap.add_argument("--receptive-field", action="store_true")
    ap.add_argument("--squares", nargs="+", type=int, default=None)
    args = ap.parse_args(argv)

    if not any([args.all, args.weekly, args.anomalies, args.receptive_field]):
        args.all = True

    squares = args.squares or dataio.load_grid().top_squares(3)

    if args.all or args.weekly:
        weekly_baseline(squares)
    if args.all or args.anomalies:
        calendar_anomalies(squares[0])
    if args.all or args.receptive_field:
        receptive_field_table()

    print(f"\nWritten to {C.TABLE_DIR} and {C.FIGURE_DIR}")


if __name__ == "__main__":
    main()
