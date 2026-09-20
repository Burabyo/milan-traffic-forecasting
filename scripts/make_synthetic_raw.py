

    python -m scripts.make_synthetic_raw --out-dir /tmp/fake_raw --days 21
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 144


def make(out_dir: Path, days: int, n_squares: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A handful of "real" square ids so the reference squares exist.
    squares = np.unique(np.concatenate([
        rng.choice(np.arange(1, 10_001), size=n_squares - 2, replace=False),
        np.array([4159, 4556]),
    ]))
    # Heavy-tailed base levels, mimicking the real concentration.
    base = rng.lognormal(mean=2.0, sigma=1.4, size=len(squares))

    start = pd.Timestamp("2013-11-01", tz="Europe/Rome")
    for d in range(days):
        day_start = start + pd.Timedelta(days=d)
        stamps = pd.date_range(day_start, periods=SLOTS_PER_DAY, freq="10min", tz="Europe/Rome")
        ms = (
            stamps.tz_convert("UTC")
            .tz_localize(None)
            .astype("datetime64[ms]")
            .astype("int64")
            .to_numpy()
        )

        hour = stamps.hour.to_numpy() + stamps.minute.to_numpy() / 60.0
        diurnal = 0.35 + np.sin((hour - 4) / 24 * 2 * np.pi) ** 2
        weekend = 0.7 if day_start.dayofweek >= 5 else 1.0

        rows = []
        for si, (sq, lvl) in enumerate(zip(squares, base)):
            signal = lvl * diurnal * weekend
            signal = signal * rng.lognormal(0, 0.18, size=SLOTS_PER_DAY)
            # Two country codes per (square, slot) to exercise the aggregation.
            for cc in (39, 33):
                share = 0.8 if cc == 39 else 0.2
                rows.append(pd.DataFrame({
                    "square_id": sq,
                    "time_interval": ms,
                    "country_code": cc,
                    "sms_in": np.nan,
                    "sms_out": np.nan,
                    "call_in": np.nan,
                    "call_out": np.nan,
                    "internet": signal * share,
                }))
        df = pd.concat(rows, ignore_index=True)
        path = out_dir / f"sms-call-internet-mi-{day_start.date()}.txt"
        df.to_csv(path, sep="\t", header=False, index=False, na_rep="")
    print(f"wrote {days} synthetic files to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--n-squares", type=int, default=60)
    make(**vars(ap.parse_args()))
