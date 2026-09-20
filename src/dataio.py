"""Read side of the pipeline.

The dense matrix is opened with ``mmap_mode="r"``: the OS pages in only the
columns actually touched, so extracting five squares costs a few hundred KiB of
resident memory instead of the full ~340 MiB. This is the second half of the
memory story in Task 1 -- ingestion shrinks what is stored, memory-mapping
shrinks what is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Grid:
    """The full (time x square) traffic matrix plus its indices."""

    matrix: np.ndarray            # (T, S) float32, possibly memory-mapped
    timestamps: pd.DatetimeIndex  # length T, tz-aware Europe/Rome
    square_ids: np.ndarray        # length S, int

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def series(self, square_id: int) -> pd.Series:
        """Traffic time series for one square, as a tz-aware pandas Series."""
        pos = int(np.searchsorted(self.square_ids, square_id))
        if pos >= len(self.square_ids) or int(self.square_ids[pos]) != int(square_id):
            raise KeyError(f"square_id {square_id} not present")
        values = np.asarray(self.matrix[:, pos], dtype=np.float64)
        return pd.Series(values, index=self.timestamps, name=f"square_{square_id}")

    def totals(self) -> pd.Series:
        """Total traffic per square over the whole observation period."""
        path = C.PROCESSED_DIR / "square_totals.parquet"
        if path.exists():
            t = pd.read_parquet(path)
            return pd.Series(
                t["total_internet"].to_numpy(), index=t["square_id"].to_numpy(), name="total_internet"
            )
        return pd.Series(
            np.asarray(self.matrix).sum(axis=0), index=self.square_ids, name="total_internet"
        )

    def top_squares(self, k: int = 3) -> list[int]:
        """The k squares with the highest total traffic, highest first."""
        return [int(s) for s in self.totals().sort_values(ascending=False).head(k).index]


@lru_cache(maxsize=1)
def load_grid(processed_dir: str | None = None, mmap: bool = True) -> Grid:
    d = Path(processed_dir) if processed_dir else C.PROCESSED_DIR
    matrix = np.load(d / "internet_matrix.npy", mmap_mode="r" if mmap else None)
    square_ids = np.load(d / "square_ids.npy").astype(np.int32)
    ts = pd.read_parquet(d / "timestamps.parquet")["timestamp"]
    index = pd.DatetimeIndex(ts)
    if index.tz is None:
        index = index.tz_localize("UTC").tz_convert(C.TIMEZONE)
    else:
        index = index.tz_convert(C.TIMEZONE)
    return Grid(matrix=matrix, timestamps=index, square_ids=square_ids)


def slice_window(series: pd.Series, start: str, end: str) -> pd.Series:
    """Inclusive date slice, e.g. ('2013-12-16', '2013-12-22')."""
    tz = series.index.tz
    lo = pd.Timestamp(start, tz=tz)
    hi = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=C.SLOT_MINUTES)
    return series.loc[lo:hi]


def train_val_test_split(
    series: pd.Series,
    test_window: tuple[str, str] = C.TEST_WINDOW,
    val_days: int = C.VAL_DAYS,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Chronological split. The test week is never seen during model selection.

    Returns (train, val, test). The split is strictly forward in time: there is
    no shuffling and no leakage of future observations into training, which
    would otherwise make one-step-ahead results meaningless.
    """
    tz = series.index.tz
    test_start = pd.Timestamp(test_window[0], tz=tz)
    test_end = pd.Timestamp(test_window[1], tz=tz) + pd.Timedelta(days=1)

    history = series.loc[:test_start - pd.Timedelta(minutes=C.SLOT_MINUTES)]
    val_start = test_start - pd.Timedelta(days=val_days)
    train = history.loc[:val_start - pd.Timedelta(minutes=C.SLOT_MINUTES)]
    val = history.loc[val_start:]
    test = series.loc[test_start:test_end - pd.Timedelta(minutes=C.SLOT_MINUTES)]
    return train, val, test


def describe_coverage(grid: Grid) -> pd.DataFrame:
    """Sanity checks worth reporting: gaps, duplicates, zero-inflation."""
    idx = grid.timestamps
    expected = pd.date_range(idx[0], idx[-1], freq=f"{C.SLOT_MINUTES}min", tz=idx.tz)
    missing = expected.difference(idx)
    m = np.asarray(grid.matrix)
    return pd.DataFrame(
        [
            {"check": "timesteps observed", "value": len(idx)},
            {"check": "timesteps expected", "value": len(expected)},
            {"check": "missing timesteps", "value": len(missing)},
            {"check": "duplicate timestamps", "value": int(idx.duplicated().sum())},
            {"check": "squares", "value": int(len(grid.square_ids))},
            {"check": "zero-valued cells (%)", "value": round(float((m == 0).mean()) * 100, 2)},
            {"check": "all-zero squares", "value": int((m.sum(axis=0) == 0).sum())},
        ]
    )
