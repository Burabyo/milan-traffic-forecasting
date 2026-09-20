"""Raw text -> compact on-disk representation.

The raw release is one tab-separated file per day, with **one row per
(square_id, 10-minute slot, country_code)**. Across 62 days that is on the order
of 3x10^8 rows and ~19 GB of text -- far beyond a naive ``pd.read_csv`` on a
laptop.

The research question does not need the country dimension, and needs only the
``internet`` activity column. The ingestion therefore reduces the data *while
streaming it*, so the full raw table is never materialised:

1. read one day at a time, in row chunks;
2. parse only columns 0, 1 and 7 (``usecols``), so the parser never allocates
   buffers for the five activity columns we discard;
3. cast on read: ``square_id`` -> int16, ``internet`` -> float32;
4. sum the country dimension away with a groupby on (square_id, time_interval);
5. write the day as Parquet with dictionary encoding + compression.

After step 4 a day holds at most 10,000 x 144 = 1.44M rows regardless of how
many countries appeared, which is a reduction of roughly an order of magnitude
before anything is kept in memory.

The final artefact is a dense ``float32`` matrix of shape
(n_timesteps, n_squares). At 8,928 x 10,000 that is ~340 MiB -- small enough to
memory-map and slice instantly, which is what the EDA and the experiments need.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .memprofile import MemoryReport, deep_memory_mb, measure, reports_to_frame

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

READ_DTYPES = {
    "square_id": "float64",     # cast to int16 after dropping partial rows
    "time_interval": "float64", # cast to int64 after dropping partial rows
    "internet": "float32",      # activity is a proportional measure, not a count
}


# --------------------------------------------------------------------------
# Loading a single day
# --------------------------------------------------------------------------
def naive_load_one_day(path: Path) -> pd.DataFrame:
    """Deliberately unoptimised baseline, used only for the Task 1 comparison.

    Reads every column, lets pandas infer dtypes (int64/float64/object), and
    keeps all country-code rows in memory.
    """
    return pd.read_csv(path, sep="\t", header=None, names=C.RAW_COLUMNS)


def optimised_load_one_day(path: Path, chunksize: int | None = 2_000_000) -> pd.DataFrame:
    """Stream one daily file into a (square_id, time_interval) -> internet frame.

    Parameters
    ----------
    path
        Daily ``sms-call-internet-mi-YYYY-MM-DD.txt`` file.
    chunksize
        Rows per chunk. ``None`` reads the file in one pass, which is faster but
        needs more transient memory. Each chunk is reduced by the groupby before
        the next is read, so peak memory is governed by ``chunksize``, not by the
        file size.
    """
    reader_kwargs = dict(
        sep="\t",
        header=None,
        names=C.RAW_COLUMNS,
        usecols=C.USED_COLUMN_IDX,
        dtype=READ_DTYPES,
        na_values=[""],
    )

    if chunksize is None:
        df = pd.read_csv(path, **reader_kwargs)
        return _reduce(df)

    partials: list[pd.DataFrame] = []
    with pd.read_csv(path, chunksize=chunksize, **reader_kwargs) as reader:
        for chunk in reader:
            partials.append(_reduce(chunk))
    if not partials:
        return pd.DataFrame(columns=C.USED_COLUMNS)
    # Chunk boundaries can split a (square, slot, country) group across two
    # chunks, so the partial sums are combined with a second reduction.
    return _reduce(pd.concat(partials, ignore_index=True))


def _reduce(df: pd.DataFrame) -> pd.DataFrame:
    """Sum the country dimension away and drop all-zero rows."""
    # Absent activity is encoded as an empty field -> NaN. For the internet
    # column an absent value means "no connection events in this bin", so it is
    # summed as zero rather than propagated as missing.
    bad = df["square_id"].isna() | df["time_interval"].isna()
    if bad.any():
        df = df.loc[~bad]
    out = (
        df.groupby(["square_id", "time_interval"], sort=False, observed=True)["internet"]
        .sum(min_count=0)
        .reset_index()
    )
    out["internet"] = out["internet"].astype("float32")
    out["square_id"] = out["square_id"].astype("int16")
    out["time_interval"] = out["time_interval"].astype("int64")
    return out


# --------------------------------------------------------------------------
# Full ingestion
# --------------------------------------------------------------------------
def discover_raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(Path(raw_dir).glob("sms-call-internet-mi-*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No raw files found in {raw_dir}. Download them from the Harvard "
            "Dataverse (doi:10.7910/DVN/EGZHFV) and pass --raw-dir."
        )
    return files


def ingest_all(raw_dir: Path, interim_dir: Path, chunksize: int | None) -> list[MemoryReport]:
    """Convert every daily text file to a reduced Parquet file (idempotent)."""
    files = discover_raw_files(raw_dir)
    reports: list[MemoryReport] = []
    incomplete = []
    min_rows = int(C.N_SQUARES * C.SLOTS_PER_DAY * 0.95)
    for path in files:
        date = DATE_RE.search(path.name).group(1)
        out_path = Path(interim_dir) / f"{date}.parquet"
        if out_path.exists():
            print(f"  skip {date} (already ingested)")
            continue
        with measure(f"ingest {date}", reports) as rep:
            day = optimised_load_one_day(path, chunksize=chunksize)
            n_slots = day["time_interval"].nunique(); n_rows = len(day)
            if n_slots != C.SLOTS_PER_DAY or n_rows < min_rows:
                print(f"  SKIP {date}: {n_slots}/{C.SLOTS_PER_DAY} slots, "
                      f"{n_rows:,} rows (expected >= {min_rows:,}) - incomplete. Re-run later.")
                incomplete.append(date)
                continue
            day.to_parquet(out_path, index=False, compression="snappy")
        rep.extra.update(
            date=date,
            rows_out=len(day),
            raw_mb=path.stat().st_size / 1024 ** 2,
            parquet_mb=out_path.stat().st_size / 1024 ** 2,
            frame_mb=deep_memory_mb(day),
        )
        print(f"  {rep}  rows={len(day):,}")
        del day
    if incomplete:
        print(f"\n  {len(incomplete)} day(s) skipped as incomplete: {', '.join(incomplete)}")
        print("  Re-run once the download has finished.")
    return reports


def build_matrix(interim_dir: Path, processed_dir: Path) -> dict:
    """Assemble the per-day Parquet files into a dense (time x square) matrix.

    A dense matrix is the right structure here: the grid is observed on a
    regular 10-minute lattice, roughly 60-70% of cells are non-zero, and every
    downstream step wants a contiguous slice of one column. The sparse
    alternative would save little and make slicing far slower.
    """
    parts = sorted(Path(interim_dir).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet files in {interim_dir}; run ingest first.")

    square_ids = np.arange(1, C.N_SQUARES + 1, dtype=np.int16)
    sq_pos = {int(s): i for i, s in enumerate(square_ids)}

    # First pass: collect the global set of timestamps.
    stamps: set[int] = set()
    for p in parts:
        stamps.update(pd.read_parquet(p, columns=["time_interval"])["time_interval"].unique().tolist())
    times = np.array(sorted(stamps), dtype=np.int64)
    t_pos = {int(t): i for i, t in enumerate(times)}

    matrix = np.zeros((len(times), len(square_ids)), dtype=np.float32)

    # Second pass: scatter each day into the matrix, one day resident at a time.
    for p in parts:
        day = pd.read_parquet(p)
        rows = day["time_interval"].map(t_pos).to_numpy(dtype=np.int64)
        cols = day["square_id"].map(sq_pos).to_numpy(dtype=np.int64)
        # Duplicate (row, col) pairs cannot occur after _reduce, but np.add.at is
        # used defensively so a re-ingested day can never silently overwrite.
        np.add.at(matrix, (rows, cols), day["internet"].to_numpy(dtype=np.float32))
        del day

    index = pd.to_datetime(times, unit="ms", utc=True).tz_convert(C.TIMEZONE)

    processed_dir = Path(processed_dir)
    np.save(processed_dir / "internet_matrix.npy", matrix)
    np.save(processed_dir / "square_ids.npy", square_ids)
    index.to_frame(index=False, name="timestamp").to_parquet(
        processed_dir / "timestamps.parquet", index=False
    )

    totals = pd.DataFrame(
        {"square_id": square_ids.astype(np.int32), "total_internet": matrix.sum(axis=0)}
    )
    totals.to_parquet(processed_dir / "square_totals.parquet", index=False)

    meta = dict(
        n_timesteps=int(matrix.shape[0]),
        n_squares=int(matrix.shape[1]),
        first_timestamp=str(index[0]),
        last_timestamp=str(index[-1]),
        matrix_mb=matrix.nbytes / 1024 ** 2,
        nonzero_fraction=float((matrix > 0).mean()),
        expected_timesteps=None,
    )
    print(
        f"  matrix {matrix.shape} = {meta['matrix_mb']:.1f} MiB float32, "
        f"{meta['nonzero_fraction']:.1%} non-zero"
    )
    return meta


# --------------------------------------------------------------------------
# Task 1 benchmark
# --------------------------------------------------------------------------
def benchmark(raw_dir: Path, chunksize: int | None) -> pd.DataFrame:
    """Naive vs optimised load of a single day -- the Task 1 evidence table."""
    path = discover_raw_files(raw_dir)[0]
    reports: list[MemoryReport] = []

    with measure("naive: all columns, inferred dtypes", reports) as rep:
        naive = naive_load_one_day(path)
    rep.extra.update(rows=len(naive), frame_mb=deep_memory_mb(naive))
    naive_mb = rep.extra["frame_mb"]
    del naive

    with measure("optimised: 3 columns, downcast, chunked+grouped", reports) as rep:
        opt = optimised_load_one_day(path, chunksize=chunksize)
    rep.extra.update(rows=len(opt), frame_mb=deep_memory_mb(opt))
    opt_mb = rep.extra["frame_mb"]
    del opt

    table = reports_to_frame(reports)
    table["file"] = path.name
    table["reduction_x"] = naive_mb / max(opt_mb, 1e-9)
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Ingest the Milan CDR dataset.")
    ap.add_argument("--raw-dir", type=Path, default=C.RAW_DIR)
    ap.add_argument("--interim-dir", type=Path, default=C.INTERIM_DIR)
    ap.add_argument("--processed-dir", type=Path, default=C.PROCESSED_DIR)
    ap.add_argument("--chunksize", type=int, default=2_000_000,
                    help="rows per read chunk; 0 disables chunking")
    ap.add_argument("--benchmark", action="store_true",
                    help="run the naive-vs-optimised comparison on day 1")
    args = ap.parse_args(argv)

    chunksize = None if args.chunksize == 0 else args.chunksize

    if args.benchmark:
        print("Task 1 benchmark (single day)")
        table = benchmark(args.raw_dir, chunksize)
        out = C.TABLE_DIR / "memory_benchmark.csv"
        table.to_csv(out, index=False)
        print(table.to_string(index=False))
        print(f"  -> {out}")

    print("Ingesting daily files ...")
    reports = ingest_all(args.raw_dir, args.interim_dir, chunksize)
    if reports:
        reports_to_frame(reports).to_csv(C.TABLE_DIR / "ingest_log.csv", index=False)

    print("Building dense matrix ...")
    meta = build_matrix(args.interim_dir, args.processed_dir)
    pd.Series(meta).to_json(C.PROCESSED_DIR / "matrix_meta.json", indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
