# Data dictionary

## Raw files

`sms-call-internet-mi-YYYY-MM-DD.txt`, tab-separated, **no header**, one file
per day for 2013-11-01 to 2014-01-01 (62 files).

| # | Column | Type | Notes |
|---|---|---|---|
| 0 | `square_id` | int | 1–10,000. Cell of the 100×100 grid over Milan; each cell is about 235×235 m. |
| 1 | `time_interval` | int | Start of the 10-minute bin, **Unix milliseconds, UTC**. Bin end = value + 600,000. |
| 2 | `country_code` | int | Country of the other party. **Summed out during ingestion** — this study is not country-resolved. |
| 3 | `sms_in` | float | Dropped. |
| 4 | `sms_out` | float | Dropped. |
| 5 | `call_in` | float | Dropped. |
| 6 | `call_out` | float | Dropped. |
| 7 | `internet` | float | The target. Activity proportional to CDR volume — a relative measure, not bytes. |

Empty fields mean no activity of that type in that bin, not a failed
measurement. For `internet` these are treated as zero.

### What "internet activity" actually measures

A CDR is generated when a user starts or ends a connection, and additionally
whenever a connection has run for more than 15 minutes or has moved more than
5 MB. The value is therefore a **proxy for connection events**, not for traffic
volume. Two consequences worth stating in the report:

1. Values are not comparable to byte counts, and only relative comparisons
   (between areas, between times) are meaningful.
2. The 15-minute / 5 MB rule injects its own periodicity into a 10-minute
   series, which is a plausible contributor to short-lag autocorrelation
   independent of human behaviour.

### Timezone

The stamps are UTC. The pipeline converts to `Europe/Rome` (`src/config.py`)
so diurnal statements describe Milanese behaviour. The period is entirely in
CET — DST ended 2013-10-27, before the data starts — so no duplicated or
missing local hours occur.

## Processed artefacts

| Path | Contents |
|---|---|
| `data/interim/<date>.parquet` | One day, country summed out: `square_id` (int16), `time_interval` (int64), `internet` (float32). |
| `data/processed/internet_matrix.npy` | Dense `(n_timesteps, 10000)` float32 matrix. |
| `data/processed/timestamps.parquet` | Row index, tz-aware `Europe/Rome`. |
| `data/processed/square_ids.npy` | Column index, 1–10,000. |
| `data/processed/square_totals.parquet` | Total traffic per square over the full period. |
