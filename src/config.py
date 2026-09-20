"""Central configuration for the Milan traffic forecasting study.

Every path and constant used by the pipeline lives here so that experiments are
reproducible and the rest of the code contains no magic numbers.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where the 62 raw `sms-call-internet-mi-YYYY-MM-DD.txt` files live.
# Override on the command line with --raw-dir if you keep them elsewhere.
RAW_DIR = PROJECT_ROOT.parent / "Dataverse"

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"     # one Parquet file per day
PROCESSED_DIR = DATA_DIR / "processed"  # dense matrix + metadata
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"

for _d in (INTERIM_DIR, PROCESSED_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset constants (Barlacchi et al., 2015)
# --------------------------------------------------------------------------
RAW_COLUMNS = [
    "square_id",
    "time_interval",   # start of the 10-minute bin, Unix milliseconds (UTC)
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
]

# We only ever need three of the eight columns for this study.
USED_COLUMNS = ["square_id", "time_interval", "internet"]
USED_COLUMN_IDX = [0, 1, 7]

N_SQUARES = 10_000          # 100 x 100 grid over Milan
SLOT_MINUTES = 10
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES     # 144
SLOTS_PER_WEEK = SLOTS_PER_DAY * 7          # 1008

# The CDRs were recorded in Italian local time. The raw millisecond stamps are
# UTC; we localise to Europe/Rome so that "the daily peak is at 21:00" is a
# statement about Milanese behaviour and not about an arbitrary offset.
TIMEZONE = "Europe/Rome"

DATE_START = "2013-11-01"
DATE_END = "2014-01-01"

# --------------------------------------------------------------------------
# Study design
# --------------------------------------------------------------------------
# Squares the assignment names explicitly, plotted alongside the top three.
REFERENCE_SQUARES = [4159, 4556]

# First two weeks of the observation period, used for the EDA time-series plot.
EDA_WINDOW = ("2013-11-01", "2013-11-14")

# Held-out evaluation week for the forecasting experiments (inclusive).
TEST_WINDOW = ("2013-12-16", "2013-12-22")

# Everything strictly before TEST_WINDOW[0] is available for fitting; the last
# VAL_DAYS of that span are held out for early stopping / hyper-parameter choice
# so the test week is never touched during model selection.
VAL_DAYS = 7

# Input representation
SEQ_LEN = SLOTS_PER_DAY      # 144 lags = 24 h of history; justified by the ACF
HORIZON = 1                  # one-step-ahead (10 minutes)

RANDOM_SEED = 42
