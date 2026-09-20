# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

One-step-ahead forecasting of mobile Internet traffic on the Telecom Italia Big
Data Challenge dataset for Milan (November 2013 – January 2014), comparing a
statistical model, a recurrent network and a convolutional network across three
geographical areas.

**Research question.** How do different sequential models compare for one-step-ahead
mobile network traffic forecasting, and how does their performance vary across
geographical areas with different traffic characteristics?

---

## 1. Data

The dataset is **not** included in this repository (it is ~19 GB and subject to
Dataverse terms of use). Download it yourself:

| Item | Source |
|---|---|
| Telecommunications – SMS, Call, Internet – MI | <https://doi.org/10.7910/DVN/EGZHFV> |
| Milan grid (GeoJSON, optional) | <https://doi.org/10.7910/DVN/QJWLFU> |

You need the 62 files `sms-call-internet-mi-YYYY-MM-DD.txt`
(2013-11-01 → 2014-01-01). Put them anywhere and point the pipeline at them.

The browser download builds a single server-side ZIP, which fails here: the
dataset is 19.4 GB and Harvard Dataverse caps ZIP bundles at 15 GB. Use the
included downloader instead — it fetches files individually from the public
API, resumes interrupted transfers, and skips anything already complete:

```bash
python -m scripts.download_dataset --out-dir ~/Documents/traffic-forecasting/Dataverse
```

Safe to re-run as often as needed.

Each row is tab-separated with no header:

```
square_id  time_interval  country_code  sms_in  sms_out  call_in  call_out  internet
```

`time_interval` is the start of a 10-minute bin in Unix **milliseconds** (UTC);
absent activity is an empty field. There is one row per
`(square_id, slot, country_code)`, so the country dimension has to be summed out.
Full notes in `docs/DATA_DICTIONARY.md`.

## 2. Setup

Python 3.10+.

```bash
git clone <your-repo-url>
cd milan-traffic-forecasting
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is only needed for the LSTM and TCN. If the default wheel is wrong for
your machine, install it from <https://pytorch.org/get-started/locally/> first,
then `pip install -r requirements.txt`.

## 3. Running the pipeline

```bash
# Step 1 - ingest raw text -> per-day Parquet -> dense matrix, with the
#          memory before/after evidence for the report
python -m src.ingest --raw-dir /path/to/Dataverse --benchmark

# Step 2 - exploratory analysis: distribution, five-area series, ACF/PACF,
#          stationarity tests, STL decomposition
python -m src.eda

# Step 3 - hyper-parameter experiments (validation split only)
python -m src.tune --model sarima --sarima-search
python -m src.tune --model lstm --params '{"hidden": 64}'  --note "starting point"
python -m src.tune --model lstm --params '{"hidden": 128}' --note "64 underfit: train and val plateaued together"
python -m src.tune --model tcn  --grid   '{"channels": [16, 32], "levels": [4, 5, 6]}'

# Step 4 - final experiments: 9 plots, 3 metric tables, timing, failure windows
python -m src.run_experiments \
    --overrides '{"lstm": {"hidden": 128}, "tcn": {"channels": 32, "levels": 6}}'
```

Steps 2–4 read the processed matrix, so step 1 only has to run once. Ingestion
is idempotent: days already converted are skipped.

### Runtime and memory

Ingestion streams one day at a time and never holds the full raw table, so peak
memory is set by `--chunksize` (default 2M rows, roughly 100–200 MB) rather than
by the dataset size. Lower it on a constrained machine:

```bash
python -m src.ingest --raw-dir /path/to/Dataverse --chunksize 500000
```

The processed matrix is 8,928 × 10,000 float32 ≈ 340 MB on disk and is opened
memory-mapped, so the EDA and experiments only page in the columns they touch.

## 4. Repository layout

```
src/
  config.py           all paths, constants and the study design in one place
  memprofile.py       peak-RSS sampler + pandas deep memory, for Task 1 evidence
  ingest.py           raw text -> Parquet -> dense matrix; memory benchmark
  dataio.py           memory-mapped reads, per-square series, chronological split
  eda.py              Task 2: distribution, five-area series, ACF/PACF, ADF/KPSS, STL
  models.py           SeasonalNaive, SARIMA, LSTM, TCN behind one interface
  evaluate.py         MAE/RMSE/MAPE/sMAPE, plots, timing, failure windows
  tune.py             hyper-parameter runs with a persistent experiment log
  run_experiments.py  Task 4 driver: everything the report needs
scripts/
  make_synthetic_raw.py   fake data with the real schema, for testing the pipeline
outputs/
  figures/  tables/  logs/  models/
report/
  REPORT_OUTLINE.md       section-by-section scaffold tied to generated outputs
  experiment_log.md       where the tuning narrative is written up
```

## 5. Models

| Model | Family | Why it is here |
|---|---|---|
| Seasonal naive | benchmark | `x̂(t+1) = x(t+1−144)`. On a strongly periodic series this is a hard baseline; a model that cannot beat it has not earned its complexity. |
| SARIMA | linear, explicitly seasonal | Encodes the daily period the ACF exposes, with no learned representation. The long-standing reference point in traffic forecasting. |
| LSTM | recurrent, gated | Learns nonlinear state from a window of lags; can represent burst-and-decay behaviour a linear model cannot. |
| TCN | dilated causal convolution | Same receptive field as the LSTM reached a fundamentally different way — parallel, fixed, no recurrent state. |

All four share one interface (`fit`, `predict_rolling`, `describe`) so the
comparison and the timing measurements are made identically.

**One-step-ahead is enforced, not assumed.** To predict `x(t+1)` every model
receives the *observed* history up to `t`; predictions are never fed back as
inputs. The split is strictly chronological and the test week (16–22 December)
is untouched during training and tuning.

### SARIMA implementation note

The seasonal period is 144. Passing `seasonal_order=(0,1,0,144)` to statsmodels'
state-space SARIMAX forces a state vector of length ≈144, which is slow and
memory-hungry (it OOMs on a 4 GB machine). Since `D=1` with no seasonal AR/MA
terms, the seasonal difference is applied explicitly and inverted by adding back
`y(t+1−144)`. This is numerically identical and runs in well under a second.

## 6. Testing without the dataset

```bash
python -m scripts.make_synthetic_raw --out-dir /tmp/fake_raw --days 60
python -m src.ingest --raw-dir /tmp/fake_raw --benchmark
python -m src.eda
python -m src.run_experiments --models seasonal_naive sarima
```

This exercises the whole pipeline on small synthetic files with the real schema.
Results from synthetic data are **not** reportable — it exists to check that the
code runs, not to produce findings.

## 7. Citation

> Barlacchi, G., De Nadai, M., Larcher, R., Casella, A., Chitic, C., Torrisi, G.,
> Antonelli, F., Vespignani, A., Pentland, A., & Lepri, B. (2015). A multi-source
> dataset of urban life in the city of Milan and the Province of Trentino.
> *Scientific Data*, 2, 150055. https://doi.org/10.1038/sdata.2015.55
