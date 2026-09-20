# Setup on a Mac (Apple Silicon) — step by step

Written for someone who has not set up a Python project on this machine before.
Do these in order. If a step fails, stop there rather than continuing.

---

## Step 0 — Find out what your machine has

Apple menu (top-left corner) → **About This Mac**. Note two things:

- **Memory** — e.g. "16 GB". This decides your chunk size in Step 6.
- **Chip** — e.g. "Apple M5". Confirms you are on Apple Silicon, which means
  PyTorch will use the MPS backend rather than CUDA.

## Step 1 — Open Terminal

Press `Cmd + Space`, type `Terminal`, press Enter. A window with a text prompt
opens. Every command below is typed there, one line at a time, pressing Enter
after each.

If you have never used it: the prompt is just a place to type commands. Nothing
you run below can damage your Mac.

## Step 2 — Check that Python is installed

```bash
python3 --version
```

- You see `Python 3.10` or higher → continue to Step 3.
- You see `Python 3.9` or lower, or "command not found" → install a current
  version from <https://www.python.org/downloads/macos/>, then close Terminal,
  reopen it, and run the check again.

macOS may prompt you to install the Xcode Command Line Tools. Accept it; it is
a normal one-off Apple download.

## Step 3 — Make a folder for the project

```bash
mkdir -p ~/Documents/traffic-forecasting
cd ~/Documents/traffic-forecasting
```

`~` means your home folder. `cd` moves into a folder. You are now working inside
`Documents/traffic-forecasting`.

## Step 4 — Unzip the repository here

Move the downloaded `milan-traffic-forecasting.zip` into that folder using
Finder, then:

```bash
unzip milan-traffic-forecasting.zip
cd milan-traffic-forecasting
ls
```

`ls` lists what is in the folder. You should see `README.md`, `src`, `scripts`,
`docs`, `report`, `requirements.txt`.

## Step 5 — Create an isolated environment and install the libraries

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

A virtual environment keeps this project's libraries separate from the rest of
your system, so nothing you install here can break another project.

After `source .venv/bin/activate` your prompt gains a `(.venv)` prefix. **Every
time you open a new Terminal window you must run that line again** (after
`cd`-ing into the project folder). Forgetting it is the single most common cause
of "it worked yesterday" problems.

The install takes a few minutes; PyTorch is a large download.

Check it worked:

```bash
python -c "import torch, pandas, statsmodels; print(torch.__version__)"
python -c "import torch; print('Apple GPU available:', torch.backends.mps.is_available())"
```

The second should print `True` on an M-series Mac. If it prints `False`, the
models still run on CPU — slower, but correct.

## Step 6 — Download the dataset

Go to <https://doi.org/10.7910/DVN/EGZHFV> in your browser. You have to accept
the terms of use; that is why this cannot be scripted.

Download the 62 files named `sms-call-internet-mi-YYYY-MM-DD.txt`
(2013-11-01 through 2014-01-01). It is roughly 19 GB, so start it early and
check you have the free disk space first (Apple menu → About This Mac →
More Info → Storage).

Put them all in one folder. A clean choice:

```
~/Documents/traffic-forecasting/Dataverse/
```

Check the count before continuing — a partial download will produce a silently
incomplete study:

```bash
ls ~/Documents/traffic-forecasting/Dataverse/*.txt | wc -l
```

This should print `62`.

## Step 7 — Run the ingestion

```bash
python -m src.ingest --raw-dir ~/Documents/traffic-forecasting/Dataverse --benchmark
```

If your Mac has **8 GB** of memory, use a smaller chunk size:

```bash
python -m src.ingest --raw-dir ~/Documents/traffic-forecasting/Dataverse --benchmark --chunksize 500000
```

This takes roughly 20–40 minutes. It prints a line per day. You can leave it
running; if it is interrupted, just run it again — days already done are
skipped.

Do not close the Terminal window while it runs. To put your Mac aside, close the
lid only if you have disabled sleep; otherwise leave it awake.

When it finishes you will have:

- `outputs/tables/memory_benchmark.csv` — the before/after evidence for Task 1
- `outputs/tables/ingest_log.csv` — per-day timings
- `data/processed/internet_matrix.npy` — the matrix everything else reads

## Step 8 — Run the exploratory analysis

```bash
python -m src.eda
```

Two minutes. It prints the top three squares and the key statistics, and writes
four figures to `outputs/figures/`. Open that folder in Finder and look at them.

## Step 9 — Run a first experiment (no neural models yet)

```bash
python -m src.run_experiments --models seasonal_naive sarima
```

Fast. This confirms the whole chain works before you spend time on training.

## Step 10 — Come back with the numbers

Send me:

- the contents of `outputs/tables/memory_benchmark.csv`
- the top-three squares the EDA printed
- `outputs/tables/table3_autocorrelation.json` and `table4_stationarity.json`
- the metrics table from Step 9

Then we do the tuning iterations, the neural models, and the interpretation.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: python3` | Python not installed | Step 2 |
| `No module named pandas` | environment not activated | `source .venv/bin/activate` |
| `No raw files found` | wrong `--raw-dir`, or files not unzipped | check `ls` prints 62 `.txt` files |
| Machine slows to a crawl during ingest | chunk size too large | re-run with `--chunksize 250000` |
| `Killed` / process disappears | out of memory | same fix as above |
| `zsh: permission denied` | running from a protected folder | work under `~/Documents` |

Copy the **full** error message when you ask about one. The last line alone is
usually the least informative part.
