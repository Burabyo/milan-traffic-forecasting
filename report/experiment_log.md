# Experiment log

The brief asks for an iterative process: after each run, analyse the result and
use it to justify the next change. `src/tune.py` appends the numbers to
`outputs/logs/experiment_log.csv`; this file is where the *reasoning* goes.
The reasoning is the part that is graded, and the part a viva will probe.

Keep one block per experiment. Fill in the numbers from your own runs.

---

## SARIMA

### S1 — order search by AIC
Command: `python -m src.tune --model sarima --sarima-search`

| Observation | |
|---|---|
| Best orders by AIC | |
| AIC spread across the top 5 | |

Reasoning for the next step:
> e.g. "AIC differences among the top three are under 3, which is not a
> meaningful separation. I carried the two simplest forward and chose on
> validation MAE instead, preferring the smaller model at a tie."

### S2 — validation comparison of the shortlist
| order | val MAE | val RMSE | fit time (s) |
|---|---|---|---|
| | | | |

Decision and why:
>

---

## LSTM

### L1 — starting point
Parameters: `hidden=64, layers=2, seq_len=144, lr=1e-3, batch=128`

Why start here:
> e.g. "144-step input follows directly from the ACF spike at lag 144
> (Section 4). 64 units is a deliberately small starting point — I want to see
> underfitting before I add capacity."

| val MAE | val RMSE | epochs run | train time (s) |
|---|---|---|---|
| | | | |

What the training curve shows (`outputs/logs/trainlog_lstm_sq<id>.csv`):
> Train and validation loss both plateauing → underfitting, add capacity.
> Validation rising while training falls → overfitting, add regularisation or
> stop earlier. Say which you saw.

### L2 — change and rationale
Changed: ______ → ______ because ______

| val MAE | val RMSE | epochs run | train time (s) |
|---|---|---|---|
| | | | |

Did it do what you predicted? If not, what does that tell you?
>

### L3 — ...

---

## TCN

### T1 — receptive field first
The receptive field must cover the daily cycle. With kernel 3 and L levels it is
`1 + 2·(k−1)·(2^L − 1)` = 1 + 4·(2^L − 1):

| levels | receptive field | covers 144? |
|---|---|---|
| 4 | 61 | no |
| 5 | 125 | no |
| 6 | 253 | yes |

Reasoning:
> e.g. "levels=5 gives 125 steps, just short of a day. If the level-5 model is
> clearly worse than level-6, that is direct evidence that the daily lag carries
> the signal the ACF suggested — which is worth reporting as a result, not just
> as a tuning step."

| channels | levels | receptive field | val MAE | train time (s) |
|---|---|---|---|---|
| | | | | |

### T2 — capacity
Changed: ______ because ______

---

## Final configuration

| Model | Parameters chosen | Selected on |
|---|---|---|
| SARIMA | | AIC shortlist + validation MAE |
| LSTM | | validation MAE |
| TCN | | validation MAE |

Confirm before the final run: the test week (16–22 December) was not used in any
of the above.
