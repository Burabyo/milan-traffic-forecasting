# Report scaffold

This is a structure and a checklist, not a draft. Every `>` prompt is a question
your prose has to answer in your own words, using numbers your own run produced.
Suggested total: **10–14 pages** including figures.

A note on how the marks fall: the brief says explicitly that the objective is
*not* to find the lowest error. Sections 2, 4 and 6 — where you justify and
interpret — carry more weight than Section 5, where you report. Budget your
writing time accordingly.

---

## 1. Introduction (~0.75 page)

- What mobile traffic forecasting is used for operationally: capacity planning,
  base-station sleep scheduling, dynamic resource allocation, anomaly detection.
  One or two sentences, cited — say why a 10-minute-ahead forecast is worth
  anything to an operator.
- The dataset and the grid, in two sentences.
- State the research question verbatim from the brief.
- Close with what you actually did and what you found — one sentence each. Write
  this paragraph last.

> What decision would a 10-minute-ahead forecast change for an operator? If you
> cannot answer that, the motivation paragraph is decoration.

## 2. Related Work (~1.5–2 pages)

Organise by **approach**, not one-paper-per-paragraph. Three or four themes:

1. Classical statistical forecasting (ARIMA/SARIMA, Holt-Winters) on network traffic.
2. Recurrent deep models (LSTM/GRU) for cellular traffic.
3. Convolutional and attention-based sequence models (TCN, Transformer).
4. Spatio-temporal models that use the grid structure — relevant context even
   though this study is single-area, and a natural "future work" thread.

For each theme: what was done, what was found, and **what it implies for your
setup**. The last clause is what turns a summary into a review.

> Which paper's finding would change your design if it were wrong? Say so.

End the section with the three-model justification (this is Task 3 folded in, as
the brief's report structure requires). For each model, three things:
- the property of *your* data, from Section 4, that motivates it;
- the prior evidence that it handles that property;
- a limitation you expect, stated **before** you see the results. Then check
  in Section 6 whether you were right. Getting a prediction wrong and saying so
  is worth more than a bland after-the-fact rationalisation.

### Starting reading list

Verify every one of these against the actual paper before citing it — do not
cite anything you have not opened.

| Theme | Reference |
|---|---|
| Dataset | Barlacchi et al. (2015), *Scientific Data* 2:150055 |
| Statistical baseline | Box & Jenkins, *Time Series Analysis: Forecasting and Control* |
| ARIMA + NN hybrids | Zhang (2003), *Neurocomputing* 50:159–175 |
| Recurrent models | Hochreiter & Schmidhuber (1997), *Neural Computation* 9(8) |
| Deep cellular traffic | Wang et al. (2017), IEEE INFOCOM |
| Long-term mobile traffic | Zhang & Patras (2018), ACM MobiHoc |
| LSTM on raw mobile traffic | Trinh et al. (2018), IEEE PIMRC |
| TCN | Bai, Kolter & Koltun (2018), arXiv:1803.01271 |
| Attention | Vaswani et al. (2017), NeurIPS |
| Baselines matter | Makridakis, Spiliotis & Assimakopoulos (2018/2020), M4/M5 findings |

Search Scopus/IEEE Xplore for "cellular traffic prediction Milan" — several
papers use this exact dataset, which lets you compare your numbers to published
ones. That comparison is worth a paragraph on its own.

## 3. Dataset and Data Preparation (~1.5 pages)

- Provenance, period, grid, 10-minute granularity, what `internet` actually
  measures (see `docs/DATA_DICTIONARY.md` — the CDR generation rule matters and
  most reports miss it).
- The row explosion: one row per (square, slot, country). State the raw size.
- Your memory strategy, as a sequence of decisions with a reason each:
  column selection → dtype downcasting → chunked reads → country aggregation →
  Parquet → dense matrix → memory-mapped access.

**Table 1** — from `outputs/tables/memory_benchmark.csv`. Report both the
object-level (`frame_mb`) and process-level (`peak_rss_mb`) numbers and explain
why they differ. Quote the reduction factor.

**Trade-offs — do not skip these, they are explicitly marked in the brief:**
- float32 vs float64: how many significant digits do you keep, and is that
  enough given the value range you actually observe? Check it, don't assert it.
- Dense vs sparse: report the non-zero fraction from `matrix_meta.json` and say
  why dense wins here (and at what sparsity it would stop winning).
- Country aggregation is irreversible — it forecloses any country-resolved
  question later.
- Chunking costs wall-clock time for lower peak memory. Quantify from
  `ingest_log.csv`.

**Table 0** — `table0_coverage.csv`: missing timesteps, duplicates, all-zero
squares. Integrity checks belong in the report, briefly.

## 4. Exploratory Analysis (~2.5 pages)

**Figure 1** — `fig1_traffic_distribution.png`, with `table1_distribution_stats.csv`.
> Quote skewness, the Gini coefficient and the busiest-1% share. Is it plausibly
> log-normal or heavy-tailed? What does that concentration imply for a model
> trained on one area and applied to another — and for your choice of loss?

**Figure 2** — `fig2_five_series_two_weeks.png`, with `table2_five_series_stats.csv`.
> Required discussion: similarities, differences, notable or unusual behaviour,
> and possible explanations. Work through: Is the daily cycle present in all
> five? Do the peak hours differ? How do weekend/weekday ratios compare
> (`weekend_over_weekday`)? Are the reference squares 4159 and 4556 noisier in
> relative terms (`cv`, `zero_fraction`) even though they are smaller in
> absolute terms? Look up where the top squares are on the grid — a business
> district and a residential area produce visibly different shapes and that is a
> real explanation, not a guess.
> Watch for the public holidays in the window and anything anomalous around
> them. Milan has a local holiday on 7 December (Sant'Ambrogio) plus the
> Christmas period inside your test week — these are prime failure-case
> candidates for Section 6.

**Figure 3** — `fig3_acf_pacf.png`, with `table3_autocorrelation.json`.
> The number that matters: ACF at lag 144. Quote it. Does a spike survive at lag
> 1008 (one week)? Then state the consequence explicitly — *this is why the input
> window is 144 steps and why the TCN needs a receptive field ≥ 144*. That
> sentence is the whole point of running the analysis.

**Figure 4** — `fig4_stl_decomposition.png`, with `table4_stationarity.json`.
> ADF and KPSS have opposite nulls; report both and say what the combination
> implies (they can disagree — if they do, that is itself informative). Quote the
> seasonal strength and the residual share of variance: the residual share is
> roughly the part a model actually has to learn, and it bounds how much better
> than seasonal-naive anything can do.

## 5. Methodology (~2 pages)

- Forecasting setup: the formal statement of one-step-ahead from the brief, and
  how `predict_rolling` enforces it.
- Input representation, **stated per model** (the brief asks for this explicitly):
  sequence length 144, log1p + z-score with statistics fitted on train only,
  why log1p (skew, exact zeros, invertibility), why scaler statistics must not
  see the test week.
- Chronological train/validation/test split with exact dates and sizes. Say
  plainly that the test week is untouched during tuning.
- The three models: structure, parameter counts, training procedure (Adam, MSE
  on the scaled scale, early stopping on validation, gradient clipping, LR
  scheduling). Include the SARIMA differencing note from the README — it is a
  real implementation decision with a real justification.
- Hyper-parameter strategy: AIC search for SARIMA orders, then the iterative
  narrative from `report/experiment_log.md`.
- Metrics, including **why MAPE is reported on strictly positive actuals only**
  and what fraction was excluded (`mape_excluded_%`). sMAPE is reported
  alongside because MAPE is unstable near zero.

## 6. Results and Discussion (~3–4 pages)

**Tables 2–4** — `metrics_sq<id>.csv`, one per area. MAE, MAPE, RMSE (plus sMAPE, R²).

**Figures 5–13** — the nine `pred_<model>_sq<id>.png`. If space is tight, put the
three `comparison_sq<id>.png` in the body and the nine in an appendix — but the
nine must exist.

**Table 5** — `timing.csv` / `timing_summary.csv` with `hardware.json`. State the
machine, that inference is the median of five repeats after a warm-up, and
whether times are from one area or averaged across three.

Then the discussion the brief actually asks for:

> **Does the ranking hold across all three areas?** If it does not, that is the
> most interesting result in the report — relate it to the per-area statistics
> in `table2_five_series_stats.csv` (coefficient of variation, zero fraction).
> A model that wins on a smooth high-traffic area and loses on a spiky one is
> telling you something about its inductive bias.
>
> **Accuracy against cost.** Put MAE next to training time and parameter count.
> If the LSTM buys 3% MAE for 200× the training time of SARIMA, say whether that
> is worth it *for the stated operational use*, and defend the answer.
>
> **Does anything beat seasonal-naive, and by how much?** Compare the margin to
> the residual share of variance from Section 4. If a model barely beats the
> benchmark, say so — that is a finding, not a failure.
>
> **Failure case (required).** Start from `failure_windows.csv` and
> `error_by_hour_sq<id>.png`. Pick one period, show it, and explain it. Strong
> candidates: the morning ramp (steepest gradient), the Christmas-period
> deviation from the normal weekly pattern inside the test week, and isolated
> spikes. Distinguish *lag* (the model reproduces the shape one step late — look
> for it in the residual panel) from *amplitude* error. A model that has learned
> a near-identity map on a smooth series looks good on MAE and is nearly useless;
> check whether yours has.
>
> **Compare to published results** on this dataset where you found them, and
> account for any gap.

## 7. Conclusion and Future Work (~0.5 page)

- Findings, answering the research question directly.
- Limitations, stated honestly: single area per model, one-step horizon only,
  one random seed (if so, say so — it weakens any small margin you report), no
  spatial information used, MAPE instability.
- Extensions: multi-step, spatio-temporal models over the grid, transfer across
  areas, exogenous inputs (weather, holidays).

## 8. References

Everything cited, consistent style.

---

## Pre-submission checklist

- [ ] Figure showing the distribution across all 10,000 areas, discussed
- [ ] Top three areas identified, with their totals stated
- [ ] Two-week series for 5 areas (top 3 + 4159 + 4556), discussed
- [ ] Two self-chosen analyses on the busiest area, each with a stated
      consequence for the forecasting approach
- [ ] Memory evidence before *and* after, with trade-offs
- [ ] Three genuinely different models, each justified from EDA + literature
- [ ] 9 actual-vs-predicted plots for the week 16–22 December
- [ ] 3 metric tables (MAE, MAPE, RMSE)
- [ ] Training and inference times + hardware + how they were measured
- [ ] Input representation described per model (sequence length, preprocessing,
      normalisation)
- [ ] Comparative analysis across accuracy, time and suitability
- [ ] Best model identified with **both** quantitative and qualitative support
- [ ] At least one failure case, shown and explained
- [ ] Tuning documented as a sequence of experiments with reasoning
- [ ] Every figure referenced in the text (delete any that is not)
- [ ] GitHub repo with README and dependencies
- [ ] 7–10 minute video: problem, data handling, model choice, findings,
      one technical decision, one limitation/failure
