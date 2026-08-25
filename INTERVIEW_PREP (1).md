# Retail Sales Forecasting & Demand Analysis — Complete Interview Prep

**Confirmed real numbers, from Mohith's own local run of `retail_forecasting.py` with real XGBoost.**

---

## 1. The 2-Minute Explanation (say this out loud, practice until natural)

> "I built a sales forecasting model on a retail dataset — daily sales for 50 products across 10 stores, over 5 years, about 900,000 records total.
>
> The core idea is: to predict tomorrow's sales, the most useful signal isn't some complicated external factor, it's the product's own recent sales history. So I engineered lag features — yesterday's sales, sales from a week ago, sales from two weeks ago — plus rolling averages over the past 7 and 28 days, which smooth out day-to-day noise and capture the underlying trend.
>
> One thing I was careful about: this is time-series data, so I couldn't just randomly split it into train and test sets the way you would for normal tabular data — that would let the model 'see the future' during training. Instead I trained on everything up to October 2017 and tested only on the last three months, which is what you'd actually face in a real deployment.
>
> I compared against a naive baseline — just predicting 'tomorrow will be the same as yesterday' — and the XGBoost model with lag features beat that baseline by about 44% in mean absolute error. When I looked at feature importance, the 7-day and 28-day rolling averages mattered far more than any single day's lag value, which makes intuitive sense — recent trend is a much more stable signal than any one day, which can be noisy."

---

## 2. Dataset — Full Detail

**Source:** Kaggle "Store Item Demand Forecasting Challenge" — a well-known, standard forecasting benchmark dataset.

**Size:** 913,000 rows, 4 raw columns, no missing values.

**Columns:**
| Column | Meaning |
|---|---|
| `date` | The calendar date of the sales record (2013-01-01 to 2017-12-31) |
| `store` | Which of 10 stores this record is for |
| `item` | Which of 50 products this record is for |
| `sales` | Units sold of that item, at that store, on that date (the target we're predicting) |

**Scale:** 10 stores × 50 items × ~1,826 days ≈ 913,000 rows — every store-item combination has a full daily time series across 5 years.

---

## 3. Why Lag Features and Rolling Windows — the Core Idea

**The problem with using only `date`, `store`, `item` directly:** a model can learn "store 3 tends to sell more" or "December is a high month," but it has no way to know if THIS particular product is currently trending up or down, running low, or had an unusual spike recently. Lag and rolling features give the model direct access to recent history for each product.

**Lag features created:**
- `lag_1` — yesterday's sales for this exact store-item combination
- `lag_7` — sales exactly one week ago (captures weekly patterns directly)
- `lag_14` — sales two weeks ago

**Rolling window features:**
- `rolling_mean_7` — average of the past 7 days (smooths out day-to-day noise, shows short-term trend)
- `rolling_mean_28` — average of the past 28 days (shows longer-term trend, smooths out even weekly cycles)
- `rolling_std_7` — how *volatile* the past 7 days were (a product with wildly swinging sales is inherently harder to predict than a stable one — this feature gives the model a sense of how much to trust recent values)

**Why both lag AND rolling features, not just one:** a single lag value (like yesterday) can be noisy — maybe yesterday was an unusually slow or busy day for no repeatable reason. A rolling average smooths that noise out. But rolling averages alone lose some immediacy — they react slowly to a real, sudden shift. Using both gives the model access to both the smoothed trend and the most recent individual data point.

**Important detail: `.shift(1)` before `.rolling()`.** The rolling mean is calculated on `group.shift(1).rolling(7).mean()`, not directly on the sales column. This shift is critical — without it, the "past 7 days" window would include *today's* actual sales value, which is exactly what we're trying to predict. That would be data leakage — an unrealistically easy problem that wouldn't work at all in real deployment, where today's sales aren't known yet at prediction time.

---

## 4. Why a Time-Based Split, Not a Random Split

This is one of the most important design decisions in the whole project, and a common mistake to explain confidently *not* making.

**The mistake:** randomly splitting rows into train/test (like `train_test_split(shuffle=True)`) would put some days from, say, March 2017 into training and other March 2017 rows into testing. Because of the lag/rolling features, information from "future" days relative to a test row could leak into training — and more fundamentally, it doesn't reflect the real problem, which is always "predict days that haven't happened yet, using only data from before."

**What was done instead:** everything before October 1, 2017 is training data; everything from October 1, 2017 onward is test data. This mirrors exactly how the model would be used in practice — trained on history, tested on the future.

---

## 5. Why XGBoost, and What Are the Alternatives

**Why XGBoost:**
- Handles tabular data with mixed feature types (categorical-ish `store`/`item` IDs alongside continuous lag values) very well, with minimal preprocessing needed.
- Captures non-linear relationships and interactions automatically (e.g., "the effect of `rolling_mean_7` might differ depending on `month`") without needing to manually engineer those interactions.
- It's fast, well-documented, and widely used in exactly this kind of forecasting competition — it's a genuinely standard, defensible choice, not an arbitrary one.

**Alternatives, and when you'd pick them instead:**
- **ARIMA / SARIMA (classical time-series models)** — good for a *single* time series with strong, well-behaved seasonality, but doesn't scale naturally to 500 different store-item series at once the way one XGBoost model handling all of them (with `store`/`item` as features) does.
- **LSTM (deep learning)** — can capture more complex sequential patterns, but needs much more data and tuning to outperform gradient boosting on a tabular-style problem like this, and is much harder to explain/interpret via feature importance.
- **Prophet (Facebook's forecasting tool)** — good for quick, interpretable single-series forecasts with strong holiday/seasonality handling, but again more suited to one series at a time rather than 500 at once.
- **Simple moving average / naive baseline** — this is literally what we compared against — useful as a sanity-check floor, not a real model.

**Honest answer if asked "why not deep learning":** For a dataset of this size and structure (~900K rows, well-defined tabular features), gradient boosting typically matches or beats deep learning approaches while being faster to train and much easier to interpret via feature importance — that trade-off mattered more here than deep learning's theoretical capacity for more complex patterns.

---

## 6. The Actual Results (confirmed — real XGBoost, run locally)

**Dataset stats confirmed from the real data:**
- 913,000 rows → 899,000 after dropping rows where lag/rolling windows aren't full yet (the first 14 days of each store-item series can't have a `lag_14` value)
- Train: 853,000 rows (through Sept 2017) | Test: 46,000 rows (Oct–Dec 2017)

**Clear seasonality found in EDA:**
- Weekday effect: sales climb steadily from Monday (41.4 avg) to Sunday (62.1 avg) — weekends are meaningfully busier
- Year-over-year growth: average daily sales grew from 43.5 (2013) to 58.8 (2017) — a genuine upward trend over the 5 years, not just noise

**Model performance:**
- Naive baseline (predict yesterday's value): MAE 10.755, RMSE 14.616
- XGBoost (real, confirmed): **MAE 5.971, RMSE 7.730, MAPE 13.11%**
- **44.5% reduction in MAE** over the naive baseline

**Feature importance (real XGBoost, gain-based):**
1. `rolling_mean_7` (0.4356) — by far the most important
2. `lag_7` (0.3129) — second most important, exact same day one week ago
3. `lag_14` (0.0853)
4. `rolling_mean_28` (0.0716)
5. `is_weekend` (0.0381), `dow` (0.0320) — calendar features, meaningful but secondary
6. `lag_1`, `month`, `day`, `rolling_std_7`, `year`, `store`, `item` — minor to negligible

**Note on feature importance methodology, worth knowing if asked:** an earlier pipeline-validation run (using a different algorithm, scikit-learn's HistGradientBoostingRegressor, as a stand-in before XGBoost was available) showed `rolling_mean_28` ranked higher and `lag_7` ranked lower than in the real XGBoost run. This isn't a contradiction — XGBoost's built-in importance is *gain-based* (how much each feature improved tree splits during training), while the other run used *permutation* importance (how much performance drops when a feature is shuffled). These two methods can legitimately disagree, especially between correlated features like `lag_7` and `rolling_mean_28`, which carry overlapping weekly-pattern information. Both runs agreed on the top-level finding though: recent history (weekly lag or rolling average) dominates over any calendar feature or distant lag.

**The one-sentence takeaway if asked to summarize the finding:** recent history — specifically the past week, whether as a 7-day rolling average or the exact value from 7 days ago — predicts tomorrow's sales far better than distant lags or calendar features alone, which makes intuitive business sense and is a genuinely useful, explainable insight.

---

## 7. Likely Interview Questions + Answers

**Q: Why didn't you just use a random train/test split?**
A: This is time-series data — a random split would mix future information into training, which doesn't reflect how the model would actually be used (predicting genuinely unseen future days). I used a time-based split instead: trained on everything before October 2017, tested only on the following three months.

**Q: What's data leakage, and where could it have happened here?**
A: Data leakage is when information that wouldn't be available at real prediction time accidentally ends up in the training features. Here, the risk was in the rolling average calculation — if I'd computed the 7-day rolling mean directly on the sales column without first shifting it by one day, "today's" rolling average would include today's actual sales, which is exactly what we're trying to predict. I used `.shift(1)` before `.rolling()` specifically to prevent that.

**Q: Why MAE and RMSE both? What's the difference?**
A: MAE (Mean Absolute Error) treats all errors proportionally — a 10-unit error counts as exactly 10. RMSE (Root Mean Squared Error) squares errors before averaging, which penalizes large errors much more heavily than small ones. Reporting both gives a fuller picture: if RMSE is much higher than MAE, that signals a few large outlier errors are dragging the average up, rather than errors being uniformly moderate.

**Q: What does the feature importance result actually tell you, practically?**
A: That recent trend (7 and 28-day averages) is a far stronger predictor than any single day's value, which makes business sense — a single day can be a noisy outlier (a random busy or slow day), but a sustained multi-week trend is a much more reliable signal of what's likely to happen next. Practically, that means simple 1-day lag features alone wouldn't have been enough — the rolling windows are doing most of the real work.

**Q: How would you improve this model further?**
A: A few directions: adding external features like holidays or promotions (which this dataset doesn't include, but real retail data usually does); trying per-store or per-item models instead of one global model, if certain series behave very differently from the average; and tuning the XGBoost hyperparameters more carefully (this was left at reasonable defaults, not extensively tuned) via cross-validation on a rolling time-window basis.

**Q: What is MAPE, and why report it alongside MAE and RMSE?**
A: MAPE (Mean Absolute Percentage Error) expresses error as a percentage of the actual value rather than an absolute number, which makes it easier to communicate to a non-technical audience ("the model is off by about 13% on average") and easier to compare across products with very different sales volumes, where a raw MAE of 6 units means something very different for a low-volume vs. high-volume item.

**Q: Why train one global model instead of a separate model per store-item combination?**
A: With 500 store-item combinations, training 500 separate models would mean each one only sees ~1,800 days of data, some of which have very sparse or low sales — not much for a model to learn from individually. A single global model, with `store` and `item` as features, lets the model share patterns learned across all 500 series (e.g., general seasonality) while still being able to differentiate between them via those ID features. It's a standard trade-off in retail forecasting between per-series and global models.

**Q: What would happen if you fed in `sales` from the *same day* as a feature by mistake?**
A: The model would achieve near-perfect accuracy on paper — because it would essentially be looking at the answer. This is a classic and easy-to-make leakage bug, and it's exactly why the `.shift(1)` step before any rolling calculation matters — I was specifically careful to make sure no feature could see the current day's actual sales value.

---

## 8. Concepts to Be Genuinely Comfortable With

- **Time series data** — what makes it different from regular tabular data (order and time matter, can't shuffle rows freely)
- **Lag features** — using past values of the target itself as predictive features
- **Rolling window statistics** (mean, std) — and why they need to be shifted to avoid leakage
- **Data leakage** — the general concept, and this project's specific example of it
- **Train/test split for time series** — why it must respect chronological order
- **MAE vs. RMSE vs. MAPE** — what each measures and when each is more informative
- **Gradient boosting (conceptually)** — an ensemble of decision trees built sequentially, where each new tree tries to correct the errors of the previous ones (doesn't need to go deeper than this level for most interviews, but should be able to say this much confidently)
- **Feature importance** — what it tells you and doesn't tell you (importance shows *how much* a feature helped predictions on this data, not necessarily direct real-world causation)
- **Baseline comparison** — why a naive baseline matters for honestly evaluating whether a "smart" model is actually adding value

---

*Same format as the previous two projects — 2-minute script, full dataset/method detail, real results, Q&A, concept checklist.*
