# Study guide: the five extra features (0.40929)

This is the **only** modelling change vs the previous public record (0.40967).  
Same LightGBM, same hyperparameters, same 10×10 average of 100 models.

Code: `BEST_0.40929.py`  
Columns added at the bottom of `build_features()`.

```python
X["n_revolve"] = (X[PAY_STATUS] == 0).sum(axis=1)
X["n_duly"] = (X[PAY_STATUS] == -1).sum(axis=1)
X["n_inactive"] = (X[PAY_STATUS] == -2).sum(axis=1)
X["log_limit"] = np.log1p(X["LIMIT_BAL"].astype(float))
X["remaining_credit"] = X["LIMIT_BAL"] - X["BILL_AMT1"]
```

`PAY_STATUS` = `PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6` (six months; there is no `PAY_1` in this dataset).  
`PAY_0` is the **most recent** month.

---

## 1. What the PAY codes mean

These are **repayment-status codes**, not “how many dollars they paid.”

| Code | Usual meaning | How to think about it |
|---|---|---|
| `-2` | No consumption that month | Account mostly unused |
| `-1` | Paid duly (paid the statement) | Settled that month |
| `0` | Revolving (used credit, paid some, carried a balance) | Normal card use |
| `1, 2, 3, …` | Delayed by that many months | Arrears |

They look like numbers on a line: `-2 < -1 < 0 < 1 < 2`.  
**They are not a clean risk ladder.** That is the whole point of the first three features.

### Default rate by most recent status (`PAY_0`) — training set, n = 24,000

| PAY_0 | n | Default rate |
|---|---|---|
| −2 | 2,241 | 13.4% |
| −1 | 4,561 | 16.4% |
| **0** | **11,739** | **12.8%** |
| 1 | 2,944 | 34.3% |
| 2 | 2,147 | 69.1% |
| 3 | 262 | 75.2% |

Revolving (`0`) is **safer** than paid-duly (`-1`), and similar to unused (`-2`).  
If you treat the code as “higher = worse,” you get the **wrong order** for `-2`, `-1`, `0`.

Delay (`≥ 1`) really is worse. We already had that via raw `PAY_*`, `delinq_count`, `severe_delinq_count`. The gap was the three **non-delay** types.

---

## 2. Why the old features were not enough

The 0.40967 model already included:

- `pay_mean` — average of the six PAY codes as **numbers**
- `pay_slope` — trend in those numbers over months

Those assume `-2`, `-1`, `0` sit on one numeric scale. Example:

| Customer | Six-month PAY pattern | `pay_mean` (rough) | Reality |
|---|---|---|---|
| A | all `0` (revolving every month) | 0 | common, relatively low default |
| B | all `-1` (paid duly every month) | −1 | looks “better” numerically, but default rate is **higher** than revolving |
| C | mix of `-2` and `2` | could land near 0 | unused months + serious delay — not the same as A |

LightGBM *can* split on raw `PAY_0` and learn “`0` is OK, `2` is bad.”  
The extra counts give it a **direct** “how many months of each *type*” signal, so it does not have to reconstruct that from a misleading average.

---

## 3. Feature by feature

### `n_revolve` — months revolving (`PAY_* == 0`)

Count of months the customer **used the card and revolving-paid** (typical utilisation).

- Range: 0 to 6  
- Large group: `PAY_0 == 0` is about half the train set and has the **lowest** default rate among common codes (12.8%)  
- Reading it: more revolving months, holding other things equal, often means a working credit relationship, not distress

### `n_duly` — months paid in full (`PAY_* == -1`)

Count of months coded “paid duly.”

- Not automatically “safer than revolving” (see table above)  
- Can mix genuine full payers with people who only appear in some months  
- We count it as its **own type**, not as “−1, therefore better than 0”

### `n_inactive` — months with no consumption (`PAY_* == -2`)

Count of unused months.

- Low default as a *recent* status (13.4% when `PAY_0 == -2`)  
- Six inactive months is a different story from “inactive now but delayed earlier”  
- Again: a **type count**, not a number on the delay scale

Together, `n_inactive + n_duly + n_revolve` plus delay counts (`delinq_count`, `severe_delinq_count`) **partition** behaviour. We did **not** also add `n_late1` (`PAY_* == 1`): that duplicate made local OOF slightly **worse** (0.42251 vs 0.42245).

### `log_limit` — `log1p(LIMIT_BAL)`

Credit limit in NT dollars is very skewed (tens of thousands to 1,000,000).

- Default rate falls as limit rises (bottom quintile ~31.6%, top ~14.0%)  
- Trees *can* split on raw `LIMIT_BAL`; `log1p` makes **proportional** differences easier (200k vs 100k, not 100k vs 0)  
- `log1p` = `log(1 + x)`, so a limit of 0 does not explode  
- We **keep** raw `LIMIT_BAL` as well; this is an extra view, not a replacement

### `remaining_credit` — `LIMIT_BAL - BILL_AMT1`

Unused limit on the **latest** bill.

- High remaining credit: plenty of headroom  
- Low or negative: bill at or above the limit (`over_limit` style stress)  
- Uses the most recent bill (`BILL_AMT1`) to match “next payment” risk  
- We already had utilisation `util_1 = BILL_AMT1 / LIMIT_BAL`. Remaining credit is the **dollar gap**, which is not the same as the ratio (50k unused on a 60k limit vs on a 500k limit)

---

## 4. What we did *not* change

- No extra_trees, bagging, CatBoost, or calibration  
- No dropping of `SEX` / `EDUCATION` / `MARRIAGE` / `AGE`  
- Undocumented category codes and negative bills left as-is  
- Probabilities only (log loss); not 0/1 labels

---

## 5. Did they help? (local + public)

Lower log loss is better.

| | 0.40967 (70 features) | 0.40929 (+ these five) |
|---|---|---|
| Local 10×10 averaged OOF log loss | 0.42287 | **0.42245** |
| Local mean of 10 repeat OOFs | 0.42513 | **0.42474** (better on 8 of 10 seeds) |
| Public log loss | 0.40967 | **0.40929** |

Gains on local OOF were largest on the PAY groups these counts describe (`PAY_0` in {−2, −1, 0, 2}). Tiny delay codes (4–8, n ≤ 56) were noisy and not the reason we shipped.

Adding **more** columns after this (`n_late1`, over-limit flag, `log(PAY_AMT1)`, …) made local OOF **worse**. Stop here.

---

## 6. One-line memory aid

**Don’t treat PAY −2 / −1 / 0 as “−2 is best.” Count each type. Also give the model log limit and leftover credit in dollars.**
