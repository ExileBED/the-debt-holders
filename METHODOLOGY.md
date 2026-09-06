# Methodology report — credit-card default probabilities (0.40929, no TE)

This packet is the **PAY-types-only** solution (public **0.40929**). It does **not** use nested target encoding and is not the 0.40900 blend packet.

**Task:** estimate `P(default = 1)` for 6,000 hidden-test customers.  
**Metric:** binary log loss (lower is better).  
**File under review:** `submission_lgbm_pay_types_0.40929.csv`  
**Public score:** 0.40929  
**Code:** `BEST_0.40929.py`

## 1. Approach

The final model is a **seed- and split-averaged LightGBM** trained to minimise binary log loss. It is a direct continuation of the team’s previous public records:

| Public log loss | What it was |
|---|---|
| 0.40981 | Teammate LightGBM 10×10 average, original engineered features |
| 0.40967 | Same loop + extra utilisation / pay-to-limit / delinquency features |
| **0.40929** | **Same loop as 0.40967 + five extra columns only** |

The 0.40929 result does **not** change hyperparameters, ensembling of different algorithms, or post-hoc calibration. The only modelling delta vs 0.40967 is feature generation (section 3).

Log loss penalises confident mistakes, so the pipeline predicts probabilities, averages many similar models, and avoids class weighting / SMOTE.

## 2. Data cleaning and preprocessing

Training: 24,000 labelled rows. Test: 6,000 rows. No missing values. No `client_id` overlap.

We **did not** drop undocumented category codes (`EDUCATION` 0/5/6, `MARRIAGE` 0), negative bill amounts, or duplicate feature rows. Those values are left as-is for LightGBM. `client_id` is never used as a predictor.

No scaling, no imputation, no target encoding. Pay-ratio columns are `PAY_AMT{i} / BILL_AMT{i+1}` with NaN where the denominator is ≤ 0 (LightGBM handles NaN natively).

Demographic fields `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are used as provided. They are source-coded categories; undocumented extra codes are not relabelled. They should not be read as a production lending policy.

## 3. Feature engineering

Row-wise only. Nothing is fitted on the training set and then applied to test.

**Carried over from 0.40967 (unchanged):**

- Payment / previous-bill ratios and summaries (`payratio_*`, months paid in full / none)
- Limit utilisation (`util_*`, mean/max/last/trend)
- Repayment-status slope, mean, recent-minus-oldest
- Delinquency max/sum/count and current streak; `severe_delinq_count`
- Bill deltas and total bill change
- Pay-to-limit ratios, zero-payment months, recent vs older pay amounts

**Added for 0.40929 (the only change):**

| Feature | Definition |
|---|---|
| `n_revolve` | count of `PAY_0, PAY_2, …, PAY_6` equal to `0` (revolving) |
| `n_duly` | count equal to `-1` (paid duly) |
| `n_inactive` | count equal to `-2` (no consumption) |
| `log_limit` | `log1p(LIMIT_BAL)` |
| `remaining_credit` | `LIMIT_BAL - BILL_AMT1` |

Reason for the PAY-type counts: default rates are not monotone in the raw codes. Revolving (`0`) defaults less than paid-duly (`-1`). Numeric summaries such as `pay_mean` treat `-2 < -1 < 0` as an ordered risk ladder; type counts do not.

Total predictors: **75** (70 previous + 5 new).

## 4. Validation strategy

Inside each of 10 repeats, `StratifiedKFold` with 10 folds (`shuffle=True`, `random_state=1000+r`). Early stopping uses that fold’s validation set. Out-of-fold probabilities are therefore **optimistic** relative to a fair holdout. We still report them because they match the procedure of the 0.40981 / 0.40967 records and are comparable across those runs.

- Local 10×10 averaged OOF log loss (0.40929 model): **0.42245**
- Local OOF AUC: **0.7900**
- Mean trees (best iteration): **~45**

The public metric is the organiser’s hidden test log loss (**0.40929**).

## 5. Models tested and final selection

The **submitted** model is LightGBM with the hyperparameters below, 10×10 averaging, and the 75-column feature set.

Other internal experiments (not submitted as the 0.40929 file):

| Attempt | Public log loss | Outcome |
|---|---|---|
| Extended LGBM 10×10 (0.40967 file) | 0.40967 | Previous record |
| Same loop + 5 PAY-type / limit features | **0.40929** | **Selected** |
| Blend extra_trees + PAY-types + bagging | 0.41037 | Worse; discarded |
| extra_trees=True, same features as 0.40967 | 0.41211 | Worse; discarded |
| Heavy regularisation (lr=0.04, 15 leaves, L2) | ~0.411 | Worse historically |
| CatBoost 50/50 blend | worse than 0.40967 | Discarded |
| Inner early-stopping then full-fold refit | under-fit (~35 trees) | Discarded |

Selection rule after those public failures: keep the 10×10 LightGBM loop that already generalised, and only add a small row-wise feature block.

## 6. Ensembling and post-processing

- **Ensemble:** unweighted average of 100 LightGBM models (same spec, different seeds/splits). Not a mix of LightGBM + CatBoost + neural nets.
- **Post-processing:** clip probabilities to `[1e-6, 1-1e-6]`. No Platt/isotonic calibration, no temperature scaling, no manual row edits.
- **Submission construction:** merge predictions onto `sample_submission.csv` by `client_id` so row order matches the organiser file.

## 7. Final model hyperparameters

```
objective: binary
metric: binary_logloss
learning_rate: 0.1
num_leaves: 31
verbose: -1
deterministic: True
force_row_wise: True
num_boost_round: 2000
early_stopping_rounds: 100
n_repeat: 10
n_splits: 10
ensemble weights: 1/100 each
```

LightGBM 4.7.0. No pretrained checkpoint.

## 8. Key results, observations, limitations

- PAY_0 is the strongest raw signal (e.g. code `2` defaults at ~69%, revolving `0` at ~13%).
- Treating PAY codes as a numeric continuum is misleading; type counts helped public log loss.
- Optimistic 10×10 OOF is **not** a reliable screen for training-procedure changes (extra_trees looked better locally and scored 0.41211 publicly).
- Feature-only changes with the established loop have now moved the public score twice (0.40981 → 0.40967 → 0.40929).
- Limitations: early stopping peeks at the scored fold; 100 correlated trees; demographics used as competition features, not as a fair lending design; no causal claim about education/sex/marriage.

## 9. Reproducibility

See `README.md`. One command: `python BEST_0.40929.py`. Intermediate processed tables are not saved; they are built in memory from the official CSVs.
