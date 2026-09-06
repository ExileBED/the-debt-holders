# Why we chose 0.40929 over the previous best (0.40967)

**Selected file:** `submission_lgbm_pay_types_0.40929.csv`  
**Code:** `BEST_0.40929.py`  
**Previous best:** `submission_lgbm_extfeat_teammate_proc.csv` (public **0.40967**)

We chose 0.40929 because it beat the previous record **on local testing first**, for a reason that makes sense in the data, and the public score then agreed. We did **not** choose extra_trees or the blend, even though those looked slightly better locally.

Lower log loss is better everywhere below.

---

## What changed

**Only five extra features.** Training loop unchanged: LightGBM, `learning_rate=0.1`, `num_leaves=31`, 10 repeats × 10 stratified folds, seeds `1000`–`1009`, early stopping 100, average of 100 models.

| New column | Definition |
|---|---|
| `n_revolve` | count of `PAY_* == 0` |
| `n_duly` | count of `PAY_* == -1` |
| `n_inactive` | count of `PAY_* == -2` |
| `log_limit` | `log1p(LIMIT_BAL)` |
| `remaining_credit` | `LIMIT_BAL - BILL_AMT1` |

70 features → **75**. No extra_trees, bagging, CatBoost, calibration, or hand-edited probabilities.

Reason: PAY codes `-2`, `-1`, `0` are not a risk ladder. Revolving (`0`) defaults **less** than paid-duly (`-1`). Type counts avoid treating that as numeric order.

---

## Local testing (this is the main evidence)

**Setup (same for both models)**

- Data: official `train.csv`, **n = 24,000**, default rate **22.1208%**
- Procedure: `StratifiedKFold(n_splits=10, shuffle=True, random_state=1000+r)` for `r = 0..9`
- 100 LightGBM fits; test/OOF probabilities averaged
- Metric: binary log loss on out-of-fold predictions
- Caveat: OOF is **optimistic** (early stopping uses the scored fold). It is still the **same** protocol used for 0.40981 and 0.40967, so the two rows below are comparable.

**Headline local result — new version is better, not worse**

| | Previous best (0.40967 recipe) | Chosen PAY-types (0.40929) | Change |
|---|---|---|---|
| Averaged 10×10 OOF log loss | 0.422873 | **0.422450** | **−0.000423** |
| Mean of 10 single-repeat OOF log losses | 0.425127 | **0.424744** | **−0.000383** |
| Std of those 10 repeats | 0.000331 | 0.000324 | slightly more stable |
| OOF ROC AUC | 0.78928 | **0.79001** | +0.00073 |
| Brier score | 0.132548 | **0.132459** | −0.000089 |
| Mean OOF probability | 0.22103 | 0.22094 | vs actual 0.22121 |
| Mean trees (best iteration) | 43.26 | 44.77 | almost the same |
| Features | 70 | 75 | +5 |
| Models averaged | 100 | 100 | same |

Source files: `outputs/record_attempt/oof_baseline.csv` + `meta_baseline.json` (old), `oof_pay_types.csv` + `meta_pay_types.json` (new).

Per-customer log loss improved for **12,739 / 24,000** rows and worsened for 11,261. Net average row log-loss change: **−0.000423**.

**Each of the 10 local repeats (OOF log loss)**

Same seeds `1000`–`1009`. PAY-types is better on **8 of 10** repeats.

| Repeat | Seed | Old (0.40967 recipe) | PAY-types | PAY-types better? |
|---|---|---|---|---|
| 0 | 1000 | 0.424801 | 0.425245 | no |
| 1 | 1001 | 0.425229 | 0.424637 | yes |
| 2 | 1002 | 0.424761 | 0.424270 | yes |
| 3 | 1003 | 0.425589 | 0.425303 | yes |
| 4 | 1004 | 0.424771 | 0.424391 | yes |
| 5 | 1005 | 0.424803 | 0.424843 | no |
| 6 | 1006 | 0.425637 | 0.424960 | yes |
| 7 | 1007 | 0.425054 | 0.424565 | yes |
| 8 | 1008 | 0.425138 | 0.424599 | yes |
| 9 | 1009 | 0.425490 | 0.424622 | yes |
| **Mean** | | **0.425127** | **0.424744** | **8 / 10** |

**Local log loss by PAY_0 (where the new features should matter)**

Gains are on the large groups that the type-counts target (`-2`, `-1`, `0`, and `2`). Tiny groups (`PAY_0 ≥ 4`, n ≤ 56) are noisy and not why we switched.

| PAY_0 | n | Actual default rate | Old OOF ll | New OOF ll | Change |
|---|---|---|---|---|---|
| −2 | 2,241 | 13.43% | 0.36708 | **0.36665** | −0.00043 |
| −1 | 4,561 | 16.44% | 0.41283 | **0.41211** | −0.00071 |
| 0 | 11,739 | 12.80% | 0.35682 | **0.35634** | −0.00048 |
| 1 | 2,944 | 34.34% | **0.59640** | 0.59683 | +0.00043 |
| 2 | 2,147 | 69.07% | 0.60097 | **0.60000** | −0.00097 |
| 3 | 262 | 75.19% | **0.52709** | 0.52733 | +0.00025 |
| 4–8 | 106 | mixed | noisy | noisy | ignore |

**Local calibration (10 probability bins), chosen model**

| Predicted bin | n | Actual default rate | Mean predicted prob. |
|---|---|---|---|
| [0.0, 0.1) | 7,240 | 6.06% | 6.80% |
| [0.1, 0.2) | 8,227 | 15.05% | 14.47% |
| [0.2, 0.3) | 3,278 | 22.15% | 23.97% |
| [0.3, 0.4) | 1,460 | 35.14% | 34.33% |
| [0.4, 0.5) | 844 | 46.45% | 44.60% |
| [0.5, 0.6) | 713 | 57.08% | 54.95% |
| [0.6, 0.7) | 1,139 | 64.79% | 65.10% |
| [0.7, 0.8) | 1,050 | 77.52% | 74.55% |
| [0.8, 0.9) | 49 | 85.71% | 81.25% |

Mean OOF probability **0.2209** vs labelled default rate **0.2212**. We submit probabilities, not 0/1 labels.

---

## Public score (supporting only)

| File | Public log loss |
|---|---|
| Previous best | 0.40967 |
| **Chosen PAY-types** | **0.40929** |

Public improvement **0.00038**, in the **same direction** as local **0.00042**. Test-set mean predicted probability **0.2191** (train rate 0.2212).

---

## What we did *not* choose — better local, worse public

These had **lower** (better-looking) local OOF than PAY-types. We still rejected them.

| Attempt | Local 10×10 OOF | Public | Decision |
|---|---|---|---|
| extra_trees (same 70 features, different trees) | 0.422420 | **0.41211** | reject — training-loop change; local OOF lied |
| Blend extra_trees + PAY-types + bagging | 0.422017 | **0.41037** | reject — extra_trees pulled it down |
| PAY-types + `n_late1` only | 0.422508 | not submitted | reject — worse than 0.422450 locally |
| PAY-types + four more columns | 0.422860 | not submitted | reject — worse locally; never uploaded |

So: **the current leaderboard best is not the model with the best local folder score.** extra_trees / blend were. We discarded those because they changed how trees are grown, and the public set punished that. PAY-types is the model that beat **0.40967 specifically**, both locally and publicly, with a justified feature add.

---

## Public / private split

If the dashboard is only ~30% of the 6,000-row test, order can still move on a hidden remainder. We cannot prove 0.40929 wins that slice.

We can say we did not pick it by fishing the dashboard:

1. Local 10×10 OOF already beat 0.40967 (tables above).
2. PAY-type counts are a data reason, not a leaderboard reason.
3. extra_trees and the blend lost publicly and were dropped.
4. Further features were worse locally and were never uploaded.

Hedge if two files are allowed: keep 0.40967. Do not put extra_trees or the blend back as a “maybe better on private” pick.

---

## One-line summary

**0.40929 = 0.40967 model + five PAY-type / limit features. Local 10×10 log loss 0.42287 → 0.42245 (8 of 10 repeats better). Public 0.40967 → 0.40929. That is the reason we switched.**
