# README — 0.40929 

**Leaderboard file:** `submission_lgbm_pay_types_0.40929.csv`  
**SHA-256:** `ee20a41ff96c6452be8761c0a41cebe240e1d67d5d1d9deacd7e8b06baaa99c1`

## What to send

Zip **this entire `final-submission-0.40929/` folder**.

| Expected item | File |
|---|---|
| 1. Methodology report | `METHODOLOGY.md` |
| 2. Complete source code | `BEST_0.40929.py` |
| 3. Final prediction file | `submission_lgbm_pay_types_0.40929.csv` |
| 4. Processed / intermediate data | `artifacts/` (OOF + test probabilities). Features are rebuilt in memory. |
| 5. Reproduction instructions | this README |
| 6. Final model information | `MODEL_INFO.md` |
| Required disclosure | `DISCLOSURE.md` |
| Why this file | `WHY_WE_CHOSE_0.40929.md`, `WHY_PUBLIC_BEST_NOT_LOCAL_BEST.md`, `FEATURE_STUDY_GUIDE.md` |

## Pipeline

```
competition CSVs
    → 75 row-wise features (including n_revolve / n_duly / n_inactive / log_limit / remaining_credit)
    → 100 LightGBM models (10 seeds × 10 folds)
    → unweighted mean of 100 probability vectors
    → clip to (1e-6, 1-1e-6)
    → submission_lgbm_pay_types_0.40929.csv
```

**No nested target encoding. No blend with a second model family.**

## Required files to re-run

Place the official competition folder next to this packet, or one directory above it:

```
inter-uni-datathon-stream-1-credit-card-clients/
  train.csv
  test.csv
  sample_submission.csv
final-submission-0.40929/
  BEST_0.40929.py
  ...
```

## Execution order

Python 3.13 (recorded run: 3.13.14).

```text
pip install -r requirements.txt
python BEST_0.40929.py
```

The script writes `outputs/submission_lgbm_pay_types_0.40929.csv`. A re-run may differ by tiny floats across machines. For review, send the CSV already in this folder, not a freshly regenerated one, unless you label it as a re-run.

## Key dependencies

- `lightgbm==4.7.0`
- `numpy==2.5.2`
- `pandas==3.0.5`
- `scikit-learn==1.9.0`

## Random seeds and settings

- Global seed: `42`
- Fold/repeat seeds: `1000 + r` for `r` in `0..9` (also `bagging_seed`, `feature_fraction_seed`, `data_random_seed`)
- `StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)`
- LightGBM: `objective=binary`, `metric=binary_logloss`, `learning_rate=0.1`, `num_leaves=31`, `deterministic=True`, `force_row_wise=True`
- `num_boost_round=2000`, early stopping `100` on the scored fold
- Test prediction = mean of **100** models
- Clip `[1e-6, 1-1e-6]`

## Which script generates the prediction file

`BEST_0.40929.py`

## Local vs public

| Set | Log loss |
|---|---|
| Local 10×10 averaged OOF (optimistic; ES uses the scored fold) | 0.42245 |
| Public leaderboard | **0.40929** |
