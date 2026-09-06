# Final model information — 0.40929 (no TE)

**Submitted file:** `submission_lgbm_pay_types_0.40929.csv`  
**Public log loss:** 0.40929  
**Type:** single LightGBM family, 100-model seed/split average. **No target encoding. No second-model blend.**

## Ensemble

| Component | Weight | Models averaged | Features |
|---|---|---|---|
| PAY-types LightGBM (`BEST_0.40929.py`) | **1.00** | 100 (10 seeds × 10 folds) | 75 row-wise columns |

Seeds: `1000` … `1009`. Each seed also sets `bagging_seed`, `feature_fraction_seed`, and `data_random_seed`.

## LightGBM hyperparameters

| Setting | Value |
|---|---|
| Package | `lightgbm==4.7.0` |
| `objective` | `binary` |
| `metric` | `binary_logloss` |
| `learning_rate` | 0.1 |
| `num_leaves` | 31 |
| `deterministic` | True |
| `force_row_wise` | True |
| `num_boost_round` | 2000 |
| Early stopping | 100 rounds on the scored fold |
| Probability clip | `[1e-6, 1-1e-6]` |

The five columns that distinguish this from the 0.40967 parent: `n_revolve`, `n_duly`, `n_inactive`, `log_limit`, `remaining_credit`.

## Not used in this file

Nested PAY target encoding, CatBoost, extra_trees, bagging as a different booster, Platt/isotonic calibration, blend weights, manual row edits, external data, pretrained checkpoints.

Saved booster files are not attached. Saved OOF/test probabilities are in `artifacts/`.
