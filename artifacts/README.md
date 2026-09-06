# Intermediate artefacts (0.40929)

| File | What it is |
|---|---|
| `submission_lgbm_pay_types_0.40929.csv` | Copy of the uploaded 0.40929 test probabilities |
| `test_pay_types.csv` | Same test vector (`client_id`, `default_probability`) |
| `oof_pay_types.csv` | 10×10 averaged OOF |
| `meta_pay_types.json` | Local OOF summary |

Row-wise feature tables are not stored. `BEST_0.40929.py` rebuilds them from `train.csv` / `test.csv`.
