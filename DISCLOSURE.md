# Required disclosure

Short provenance statement for the **0.40929** leaderboard file `submission_lgbm_pay_types_0.40929.csv`.

This packet is the **no target-encoding** alternative. It is not the 0.40900 TE-blend packet.

## External datasets

None. Only the competition files `train.csv`, `test.csv`, and `sample_submission.csv` were used.

## External code, notebooks, repositories, or public solutions

The training loop (LightGBM, learning rate 0.1, 31 leaves, 10×10 seed/split averaging with early stopping on the scored fold) was adapted from a teammate baseline that previously scored **0.40981**, then from the team’s **0.40967** extended-feature script (`improve_lgbm.py` / `improve_lgbm_ext.py`). No public Kaggle kernel or third-party repository was copied as the final pipeline.

The five new features (`n_revolve`, `n_duly`, `n_inactive`, `log_limit`, `remaining_credit`) were added on top of that internal baseline. They use only columns already in the competition data.

## Pretrained models

None. All 100 LightGBM models are trained from scratch on the competition training set.

## AutoML / external modelling systems

None. Training uses the `lightgbm` Python package (`lgb.train`) and `scikit-learn` `StratifiedKFold`.

## AI tools or coding agents

Yes. Cursor (coding agent) was used to inspect the competition brief, compare prior internal experiments, implement the five extra features, run training, and draft this submission packet. The submitted prediction file was produced by executing the LightGBM script on the competition data, not by an AI model emitting probabilities directly.

## Manual modification or post-processing of predictions

None. Test probabilities are the unweighted mean of 100 LightGBM models, clipped to `(1e-6, 1-1e-6)` only to keep log loss defined. No hand edits, no spreadsheet changes, no blending with other algorithms in the submitted file.

## Information beyond the competition-provided files

None for the fitted model. Organiser documentation (task description, data dictionary, evaluation as binary log loss) was used to choose the metric and to interpret `PAY_*` codes. No extra customer data were added.
