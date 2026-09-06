# Why we shipped the public-best file, not the local-best file

This is **not** about the friend’s 0.40981 model.  
Among *our* experiments, the **best local 10×10 OOF** and the **best public log loss** were **different files**.

We submitted **PAY-types (public 0.40929)**, not extra_trees and not the blend.

Lower log loss is better.

---

## The two “bests”

| Rank by… | Model | Local 10×10 averaged OOF | Public log loss | Mean trees |
|---|---|---|---|---|
| **Best local** | Blend (~50% extra_trees + ~46% PAY-types + ~4% bagging) | **0.42202** | 0.41037 | mixed |
| 2nd local | extra_trees (`extra_trees=True`, same 70 features as 0.40967) | **0.42242** | **0.41211** | ~62 |
| **Best public (shipped)** | PAY-types (five extra features, same loop as 0.40967) | 0.42245 | **0.40929** | ~45 |
| Previous record | Extended LGBM 0.40967 | 0.42287 | 0.40967 | ~43 |

Local gap extra_trees vs PAY-types: **0.00003** (noise).  
Public gap: **0.00282** the other way (extra_trees much worse).

The blend’s local win is almost entirely extra_trees in the mix. On the public board the blend landed **between** extra_trees (0.41211) and PAY-types (0.40929), at **0.41037**. extra_trees dragged PAY-types **down**.

---

## Why local OOF liked extra_trees / the blend

Our 10×10 OOF is **optimistic**. Early stopping uses the fold that is later scored as “OOF.” The model is allowed to peek at those labels when choosing how many trees to grow.

extra_trees changes **how splits are chosen** and grew **~62 trees** instead of **~45**. That extra flexibility fits the peeked fold more tightly, so OOF log loss falls. The hidden test never used those labels for early stopping, so the same change **hurt** public log loss (0.41211).

We had already seen the same pattern: heavy regularisation looked better on local holdouts and scored **~0.411** publicly.

PAY-types does **not** change the optimiser. It only adds five row-wise columns and keeps ~45 trees. That local comparison vs 0.40967 (0.42287 → 0.42245) is a fairer “same recipe” test — and public moved with it (0.40967 → 0.40929).

---

## Why we still choose PAY-types (public best)

**1. The contest metric is hidden-test log loss, not training OOF.**  
When a *training-procedure* change wins in-folder and loses on the board, the board is the evidence for that family of change.

**2. The local “win” is not comparable.**  
Ranking extra_trees above PAY-types because 0.42242 < 0.42245 treats two different recipes as one CV. 0.00003 is smaller than repeat-to-repeat noise (sd ~0.0003). Public 0.41211 vs 0.40929 is not noise.

**3. The blend failed a simple check.**  
If extra_trees were truly better, mixing it with PAY-types should not make public **worse** than PAY-types alone. It did (0.41037 vs 0.40929). That is extra_trees adding error, not diversity.

**4. PAY-types has an independent reason to exist.**  
PAY codes −2 / −1 / 0 are not a numeric risk ladder (revolving defaults less than paid-duly). Type counts + log limit + remaining credit. extra_trees has no new data story — only a different tree-growing rule that overfit ES.

**5. We did not pick PAY-types by mining the dashboard.**  
It was queued from (same loop as 0.40967) + (PAY-type logic) + (better local OOF vs 0.40967). Public 0.40929 confirmed it. extra_trees was dropped **after** 0.41211.

**6. Public-best ≠ “always chase public.”**  
We submitted few files. Further features after PAY-types were **worse locally** (0.42251 / 0.42286) and were **never** uploaded. We are not using the public slice as a second validation set.

---

## Rule

| Kind of change | Local better, public worse? | What we do |
|---|---|---|
| Same 10×10 LightGBM, small justified features (PAY-types vs 0.40967) | Did not happen — both improved | Ship the new file |
| Different training dynamics (extra_trees, heavy reg, blend with those) | Yes | **Do not ship the local winner; trust public** |

**Shipped file:** `submission_lgbm_pay_types_0.40929.csv`  
**Not shipped:** `submission_extra_trees.csv`, `outputs/submission.csv` (the blend)

---

## One-line summary

**Local-best (blend / extra_trees) exploited optimistic early-stopping OOF. Public-best (PAY-types) is the same proven loop plus five justified features; it beat 0.40967 both locally and publicly. We submit that, not the in-folder winner.**
