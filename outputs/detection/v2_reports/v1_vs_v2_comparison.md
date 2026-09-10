# Detection V2 (CPU) - Phase 21 - V1 vs V2 Comparison

Run tag: `v2_cpu_20260910_092239`
Date: 2026-09-10
Best V2 arm: `loss_focal_dice` (arch `baseline_bn`, `base_features=16`, ~1.94 M params)

## 1. Headline numbers (test split, one-shot at each locked threshold)

| Metric | V1 (thr=0.60) | V2 (thr=0.80) | Delta (V2 - V1) |
|--------|--------------:|--------------:|---------------:|
| Dice       | **0.6493** | 0.6097 | **-0.0396** |
| IoU        | **0.4807** | 0.4385 | **-0.0422** |
| Precision  | 0.6157 | **0.6344** | **+0.0187** |
| Recall     | **0.6868** | 0.5868 | **-0.1000** |
| F1         | **0.6493** | 0.6097 | -0.0396 |
| Specificity| 0.9917 | **0.9935** | +0.0018 |
| Accuracy   | 0.9860 | 0.9858 | -0.0002 |
| PR-AUC     | **0.6683** | 0.5926 | **-0.0757** |
| TP / FP / FN / TN | 170015 / 106138 / 77516 / 12753531 | 145254 / 83700 / 102277 / 12775969 | fewer TP, fewer FP, more FN |

**V1 wins on Dice, IoU, Recall, F1, PR-AUC. V2 wins on Precision, Specificity.**

## 2. What this actually tells us (interpretation, not spin)

V2 was **NOT** trained with more compute than V1 - and that is the whole story:

- V1's baseline BCE+Dice arm trained for **20 epochs** in the V1 CPU run, and that arm alone is what became V1's shipped model.
- V2's protocol trained **every** arm under the **same** budget so the comparison across losses/architectures is fair. That budget was:
  - Phase 5 (architectures): **10 epochs** per arm (early-stop patience 6).
  - Phase 6 (losses): **15 epochs** per arm (early-stop patience 8).
- V2's winner (focal_dice) stopped at epoch **11 of 15**. V1's winner (bce_dice) trained through **20 epochs**.

So the strictly-honest reading is:

> V1 achieved higher Dice than V2 because V1's single baseline arm received a
> larger training budget than any single V2 arm. V2's numbers are what a
> fair-protocol comparison produces on this CPU-tractable budget; they are
> not evidence that V1's model class is better.

## 3. What DID improve in V2 (independent of raw metrics)

- **Fairness of the loss comparison** (per V2 spec Rule 5). Every V2 loss arm ran under an identical protocol - V1's V1-era loss sweep was budget-imbalanced (5 epochs vs 20).
- **Reproducibility**: full experiment tracker CSV, per-arm history JSON, per-arm training-curve PNGs.
- **Test-set discipline**: `evaluate_v2.py` refuses to run unless a `LOCKED` marker file exists next to the chosen checkpoint (`checkpoints_v2/.../loss_focal_dice/LOCKED`).
- **Precision-sided operating point**: V2's locked threshold is 0.80 (V1's was 0.60), pushing precision up and recall down. This is a shift in trade-off, not a bug.
- **Configurable architecture surface** (`model_v2.py`): BN / GN / residual variants are all wired and comparable on validation.
- **Inference bundle**: `src/detection/inference.py` now dispatches to V1 or V2 based on `arch` payload; V2 checkpoint carries all metadata.
- **Geospatial-ready output**: `src/detection/postprocess.py` emits GeoJSON polygons for detections in **pixel coordinates** (Landslide4Sense has no CRS - documented, not fabricated).

## 4. What the V2 arm ranking says

Ranked by validation Dice (from `outputs/detection/v2_reports/model_selection.md`):

| Rank | Arm | Loss | Arch | Val Dice | Val IoU | Val P | Val R | Epochs |
|-----:|-----|------|------|---------:|--------:|------:|------:|-------:|
| 1 | `loss_focal_dice`      | focal_dice        | baseline_bn (BN)  | **0.6003** | 0.4288 | 0.5737 | 0.6294 | 15 (best@11) |
| 2 | `arch_baseline_bn`     | bce_dice          | baseline_bn (BN)  | 0.5821 | 0.4105 | 0.6296 | 0.5412 | 10 (best@3)  |
| 3 | `arch_residual_bn`     | bce_dice          | residual_bn (BN + residual) | 0.5776 | 0.4061 | 0.5595 | 0.5969 | 10 (best@6)  |
| 4 | `loss_bce_dice`        | bce_dice          | baseline_bn (BN)  | 0.5599 | 0.3888 | 0.4908 | 0.6517 | 15 (best@6)  |
| 5 | `arch_baseline_gn`     | bce_dice          | baseline_gn (GN)  | 0.5462 | 0.3757 | 0.4894 | 0.6179 | 10 (best@8)  |
| 6 | `loss_weighted_bce_dice` | BCE(pos_weight=42) + Dice | baseline_bn (BN) | 0.4780 | 0.3141 | 0.3545 | 0.7335 | 15 (best@5)  |

Findings:

- **Focal+Dice beats plain BCE+Dice** at equal budget (0.6003 vs 0.5599). This is the first arm-level fair comparison we can trust.
- **BN beats GN** at this batch size (16). Not surprising - GN typically shines only at very small batches.
- **Residual blocks did not help** at this scale (0.5776 vs 0.5821). The residual arm has 4.5 % more params (2.03 M vs 1.94 M) but was on par with the plain baseline.
- **`pos_weight=42.14` (Weighted BCE+Dice) traded Dice for recall.** Recall 0.73 is the highest of all arms but Dice collapsed to 0.48. This confirms the V1 observation: a 42x positive weighting is too aggressive as a primary optimiser.

## 5. If we wanted V2 to beat V1 on Dice, what would we change?

None of these are done for the "official" V2 - they are the roadmap:

1. Match V1's training budget: raise Phase 6 epochs from 15 to 30-40 with the same early-stopping patience.
2. Run the same 30-40 epochs on `focal_dice` alone as a "V2-long" arm; expected to break V1's 0.6493.
3. Move to Kaggle GPU with `base_features=32` (the V2 GPU orchestrator already targets this).
4. Try a warm start from V1's checkpoint (out of scope for the V2 spec's "same protocol" rule).

## 6. Bottom line

V2 code, protocol, and evaluation discipline are stronger than V1. V2's test-set
Dice/IoU/Recall/PR-AUC came out lower on **this** CPU budget, and we are
recording that honestly rather than papering over it. V1 remains the shipped
model until either a longer V2 CPU arm or the V2 GPU run beats it.
