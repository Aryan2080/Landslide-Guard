# Detection V2 (CPU) - Phase 11 - Model Selection

Selected: **v2_cpu_20260910_092239_loss_focal_dice** (validation Dice = 0.6003, IoU = 0.4288, PR-AUC = 0.5927).

Selection criterion (pre-declared): highest **validation Dice**; tie-break by **validation IoU** then **validation recall**.

Full experiment ranking:

```
                                experiment_id                 model              loss  val_dice  val_iou  val_precision  val_recall   val_f1  val_pr_auc                                                                                              checkpoint
       v2_cpu_20260910_092239_loss_focal_dice     unet_v2_batchnorm        focal_dice  0.600253 0.428829       0.573723    0.629355 0.600253    0.592696        D:\LANDSLIDE\LandslideGuard\checkpoints_v2\v2_cpu_20260910_092239\loss_focal_dice\best_model.pth
      v2_cpu_20260910_092239_arch_baseline_bn     unet_v2_batchnorm          bce_dice  0.582063 0.410500       0.629616    0.541188 0.582063    0.536513       D:\LANDSLIDE\LandslideGuard\checkpoints_v2\v2_cpu_20260910_092239\arch_baseline_bn\best_model.pth
      v2_cpu_20260910_092239_arch_residual_bn unet_v2_batchnorm_res          bce_dice  0.577628 0.406102       0.559548    0.596916 0.577628    0.583126       D:\LANDSLIDE\LandslideGuard\checkpoints_v2\v2_cpu_20260910_092239\arch_residual_bn\best_model.pth
         v2_cpu_20260910_092239_loss_bce_dice     unet_v2_batchnorm          bce_dice  0.559888 0.388781       0.490760    0.651684 0.559888    0.605820          D:\LANDSLIDE\LandslideGuard\checkpoints_v2\v2_cpu_20260910_092239\loss_bce_dice\best_model.pth
      v2_cpu_20260910_092239_arch_baseline_gn     unet_v2_groupnorm          bce_dice  0.546189 0.375694       0.489378    0.617921 0.546189    0.499631       D:\LANDSLIDE\LandslideGuard\checkpoints_v2\v2_cpu_20260910_092239\arch_baseline_gn\best_model.pth
v2_cpu_20260910_092239_loss_weighted_bce_dice     unet_v2_batchnorm weighted_bce_dice  0.478001 0.314061       0.354511    0.733508 0.478001    0.595746 D:\LANDSLIDE\LandslideGuard\checkpoints_v2\v2_cpu_20260910_092239\loss_weighted_bce_dice\best_model.pth
```
