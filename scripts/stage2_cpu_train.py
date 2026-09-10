"""Stage-2 CPU training driver (real run, no fabricated numbers).

Trains a smaller-but-real U-Net (base_features=16) on the full Landslide4Sense
training split for a realistic wall-clock on 12 CPU threads, runs a
class-imbalance loss sweep, sweeps the threshold on validation only, then
evaluates ONCE on the test split. All artifacts are written to disk under
outputs/detection/ and checkpoints/detection/.

Design decisions and their rationale are documented as they occur.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.dataset       import Landslide4SenseDataset, build_dataloader
from src.detection.losses        import build_loss
from src.detection.metrics       import (
    BinaryMetricAccumulator, PRAUCAccumulator, sweep_thresholds)
from src.detection.model         import UNet, count_parameters
from src.detection.preprocessing import NormalizationStats
from src.detection.train         import Trainer, evaluate, fit
from src.detection.utils         import (
    ExperimentRow, ExperimentTracker, EarlyStopping, device_summary, set_seed)
from src.detection.validate      import compute_metrics, sweep_threshold

# ---------------------------------------------------------------------------
# CPU-adapted configuration - deliberately smaller than the Kaggle notebook
# because ~2 min/epoch on this machine sets a hard time budget.
# ---------------------------------------------------------------------------
CFG = {
    "seed": 42,
    "base_features": 16,      # 1.94 M params vs 7.77 M at 32; ~2 min/epoch here
    "batch_size": 16,
    "num_workers": 0,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "baseline_epochs": 20,    # BCE+Dice arm - the "strong reproducible baseline"
    "imbalance_epochs": 5,    # each class-imbalance arm; seeded identically
    "threshold_grid": np.round(np.arange(0.10, 0.905, 0.05), 3).tolist(),
    "select_by": "val_dice",
    "early_stop_patience": 12,
}

DATA_ROOT = ROOT / "data" / "raw" / "landslide4sense"
CKPT_ROOT = ROOT / "checkpoints" / "detection"
OUT_ROOT  = ROOT / "outputs" / "detection"

TRAIN_IMG  = DATA_ROOT / "TrainData" / "TrainData" / "img"
TRAIN_MASK = DATA_ROOT / "TrainData" / "TrainData" / "mask"
VALID_IMG  = DATA_ROOT / "ValidData" / "ValidData" / "img"
VALID_MASK = DATA_ROOT / "ValidData" / "ValidData" / "mask"
TEST_IMG   = DATA_ROOT / "TestData"  / "TestData"  / "img"
TEST_MASK  = DATA_ROOT / "TestData"  / "TestData"  / "mask"

RUN_TAG = "cpu_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def make_loaders(stats):
    ds_train = Landslide4SenseDataset(TRAIN_IMG, TRAIN_MASK, stats,
                                      split="train", augment_seed=CFG["seed"])
    ds_valid = Landslide4SenseDataset(VALID_IMG, VALID_MASK, stats, split="valid")
    ds_test  = Landslide4SenseDataset(TEST_IMG,  TEST_MASK,  stats, split="test")
    lt = build_dataloader(ds_train, batch_size=CFG["batch_size"],
                          num_workers=CFG["num_workers"], pin_memory=False)
    lv = build_dataloader(ds_valid, batch_size=CFG["batch_size"],
                          num_workers=CFG["num_workers"], pin_memory=False)
    le = build_dataloader(ds_test,  batch_size=CFG["batch_size"],
                          num_workers=CFG["num_workers"], pin_memory=False)
    return ds_train, ds_valid, ds_test, lt, lv, le


def build_model() -> UNet:
    set_seed(CFG["seed"])
    return UNet(in_channels=14, out_channels=1,
                base_features=CFG["base_features"])


def train_arm(name: str, loss_spec: dict, epochs: int,
              train_loader, valid_loader, device,
              tracker: ExperimentTracker,
              pos_weight: torch.Tensor | None = None,
              early_stop: bool = False) -> tuple[Path, dict]:
    log(f"=== ARM {name}: loss={loss_spec} epochs={epochs} ===")
    set_seed(CFG["seed"])
    model = build_model().to(device)
    loss_fn = build_loss(loss_spec, pos_weight=pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = Trainer(model=model, loss_fn=loss_fn, optimizer=opt,
                 device=device, scheduler=sched,
                 use_amp=False, grad_clip=CFG["grad_clip"],
                 threshold=0.5)
    ckpt_dir = CKPT_ROOT / RUN_TAG / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "best_model.pth"
    hist = ckpt_dir / "history.json"

    stopper = (EarlyStopping(patience=CFG["early_stop_patience"], mode="max")
               if early_stop else None)
    t0 = time.time()
    h = fit(tr, train_loader, valid_loader,
            epochs=epochs, checkpoint_path=ckpt,
            history_path=hist, select_by=CFG["select_by"],
            early_stopping=stopper,
            extra_state={"loss": name, "loss_params": loss_spec,
                         "seed": CFG["seed"], "cfg": CFG})
    sec = time.time() - t0
    best = max(h.epochs, key=lambda e: e.val_dice)
    log(f"    done: best epoch {best.epoch}  "
        f"val_dice={best.val_dice:.4f}  val_iou={best.val_iou:.4f}  "
        f"({sec/60:.1f} min)")

    tracker.append(ExperimentRow(
        experiment_id=f"{RUN_TAG}_{name}", model="unet",
        in_channels=14, out_channels=1,
        base_features=CFG["base_features"],
        loss=name, loss_params=json.dumps(loss_spec),
        optimizer="adamw", learning_rate=CFG["lr"],
        weight_decay=CFG["weight_decay"], scheduler="cosine",
        batch_size=CFG["batch_size"],
        epochs_planned=epochs, epochs_actually_ran=len(h.epochs),
        best_epoch=best.epoch, seed=CFG["seed"],
        val_loss=best.val_loss, val_dice=best.val_dice, val_iou=best.val_iou,
        val_precision=best.val_precision, val_recall=best.val_recall,
        val_f1=best.val_f1, val_specificity=best.val_specificity,
        val_accuracy=best.val_accuracy, val_pr_auc=best.val_pr_auc,
        threshold=0.5, checkpoint=str(ckpt), seconds_total=sec,
        notes=("baseline (bce_dice, longer)" if name == "baseline_bce_dice"
               else "class-imbalance arm (shorter)"),
    ))
    return ckpt, best.as_dict()


def main() -> None:
    device = torch.device("cpu")
    torch.set_num_threads(12)
    log(f"device: {device_summary(device)}")
    log(f"config: {json.dumps(CFG)}")

    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "experiments").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "training").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "validation").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "test").mkdir(parents=True, exist_ok=True)

    stats = NormalizationStats.from_json(
        OUT_ROOT / "data_verification" / "normalization_statistics.json")
    log("normalization stats: train-only, loaded (not recomputed)")

    ds_train, ds_valid, ds_test, train_loader, valid_loader, test_loader = make_loaders(stats)
    log(f"splits: train={len(ds_train)}  valid={len(ds_valid)}  test={len(ds_test)}")

    # training positive-pixel ratio for pos_weight
    mask_dist = json.loads(
        (OUT_ROOT / "data_verification" / "mask_class_distribution.json").read_text())
    p = mask_dist["train"]["positive_pixel_ratio"]
    pw = torch.tensor([(1 - p) / max(p, 1e-8)], device=device)
    log(f"train positive-pixel ratio: {p:.5f}  pos_weight: {float(pw.item()):.2f}")

    tracker = ExperimentTracker(OUT_ROOT / "experiments" / f"{RUN_TAG}_results.csv")

    # ---- 1. BASELINE: bce_dice, longer training ----
    baseline_ckpt, baseline_best = train_arm(
        name="baseline_bce_dice",
        loss_spec={"name": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5},
        epochs=CFG["baseline_epochs"],
        train_loader=train_loader, valid_loader=valid_loader,
        device=device, tracker=tracker, early_stop=True)

    # save baseline curves as CSV + PNG
    hist_data = json.loads((baseline_ckpt.parent / "history.json").read_text())
    df = pd.DataFrame(hist_data)
    curves_csv = OUT_ROOT / "training" / f"{RUN_TAG}_baseline_history.csv"
    df.to_csv(curves_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0,0].plot(df["epoch"], df["train_loss"], label="train")
    ax[0,0].plot(df["epoch"], df["val_loss"],   label="val")
    ax[0,0].set_title("Loss (BCE+Dice)"); ax[0,0].legend(); ax[0,0].grid(alpha=0.3)
    ax[0,1].plot(df["epoch"], df["val_dice"], label="Dice")
    ax[0,1].plot(df["epoch"], df["val_iou"],  label="IoU")
    ax[0,1].set_title("Validation overlap"); ax[0,1].legend(); ax[0,1].grid(alpha=0.3)
    ax[1,0].plot(df["epoch"], df["val_precision"], label="P")
    ax[1,0].plot(df["epoch"], df["val_recall"],    label="R")
    ax[1,0].plot(df["epoch"], df["val_f1"],        label="F1")
    ax[1,0].set_title("Validation P/R/F1"); ax[1,0].legend(); ax[1,0].grid(alpha=0.3)
    ax[1,1].plot(df["epoch"], df["val_pr_auc"], color="C3", label="PR-AUC")
    ax[1,1].set_title("Validation PR-AUC"); ax[1,1].legend(); ax[1,1].grid(alpha=0.3)
    fig.tight_layout()
    curves_png = OUT_ROOT / "training" / f"{RUN_TAG}_baseline_curves.png"
    fig.savefig(curves_png, dpi=120); plt.close(fig)
    log(f"wrote {curves_csv.name}  {curves_png.name}")

    # ---- 2. CLASS-IMBALANCE SWEEP ----
    imbalance_arms = [
        ("imb_bce",         {"name": "bce"},         None),
        ("imb_bce_pw",      {"name": "bce"},         pw),
        ("imb_dice",        {"name": "dice"},        None),
        ("imb_focal_dice",  {"name": "focal_dice",
                             "gamma": 2.0, "alpha": 0.25,
                             "focal_weight": 0.5, "dice_weight": 0.5}, None),
    ]
    imb_records = [{"arm": "baseline_bce_dice", **baseline_best,
                    "checkpoint": str(baseline_ckpt)}]
    for arm_name, spec, pw_arg in imbalance_arms:
        ck, best = train_arm(
            name=arm_name, loss_spec=spec, epochs=CFG["imbalance_epochs"],
            train_loader=train_loader, valid_loader=valid_loader,
            device=device, tracker=tracker, pos_weight=pw_arg)
        imb_records.append({"arm": arm_name, **best, "checkpoint": str(ck)})

    imb_df = pd.DataFrame(imb_records)
    imb_csv = OUT_ROOT / "experiments" / f"{RUN_TAG}_class_imbalance_summary.csv"
    imb_df.to_csv(imb_csv, index=False)
    log(f"class-imbalance summary:\n{imb_df[['arm','val_dice','val_iou','val_precision','val_recall','val_f1','val_pr_auc']].to_string(index=False)}")

    # ---- 3. BEST MODEL SELECTION ----
    all_rows = pd.DataFrame(tracker.rows())
    for c in ("val_dice","val_iou","val_precision","val_recall","val_f1","val_pr_auc"):
        all_rows[c] = all_rows[c].astype(float)
    ranked = all_rows.sort_values(["val_dice","val_iou","val_recall"], ascending=False)
    top = ranked.iloc[0]
    best_ckpt = Path(top["checkpoint"])
    log(f"BEST: {top['experiment_id']}  "
        f"val_dice={top['val_dice']:.4f}  val_iou={top['val_iou']:.4f}  "
        f"ckpt={best_ckpt.name}")

    best_model = build_model().to(device)
    best_model.load_state_dict(
        torch.load(best_ckpt, map_location=device, weights_only=False)["model"])

    # ---- 4. THRESHOLD SWEEP (validation only) ----
    log("threshold sweep on validation ...")
    grid = np.array(CFG["threshold_grid"], dtype=np.float64)
    sw = sweep_threshold(best_model, valid_loader, device, thresholds=grid)
    best_i = int(sw["dice"].argmax())
    LOCKED_THRESHOLD = float(sw["thresholds"][best_i])
    thr_csv = OUT_ROOT / "experiments" / f"{RUN_TAG}_threshold_sweep_validation.csv"
    pd.DataFrame({k: sw[k] for k in ("thresholds","dice","iou","precision",
                                     "recall","f1","tp","fp","fn","tn")}).to_csv(thr_csv, index=False)
    log(f"LOCKED THRESHOLD = {LOCKED_THRESHOLD}  "
        f"(val dice at that thr = {sw['dice'][best_i]:.4f})")

    # ---- 5. FINAL MODEL LOCK ----
    final_cfg = {
        "run_tag": RUN_TAG,
        "model": {"name": "unet", "in_channels": 14, "out_channels": 1,
                  "base_features": CFG["base_features"]},
        "loss": top["loss"], "loss_params": top["loss_params"],
        "optimizer": "adamw", "learning_rate": CFG["lr"],
        "weight_decay": CFG["weight_decay"], "scheduler": "cosine",
        "batch_size": CFG["batch_size"], "seed": CFG["seed"],
        "normalization_stats_file":
            "outputs/detection/data_verification/normalization_statistics.json",
        "augmentation": {"train_only": True,
                         "methods": ["hflip", "vflip", "rot90"]},
        "checkpoint": str(best_ckpt),
        "threshold": LOCKED_THRESHOLD,
        "training_positive_pixel_ratio": p,
        "selected_by": "val_dice (tiebreak val_iou, val_recall)",
        "notes": "CPU-adapted Stage-2 run; smaller U-Net (base_features=16)",
    }
    (ROOT / "configs" / "detection_final.yaml").parent.mkdir(exist_ok=True)
    import yaml
    (ROOT / "configs" / "detection_final.yaml").write_text(
        yaml.safe_dump(final_cfg, sort_keys=False))
    log(f"wrote configs/detection_final.yaml")

    # export bundle
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    import shutil
    export_ckpt = CKPT_ROOT / "best_model.pth"
    shutil.copyfile(best_ckpt, export_ckpt)
    shutil.copyfile(OUT_ROOT / "data_verification" / "normalization_statistics.json",
                    CKPT_ROOT / "normalization_statistics.json")
    (CKPT_ROOT / "final_model_config.json").write_text(
        json.dumps(final_cfg, indent=2, default=str))
    log(f"exported best model to {export_ckpt}")

    # ---- 6. TEST EVALUATION (one-shot) ----
    log("=== TEST EVALUATION (one-shot at locked threshold) ===")
    test_metrics = evaluate(best_model, test_loader, device,
                            threshold=LOCKED_THRESHOLD)
    test_out = OUT_ROOT / "test" / f"{RUN_TAG}_test_metrics.json"
    test_out.write_text(json.dumps(test_metrics, indent=2))
    log(f"TEST @ thr={LOCKED_THRESHOLD}:")
    for k in ("dice","iou","precision","recall","f1","specificity","accuracy","pr_auc"):
        log(f"  test_{k:10s} = {test_metrics[k]:.4f}")
    log(f"  tp/fp/fn/tn = {test_metrics['tp']}/{test_metrics['fp']}"
        f"/{test_metrics['fn']}/{test_metrics['tn']}")
    log(f"wrote {test_out.name}")

    # ---- 7. FINAL REPORT ----
    val_final = compute_metrics(best_model, valid_loader, device,
                                threshold=LOCKED_THRESHOLD)
    report = {
        "run_tag": RUN_TAG,
        "device": device_summary(device),
        "config": CFG,
        "best_experiment": top["experiment_id"],
        "best_checkpoint": str(export_ckpt),
        "locked_threshold": LOCKED_THRESHOLD,
        "training_positive_pixel_ratio": p,
        "val_metrics_at_locked_threshold": {
            k: val_final[k] for k in
            ("dice","iou","precision","recall","f1","specificity","accuracy","pr_auc")},
        "test_metrics_at_locked_threshold": {
            k: test_metrics[k] for k in
            ("dice","iou","precision","recall","f1","specificity","accuracy","pr_auc",
             "tp","fp","fn","tn","threshold")},
    }
    report_out = OUT_ROOT / "test" / f"{RUN_TAG}_stage2_report.json"
    report_out.write_text(json.dumps(report, indent=2))
    log(f"wrote {report_out.name}")
    log("STAGE 2 CPU RUN COMPLETE")


if __name__ == "__main__":
    main()
