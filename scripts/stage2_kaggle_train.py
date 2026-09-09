"""Stage-2 Kaggle GPU training driver (real run on Landslide4Sense).

Runs on Kaggle T4 x2 with the Tek Bahadur Kshetri dataset:
    /kaggle/input/datasets/tekbahadurkshetri/landslide4sense/{TrainData,ValidData,TestData}/{img,mask}/*.h5

Same experimental design as scripts/stage2_cpu_train.py but sized for GPU:
    - full-size U-Net (base_features=32, ~7.77M params)
    - batch_size=32, AMP on, num_workers=2
    - Baseline BCE+Dice for 25 epochs, class-imbalance sweep 5 arms x 8 epochs
    - Threshold sweep on validation only, one-shot test evaluation

All artifacts written under /kaggle/working/ (checkpoints/, outputs/).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_DIR = Path(os.environ.get("LANDSLIDE_REPO_DIR", "/kaggle/working/Landslide-Guard"))
# Default DATA_ROOT is Aryan's own upload (has masks for all three splits).
# Override with LANDSLIDE_DATA_ROOT env var when using a different dataset.
DATA_ROOT = Path(os.environ.get(
    "LANDSLIDE_DATA_ROOT",
    "/kaggle/input/landslide4sense-full/landslide4sense"))
WORK = Path(os.environ.get("LANDSLIDE_WORK_DIR", "/kaggle/working"))
CKPT_ROOT = WORK / "checkpoints"
OUT_ROOT = WORK / "outputs" / "detection"

sys.path.insert(0, str(REPO_DIR))

from src.detection.dataset       import Landslide4SenseDataset, build_dataloader
from src.detection.losses        import build_loss
from src.detection.model         import UNet, count_parameters
from src.detection.preprocessing import NormalizationStats
from src.detection.train         import Trainer, evaluate, fit
from src.detection.utils         import (
    EarlyStopping, ExperimentRow, ExperimentTracker, device_summary, set_seed)
from src.detection.validate      import compute_metrics, sweep_threshold


CFG = {
    "seed": 42,
    "base_features": 32,
    "batch_size": 32,
    "num_workers": 2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "baseline_epochs": 25,
    "imbalance_epochs": 8,
    "select_by": "val_dice",
    "early_stop_patience": 15,
    "threshold_grid": np.round(np.arange(0.10, 0.905, 0.05), 3).tolist(),
}

RUN_TAG = "kaggle_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve(split: str) -> tuple[Path, Path]:
    """Find the img/ + mask/ subdirectories for a split.

    Handles both flat layout  <root>/<split>/{img,mask}
    and archive-nested layout <root>/<split>/<split>/{img,mask}.
    """
    candidates = [
        (DATA_ROOT / split / "img",         DATA_ROOT / split / "mask"),
        (DATA_ROOT / split / split / "img", DATA_ROOT / split / split / "mask"),
    ]
    for img, mask in candidates:
        if img.is_dir() and mask.is_dir():
            return img, mask
    raise FileNotFoundError(
        f"Could not resolve {split}. Checked: "
        + ", ".join(str(i) + " / " + str(m) for i, m in candidates)
    )


def make_loaders(stats):
    ds_train = Landslide4SenseDataset(*resolve("TrainData"), stats=stats,
                                      split="train", augment_seed=CFG["seed"])
    ds_valid = Landslide4SenseDataset(*resolve("ValidData"), stats=stats, split="valid")
    ds_test  = Landslide4SenseDataset(*resolve("TestData"),  stats=stats, split="test")
    lt = build_dataloader(ds_train, batch_size=CFG["batch_size"],
                          num_workers=CFG["num_workers"], pin_memory=True)
    lv = build_dataloader(ds_valid, batch_size=CFG["batch_size"],
                          num_workers=CFG["num_workers"], pin_memory=True)
    le = build_dataloader(ds_test,  batch_size=CFG["batch_size"],
                          num_workers=CFG["num_workers"], pin_memory=True)
    return ds_train, ds_valid, ds_test, lt, lv, le


def train_arm(name, loss_spec, epochs, train_loader, valid_loader, device,
              tracker, pos_weight=None, early_stop=False):
    log(f"=== ARM {name}: loss={loss_spec} epochs={epochs} ===")
    set_seed(CFG["seed"])
    model = UNet(in_channels=14, out_channels=1,
                 base_features=CFG["base_features"]).to(device)
    loss_fn = build_loss(loss_spec, pos_weight=pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = Trainer(model=model, loss_fn=loss_fn, optimizer=opt,
                 device=device, scheduler=sched,
                 use_amp=True, grad_clip=CFG["grad_clip"],
                 threshold=0.5)
    ck_dir = CKPT_ROOT / RUN_TAG / name; ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / "best_model.pth"
    stopper = EarlyStopping(patience=CFG["early_stop_patience"], mode="max") if early_stop else None
    t0 = time.time()
    h = fit(tr, train_loader, valid_loader,
            epochs=epochs, checkpoint_path=ckpt,
            history_path=ck_dir / "history.json",
            select_by=CFG["select_by"], early_stopping=stopper,
            extra_state={"loss": name, "loss_params": loss_spec,
                         "seed": CFG["seed"]})
    sec = time.time() - t0
    best = max(h.epochs, key=lambda e: e.val_dice)
    log(f"    done: best epoch {best.epoch}  val_dice={best.val_dice:.4f} "
        f"val_iou={best.val_iou:.4f}  ({sec/60:.1f} min)")
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
        notes=("baseline (longer)" if name == "baseline_bce_dice"
               else "class-imbalance arm"),
    ))
    return ckpt, best.as_dict()


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    log(f"device: {device_summary(device)}")
    log(f"REPO_DIR : {REPO_DIR}")
    log(f"DATA_ROOT: {DATA_ROOT}")
    log(f"WORK     : {WORK}")
    log(f"config   : {json.dumps(CFG)}")
    assert DATA_ROOT.is_dir(), \
        f"DATA_ROOT does not exist: {DATA_ROOT}. Set LANDSLIDE_DATA_ROOT."

    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "experiments").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "training").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "validation").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "test").mkdir(parents=True, exist_ok=True)

    stats = NormalizationStats.from_json(
        REPO_DIR / "outputs/detection/data_verification/normalization_statistics.json")
    log("normalization stats: train-only, loaded (not recomputed)")

    ds_train, ds_valid, ds_test, train_loader, valid_loader, test_loader = make_loaders(stats)
    log(f"splits: train={len(ds_train)} valid={len(ds_valid)} test={len(ds_test)}")

    mask_dist = json.loads(
        (REPO_DIR / "outputs/detection/data_verification/mask_class_distribution.json").read_text())
    p = mask_dist["train"]["positive_pixel_ratio"]
    pw = torch.tensor([(1 - p) / max(p, 1e-8)], device=device)
    log(f"train positive-pixel ratio: {p:.5f}  pos_weight: {float(pw.item()):.2f}")

    tracker = ExperimentTracker(OUT_ROOT / "experiments" / f"{RUN_TAG}_results.csv")

    # 1. Baseline
    baseline_ckpt, baseline_best = train_arm(
        name="baseline_bce_dice",
        loss_spec={"name": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5},
        epochs=CFG["baseline_epochs"],
        train_loader=train_loader, valid_loader=valid_loader,
        device=device, tracker=tracker, early_stop=True)

    hist_data = json.loads((baseline_ckpt.parent / "history.json").read_text())
    df = pd.DataFrame(hist_data)
    df.to_csv(OUT_ROOT / "training" / f"{RUN_TAG}_baseline_history.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0,0].plot(df["epoch"], df["train_loss"], label="train")
    ax[0,0].plot(df["epoch"], df["val_loss"],   label="val")
    ax[0,0].set_title("Loss"); ax[0,0].legend(); ax[0,0].grid(alpha=0.3)
    ax[0,1].plot(df["epoch"], df["val_dice"], label="Dice")
    ax[0,1].plot(df["epoch"], df["val_iou"],  label="IoU")
    ax[0,1].set_title("Validation overlap"); ax[0,1].legend(); ax[0,1].grid(alpha=0.3)
    ax[1,0].plot(df["epoch"], df["val_precision"], label="P")
    ax[1,0].plot(df["epoch"], df["val_recall"],    label="R")
    ax[1,0].plot(df["epoch"], df["val_f1"],        label="F1")
    ax[1,0].set_title("P/R/F1"); ax[1,0].legend(); ax[1,0].grid(alpha=0.3)
    ax[1,1].plot(df["epoch"], df["val_pr_auc"], color="C3", label="PR-AUC")
    ax[1,1].set_title("PR-AUC"); ax[1,1].legend(); ax[1,1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "training" / f"{RUN_TAG}_baseline_curves.png", dpi=120)
    plt.close(fig)

    # 2. Class-imbalance sweep
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
    imb_df.to_csv(OUT_ROOT / "experiments" / f"{RUN_TAG}_class_imbalance_summary.csv", index=False)
    log("class-imbalance summary:\n" + imb_df[
        ["arm","val_dice","val_iou","val_precision","val_recall","val_f1","val_pr_auc"]
    ].to_string(index=False))

    # 3. Best model selection
    all_rows = pd.DataFrame(tracker.rows())
    for c in ("val_dice","val_iou","val_precision","val_recall","val_f1","val_pr_auc"):
        all_rows[c] = all_rows[c].astype(float)
    ranked = all_rows.sort_values(["val_dice","val_iou","val_recall"], ascending=False)
    top = ranked.iloc[0]
    best_ckpt = Path(top["checkpoint"])
    log(f"BEST: {top['experiment_id']}  val_dice={top['val_dice']:.4f} "
        f"val_iou={top['val_iou']:.4f}  ckpt={best_ckpt.name}")

    best_model = UNet(in_channels=14, out_channels=1,
                      base_features=CFG["base_features"]).to(device)
    best_model.load_state_dict(
        torch.load(best_ckpt, map_location=device, weights_only=False)["model"])

    # 4. Threshold sweep (validation only)
    grid = np.array(CFG["threshold_grid"], dtype=np.float64)
    sw = sweep_threshold(best_model, valid_loader, device, thresholds=grid)
    best_i = int(sw["dice"].argmax())
    LOCKED_THRESHOLD = float(sw["thresholds"][best_i])
    pd.DataFrame({k: sw[k] for k in
                  ("thresholds","dice","iou","precision","recall","f1","tp","fp","fn","tn")}
                 ).to_csv(OUT_ROOT / "experiments" / f"{RUN_TAG}_threshold_sweep.csv",
                          index=False)
    log(f"LOCKED THRESHOLD = {LOCKED_THRESHOLD}  "
        f"(val dice at that thr = {sw['dice'][best_i]:.4f})")

    # 5. Final model lock
    final_cfg = {
        "run_tag": RUN_TAG,
        "model": {"name": "unet", "in_channels": 14, "out_channels": 1,
                  "base_features": CFG["base_features"]},
        "loss": top["loss"], "loss_params": top["loss_params"],
        "optimizer": "adamw", "learning_rate": CFG["lr"],
        "weight_decay": CFG["weight_decay"], "scheduler": "cosine",
        "batch_size": CFG["batch_size"], "seed": CFG["seed"],
        "checkpoint": str(best_ckpt),
        "threshold": LOCKED_THRESHOLD,
        "training_positive_pixel_ratio": p,
        "selected_by": "val_dice (tiebreak val_iou, val_recall)",
    }
    import shutil, yaml
    export_dir = CKPT_ROOT
    export_ckpt = export_dir / "best_model.pth"
    shutil.copyfile(best_ckpt, export_ckpt)
    shutil.copyfile(REPO_DIR / "outputs/detection/data_verification/normalization_statistics.json",
                    export_dir / "normalization_statistics.json")
    (export_dir / "final_model_config.json").write_text(json.dumps(final_cfg, indent=2, default=str))
    (WORK / "detection_final.yaml").write_text(yaml.safe_dump(final_cfg, sort_keys=False))

    # 6. Test evaluation (one-shot)
    log("=== TEST EVALUATION (one-shot at locked threshold) ===")
    test_metrics = evaluate(best_model, test_loader, device, threshold=LOCKED_THRESHOLD)
    (OUT_ROOT / "test" / f"{RUN_TAG}_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2))
    log(f"TEST @ thr={LOCKED_THRESHOLD}:")
    for k in ("dice","iou","precision","recall","f1","specificity","accuracy","pr_auc"):
        log(f"  test_{k:10s} = {test_metrics[k]:.4f}")
    log(f"  tp/fp/fn/tn = {test_metrics['tp']}/{test_metrics['fp']}"
        f"/{test_metrics['fn']}/{test_metrics['tn']}")

    # 7. Final report
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
    (OUT_ROOT / "test" / f"{RUN_TAG}_stage2_report.json").write_text(
        json.dumps(report, indent=2))
    log("STAGE 2 KAGGLE RUN COMPLETE")


if __name__ == "__main__":
    main()
