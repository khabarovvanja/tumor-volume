import tempfile
from datetime import datetime
from pathlib import Path
import json

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from tumor_volume.data.dataset import PETCTPatchDataset, list_case_triplets, load_volume
from tumor_volume.data.dvc_utils import download_data
from tumor_volume.data.preprocess import TARGET_SPACING, nifti2npy
from tumor_volume.inference.postprocess import logits_to_mask
from tumor_volume.inference.sliding_window import sliding_window_inference
from tumor_volume.models.losses import DiceLoss
from tumor_volume.models.metrics import (
    absolute_volume_difference,
    dice_coefficient,
    hd95,
    relative_absolute_volume_difference,
)
from tumor_volume.models.unet_3d import UNet3D
from tumor_volume.utils.logging import setup_mlflow


def run_training(cfg: DictConfig) -> None:
    _prepare_data(cfg)
    all_cases = list_case_triplets(cfg.data.pet_dir, cfg.data.ct_dir, cfg.data.masks_dir)
    split_plan = build_split_plan(all_cases, cfg)

    fold_summaries = []
    for split in split_plan:
        run_name = cfg.logging.run_name
        if split["name"] != "holdout":
            run_name = f"{run_name}_{split['name']}"

        setup_mlflow(cfg.logging, run_name=run_name)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg_path = Path(tmpdir) / "config.yaml"
                split_manifest_path = Path(tmpdir) / "split_manifest.json"
                OmegaConf.save(cfg, cfg_path)
                _save_split_manifest(split, split_manifest_path)
                mlflow.log_artifact(str(cfg_path))
                mlflow.log_artifact(str(split_manifest_path))

            mlflow.set_tags(
                {
                    "split_strategy": cfg.training.split_strategy,
                    "fold_name": split["name"],
                    "train_cases": len(split["train"]),
                    "val_cases": len(split["val"]),
                    "test_cases": len(split["test"]),
                }
            )

            summary = _train_single_split(cfg, split)
            fold_summaries.append(summary)
        finally:
            mlflow.end_run()

    if len(fold_summaries) > 1:
        mean_dice = float(np.mean([summary["test_dice"] for summary in fold_summaries]))
        mean_hd95 = float(np.mean([summary["test_hd95"] for summary in fold_summaries]))
        mean_avd = float(np.mean([summary["test_avd_ml"] for summary in fold_summaries]))
        mean_ravd = float(np.mean([summary["test_ravd_percent"] for summary in fold_summaries]))
        print(
            f"K-fold summary: mean test Dice={mean_dice:.4f}, "
            f"mean test HD95={mean_hd95:.4f}, "
            f"mean test AVD={mean_avd:.4f} ml, "
            f"mean test rAVD={mean_ravd:.2f}%"
        )


def _prepare_data(cfg: DictConfig) -> None:
    download_data(cfg.data)

    pet_dir = f"{cfg.data.root_dir}/raw/pet"
    ct_dir = f"{cfg.data.root_dir}/raw/ct"
    mask_dir = f"{cfg.data.root_dir}/raw/masks"
    out_pet_dir = f"{cfg.data.root_dir}/processed/pet"
    out_ct_dir = f"{cfg.data.root_dir}/processed/ct"
    out_mask_dir = f"{cfg.data.root_dir}/processed/masks"

    print("Converting multimodal NIfTI to NumPy (incremental)...")
    nifti2npy(
        pet_dir=pet_dir,
        ct_dir=ct_dir,
        mask_dir=mask_dir,
        out_pet_dir=out_pet_dir,
        out_ct_dir=out_ct_dir,
        out_mask_dir=out_mask_dir,
        force=False,
    )


def _case_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".npy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _save_split_manifest(split: dict[str, object], output_path: Path) -> None:
    manifest = {"split_name": split["name"]}
    for key in ("train", "val", "test"):
        manifest[key] = [_case_id_from_path(case[0]) for case in split[key]]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)


def build_split_plan(
    all_cases: list[tuple[Path, Path, Path]],
    cfg: DictConfig,
) -> list[dict[str, object]]:
    if cfg.training.split_strategy == "kfold":
        return _build_kfold_plan(all_cases, cfg)
    return [_build_holdout_split(all_cases, cfg)]


def _build_holdout_split(
    all_cases: list[tuple[Path, Path, Path]],
    cfg: DictConfig,
) -> dict[str, object]:
    train_ratio = float(cfg.training.train_split)
    val_ratio = float(cfg.training.val_split)
    test_ratio = float(cfg.training.test_split)
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("train_split + val_split + test_split must sum to 1.0")

    shuffled = list(all_cases)
    rng = np.random.default_rng(cfg.data.seed)
    rng.shuffle(shuffled)

    num_cases = len(shuffled)
    train_count = max(1, int(round(num_cases * train_ratio)))
    val_count = max(1, int(round(num_cases * val_ratio)))
    test_count = num_cases - train_count - val_count

    if test_count < 1:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        else:
            val_count -= 1

    train_cases = shuffled[:train_count]
    val_cases = shuffled[train_count : train_count + val_count]
    test_cases = shuffled[train_count + val_count :]

    if not train_cases or not val_cases or not test_cases:
        raise ValueError("Patient-level split must produce non-empty train/val/test sets.")

    return {
        "name": "holdout",
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
    }


def _build_kfold_plan(
    all_cases: list[tuple[Path, Path, Path]],
    cfg: DictConfig,
) -> list[dict[str, object]]:
    num_folds = int(cfg.training.num_folds)
    if num_folds < 3:
        raise ValueError("num_folds must be at least 3 to build train/val/test folds.")
    if len(all_cases) < num_folds:
        raise ValueError("Number of cases must be >= num_folds.")

    shuffled = list(all_cases)
    rng = np.random.default_rng(cfg.data.seed)
    rng.shuffle(shuffled)
    fold_sizes = [len(shuffled) // num_folds] * num_folds
    for idx in range(len(shuffled) % num_folds):
        fold_sizes[idx] += 1

    folds = []
    start = 0
    for fold_size in fold_sizes:
        folds.append(shuffled[start : start + fold_size])
        start += fold_size

    target_fold_indices = range(num_folds)
    if not cfg.training.run_all_folds:
        target_fold_indices = [int(cfg.training.fold_index) % num_folds]

    split_plan = []
    for test_idx in target_fold_indices:
        val_idx = (test_idx + 1) % num_folds
        train_cases = [
            case
            for fold_idx, fold_cases in enumerate(folds)
            if fold_idx not in {test_idx, val_idx}
            for case in fold_cases
        ]
        val_cases = folds[val_idx]
        test_cases = folds[test_idx]

        split_plan.append(
            {
                "name": f"fold_{test_idx}",
                "train": train_cases,
                "val": val_cases,
                "test": test_cases,
            }
        )

    return split_plan


def _train_single_split(cfg: DictConfig, split: dict[str, object]) -> dict[str, float]:
    model = UNet3D(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
    ).to(cfg.training.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    dice_loss = DiceLoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    train_dataset = PETCTPatchDataset(
        patch_size=tuple(cfg.data.patch_size),
        samples_per_volume=cfg.data.samples_per_volume,
        cases=split["train"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    best_metric = None
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = (
        Path(cfg.training.checkpoint_dir)
        / f"exp_{cfg.logging.run_name}_{split['name']}_{session_id}"
    )
    experiment_folder.mkdir(exist_ok=True, parents=True)

    stable_checkpoint_path = Path(cfg.training.checkpoint_dir) / "best_model.pth"
    if cfg.training.split_strategy == "kfold":
        stable_checkpoint_path = (
            Path(cfg.training.checkpoint_dir) / f"best_model_{split['name']}.pth"
        )

    best_checkpoint_path = experiment_folder / "best_model.pth"

    for epoch in range(cfg.training.epochs):
        train_metrics = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            dice_loss,
            ce_loss,
            cfg.training.device,
            epoch,
            cfg.training.epochs,
        )
        mlflow.log_metric("train/loss_epoch", train_metrics["loss"], step=epoch)
        mlflow.log_metric("train/dice_loss_epoch", train_metrics["dice_loss"], step=epoch)
        mlflow.log_metric("train/ce_loss_epoch", train_metrics["ce_loss"], step=epoch)

        val_metrics = evaluate_cases(model, split["val"], cfg, dice_loss, ce_loss, "Validation")
        mlflow.log_metric("val/loss_epoch", val_metrics["loss"], step=epoch)
        mlflow.log_metric("val/dice_epoch", val_metrics["dice"], step=epoch)
        mlflow.log_metric("val/hd95_epoch", val_metrics["hd95"], step=epoch)
        mlflow.log_metric("val/avd_ml_epoch", val_metrics["avd_ml"], step=epoch)
        mlflow.log_metric("val/ravd_percent_epoch", val_metrics["ravd_percent"], step=epoch)
        mlflow.log_metric("val/dice_loss_epoch", val_metrics["dice_loss"], step=epoch)
        mlflow.log_metric("val/ce_loss_epoch", val_metrics["ce_loss"], step=epoch)

        current_metric = _select_monitor_metric(val_metrics, cfg.training.monitor_metric)
        if cfg.training.save_best and _is_better(current_metric, best_metric, cfg.training.monitor_metric):
            best_metric = current_metric
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "monitor_metric": cfg.training.monitor_metric,
                "monitor_value": current_metric,
                "split_name": split["name"],
            }
            torch.save(checkpoint, best_checkpoint_path)
            torch.save(checkpoint, stable_checkpoint_path)
            mlflow.log_artifact(str(best_checkpoint_path))
            print(
                f"New best model saved (epoch={epoch}, "
                f"{cfg.training.monitor_metric}={current_metric:.4f})"
            )

    if not best_checkpoint_path.exists():
        checkpoint = {
            "epoch": cfg.training.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "monitor_metric": cfg.training.monitor_metric,
            "monitor_value": best_metric,
            "split_name": split["name"],
        }
        torch.save(checkpoint, best_checkpoint_path)
        torch.save(checkpoint, stable_checkpoint_path)

    checkpoint = torch.load(best_checkpoint_path, map_location=cfg.training.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.training.device)
    model.eval()

    test_metrics = evaluate_cases(model, split["test"], cfg, dice_loss, ce_loss, "Test")
    mlflow.log_metrics(
        {
            "test/loss": test_metrics["loss"],
            "test/dice": test_metrics["dice"],
            "test/hd95": test_metrics["hd95"],
            "test/avd_ml": test_metrics["avd_ml"],
            "test/ravd_percent": test_metrics["ravd_percent"],
            "test/dice_loss": test_metrics["dice_loss"],
            "test/ce_loss": test_metrics["ce_loss"],
        }
    )

    print(
        f"{split['name']} summary: "
        f"test Dice={test_metrics['dice']:.4f}, "
        f"test HD95={test_metrics['hd95']:.4f}, "
        f"test AVD={test_metrics['avd_ml']:.4f} ml, "
        f"test rAVD={test_metrics['ravd_percent']:.2f}%"
    )

    return {
        "test_dice": test_metrics["dice"],
        "test_hd95": test_metrics["hd95"],
        "test_avd_ml": test_metrics["avd_ml"],
        "test_ravd_percent": test_metrics["ravd_percent"],
    }


def _run_train_epoch(
    model,
    train_loader,
    optimizer,
    dice_loss,
    ce_loss,
    device,
    epoch,
    total_epochs,
):
    model.train()

    losses = []
    dice_losses = []
    ce_losses = []

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        dice = dice_loss(logits, masks)
        ce = ce_loss(logits, masks)
        loss = dice + ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        dice_losses.append(dice.item())
        ce_losses.append(ce.item())

    return {
        "loss": float(np.mean(losses)),
        "dice_loss": float(np.mean(dice_losses)),
        "ce_loss": float(np.mean(ce_losses)),
    }


@torch.no_grad()
def evaluate_cases(
    model,
    cases: list[tuple[Path, Path, Path]],
    cfg: DictConfig,
    dice_loss,
    ce_loss,
    desc: str,
):
    model.eval()

    losses = []
    dice_losses = []
    ce_losses = []
    dice_scores = []
    hd95_scores = []
    avd_scores = []
    ravd_scores = []

    for pet_path, ct_path, mask_path in tqdm(cases, desc=desc):
        pet = load_volume(pet_path).astype(np.float32)
        ct = load_volume(ct_path).astype(np.float32)
        mask = load_volume(mask_path).astype(np.int64)
        pet = (pet - pet.mean()) / (pet.std() + 1e-6)
        ct = (ct - ct.mean()) / (ct.std() + 1e-6)
        volume = np.stack([pet, ct], axis=0)

        logits = sliding_window_inference(
            volume=volume,
            model=model,
            patch_size=tuple(cfg.inference.sliding_window.patch_size),
            overlap=cfg.inference.sliding_window.overlap,
            batch_size=cfg.inference.sliding_window.batch_size,
            device=cfg.training.device,
        )

        logits_tensor = torch.from_numpy(logits).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).long()

        dice = dice_loss(logits_tensor, mask_tensor)
        ce = ce_loss(logits_tensor, mask_tensor)
        loss = dice + ce

        prediction = logits_to_mask(logits_tensor)
        dice_scores.append(dice_coefficient(prediction, mask))
        hd95_value = hd95(prediction, mask, TARGET_SPACING)
        if not np.isfinite(hd95_value):
            hd95_value = float(np.linalg.norm(np.multiply(mask.shape, TARGET_SPACING)))
        hd95_scores.append(hd95_value)
        avd_scores.append(
            absolute_volume_difference(prediction, mask, TARGET_SPACING)
        )
        ravd_scores.append(
            relative_absolute_volume_difference(prediction, mask)
        )
        losses.append(loss.item())
        dice_losses.append(dice.item())
        ce_losses.append(ce.item())

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dice_scores)),
        "hd95": float(np.mean(hd95_scores)),
        "avd_ml": float(np.mean(avd_scores)),
        "ravd_percent": float(np.mean(ravd_scores)),
        "dice_loss": float(np.mean(dice_losses)),
        "ce_loss": float(np.mean(ce_losses)),
    }


def _select_monitor_metric(metrics: dict[str, float], monitor_metric: str) -> float:
    key_map = {
        "val/loss_epoch": "loss",
        "val/dice_epoch": "dice",
        "val/hd95_epoch": "hd95",
        "val/avd_ml_epoch": "avd_ml",
        "val/ravd_percent_epoch": "ravd_percent",
    }
    if monitor_metric not in key_map:
        raise ValueError(f"Unsupported monitor metric: {monitor_metric}")
    return float(metrics[key_map[monitor_metric]])


def _is_better(current: float, best: float | None, monitor_metric: str) -> bool:
    if best is None:
        return True
    if (
        "loss" in monitor_metric
        or "hd95" in monitor_metric
        or "avd" in monitor_metric
        or "ravd" in monitor_metric
    ):
        return current < best
    return current > best
