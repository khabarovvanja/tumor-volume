import numpy as np
import hydra
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm
import tempfile
from omegaconf import OmegaConf
import mlflow
from pathlib import Path
from torch.utils.data import random_split

from tumor_volume.data.dataset import PETPatchDataset
from tumor_volume.data.dvc_utils import download_data
from tumor_volume.models.unet_3d import UNet3D
from tumor_volume.models.losses import DiceLoss
from tumor_volume.utils.logging import setup_mlflow
from tumor_volume.data.preprocess import nifti2npy

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def train_entrypoint(cfg: DictConfig) -> None:
    run_training(cfg)

def run_training(cfg: DictConfig) -> None:
    download_data(cfg.data)

    img_dir = cfg.data.root_dir + "/raw/images"
    mask_dir = cfg.data.root_dir + "/raw/masks"
    out_img_dir = cfg.data.root_dir + "/processed/images"
    out_mask_dir = cfg.data.root_dir + "/processed/masks"

    # создаём npy файлы только если их нет
    if not Path(out_img_dir).exists() or not any(Path(out_img_dir).iterdir()):
        print("Converting NIfTI to NumPy...")
        nifti2npy(img_dir, mask_dir, out_img_dir, out_mask_dir)
        print("Done.")


    setup_mlflow(cfg.logging)
    with tempfile.TemporaryDirectory() as tmpdir: # save config to mlflow
        cfg_path = Path(tmpdir) / "config.yaml"
        OmegaConf.save(cfg, cfg_path)
        mlflow.log_artifact(str(cfg_path))

    model = UNet3D(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
    ).to(cfg.training.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr
    )

    dice_loss = DiceLoss()
    ce_loss = torch.nn.CrossEntropyLoss(
        # weight=torch.tensor(cfg.training.class_weights).to(cfg.training.device)
    )

    dataset = PETPatchDataset(**cfg.data.dataset)
    train_size = int(len(dataset) * cfg.data.dataset.train_split)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(cfg.data.dataset.seed)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    best_val_loss = float("inf")

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    global_step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}"):
            images = images.to(cfg.training.device)
            masks = masks.to(cfg.training.device)

            logits = model(images)
            dice = dice_loss(logits, masks)
            ce = ce_loss(logits, masks)
            loss = dice + ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add to mlflow online loss graph
            mlflow.log_metric("train/loss_step", loss.item(), step=global_step)
            mlflow.log_metric("train/dice_loss_step", dice.item(), step=global_step)
            mlflow.log_metric("train/ce_loss_step", ce.item(), step=global_step)

            global_step += 1

        val_metrics = run_validation(
            model,
            val_loader,
            dice_loss,
            ce_loss,
            cfg.training.device,
        )

        current_val_loss = val_metrics["loss"]

        if cfg.training.save_best and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss

            checkpoint_path = checkpoint_dir / "best_model.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                checkpoint_path,
            )

            mlflow.log_artifact(str(checkpoint_path))

            print(
                f"✅ New best model saved "
                f"(epoch={epoch}, val_loss={best_val_loss:.4f})"
            )

        mlflow.log_metric("val/loss_epoch", val_metrics["loss"], step=epoch)
        mlflow.log_metric("val/dice_loss_epoch", val_metrics["dice_loss"], step=epoch)
        mlflow.log_metric("val/ce_loss_epoch", val_metrics["ce_loss"], step=epoch)

@torch.no_grad()
def run_validation(model, loader, dice_loss, ce_loss, device):
    model.eval()

    losses = []
    dice_losses = []
    ce_losses = []

    for images, masks in tqdm(loader, desc="Validation"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        dice = dice_loss(logits, masks)
        ce = ce_loss(logits, masks)
        loss = dice + ce

        losses.append(loss.item())
        dice_losses.append(dice.item())
        ce_losses.append(ce.item())

    model.train()

    return {
        "loss": float(np.mean(losses)),
        "dice_loss": float(np.mean(dice_losses)),
        "ce_loss": float(np.mean(ce_losses)),
    }