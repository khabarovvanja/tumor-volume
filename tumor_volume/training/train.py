import hydra
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from tumor_volume.data.dataset import PETPatchDataset
from tumor_volume.data.dvc_utils import download_data
from tumor_volume.models.unet_3d import UNet3D
from tumor_volume.models.losses import DiceLoss
from tumor_volume.utils.logging import setup_mlflow


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def train_entrypoint(cfg: DictConfig) -> None:
    run_training(cfg)


def run_training(cfg: DictConfig) -> None:
    # download_data(cfg.data)

    setup_mlflow(cfg.logging)

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
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    model.train()
    for epoch in range(cfg.training.epochs):
        for images, masks in tqdm(loader):
            images = images.to(cfg.training.device)
            masks = masks.to(cfg.training.device)

            logits = model(images)
            loss = dice_loss(logits, masks) + ce_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()