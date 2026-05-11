from omegaconf import DictConfig, ListConfig, OmegaConf

from tumor_volume.models.unet_3d import UNet3D


def _to_tuple(value):
    if isinstance(value, (ListConfig, DictConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, list):
        return tuple(value)
    return value


def build_model(cfg: DictConfig):
    architecture = getattr(cfg, "architecture", "unet3d")

    if architecture == "unet3d":
        return UNet3D(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
        )

    if architecture == "swin_unetr":
        from tumor_volume.models.swin_unetr import SwinUNETR3D

        return SwinUNETR3D(
            img_size=tuple(cfg.img_size),
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            feature_size=cfg.feature_size,
            depths=_to_tuple(cfg.depths),
            num_heads=_to_tuple(cfg.num_heads),
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            dropout_path_rate=cfg.dropout_path_rate,
            use_checkpoint=cfg.use_checkpoint,
            use_v2=cfg.use_v2,
        )

    raise ValueError(f"Unsupported model architecture: {architecture}")
