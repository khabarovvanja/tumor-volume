import fire
import hydra
from hydra.core.global_hydra import GlobalHydra

from tumor_volume.training.train import run_training
from tumor_volume.inference.infer import run_inference


def _compose_cfg(overrides: tuple[str, ...]):
    """
    Safe Hydra compose wrapper for Fire.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with hydra.initialize(config_path="../configs", version_base="1.3"):
        cfg = hydra.compose(
            config_name="config",
            overrides=list(overrides),
        )
    return cfg


def train(*overrides: str):
    """
    Example:
    python -m tumor_volume.commands train training.epochs=2 data.dataset.train_split=0.3
    """
    cfg = _compose_cfg(overrides)
    run_training(cfg)


def infer(*overrides: str):
    cfg = _compose_cfg(overrides)
    run_inference(cfg)


def main():
    fire.Fire(
        {
            "train": train,
            "infer": infer,
        }
    )


if __name__ == "__main__":
    main()