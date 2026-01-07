import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from tumor_volume.data.preprocess import TARGET_SPACING
from tumor_volume.inference.postprocess import logits_to_mask
from tumor_volume.inference.sliding_window import sliding_window_inference
from tumor_volume.inference.volume import compute_tumor_volume
from tumor_volume.models.unet_3d import UNet3D


def run_inference(cfg: DictConfig):

    input_path = Path(cfg.inference.input.volume)
    filename = input_path.stem
    output_dir = Path(cfg.inference.output.dir) / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = np.load(input_path)
    volume = (volume - volume.mean()) / (volume.std() + 1e-6)

    model = UNet3D(
        in_channels=1,
        num_classes=cfg.inference.num_classes,
    )

    ckpt = torch.load(cfg.inference.checkpoint.path, map_location=cfg.inference.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(cfg.inference.device)
    model.eval()

    logits = sliding_window_inference(
        volume=volume,
        model=model,
        patch_size=tuple(cfg.inference.sliding_window.patch_size),
        overlap=cfg.inference.sliding_window.overlap,
        batch_size=cfg.inference.sliding_window.batch_size,
        device=cfg.inference.device,
    )

    mask = logits_to_mask(torch.from_numpy(logits))
    tumor_volume_ml = compute_tumor_volume(mask, TARGET_SPACING)

    np.save(output_dir / "mask.npy", mask)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {"tumor_volume_ml": float(tumor_volume_ml)},
            f,
            indent=2,
        )

    print(f"Tumor volume: {tumor_volume_ml:.2f} ml")


if __name__ == "__main__":
    run_inference()
