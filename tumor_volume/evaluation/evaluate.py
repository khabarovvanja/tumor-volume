import json
from pathlib import Path

import torch
from omegaconf import DictConfig

from tumor_volume.data.dataset import list_case_triplets
from tumor_volume.inference.infer import _resolve_checkpoint_path
from tumor_volume.models.losses import DiceLoss
from tumor_volume.models.unet_3d import UNet3D
from tumor_volume.training.train import evaluate_cases


def run_evaluation(cfg: DictConfig) -> None:
    if cfg.evaluation.override_overlap is not None:
        cfg.inference.sliding_window.overlap = cfg.evaluation.override_overlap
    if cfg.evaluation.override_batch_size is not None:
        cfg.inference.sliding_window.batch_size = cfg.evaluation.override_batch_size

    cases = list_case_triplets(
        cfg.data.pet_dir,
        cfg.data.ct_dir,
        cfg.data.masks_dir,
    )
    if not cases:
        raise RuntimeError("No PET/CT/mask cases found for evaluation.")

    checkpoint_path = _resolve_checkpoint_path(cfg.inference.checkpoint.path)

    model = UNet3D(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
    ).to(cfg.training.device)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.training.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = evaluate_cases(
        model=model,
        cases=cases,
        cfg=cfg,
        dice_loss=DiceLoss(),
        ce_loss=torch.nn.CrossEntropyLoss(),
        desc=cfg.evaluation.progress_desc,
    )

    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = checkpoint_path.parent.name
    output_path = output_dir / f"{cfg.evaluation.run_name}_{checkpoint_name}.json"

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "pet_dir": str(cfg.data.pet_dir),
        "ct_dir": str(cfg.data.ct_dir),
        "masks_dir": str(cfg.data.masks_dir),
        "num_cases": len(cases),
        "compute_hd95": bool(cfg.evaluation.compute_hd95),
        "effective_overlap": float(cfg.inference.sliding_window.overlap),
        "effective_batch_size": int(cfg.inference.sliding_window.batch_size),
        "metrics": {
            "loss": float(metrics["loss"]),
            "dice": float(metrics["dice"]),
            "hd95": float(metrics["hd95"]),
            "avd_ml": float(metrics["avd_ml"]),
            "ravd_percent": float(metrics["ravd_percent"]),
            "dice_loss": float(metrics["dice_loss"]),
            "ce_loss": float(metrics["ce_loss"]),
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(
        f"Evaluation completed on {len(cases)} cases. "
        f"Dice={metrics['dice']:.4f}, HD95={metrics['hd95']:.4f}, "
        f"AVD={metrics['avd_ml']:.4f} ml, rAVD={metrics['ravd_percent']:.2f}%"
    )
    print(f"Saved evaluation report to {output_path}")
