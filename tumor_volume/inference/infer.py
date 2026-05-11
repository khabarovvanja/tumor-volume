import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from tumor_volume.data.preprocess import (
    TARGET_SPACING,
    compute_nonzero_bbox,
    crop_to_bbox,
    restore_from_bbox,
    resample_to_spacing,
)
from tumor_volume.inference.postprocess import logits_to_mask
from tumor_volume.inference.sliding_window import sliding_window_inference
from tumor_volume.inference.volume import compute_tumor_volume
from tumor_volume.models.factory import build_model


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.exists():
        return path

    parent = path.parent if path.parent != Path("") else Path(".")
    candidates = sorted(
        parent.glob("exp_*/best_model*.pth"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Checkpoint '{checkpoint_path}' was not found and no saved experiment checkpoints were discovered."
    )


def _case_name(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".npy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    return (volume - volume.mean()) / (volume.std() + 1e-6)


def _prepare_nifti_for_inference(input_path: Path):
    original_nii = nib.load(str(input_path))
    original_data = original_nii.get_fdata().astype(np.float32)
    original_spacing = tuple(float(s) for s in original_nii.header.get_zooms()[:3])

    canonical_nii = nib.as_closest_canonical(original_nii)
    canonical_data = canonical_nii.get_fdata().astype(np.float32)
    canonical_spacing = tuple(float(s) for s in canonical_nii.header.get_zooms()[:3])

    original_ornt = nib.orientations.io_orientation(original_nii.affine)
    canonical_ornt = nib.orientations.io_orientation(canonical_nii.affine)
    canonical_to_original = nib.orientations.ornt_transform(
        canonical_ornt,
        original_ornt,
    )

    resampled = resample_to_spacing(
        volume=canonical_data,
        input_spacing=canonical_spacing,
        output_spacing=TARGET_SPACING,
        is_mask=False,
    ).astype(np.float32)
    crop_bbox = compute_nonzero_bbox(resampled)
    processed = crop_to_bbox(resampled, crop_bbox).astype(np.float32)

    metadata = {
        "input_path": str(input_path),
        "original_nii": original_nii,
        "original_data": original_data,
        "original_shape": tuple(int(v) for v in original_data.shape),
        "original_spacing": original_spacing,
        "canonical_shape": tuple(int(v) for v in canonical_data.shape),
        "canonical_spacing": canonical_spacing,
        "resampled_shape": tuple(int(v) for v in resampled.shape),
        "crop_bbox": crop_bbox,
        "canonical_to_original": canonical_to_original,
    }
    return _normalize_volume(processed), metadata


def _restore_mask_to_original_space(mask: np.ndarray, metadata: dict) -> np.ndarray:
    mask_resampled_full = restore_from_bbox(
        cropped=mask.astype(np.uint8),
        full_shape=metadata["resampled_shape"],
        bbox=metadata["crop_bbox"],
        fill_value=0,
    )

    canonical_mask = resample_to_spacing(
        volume=mask_resampled_full,
        input_spacing=TARGET_SPACING,
        output_spacing=metadata["canonical_spacing"],
        is_mask=True,
        output_shape=metadata["canonical_shape"],
    ).astype(np.uint8)

    original_mask = nib.orientations.apply_orientation(
        canonical_mask,
        metadata["canonical_to_original"],
    )
    return original_mask.astype(np.uint8)


def _save_nifti_outputs(output_dir: Path, metadata: dict, original_mask: np.ndarray) -> None:
    image_output_path = output_dir / "image.nii.gz"
    mask_output_path = output_dir / "mask.nii.gz"

    nib.save(metadata["original_nii"], str(image_output_path))

    mask_header = metadata["original_nii"].header.copy()
    mask_header.set_data_dtype(np.uint8)
    mask_nii = nib.Nifti1Image(
        original_mask.astype(np.uint8),
        metadata["original_nii"].affine,
        mask_header,
    )
    nib.save(mask_nii, str(mask_output_path))


def _render_preview_gif(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    duration_ms: int,
) -> None:
    target_height, target_width = image.shape[1], image.shape[2]
    image = image.astype(np.float32)
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_max - image_min < 1e-6:
        image_scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        image_scaled = ((image - image_min) / (image_max - image_min) * 255.0).astype(
            np.uint8
        )
    image_scaled = 255 - image_scaled

    frames = []
    for slice_idx in range(image_scaled.shape[0]):
        rotated_image = np.rot90(image_scaled[slice_idx], k=1)
        rotated_mask = np.rot90(mask[slice_idx] > 0, k=1)

        frame_image = Image.fromarray(rotated_image).resize(
            (target_width, target_height),
            resample=Image.Resampling.BILINEAR,
        )
        mask_image = Image.fromarray(rotated_mask.astype(np.uint8) * 255).resize(
            (target_width, target_height),
            resample=Image.Resampling.NEAREST,
        )

        frame = np.stack([np.asarray(frame_image)] * 3, axis=-1)
        mask_slice = np.asarray(mask_image) > 0

        if mask_slice.any():
            overlay = frame.copy()
            overlay[mask_slice] = np.array([255, 0, 0], dtype=np.uint8)
            frame = np.where(
                mask_slice[..., None],
                (0.45 * frame + 0.55 * overlay),
                frame,
            )
            frame = frame.astype(np.uint8)

        frames.append(Image.fromarray(frame))

    if not frames:
        raise RuntimeError("Cannot create preview GIF: no slices were generated.")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _run_npy_inference(
    cfg: DictConfig,
    input_path: Path,
    output_dir: Path,
    model,
    checkpoint_path: Path,
) -> None:
    volume = np.load(input_path).astype(np.float32)
    volume = _normalize_volume(volume)

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
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_path": str(input_path),
                "checkpoint_path": str(checkpoint_path),
                "tumor_volume_ml": float(tumor_volume_ml),
                "output_mask_path": str(output_dir / "mask.npy"),
            },
            f,
            indent=2,
        )


def run_inference(cfg: DictConfig):
    input_path = Path(cfg.inference.input.volume)
    filename = _case_name(input_path)
    output_dir = Path(cfg.inference.output.dir) / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg.model)

    checkpoint_path = _resolve_checkpoint_path(cfg.inference.checkpoint.path)
    ckpt = torch.load(checkpoint_path, map_location=cfg.inference.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(cfg.inference.device)
    model.eval()

    if input_path.suffix == ".npy":
        _run_npy_inference(cfg, input_path, output_dir, model, checkpoint_path)
        print(f"Tumor volume saved to {output_dir / 'metrics.json'}")
        return

    if not (input_path.name.endswith(".nii.gz") or input_path.suffix == ".nii"):
        raise ValueError("Inference input must be a .nii.gz, .nii, or .npy file.")

    volume, metadata = _prepare_nifti_for_inference(input_path)
    logits = sliding_window_inference(
        volume=volume,
        model=model,
        patch_size=tuple(cfg.inference.sliding_window.patch_size),
        overlap=cfg.inference.sliding_window.overlap,
        batch_size=cfg.inference.sliding_window.batch_size,
        device=cfg.inference.device,
    )

    processed_mask = logits_to_mask(torch.from_numpy(logits))
    original_mask = _restore_mask_to_original_space(processed_mask, metadata)
    if tuple(original_mask.shape) != tuple(metadata["original_shape"]):
        raise RuntimeError(
            "Restored mask shape does not match the original input shape: "
            f"{original_mask.shape} != {metadata['original_shape']}"
        )
    tumor_volume_ml = compute_tumor_volume(original_mask, metadata["original_spacing"])

    _save_nifti_outputs(output_dir, metadata, original_mask)
    if cfg.inference.output.save_preview_gif:
        _render_preview_gif(
            image=metadata["original_data"],
            mask=original_mask,
            output_path=output_dir / "preview.gif",
            duration_ms=cfg.inference.output.preview_gif_duration_ms,
        )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_path": str(input_path),
                "checkpoint_path": str(checkpoint_path),
                "tumor_volume_ml": float(tumor_volume_ml),
                "input_image_path": str(output_dir / "image.nii.gz"),
                "output_mask_path": str(output_dir / "mask.nii.gz"),
                "preview_gif_path": (
                    str(output_dir / "preview.gif")
                    if cfg.inference.output.save_preview_gif
                    else None
                ),
                "input_shape": list(metadata["original_shape"]),
                "output_mask_shape": list(original_mask.shape),
                "input_spacing": list(metadata["original_spacing"]),
                "target_spacing": list(TARGET_SPACING),
            },
            f,
            indent=2,
        )

    print(f"Tumor volume: {tumor_volume_ml:.2f} ml")


if __name__ == "__main__":
    run_inference()
