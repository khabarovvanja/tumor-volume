from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def _strip_known_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".npy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def load_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.name.endswith(".nii.gz") or path.suffix == ".nii":
        return nib.load(str(path)).get_fdata()
    return np.load(path)


def list_case_files(
    images_dir: str | Path,
    masks_dir: str | Path,
) -> list[tuple[Path, Path]]:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    images = {
        _strip_known_suffix(path): path
        for path in images_dir.glob("*")
        if path.is_file()
    }
    masks = {
        _strip_known_suffix(path): path
        for path in masks_dir.glob("*")
        if path.is_file()
    }

    shared_case_ids = sorted(images.keys() & masks.keys())
    if not shared_case_ids:
        raise RuntimeError("No matching image-mask pairs were found.")

    missing_images = sorted(masks.keys() - images.keys())
    missing_masks = sorted(images.keys() - masks.keys())
    if missing_images or missing_masks:
        raise RuntimeError(
            "Image/mask pairs are inconsistent. "
            f"Missing images for: {missing_images[:3]}; "
            f"missing masks for: {missing_masks[:3]}"
        )

    return [(images[case_id], masks[case_id]) for case_id in shared_case_ids]


class PETPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir=None,
        masks_dir=None,
        patch_size=(96, 96, 96),
        samples_per_volume=16,
        cases: list[tuple[Path, Path]] | None = None,
    ):
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume

        if cases is not None:
            self.cases = [(Path(img), Path(mask)) for img, mask in cases]
        else:
            if images_dir is None or masks_dir is None:
                raise ValueError("Either cases or both images_dir and masks_dir must be provided.")
            self.cases = list_case_files(images_dir, masks_dir)

        self.images = [img for img, _ in self.cases]
        self.masks = [mask for _, mask in self.cases]

        assert len(self.images) == len(
            self.masks
        ), "The number of images and masks does not match"

    def __len__(self):
        return len(self.images) * self.samples_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.samples_per_volume

        img = load_volume(self.images[vol_idx])
        mask = load_volume(self.masks[vol_idx])

        # z-core normalization
        img = (img - img.mean()) / (img.std() + 1e-6)

        center = self._get_random_patch_center(mask, p_tumor=0.7)
        img_patch = self._crop_patch(img, center, self.patch_size)
        mask_patch = self._crop_patch(mask, center, self.patch_size)

        img_patch = torch.from_numpy(img_patch).float().unsqueeze(0)  # [1, D, H, W]
        mask_patch = torch.from_numpy(mask_patch).long()  # [D, H, W]

        return img_patch, mask_patch

    def _get_random_patch_center(self, mask, p_tumor=0.5):
        if np.random.rand() < p_tumor and mask.sum() > 0:
            coords = np.argwhere(mask > 0)
            return coords[np.random.randint(len(coords))]
        else:
            return np.array(
                [
                    np.random.randint(mask.shape[0]),
                    np.random.randint(mask.shape[1]),
                    np.random.randint(mask.shape[2]),
                ]
            )

    def _crop_patch(self, vol, center, size):
        d, h, w = vol.shape
        sd, sh, sw = size

        z0 = np.clip(center[0] - sd // 2, 0, d - sd)
        y0 = np.clip(center[1] - sh // 2, 0, h - sh)
        x0 = np.clip(center[2] - sw // 2, 0, w - sw)

        return vol[z0 : z0 + sd, y0 : y0 + sh, x0 : x0 + sw]
