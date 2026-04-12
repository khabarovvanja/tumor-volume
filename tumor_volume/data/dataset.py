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


def list_case_triplets(
    pet_dir: str | Path,
    ct_dir: str | Path,
    masks_dir: str | Path,
) -> list[tuple[Path, Path, Path]]:
    pet_dir = Path(pet_dir)
    ct_dir = Path(ct_dir)
    masks_dir = Path(masks_dir)

    pet = {
        _strip_known_suffix(path): path
        for path in pet_dir.glob("*")
        if path.is_file()
    }
    ct = {
        _strip_known_suffix(path): path
        for path in ct_dir.glob("*")
        if path.is_file()
    }
    masks = {
        _strip_known_suffix(path): path
        for path in masks_dir.glob("*")
        if path.is_file()
    }

    shared_case_ids = sorted(pet.keys() & ct.keys() & masks.keys())
    if not shared_case_ids:
        raise RuntimeError("No matching PET-CT-mask triplets were found.")

    missing_pet = sorted((ct.keys() | masks.keys()) - pet.keys())
    missing_ct = sorted((pet.keys() | masks.keys()) - ct.keys())
    missing_masks = sorted((pet.keys() | ct.keys()) - masks.keys())
    if missing_pet or missing_ct or missing_masks:
        raise RuntimeError(
            "PET/CT/mask triplets are inconsistent. "
            f"Missing PET for: {missing_pet[:3]}; "
            f"missing CT for: {missing_ct[:3]}; "
            f"missing masks for: {missing_masks[:3]}"
        )

    return [(pet[case_id], ct[case_id], masks[case_id]) for case_id in shared_case_ids]


class PETCTPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pet_dir=None,
        ct_dir=None,
        masks_dir=None,
        patch_size=(96, 96, 96),
        samples_per_volume=16,
        cases: list[tuple[Path, Path, Path]] | None = None,
    ):
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume

        if cases is not None:
            self.cases = [(Path(pet), Path(ct), Path(mask)) for pet, ct, mask in cases]
        else:
            if pet_dir is None or ct_dir is None or masks_dir is None:
                raise ValueError(
                    "Either cases or pet_dir, ct_dir and masks_dir must be provided."
                )
            self.cases = list_case_triplets(pet_dir, ct_dir, masks_dir)

        self.pet = [pet for pet, _, _ in self.cases]
        self.ct = [ct for _, ct, _ in self.cases]
        self.masks = [mask for _, _, mask in self.cases]

        assert len(self.pet) == len(self.ct) == len(
            self.masks
        ), "The number of PET, CT and mask volumes does not match"

    def __len__(self):
        return len(self.pet) * self.samples_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.samples_per_volume

        pet = load_volume(self.pet[vol_idx]).astype(np.float32)
        ct = load_volume(self.ct[vol_idx]).astype(np.float32)
        mask = load_volume(self.masks[vol_idx])

        pet = (pet - pet.mean()) / (pet.std() + 1e-6)
        ct = (ct - ct.mean()) / (ct.std() + 1e-6)

        center = self._get_random_patch_center(mask, p_tumor=0.7)
        pet_patch = self._crop_patch(pet, center, self.patch_size)
        ct_patch = self._crop_patch(ct, center, self.patch_size)
        mask_patch = self._crop_patch(mask, center, self.patch_size)

        img_patch = torch.from_numpy(
            np.stack([pet_patch, ct_patch], axis=0)
        ).float()  # [2, D, H, W]
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
