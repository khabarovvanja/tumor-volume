import numpy as np
import torch
from pathlib import Path
import nibabel as nib  # для .nii/.nii.gz (опционально)

class PETPatchDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            patch_size=(96, 96, 96), 
            samples_per_volume=16,
            train_split=0.8,
            seed=42
        ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume

        self.images = sorted(list(self.images_dir.glob("*")))
        self.masks = sorted(list(self.masks_dir.glob("*")))

        assert len(self.images) == len(self.masks), "The number of images and masks does not match"

    def __len__(self):
        return len(self.images) * self.samples_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.samples_per_volume

        img = self._load_volume(self.images[vol_idx])
        mask = self._load_volume(self.masks[vol_idx])

        # z-core normalization
        img = (img - img.mean()) / (img.std() + 1e-6)

        center = self._get_random_patch_center(mask, p_tumor=0.7)
        img_patch = self._crop_patch(img, center, self.patch_size)
        mask_patch = self._crop_patch(mask, center, self.patch_size)

        img_patch = torch.from_numpy(img_patch).float().unsqueeze(0)  # [1, D, H, W]
        mask_patch = torch.from_numpy(mask_patch).long()              # [D, H, W]

        return img_patch, mask_patch

    def _load_volume(self, path):
        path = Path(path)
        if path.suffix in [".nii", ".gz", ".nii.gz"]:
            vol = nib.load(str(path)).get_fdata()
        else:
            vol = np.load(path)
        return vol

    def _get_random_patch_center(self, mask, p_tumor=0.5):
        if np.random.rand() < p_tumor and mask.sum() > 0:
            coords = np.argwhere(mask > 0)
            return coords[np.random.randint(len(coords))]
        else:
            return np.array([
                np.random.randint(mask.shape[0]),
                np.random.randint(mask.shape[1]),
                np.random.randint(mask.shape[2]),
            ])

    def _crop_patch(self, vol, center, size):
        d, h, w = vol.shape
        sd, sh, sw = size

        z0 = np.clip(center[0] - sd // 2, 0, d - sd)
        y0 = np.clip(center[1] - sh // 2, 0, h - sh)
        x0 = np.clip(center[2] - sw // 2, 0, w - sw)

        return vol[z0:z0+sd, y0:y0+sh, x0:x0+sw]