import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

TARGET_SPACING = (2.0, 2.0, 2.0)
MIN_NONZERO = 1e-6
SAVE_FORMAT = "npy"   # "npy" or "nii"


def load_nii(path):
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    data = nii.get_fdata().astype(np.float32)
    spacing = nii.header.get_zooms()
    return data, spacing, nii.affine


def resample(volume, spacing, target_spacing, is_mask=False):
    sitk_vol = sitk.GetImageFromArray(volume)
    sitk_vol.SetSpacing(spacing[::-1])

    new_size = [
        int(round(volume.shape[i] * spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing[::-1])
    resampler.SetSize(new_size[::-1])
    resampler.SetInterpolator(
        sitk.sitkNearest if is_mask else sitk.sitkLinear
    )
    resampler.SetOutputDirection(sitk_vol.GetDirection())
    resampler.SetOutputOrigin(sitk_vol.GetOrigin())

    out = resampler.Execute(sitk_vol)
    return sitk.GetArrayFromImage(out)


def crop_nonzero(img, mask):
    coords = np.where(img > MIN_NONZERO)
    if len(coords[0]) == 0:
        return img, mask

    z0, z1 = coords[0].min(), coords[0].max()
    y0, y1 = coords[1].min(), coords[1].max()
    x0, x1 = coords[2].min(), coords[2].max()

    return (
        img[z0:z1+1, y0:y1+1, x0:x1+1],
        mask[z0:z1+1, y0:y1+1, x0:x1+1],
    )


def save(volume, affine, path):
    if SAVE_FORMAT == "npy":
        np.save(path, volume)
    else:
        nii = nib.Nifti1Image(volume, affine)
        nib.save(nii, path)


def preprocess_case(img_path, mask_path, out_img, out_mask):
    img, spacing, affine = load_nii(img_path)
    mask, _, _ = load_nii(mask_path)

    # binarize mask
    mask = (mask > 0).astype(np.uint8)

    # resample
    img = resample(img, spacing, TARGET_SPACING, is_mask=False)
    mask = resample(mask, spacing, TARGET_SPACING, is_mask=True)

    # crop background
    img, mask = crop_nonzero(img, mask)

    save(img, affine, out_img)
    save(mask, affine, out_mask)


def main(img_dir, mask_dir, out_img_dir, out_mask_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    images = sorted(os.listdir(img_dir))

    for name in tqdm(images):
        img_path = os.path.join(img_dir, name)
        mask_path = os.path.join(mask_dir, name)

        out_img = os.path.join(out_img_dir, name.replace(".nii.gz", f".{SAVE_FORMAT}"))
        out_mask = os.path.join(out_mask_dir, name.replace(".nii.gz", f".{SAVE_FORMAT}"))

        preprocess_case(img_path, mask_path, out_img, out_mask)


if __name__ == "__main__":
    main(
        img_dir="data/images_nii",
        mask_dir="data/masks_nii",
        out_img_dir="data/images_preprocessed",
        out_mask_dir="data/masks_preprocessed",
    )