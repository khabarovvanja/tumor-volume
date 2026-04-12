import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.processing import resample_from_to
from tqdm import tqdm

TARGET_SPACING = (2.0, 2.0, 2.0)
MIN_NONZERO = 1e-6
SAVE_FORMAT = "npy"  # "npy" or "nii"


def load_nii(path):
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    data = nii.get_fdata().astype(np.float32)
    spacing = nii.header.get_zooms()[:3]
    spacing = tuple(float(s) for s in spacing)
    return data, spacing, nii.affine


def resample(volume, spacing, target_spacing, is_mask=False):
    return resample_to_spacing(
        volume=volume,
        input_spacing=spacing,
        output_spacing=target_spacing,
        is_mask=is_mask,
    )


def resample_to_spacing(
    volume,
    input_spacing,
    output_spacing,
    is_mask=False,
    output_shape=None,
):
    sitk_vol = sitk.GetImageFromArray(volume)
    sitk_vol.SetSpacing(tuple(input_spacing[::-1]))  # z,y,x → x,y,z

    if output_shape is None:
        output_shape = [
            int(round(volume.shape[i] * input_spacing[i] / output_spacing[i]))
            for i in range(3)
        ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(output_spacing[::-1]))
    resampler.SetSize(list(output_shape[::-1]))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    resampler.SetOutputDirection(sitk_vol.GetDirection())
    resampler.SetOutputOrigin(sitk_vol.GetOrigin())

    out = resampler.Execute(sitk_vol)
    return sitk.GetArrayFromImage(out)


def crop_nonzero(img, mask):
    bbox = compute_nonzero_bbox(img)
    if bbox is None:
        return img, mask

    return crop_to_bbox(img, bbox), crop_to_bbox(mask, bbox)


def compute_nonzero_bbox(img):
    coords = np.where(img > MIN_NONZERO)
    if len(coords[0]) == 0:
        return None

    return (
        int(coords[0].min()),
        int(coords[0].max()) + 1,
        int(coords[1].min()),
        int(coords[1].max()) + 1,
        int(coords[2].min()),
        int(coords[2].max()) + 1,
    )


def crop_to_bbox(vol, bbox):
    if bbox is None:
        return vol

    z0, z1, y0, y1, x0, x1 = bbox
    return vol[z0:z1, y0:y1, x0:x1]


def restore_from_bbox(cropped, full_shape, bbox, fill_value=0):
    restored = np.full(full_shape, fill_value, dtype=cropped.dtype)
    if bbox is None:
        restored[...] = cropped
        return restored

    z0, z1, y0, y1, x0, x1 = bbox
    restored[z0:z1, y0:y1, x0:x1] = cropped
    return restored


def save(volume, affine, path):
    if SAVE_FORMAT == "npy":
        np.save(path, volume)
    else:
        nii = nib.Nifti1Image(volume, affine)
        nib.save(nii, path)


def preprocess_case(img_path, mask_path, out_img, out_mask):
    img, spacing, affine = load_nii(img_path)
    mask, _, _ = load_nii(mask_path)

    mask = (mask > 0).astype(np.uint8)

    img = resample(img, spacing, TARGET_SPACING, is_mask=False)
    mask = resample(mask, spacing, TARGET_SPACING, is_mask=True)

    img, mask = crop_nonzero(img, mask)

    save(img, affine, out_img)
    save(mask, affine, out_mask)


def preprocess_multimodal_case(
    pet_path,
    ct_path,
    mask_path,
    out_pet,
    out_ct,
    out_mask,
):
    pet_nii = nib.as_closest_canonical(nib.load(str(pet_path)))
    ct_nii = nib.as_closest_canonical(nib.load(str(ct_path)))
    mask_nii = nib.as_closest_canonical(nib.load(str(mask_path)))

    pet = pet_nii.get_fdata().astype(np.float32)
    mask = (mask_nii.get_fdata() > 0).astype(np.uint8)
    pet_spacing = tuple(float(s) for s in pet_nii.header.get_zooms()[:3])
    pet_affine = pet_nii.affine

    if pet.shape != mask.shape:
        raise ValueError(
            "PET and mask must have identical shapes before preprocessing. "
            f"Got PET={pet.shape}, mask={mask.shape}"
        )

    ct_in_pet_space = resample_from_to(
        ct_nii,
        (pet_nii.shape, pet_nii.affine),
        order=1,
    )
    ct = ct_in_pet_space.get_fdata().astype(np.float32)

    pet = resample(pet, pet_spacing, TARGET_SPACING, is_mask=False)
    target_shape = pet.shape
    ct = resample_to_spacing(
        ct,
        input_spacing=pet_spacing,
        output_spacing=TARGET_SPACING,
        is_mask=False,
        output_shape=target_shape,
    )
    mask = resample_to_spacing(
        mask,
        input_spacing=pet_spacing,
        output_spacing=TARGET_SPACING,
        is_mask=True,
        output_shape=target_shape,
    )

    bbox = compute_nonzero_bbox(pet)
    pet = crop_to_bbox(pet, bbox)
    ct = crop_to_bbox(ct, bbox)
    mask = crop_to_bbox(mask, bbox)

    save(pet, pet_affine, out_pet)
    save(ct, pet_affine, out_ct)
    save(mask, pet_affine, out_mask)


def _is_valid_output_pair(out_img, out_mask):
    return (
        os.path.isfile(out_img)
        and os.path.isfile(out_mask)
        and os.path.getsize(out_img) > 0
        and os.path.getsize(out_mask) > 0
    )

def _is_valid_output_triplet(out_pet, out_ct, out_mask):
    return (
        os.path.isfile(out_pet)
        and os.path.isfile(out_ct)
        and os.path.isfile(out_mask)
        and os.path.getsize(out_pet) > 0
        and os.path.getsize(out_ct) > 0
        and os.path.getsize(out_mask) > 0
    )


def nifti2npy(
    pet_dir,
    ct_dir,
    mask_dir,
    out_pet_dir,
    out_ct_dir,
    out_mask_dir,
    force=False,
):
    os.makedirs(out_pet_dir, exist_ok=True)
    os.makedirs(out_ct_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    pet_files = sorted(os.listdir(pet_dir))
    processed = 0
    skipped = 0

    for name in tqdm(pet_files):
        pet_path = os.path.join(pet_dir, name)
        ct_path = os.path.join(ct_dir, name)
        mask_path = os.path.join(mask_dir, name)

        if not os.path.isfile(ct_path):
            raise FileNotFoundError(f"Missing CT file for case '{name}' in '{ct_dir}'")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Missing mask file for case '{name}' in '{mask_dir}'"
            )

        out_pet = os.path.join(out_pet_dir, name.replace(".nii.gz", f".{SAVE_FORMAT}"))
        out_ct = os.path.join(out_ct_dir, name.replace(".nii.gz", f".{SAVE_FORMAT}"))
        out_mask = os.path.join(
            out_mask_dir, name.replace(".nii.gz", f".{SAVE_FORMAT}")
        )

        if not force and _is_valid_output_triplet(out_pet, out_ct, out_mask):
            skipped += 1
            continue

        preprocess_multimodal_case(
            pet_path,
            ct_path,
            mask_path,
            out_pet,
            out_ct,
            out_mask,
        )
        processed += 1

    print(
        f"nifti2npy completed: processed={processed}, skipped={skipped}, force={force}"
    )


if __name__ == "__main__":
    nifti2npy(
        pet_dir="data/raw/pet",
        ct_dir="data/raw/ct",
        mask_dir="data/raw/masks",
        out_pet_dir="data/processed/pet",
        out_ct_dir="data/processed/ct",
        out_mask_dir="data/processed/masks",
        force=False,
    )
