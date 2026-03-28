import numpy as np
import SimpleITK as sitk


def absolute_volume_difference(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    positive_label: int = 1,
) -> float:
    pred_voxels = int((prediction == positive_label).sum())
    target_voxels = int((target == positive_label).sum())
    voxel_volume_ml = float(np.prod(spacing)) / 1000.0
    return float(abs(pred_voxels - target_voxels) * voxel_volume_ml)


def dice_coefficient(
    prediction: np.ndarray,
    target: np.ndarray,
    positive_label: int = 1,
) -> float:
    pred = prediction == positive_label
    truth = target == positive_label

    pred_sum = int(pred.sum())
    truth_sum = int(truth.sum())
    if pred_sum == 0 and truth_sum == 0:
        return 1.0

    intersection = int(np.logical_and(pred, truth).sum())
    return float((2.0 * intersection) / (pred_sum + truth_sum + 1e-6))


def hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    positive_label: int = 1,
) -> float:
    pred = (prediction == positive_label).astype(np.uint8)
    truth = (target == positive_label).astype(np.uint8)

    if pred.sum() == 0 and truth.sum() == 0:
        return 0.0
    if pred.sum() == 0 or truth.sum() == 0:
        return float("inf")

    pred_img = sitk.GetImageFromArray(pred)
    truth_img = sitk.GetImageFromArray(truth)
    pred_img.SetSpacing(tuple(spacing[::-1]))
    truth_img.SetSpacing(tuple(spacing[::-1]))

    pred_surface = sitk.LabelContour(pred_img)
    truth_surface = sitk.LabelContour(truth_img)

    pred_surface_arr = sitk.GetArrayViewFromImage(pred_surface).astype(bool)
    truth_surface_arr = sitk.GetArrayViewFromImage(truth_surface).astype(bool)

    if not pred_surface_arr.any() and not truth_surface_arr.any():
        return 0.0

    pred_distance = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            pred_surface,
            squaredDistance=False,
            useImageSpacing=True,
        )
    )
    truth_distance = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            truth_surface,
            squaredDistance=False,
            useImageSpacing=True,
        )
    )

    pred_to_truth = sitk.GetArrayViewFromImage(truth_distance)[pred_surface_arr]
    truth_to_pred = sitk.GetArrayViewFromImage(pred_distance)[truth_surface_arr]

    distances = np.concatenate([pred_to_truth, truth_to_pred])
    if distances.size == 0:
        return 0.0

    return float(np.percentile(distances, 95))
