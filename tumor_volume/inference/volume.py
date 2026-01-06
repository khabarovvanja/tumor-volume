import numpy as np

def compute_tumor_volume(mask: np.ndarray, spacing: tuple[float, float, float] = (2.0, 2.0, 2.0)) -> float:
    """
    mask: [D, H, W] binary
    spacing: (z, y, x) in mm
    returns volume in ml
    """
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    tumor_voxels = (mask > 0).sum()
    volume_ml = tumor_voxels * voxel_volume_mm3 / 1000.0
    return volume_ml