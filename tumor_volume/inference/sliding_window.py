import numpy as np
import torch


def _compute_starts(size: int, patch: int, stride: int) -> list[int]:
    if size <= patch:
        return [0]

    starts = list(range(0, size - patch + 1, stride))
    last_start = size - patch
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


@torch.no_grad()
def sliding_window_inference(
    volume: np.ndarray,
    model: torch.nn.Module,
    patch_size: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    device: str,
):
    model.eval()

    if hasattr(model, "num_classes"):
        C = model.num_classes
    else:
        C = model.out.out_channels
    has_channels = volume.ndim == 4
    if has_channels:
        _, orig_d, orig_h, orig_w = volume.shape
    else:
        orig_d, orig_h, orig_w = volume.shape
    pd, ph, pw = patch_size

    pad_d = max(pd - orig_d, 0)
    pad_h = max(ph - orig_h, 0)
    pad_w = max(pw - orig_w, 0)
    if pad_d or pad_h or pad_w:
        pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
        if has_channels:
            pad_width = ((0, 0), *pad_width)
        volume = np.pad(
            volume,
            pad_width,
            mode="constant",
        )

    if has_channels:
        _, D, H, W = volume.shape
    else:
        D, H, W = volume.shape

    stride = (
        max(int(pd * (1 - overlap)), 1),
        max(int(ph * (1 - overlap)), 1),
        max(int(pw * (1 - overlap)), 1),
    )

    logits_sum = np.zeros((C, D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    patches, coords = [], []

    z_starts = _compute_starts(D, pd, stride[0])
    y_starts = _compute_starts(H, ph, stride[1])
    x_starts = _compute_starts(W, pw, stride[2])

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                if has_channels:
                    patch = volume[:, z : z + pd, y : y + ph, x : x + pw]
                else:
                    patch = volume[z : z + pd, y : y + ph, x : x + pw]
                patches.append(patch)
                coords.append((z, y, x))

                if len(patches) == batch_size:
                    _run_batch(patches, coords, logits_sum, count_map, model, device)
                    patches, coords = [], []

    if patches:
        _run_batch(patches, coords, logits_sum, count_map, model, device)

    logits = logits_sum / np.clip(count_map[None], 1e-6, None)
    return logits[:, :orig_d, :orig_h, :orig_w]


def _run_batch(patches, coords, logits_sum, count_map, model, device):
    batch = np.stack(patches)
    x = torch.from_numpy(batch).float()
    if x.ndim == 4:
        x = x.unsqueeze(1)
    x = x.to(device)
    out = model(x).cpu().numpy()

    for i, (z, y, x0) in enumerate(coords):
        pd, ph, pw = out.shape[2:]
        logits_sum[:, z : z + pd, y : y + ph, x0 : x0 + pw] += out[i]
        count_map[z : z + pd, y : y + ph, x0 : x0 + pw] += 1
