import numpy as np
import torch


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

    C = model.out.out_channels  # num_classes
    D, H, W = volume.shape
    pd, ph, pw = patch_size

    stride = (
        int(pd * (1 - overlap)),
        int(ph * (1 - overlap)),
        int(pw * (1 - overlap)),
    )

    logits_sum = np.zeros((C, D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    patches, coords = [], []

    for z in range(0, max(D - pd + 1, 1), stride[0]):
        for y in range(0, max(H - ph + 1, 1), stride[1]):
            for x in range(0, max(W - pw + 1, 1), stride[2]):
                patch = volume[z : z + pd, y : y + ph, x : x + pw]
                patches.append(patch)
                coords.append((z, y, x))

                if len(patches) == batch_size:
                    _run_batch(patches, coords, logits_sum, count_map, model, device)
                    patches, coords = [], []

    if patches:
        _run_batch(patches, coords, logits_sum, count_map, model, device)

    return logits_sum / np.clip(count_map[None], 1e-6, None)


def _run_batch(patches, coords, logits_sum, count_map, model, device):
    x = torch.from_numpy(np.stack(patches)).unsqueeze(1).float().to(device)
    out = model(x).cpu().numpy()

    for i, (z, y, x0) in enumerate(coords):
        pd, ph, pw = out.shape[2:]
        logits_sum[:, z : z + pd, y : y + ph, x0 : x0 + pw] += out[i]
        count_map[z : z + pd, y : y + ph, x0 : x0 + pw] += 1
