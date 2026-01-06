import numpy as np
import torch

@torch.no_grad()
def logits_to_mask(logits: torch.Tensor) -> np.ndarray:
    """
    logits: [C, D, H, W] or [1, C, D, H, W]
    returns: [D, H, W] uint8
    """
    if logits.dim() == 5:
        logits = logits[0]

    mask = torch.argmax(logits, dim=0)
    return mask.cpu().numpy().astype("uint8")