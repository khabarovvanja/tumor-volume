from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


def load_swin_unetr_pretrained(
    model: nn.Module,
    checkpoint_path: str | Path,
    adapt_input: bool = True,
    ct_channel_index: int = 1,
) -> dict[str, object]:
    """
    Load compatible MONAI/NVIDIA Swin-UNETR weights into our wrapper.

    The BTCV bundle is CT-only and has 14 output classes. For PET+CT training we
    keep PET input and tumor head randomly initialized, while reusing compatible
    Swin-UNETR weights and placing CT pretrained kernels into the CT channel.
    """
    target = model.net if hasattr(model, "net") else model
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint was not found: {checkpoint_path}")

    pretrained_state = _load_state_dict(checkpoint_path)
    target_state = target.state_dict()
    loadable_state = {}
    adapted_keys = []
    skipped_shape = []
    skipped_missing = []

    for key, value in pretrained_state.items():
        if not torch.is_tensor(value):
            skipped_missing.append(key)
            continue

        normalized_key = _normalize_key(key, target_state)
        if normalized_key not in target_state:
            skipped_missing.append(key)
            continue

        target_value = target_state[normalized_key]
        if tuple(value.shape) == tuple(target_value.shape):
            loadable_state[normalized_key] = _to_target_dtype(value, target_value)
            continue

        adapted_value = None
        if adapt_input:
            adapted_value = _adapt_single_channel_conv_to_multichannel(
                pretrained_value=value,
                target_value=target_value,
                ct_channel_index=ct_channel_index,
            )

        if adapted_value is not None:
            loadable_state[normalized_key] = adapted_value
            adapted_keys.append(normalized_key)
            continue

        skipped_shape.append(
            {
                "key": normalized_key,
                "pretrained_shape": tuple(value.shape),
                "target_shape": tuple(target_value.shape),
            }
        )

    updated_state = dict(target_state)
    updated_state.update(loadable_state)
    target.load_state_dict(updated_state, strict=True)

    return {
        "path": str(checkpoint_path),
        "loaded": len(loadable_state),
        "adapted": len(adapted_keys),
        "skipped_shape": len(skipped_shape),
        "skipped_missing": len(skipped_missing),
        "adapted_keys": adapted_keys,
        "skipped_shape_keys": [item["key"] for item in skipped_shape],
        "skipped_missing_keys": skipped_missing,
    }


def _load_state_dict(checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "network_weights"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return checkpoint

    raise TypeError(
        f"Unsupported pretrained checkpoint format at {checkpoint_path}: "
        f"{type(checkpoint)!r}"
    )


def _normalize_key(key: str, target_state: Mapping[str, torch.Tensor]) -> str:
    if key in target_state:
        return key

    for prefix in ("module.", "net."):
        if key.startswith(prefix):
            candidate = key[len(prefix) :]
            if candidate in target_state:
                return candidate

    return key


def _adapt_single_channel_conv_to_multichannel(
    pretrained_value: torch.Tensor,
    target_value: torch.Tensor,
    ct_channel_index: int,
) -> torch.Tensor | None:
    if pretrained_value.ndim != 5 or target_value.ndim != 5:
        return None
    if pretrained_value.shape[1] != 1 or target_value.shape[1] <= 1:
        return None
    if pretrained_value.shape[0] != target_value.shape[0]:
        return None
    if tuple(pretrained_value.shape[2:]) != tuple(target_value.shape[2:]):
        return None

    adapted = target_value.detach().clone()
    channel_index = max(0, min(int(ct_channel_index), target_value.shape[1] - 1))
    adapted[:, channel_index : channel_index + 1] = _to_target_dtype(
        pretrained_value,
        target_value,
    )
    return adapted


def _to_target_dtype(
    value: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor:
    if value.dtype == target_value.dtype:
        return value
    return value.to(dtype=target_value.dtype)
