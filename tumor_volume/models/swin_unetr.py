import inspect

import torch.nn as nn


class SwinUNETR3D(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: int,
        num_classes: int,
        feature_size: int = 24,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        use_v2: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:
            raise ImportError(
                "SwinUNETR requires MONAI. Install dependencies with Poetry "
                "after adding monai to pyproject.toml."
            ) from exc

        kwargs = dict(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            use_v2=use_v2,
        )
        signature_params = inspect.signature(SwinUNETR).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in signature_params}
        if "img_size" in signature_params:
            kwargs["img_size"] = img_size

        self.net = SwinUNETR(**kwargs)

    def forward(self, x):
        return self.net(x)
