import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class AOTDecoder(nn.Module):
    """Decoder used in AOT-GAN model.

    This implementation follows:
    Aggregated Contextual Transformations for High-Resolution Image Inpainting

    Args:
        in_channels (int, optional): Channel number of input feature.
            Default: 256.
        mid_channels (int, optional): Channel number of middle features.
            Default: 128.
        act_cfg (dict, optional): Config dict for activation layer,
            "relu" by default.
    """

    def __init__(self,
                 in_channels=256,
                 mid_channels=128,
                 act_cfg=dict(type='ReLU')):
        super().__init__()

        self.decoder = nn.ModuleList([
            ConvModule(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                act_cfg=act_cfg),
            ConvModule(
                mid_channels,
                mid_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                act_cfg=act_cfg),
            ConvModule(
                mid_channels // 2,
                3,
                kernel_size=3,
                stride=1,
                padding=1,
                act_cfg=None)
        ])
        self.output_act = nn.Tanh()

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        for i in range(0, len(self.decoder)):
            if i <= 1:
                x = F.interpolate(
                    x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.decoder[i](x)

        return self.output_act(x)
