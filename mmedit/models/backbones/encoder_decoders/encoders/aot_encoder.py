import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class AOTEncoder(nn.Module):
    """Encoder used in AOT-GAN model.

    This implementation follows:
    Aggregated Contextual Transformations for High-Resolution Image Inpainting

    Args:
        in_channels (int, optional): Channel number of input feature.
            Default: 4.
        mid_channels (int, optional): Channel number of middle feature.
            Default: 64.
        out_channels (int, optional): Channel number of output feature.
            Default: 256.
        act_cfg (dict, optional): Config dict for activation layer,
            "relu" by default.
    """

    def __init__(self,
                 in_channels=4,
                 mid_channels=64,
                 out_channels=256,
                 act_cfg=dict(type='ReLU')):
        super().__init__()

        self.reflectionpad2d = nn.ReflectionPad2d(3)
        self.encoder = nn.ModuleList([
            ConvModule(
                in_channels,
                mid_channels,
                kernel_size=7,
                stride=1,
                act_cfg=act_cfg),
            ConvModule(
                mid_channels,
                mid_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                act_cfg=act_cfg),
            ConvModule(
                mid_channels * 2,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                act_cfg=act_cfg)
        ])

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """

        x = self.reflectionpad2d(x)
        for i in range(0, len(self.encoder)):
            x = self.encoder[i](x)

        return x
