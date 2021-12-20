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
        in_channels (int): Channel number of input feature.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
    """

    def __init__(self, in_channels=256, act_cfg=dict(type='ReLU')):
        super().__init__()

        self.dec1 = ConvModule(
            in_channels,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=act_cfg)
        self.dec2 = ConvModule(
            128, 64, kernel_size=3, stride=1, padding=1, act_cfg=act_cfg)
        self.dec3 = ConvModule(
            64, 3, kernel_size=3, stride=1, padding=1, act_cfg=None)

        self.output_act = nn.Tanh()

    def forward(self, x):
        """Forward Function.
        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        for i in range(3):
            if i <= 1:
                x = F.interpolate(
                    x, scale_factor=2, mode='bilinear', align_corners=True)
            x = getattr(self, f'dec{i + 1}')(x)
        x = self.output_act(x)

        return x
