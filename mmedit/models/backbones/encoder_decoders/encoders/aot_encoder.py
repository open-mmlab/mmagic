import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class AOTEncoder(nn.Module):
    """Encoder used in AOT-GAN model.

    This implementation follows:
    Aggregated Contextual Transformations for High-Resolution Image Inpainting

    Args:
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
    """

    def __init__(self, in_channels=4, act_cfg=dict(type='ReLU')):
        super().__init__()

        self.reflectionpad2d = nn.ReflectionPad2d(3)

        self.enc1 = ConvModule(
            in_channels, 64, kernel_size=7, stride=1, act_cfg=act_cfg)

        self.enc2 = ConvModule(
            64, 128, kernel_size=4, stride=2, padding=1, act_cfg=act_cfg)

        self.enc3 = ConvModule(
            128, 256, kernel_size=4, stride=2, padding=1, act_cfg=act_cfg)

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """

        x = self.reflectionpad2d(x)
        for i in range(3):
            x = getattr(self, f'enc{i + 1}')(x)

        return x
