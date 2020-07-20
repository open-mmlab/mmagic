import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class GLEncoder(nn.Module):
    """Encoder used in Global&Local model.

    This implementation follows:
    Globally and locally Consistent Image Completion

    Args:
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
    """

    def __init__(self, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super(GLEncoder, self).__init__()

        channel_list = [64, 128, 128, 256, 256, 256]
        kernel_size_list = [5, 3, 3, 3, 3, 3]
        stride_list = [1, 2, 1, 2, 1, 1]
        in_channels = 4
        for i in range(6):
            ks = kernel_size_list[i]
            padding = (ks - 1) // 2
            self.add_module(
                f'enc{i + 1}',
                ConvModule(
                    in_channels,
                    channel_list[i],
                    kernel_size=ks,
                    stride=stride_list[i],
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channels = channel_list[i]

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        for i in range(6):
            x = getattr(self, f'enc{i + 1}')(x)
        return x
