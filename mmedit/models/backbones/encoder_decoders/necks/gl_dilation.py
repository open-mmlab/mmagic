import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.common import SimpleGatedConvModule
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class GLDilationNeck(nn.Module):
    """Dilation Backbone used in Global&Local model.

    This implementation follows:
    Globally and locally Consistent Image Completion

    Args:
        in_channels (int): Channel number of input feature.
        conv_type (str): The type of conv module. In DeepFillv1 model, the
            `conv_type` should be 'conv'. In DeepFillv2 model, the `conv_type`
            should be 'gated_conv'.
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        kwargs (keyword arguments).
    """
    _conv_type = dict(conv=ConvModule, gated_conv=SimpleGatedConvModule)

    def __init__(self,
                 in_channels=256,
                 conv_type='conv',
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(GLDilationNeck, self).__init__()
        conv_module = self._conv_type[conv_type]
        dilation_convs_ = []
        for i in range(4):
            dilation_ = int(2**(i + 1))
            dilation_convs_.append(
                conv_module(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=dilation_,
                    dilation=dilation_,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
        self.dilation_convs = nn.Sequential(*dilation_convs_)

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        x = self.dilation_convs(x)
        return x
