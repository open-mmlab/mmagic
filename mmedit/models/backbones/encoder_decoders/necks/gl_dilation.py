import torch.nn as nn
from mmedit.models.common import ConvModule
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module
class GLDilationNeck(nn.Module):
    """Dilation Backbone used in Global&Local model.

    This implementation follows:
    Globally and locally Consistent Image Completion

    Args:
        in_channels (int): Channel number of input feature.
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
    """

    def __init__(self,
                 in_channels=256,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(GLDilationNeck, self).__init__()

        dilation_convs_ = []
        for i in range(4):
            dilation_ = int(2**(i + 1))
            dilation_convs_.append(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=dilation_,
                    dilation=dilation_,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.dilation_convs = nn.Sequential(*dilation_convs_)

    def forward(self, x):
        x = self.dilation_convs(x)
        return x
