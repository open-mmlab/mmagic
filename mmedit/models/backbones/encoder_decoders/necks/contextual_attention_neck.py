import torch.nn as nn
from mmedit.models.common import ConvModule
from mmedit.models.common.contextual_attention import ContextualAttentionModule
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module
class ContextualAttentionNeck(nn.Module):
    """Neck with contextual attention module.

    Args:
        in_channels (int): The number of input channels. In DeepFill model,
            they use 128 as default.
        conv_cfg (dict | None): Config of conv module. Default: None.
        norm_cfg (dict | None): Config of norm module. Default: None.
        act_cfg (dict | None): Config of activation layer.
            Default: dict(type='ELU').
        ca_args (dict): Config of contextual attention module.
            Default: dict(softmax_scale=10.).
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ELU'),
                 ca_args=dict(softmax_scale=10.)):
        super(ContextualAttentionNeck, self).__init__()
        self.contextual_attention = ContextualAttentionModule(**ca_args)
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x, mask):
        x, offset = self.contextual_attention(x, x, mask)
        x = self.conv1(x)
        x = self.conv2(x)

        return x, offset
