import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.common import SimpleGatedConvModule
from mmedit.models.common.contextual_attention import ContextualAttentionModule
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class ContextualAttentionNeck(nn.Module):
    """Neck with contextual attention module.

    Args:
        in_channels (int): The number of input channels.
        conv_type (str): The type of conv module. In DeepFillv1 model, the
            `conv_type` should be 'conv'. In DeepFillv2 model, the `conv_type`
            should be 'gated_conv'.
        conv_cfg (dict | None): Config of conv module. Default: None.
        norm_cfg (dict | None): Config of norm module. Default: None.
        act_cfg (dict | None): Config of activation layer. Default:
            dict(type='ELU').
        contextual_attention_args (dict): Config of contextual attention
            module. Default: dict(softmax_scale=10.).
        kwargs (keyword arguments).
    """
    _conv_type = dict(conv=ConvModule, gated_conv=SimpleGatedConvModule)

    def __init__(self,
                 in_channels,
                 conv_type='conv',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ELU'),
                 contextual_attention_args=dict(softmax_scale=10.),
                 **kwargs):
        super(ContextualAttentionNeck, self).__init__()
        self.contextual_attention = ContextualAttentionModule(
            **contextual_attention_args)
        conv_module = self._conv_type[conv_type]
        self.conv1 = conv_module(
            in_channels,
            in_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        self.conv2 = conv_module(
            in_channels,
            in_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

    def forward(self, x, mask):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).
            mask (torch.Tensor): Input tensor with shape of (n, 1, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        x, offset = self.contextual_attention(x, x, mask)
        x = self.conv1(x)
        x = self.conv2(x)

        return x, offset
