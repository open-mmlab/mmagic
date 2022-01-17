# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.common import SimpleGatedConvModule
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class DeepFillEncoder(nn.Module):
    """Encoder used in DeepFill model.

    This implementation follows:
    Generative Image Inpainting with Contextual Attention

    Args:
        in_channels (int): The number of input channels. Default: 5.
        conv_type (str): The type of conv module. In DeepFillv1 model, the
            `conv_type` should be 'conv'. In DeepFillv2 model, the `conv_type`
            should be 'gated_conv'.
        norm_cfg (dict): Config dict to build norm layer. Default: None.
        act_cfg (dict): Config dict for activation layer, "elu" by default.
        encoder_type (str): Type of the encoder. Should be one of ['stage1',
            'stage2_conv', 'stage2_attention']. Default: 'stage1'.
        channel_factor (float): The scale factor for channel size.
            Default: 1.
        kwargs (keyword arguments).
    """
    _conv_type = dict(conv=ConvModule, gated_conv=SimpleGatedConvModule)

    def __init__(self,
                 in_channels=5,
                 conv_type='conv',
                 norm_cfg=None,
                 act_cfg=dict(type='ELU'),
                 encoder_type='stage1',
                 channel_factor=1.,
                 **kwargs):
        super().__init__()
        conv_module = self._conv_type[conv_type]
        channel_list_dict = dict(
            stage1=[32, 64, 64, 128, 128, 128],
            stage2_conv=[32, 32, 64, 64, 128, 128],
            stage2_attention=[32, 32, 64, 128, 128, 128])
        channel_list = channel_list_dict[encoder_type]
        channel_list = [int(x * channel_factor) for x in channel_list]
        kernel_size_list = [5, 3, 3, 3, 3, 3]
        stride_list = [1, 2, 1, 2, 1, 1]
        for i in range(6):
            ks = kernel_size_list[i]
            padding = (ks - 1) // 2
            self.add_module(
                f'enc{i + 1}',
                conv_module(
                    in_channels,
                    channel_list[i],
                    kernel_size=ks,
                    stride=stride_list[i],
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
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
        outputs = dict(out=x)
        return outputs
