import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmedit.models.common import ConvModule, build_activation_layer
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module
class DeepFillDecoder(nn.Module):
    """Decoder used in DeepFill model.

    This implementation follows:
    Generative Image Inpainting with Contextual Attention

    Args:
        in_channels (int): The number of input channels..
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "elu" by default.
        out_act_cfg (dict): Config dict for output activation layer. Here, we
            provide commonly used `clamp` or `clip` operation.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=None,
                 act_cfg=dict(type='ELU'),
                 out_act_cfg=dict(type='clip', min=-1., max=1.)):
        super(DeepFillDecoder, self).__init__()
        self.with_out_activation = out_act_cfg is not None

        channel_list = [128, 128, 64, 64, 32, 16, 3]
        for i in range(7):
            if i == 6:
                act_cfg = None
            self.add_module(
                f'dec{i + 1}',
                ConvModule(
                    in_channels,
                    channel_list[i],
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channels = channel_list[i]

        if self.with_out_activation:
            act_type = out_act_cfg['type']
            if act_type in 'clip':
                act_cfg_ = copy.deepcopy(out_act_cfg)
                act_cfg_.pop('type')
                self.out_act = partial(torch.clamp, **act_cfg_)
            else:
                self.out_act = build_activation_layer(out_act_cfg)

    def forward(self, input_dict):
        if isinstance(input_dict, dict):
            x = input_dict['out']
        else:
            x = input_dict
        for i in range(7):
            x = getattr(self, f'dec{i + 1}')(x)
            if i == 1 or i == 3:
                x = F.interpolate(x, scale_factor=2)

        if self.with_out_activation:
            x = self.out_act(x)
        return x
