# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import LinearModule
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class MultiLayerDiscriminator(nn.Module):
    """Multilayer Discriminator.

    This is a commonly used structure with stacked multiply convolution layers.

    Args:
        in_channels (int): Input channel of the first input convolution.
        max_channels (int): The maximum channel number in this structure.
        num_conv (int): Number of stacked intermediate convs (including input
            conv but excluding output conv).
        fc_in_channels (int | None): Input dimension of the fully connected
            layer. If `fc_in_channels` is None, the fully connected layer will
            be removed.
        fc_out_channels (int): Output dimension of the fully connected layer.
        kernel_size (int): Kernel size of the conv modules. Default to 5.
        conv_cfg (dict): Config dict to build conv layer.
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        out_act_cfg (dict): Config dict for output activation, "relu" by
            default.
        with_input_norm (bool): Whether add normalization after the input conv.
            Default to True.
        with_out_convs (bool): Whether add output convs to the discriminator.
            The output convs contain two convs. The first out conv has the same
            setting as the intermediate convs but a stride of 1 instead of 2.
            The second out conv is a conv similar to the first out conv but
            reduces the number of channels to 1 and has no activation layer.
            Default to False.
        with_spectral_norm (bool): Whether use spectral norm after the conv
            layers. Default to False.
        kwargs (keyword arguments).
    """

    def __init__(self,
                 in_channels,
                 max_channels,
                 num_convs=5,
                 fc_in_channels=None,
                 fc_out_channels=1024,
                 kernel_size=5,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='ReLU'),
                 with_input_norm=True,
                 with_out_convs=False,
                 with_spectral_norm=False,
                 **kwargs):
        super().__init__()
        if fc_in_channels is not None:
            assert fc_in_channels > 0

        self.max_channels = max_channels
        self.with_fc = fc_in_channels is not None
        self.num_convs = num_convs
        self.with_out_act = out_act_cfg is not None
        self.with_out_convs = with_out_convs

        cur_channels = in_channels
        for i in range(num_convs):
            out_ch = min(64 * 2**i, max_channels)
            norm_cfg_ = norm_cfg
            act_cfg_ = act_cfg
            if i == 0 and not with_input_norm:
                norm_cfg_ = None
            elif (i == num_convs - 1 and not self.with_fc
                  and not self.with_out_convs):
                norm_cfg_ = None
                act_cfg_ = out_act_cfg
            self.add_module(
                f'conv{i + 1}',
                ConvModule(
                    cur_channels,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg_,
                    act_cfg=act_cfg_,
                    with_spectral_norm=with_spectral_norm,
                    **kwargs))
            cur_channels = out_ch

        if self.with_out_convs:
            cur_channels = min(64 * 2**(num_convs - 1), max_channels)
            out_ch = min(64 * 2**num_convs, max_channels)
            self.add_module(
                f'conv{num_convs + 1}',
                ConvModule(
                    cur_channels,
                    out_ch,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_spectral_norm=with_spectral_norm,
                    **kwargs))
            self.add_module(
                f'conv{num_convs + 2}',
                ConvModule(
                    out_ch,
                    1,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    act_cfg=None,
                    with_spectral_norm=with_spectral_norm,
                    **kwargs))

        if self.with_fc:
            self.fc = LinearModule(
                fc_in_channels,
                fc_out_channels,
                bias=True,
                act_cfg=out_act_cfg,
                with_spectral_norm=with_spectral_norm)

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w') or (n, c).
        """
        input_size = x.size()
        # out_convs has two additional ConvModules
        num_convs = self.num_convs + 2 * self.with_out_convs
        for i in range(num_convs):
            x = getattr(self, f'conv{i + 1}')(x)

        if self.with_fc:
            x = x.view(input_size[0], -1)
            x = self.fc(x)

        return x

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                # Here, we only initialize the module with fc layer since the
                # conv and norm layers has been initialized in `ConvModule`.
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
        else:
            raise TypeError('pretrained must be a str or None')
