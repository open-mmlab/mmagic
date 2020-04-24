import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmedit.models.common import ConvModule, LinearModule
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module
class MultiLayerDiscriminator(nn.Module):
    """Multilayer Discriminator.

    This is a commonly used structure with stacked multiply convolution layers.

    Args:
        in_channels (int): Input channel of the first input convolution.
        max_channels (int): The maxinum channel number in this structure.
        fc_in_channels (int | None): Input dimension of the fully connected
            layer. If `fc_in_channels` is None, the fully connected layer will
            be removed.
        fc_out_channels (int): Output dimension of the fully connected layer.
        num_convs (int): The number of the stacked convolution layers.
        conv_cfg (dict): Config dict to build conv layer.
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        out_act_cfg (dict): Config dict for output activation, "relu" by
            default.
        kwargs (keyword arguments).
    """

    def __init__(self,
                 in_channels,
                 max_channels,
                 num_convs=5,
                 fc_in_channels=None,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='ReLU'),
                 with_spectral_norm=False,
                 **kwargs):
        super(MultiLayerDiscriminator, self).__init__()
        if fc_in_channels is not None:
            assert fc_in_channels > 0

        self.max_channels = max_channels
        self.with_fc = fc_in_channels is not None
        self.num_convs = num_convs
        self.with_out_act = out_act_cfg is not None

        cur_channels = in_channels
        for i in range(num_convs):
            out_ch = min(64 * 2**i, max_channels)
            if not self.with_fc and i == num_convs - 1:
                act_cfg_ = out_act_cfg
            else:
                act_cfg_ = act_cfg
            self.add_module(
                f'conv{i + 1}',
                ConvModule(
                    cur_channels,
                    out_ch,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg_,
                    with_spectral_norm=with_spectral_norm,
                    **kwargs))
            cur_channels = out_ch

        if self.with_fc:
            self.fc = LinearModule(
                fc_in_channels,
                fc_out_channels,
                bias=True,
                act_cfg=out_act_cfg,
                with_spectral_norm=with_spectral_norm)

    def forward(self, x):
        input_size = x.size()
        for i in range(self.num_convs):
            x = getattr(self, f'conv{i + 1}')(x)

        if self.with_fc:
            x = x.view(input_size[0], -1)
            x = self.fc(x)

        return x

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                # Here, we only initialize the module with fc layer since the
                # conv and norm layers has been intialized in `ConvModule`.
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
        else:
            raise TypeError('pretrained must be a str or None')
