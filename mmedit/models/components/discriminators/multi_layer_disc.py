import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmedit.models.common import ConvModule
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module
class MultiLayerDiscriminator(nn.Module):
    """Multilayer Discriminator.

    This is a commonly used structure with stacked multiply convolution layers.

    Args:
        in_channels (int): Input channel of the first input convolution.
        max_channels (int): The maxinum channel number in this structure.
        fc_in_channels (int): Input dimension of the fully connected layer.
            If `fc_in_channels` is zero, the fully connected layer will be
            removed.
        fc_out_channels (int): Output dimension of the fully connected layer.
        num_convs (int): The number of the stacked convolution layers.
        conv_cfg (dict): Config dict to build conv layer.
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
    """

    def __init__(self,
                 in_channels,
                 max_channels,
                 fc_in_channels=0,
                 fc_out_channels=1024,
                 num_convs=5,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(MultiLayerDiscriminator, self).__init__()

        self.max_channels = max_channels
        self.with_fc = fc_in_channels > 0
        self.num_convs = num_convs

        cur_channels = in_channels
        for i in range(num_convs):
            out_ch = min(64 * 2**i, max_channels)
            self.add_module(
                f'conv{i + 1}',
                ConvModule(
                    cur_channels,
                    out_ch,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            cur_channels = out_ch

        if self.with_fc:
            self.fc = nn.Linear(fc_in_channels, fc_out_channels, bias=True)
            self.fc_act = nn.ReLU()

    def forward(self, x):
        input_size = x.size()
        for i in range(self.num_convs):
            x = getattr(self, f'conv{i + 1}')(x)

        if self.with_fc:
            x = x.view(input_size[0], -1)
            x = self.fc(x)
            x = self.fc_act(x)

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
