import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmedit.models.common import build_norm_layer, generation_init_weights
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module
class PatchDiscriminator(nn.Module):
    """A PatchGAN discriminator.

    Args:
        in_channels (int): Number of channels in input images.
        base_channels (int): Number of filters at the first conv layer.
            Default: 64.
        num_conv (int): Number of stacked intermediate convs (excluding input
            and output conv). Default: 3.
        norm_cfg (dict): Config dict to build norm layer.
            Default: `dict(type='BN')`.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
                Default: 0.02.
    """

    def __init__(self,
                 in_channels,
                 base_channels=64,
                 num_conv=3,
                 norm_cfg=dict(type='BN'),
                 init_cfg=dict(type='normal', gain=0.02)):
        super(PatchDiscriminator, self).__init__()
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the patch discriminator.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        kernel_size = 4
        padding = 1

        # input layer
        sequence = [
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        # stacked intermediate layers,
        # gradually increasing the number of filters
        multiple_now = 1
        multiple_prev = 1
        for n in range(1, num_conv):
            multiple_prev = multiple_now
            multiple_now = min(2**n, 8)
            _, norm = build_norm_layer(norm_cfg, base_channels * multiple_now)
            sequence += [
                nn.Conv2d(
                    base_channels * multiple_prev,
                    base_channels * multiple_now,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=use_bias), norm,
                nn.LeakyReLU(0.2, True)
            ]
        multiple_prev = multiple_now
        multiple_now = min(2**num_conv, 8)
        _, norm = build_norm_layer(norm_cfg, base_channels * multiple_now)
        sequence += [
            nn.Conv2d(
                base_channels * multiple_prev,
                base_channels * multiple_now,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=use_bias), norm,
            nn.LeakyReLU(0.2, True)
        ]

        # output one-channel prediction map
        sequence += [
            nn.Conv2d(
                base_channels * multiple_now,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
        ]

        self.model = nn.Sequential(*sequence)
        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, input):
        return self.model(input)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
