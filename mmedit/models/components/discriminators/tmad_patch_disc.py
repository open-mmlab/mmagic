import torch.nn as nn
from mmedit.models.backbones import ResidualDilationBlock, TMADEncoder
from mmedit.models.common import ConvModule


class TMADPatchDiscriminator(nn.Module):
    """Patch discriminator used in TMAD.

    The details can be found in this paper:
    Texture Memory Augmented Deep Image Inpainting.

    Args:
        in_channels (int): Channels of input feature or image.
        channel_factor (int, optional): The output channels of the input conv
            module and the width of the encoder is computed by multiplying a
            constant with channel_factor. Defaults to 16.
        num_enc_blocks (int, optional): The number of downsampling blocks used
            in the encoder. Defaults to 3.
        num_dilation_blocks (int, optional): The number of residual dialtion
            blocks used in the dilation neck. Defaults to 3.
        residual_dialtion (int): Dilation used residual dilation neck.
            Defalut to 2.
        act_cfg (None | dict, optional): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.2).
        norm_cfg (None | dict, optional): Config dict for normalization layer.
            Defaults to None.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
        output_act_cfg (dict): Activation config for the output feature.
        kwargs (keyword arguments): Keyword arguments for `ConvModule`.
    """

    def __init__(self,
                 in_channels,
                 channel_factor,
                 num_enc_blocks=3,
                 num_dilation_blocks=2,
                 residual_dialtion=2,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 norm_cfg=None,
                 with_spectral_norm=False,
                 output_act_cfg=None,
                 **kwargs):
        super(TMADPatchDiscriminator, self).__init__()

        self.encoder = TMADEncoder(
            in_channels,
            channel_factor,
            num_blocks=num_enc_blocks,
            norm_cfg=norm_cfg,
            with_spectral_norm=with_spectral_norm,
            act_cfg=act_cfg,
            **kwargs)

        mid_channels = channel_factor * 2**num_dilation_blocks
        dilation_blocks_ = []

        for _ in range(num_dilation_blocks):
            dilation_blocks_.append(
                ResidualDilationBlock(
                    mid_channels,
                    mid_channels // 2,
                    dilation=residual_dialtion,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_spectral_norm=with_spectral_norm,
                    **kwargs))
            mid_channels = mid_channels // 2
        self.dilation_necks = nn.Sequential(*dilation_blocks_)

        self.output_conv = ConvModule(
            mid_channels,
            1,
            kernel_size=3,
            padding=1,
            act_cfg=output_act_cfg,
            norm_cfg=norm_cfg,
            with_spectral_norm=with_spectral_norm,
            **kwargs)

    def forward(self, x):
        x = self.encoder(x)['out']
        x = self.dilation_necks(x)
        x = self.output_conv(x)

        return x
