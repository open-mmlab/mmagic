import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, xavier_init
from mmedit.models.registry import COMPONENTS


class BasicBlock(nn.Module):
    """Basic residual block for BGMattingDecoder.

    Args:
        channels (int): Input channels and output channels of the conv layer.
        norm_cfg (dict): Config dict for normalization layer.
        padding_mode (str): Padding mode of the conv layers. See `ConvModule`
            for more information.
        use_dropout (bool): Whether use dropout after the first conv layer.
    """

    def __init__(self, channels, norm_cfg, padding_mode, use_dropout):
        super(BasicBlock, self).__init__()

        self.conv1 = self.build_conv_layer(channels, norm_cfg,
                                           dict(type='ReLU'), padding_mode)
        self.conv2 = self.build_conv_layer(channels, norm_cfg, None,
                                           padding_mode)

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)

    def build_conv_layer(self, channels, norm_cfg, act_cfg, padding_mode):
        return ConvModule(
            channels,
            channels,
            3,
            padding=1,
            bias=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            padding_mode=padding_mode)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        out = out + x
        return out


@COMPONENTS.register_module()
class BGMattingDecoder(nn.Module):
    """Decoder for Background Matting.

    This implementation follows:
    Background Matting: The World is Your Green Screen

    Args:
        in_channels (int): Input channels of the decoder.
        mid_channels (int, optional): Base channels of the decoder. Most of the
            intermediate channels will be a multiple of it. Defaults to 64.
        fg_channels (int, optional): Predicted foreground channels.
            Defaults to 3.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
        padding_mode (str, optional): Padding mode of the conv layers. See
            `ConvModule` for more information. Defaults to 'reflect'.
        num_shared_dec_blocks (int, optional): Number of shared decoder
            residual basic block. Defaults to 7.
        num_sep_dec_blocks (int, optional): Nubmer of decoder residual basic
            block for alpha decoder and foreground decoder. Defaults to 3.
        use_dropout (bool, optional): Whether use dropout after the first conv
            layer. Defaults to False.
    """

    def __init__(self,
                 in_channels,
                 mid_channels=64,
                 fg_channels=3,
                 norm_cfg=dict(type='BN'),
                 padding_mode='reflect',
                 num_shared_dec_blocks=7,
                 num_sep_dec_blocks=3,
                 use_dropout=False):
        super(BGMattingDecoder, self).__init__()
        self.mid_channels = mid_channels
        # number of dowmsample of the encoders
        self.num_downsampling = 2

        # build shared decoder part
        expand_ratio = 2**self.num_downsampling
        self.shared_conv = ConvModule(
            in_channels, mid_channels * expand_ratio, 1, norm_cfg=norm_cfg)
        self.shared_res_dec = self.build_res_blocks(num_shared_dec_blocks,
                                                    norm_cfg, padding_mode,
                                                    use_dropout)

        # build alpha decoder part
        self.alpha_res_dec = self.build_res_blocks(num_sep_dec_blocks,
                                                   norm_cfg, padding_mode,
                                                   use_dropout)
        alpha_conv_layers = []
        for i in range(self.num_downsampling):
            expand_ratio = 2**(self.num_downsampling - i)
            alpha_conv_layers += [
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True),
                ConvModule(
                    mid_channels * expand_ratio,
                    mid_channels * expand_ratio // 2,
                    3,
                    padding=1,
                    bias=True,
                    norm_cfg=norm_cfg)
            ]
        alpha_conv_layers.append(
            ConvModule(
                mid_channels,
                1,
                7,
                padding=3,
                act_cfg=dict(type='Tanh'),
                padding_mode='reflect'))
        self.alpha_conv_layers = nn.Sequential(*alpha_conv_layers)

        # build fg decoder part
        self.fg_res_dec = self.build_res_blocks(num_sep_dec_blocks, norm_cfg,
                                                padding_mode, use_dropout)
        self.fg_conv_layers1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvModule(
                mid_channels * 4,
                mid_channels * 2,
                3,
                padding=1,
                bias=True,
                norm_cfg=norm_cfg))
        self.fg_conv_layers2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvModule(
                mid_channels * 4,
                mid_channels,
                3,
                padding=1,
                bias=True,
                norm_cfg=norm_cfg),
            ConvModule(
                mid_channels,
                fg_channels,
                7,
                padding=3,
                act_cfg=None,
                padding_mode='reflect'))

    def build_res_blocks(self, num_blocks, norm_cfg, padding_mode,
                         use_dropout):
        expand_ratio = 2**self.num_downsampling
        res_blocks = []
        for i in range(num_blocks):
            res_blocks.append(
                BasicBlock(self.mid_channels * expand_ratio, norm_cfg,
                           padding_mode, use_dropout))
        return nn.Sequential(*res_blocks)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, gain=np.sqrt(2))
            elif isinstance(m, nn.BatchNorm2d):
                normal_init(m, 1.0, 0.2)

    def forward(self, inputs):
        out = inputs['out']
        img_feat1 = inputs['img_feat1']

        shared_out = self.shared_conv(out)
        shared_out = self.shared_res_dec(shared_out)

        alpha_out = self.alpha_res_dec(shared_out)
        alpha_out = self.alpha_conv_layers(alpha_out)

        fg_out = self.fg_res_dec(shared_out)
        fg_out = self.fg_conv_layers1(fg_out)
        fg_out = self.fg_conv_layers2(torch.cat([fg_out, img_feat1], dim=1))

        return {'alpha_out': alpha_out, 'fg_out': fg_out}
