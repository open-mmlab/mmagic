import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, xavier_init
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module
class BGMattingEncoder(nn.Module):
    """Encoder for Background Matting.

    This implementation follows:
    Background Matting: The World is Your Green Screen

    Args:
        in_channels_list (tuple[int]): Input channels of the encoder. It
            should be a list of int with length 3 indicating the channels of
            (image, bg, seg).
        mid_channels (int, optional): Base channels of the encoder. Most of the
            intermediate channels will be a multiple of it. Defaults to 64.
        context_channels (int, optional): Output channels of the conv layers in
            the context switching block. Defaults to 64.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
    """

    def __init__(self,
                 in_channels_list,
                 mid_channels=64,
                 context_channels=64,
                 norm_cfg=dict(type='BN')):
        super(BGMattingEncoder, self).__init__()
        if (not mmcv.is_tuple_of(in_channels_list, int)
                or len(in_channels_list) != 3):
            raise ValueError('in_channels_list must be a tuple of 3 int, '
                             f'but got {in_channels_list}')

        img_c, bg_c, seg_c = in_channels_list
        self.mid_channels = mid_channels
        self.context_channels = context_channels
        # number of dowmsampling of the encoders
        self.num_downsampling = 2

        img_enc = self.build_encoder(img_c, norm_cfg)
        self.img_enc1 = nn.Sequential(*img_enc[:2])
        self.img_enc2 = img_enc[2]

        self.bg_enc = nn.Sequential(*self.build_encoder(bg_c, norm_cfg))
        self.seg_enc = nn.Sequential(*self.build_encoder(seg_c, norm_cfg))

        # conv layers in context switching block
        self.bg_context_conv1 = self.build_context_conv(norm_cfg)
        self.seg_context_conv = self.build_context_conv(norm_cfg)
        # In the original code, bg_context_conv2 is named as comb_multi whereas
        # bg_context_conv1 is named as comb_back and seg_context_conv is named
        # as comb_seg. However, it appears that comb_multi is not used to
        # conv over motion cue but background feature due to a bug. Thus, we
        # rename this conv as bg_context_conv2.
        self.bg_context_conv2 = self.build_context_conv(norm_cfg)

        self.out_channels = (
            self.mid_channels * 2**self.num_downsampling +
            self.context_channels * 3)

    def build_encoder(self, in_channels, norm_cfg):
        enc = [
            ConvModule(
                in_channels,
                self.mid_channels,
                7,
                padding=3,
                bias=True,
                norm_cfg=norm_cfg,
                padding_mode='reflect')
        ]
        for i in range(self.num_downsampling):
            expand_ratio = 2**i
            enc.append(
                ConvModule(
                    self.mid_channels * expand_ratio,
                    self.mid_channels * expand_ratio * 2,
                    3,
                    stride=2,
                    padding=1,
                    bias=True,
                    norm_cfg=norm_cfg))
        return enc

    def build_context_conv(self, norm_cfg):
        expand_ratio = 2**self.num_downsampling
        context_conv = ConvModule(
            self.mid_channels * expand_ratio * 2,
            self.context_channels,
            1,
            norm_cfg=norm_cfg)
        return context_conv

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, gain=np.sqrt(2))
            elif isinstance(m, nn.BatchNorm2d):
                normal_init(m, 1.0, 0.2)

    def forward(self, img, bg, seg):
        img_feat1 = self.img_enc1(img)
        img_feat = self.img_enc2(img_feat1)

        bg_feat = self.bg_enc(bg)
        seg_feat = self.seg_enc(seg)

        bg_context_feat1 = self.bg_context_conv1(
            torch.cat([img_feat, bg_feat], dim=1))
        seg_context_feat = self.seg_context_conv(
            torch.cat([img_feat, seg_feat], dim=1))
        # background feature is used twice with different convs
        bg_context_feat2 = self.bg_context_conv2(
            torch.cat([img_feat, bg_feat], dim=1))

        context_feat = torch.cat(
            [bg_context_feat1, seg_context_feat, bg_context_feat2], dim=1)

        out = torch.cat([img_feat, context_feat], dim=1)
        return {'out': out, 'img_feat1': img_feat1}
