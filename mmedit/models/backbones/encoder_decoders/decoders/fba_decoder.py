# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class FBADecoder(nn.Module):
    """Decoder for FBA matting.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self,
                 pool_scales,
                 in_channels,
                 channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super().__init__()

        assert isinstance(pool_scales, (list, tuple))
        # Pyramid Pooling Module
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.batch_norm = False

        self.ppm = []
        for scale in self.pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    *(ConvModule(
                        self.in_channels,
                        self.channels,
                        kernel_size=1,
                        bias=True,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg).children())))
        self.ppm = nn.ModuleList(self.ppm)

        # Followed the author's implementation that
        # concatenate conv layers described in the supplementary
        # material between up operations
        self.conv_up1 = nn.Sequential(*(list(
            ConvModule(
                self.in_channels + len(pool_scales) * 256,
                self.channels,
                padding=1,
                kernel_size=3,
                bias=True,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg).children()) + list(
                    ConvModule(
                        self.channels,
                        self.channels,
                        padding=1,
                        bias=True,
                        kernel_size=3,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg).children())))

        self.conv_up2 = nn.Sequential(*(list(
            ConvModule(
                self.channels * 2,
                self.channels,
                padding=1,
                kernel_size=3,
                bias=True,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg).children())))

        if (self.norm_cfg['type'] == 'BN'):
            d_up3 = 128
        else:
            d_up3 = 64

        self.conv_up3 = nn.Sequential(*(list(
            ConvModule(
                self.channels + d_up3,
                64,
                padding=1,
                kernel_size=3,
                bias=True,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg).children())))

        self.unpool = nn.MaxUnpool2d(2, stride=2)

        conv_up4_list = list(
            ConvModule(
                64 + 3 + 3 + 2,
                32,
                padding=1,
                kernel_size=3,
                bias=True,
                act_cfg=self.act_cfg).children())
        conv_up4_list += list(
            ConvModule(
                32,
                16,
                padding=1,
                kernel_size=3,
                bias=True,
                act_cfg=self.act_cfg).children())
        conv_up4_list += list(
            ConvModule(
                16, 7, padding=0, kernel_size=1, bias=True,
                act_cfg=None).children())
        self.conv_up4 = nn.Sequential(*conv_up4_list)

    def init_weights(self, pretrained=None):
        """Init weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (dict): Output dict of FbaEncoder.
        Returns:
            Tensor: Predicted alpha, fg and bg of the current batch.
        """

        conv_out = inputs['conv_out']
        img = inputs['merged']
        two_channel_trimap = inputs['two_channel_trimap']
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(
                nn.functional.interpolate(
                    pool_scale(conv5), (input_size[2], input_size[3]),
                    mode='bilinear',
                    align_corners=self.align_corners))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_up1(ppm_out)

        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=self.align_corners)

        x = torch.cat((x, conv_out[-4]), 1)

        x = self.conv_up2(x)
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=self.align_corners)

        x = torch.cat((x, conv_out[-5]), 1)
        x = self.conv_up3(x)

        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=self.align_corners)

        x = torch.cat((x, conv_out[-6][:, :3], img, two_channel_trimap), 1)
        output = self.conv_up4(x)
        alpha = torch.clamp(output[:, 0:1], 0, 1)
        F = torch.sigmoid(output[:, 1:4])
        B = torch.sigmoid(output[:, 4:7])

        return alpha, F, B
