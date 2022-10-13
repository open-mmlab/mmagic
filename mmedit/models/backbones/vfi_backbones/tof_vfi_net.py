# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import flow_warp
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class BasicModule(nn.Module):
    """Basic module of SPyNet.

    Note that unlike the common spynet architecture, the basic module
    here could contain batch normalization.

    Args:
        norm_cfg (dict | None): Config of normalization.
    """

    def __init__(self, norm_cfg):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Estimated flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    """SPyNet architecture.

    Note that this implementation is specifically for TOFlow. It differs from
    the common SPyNet in the following aspects:
        1. The basic modules in paper of TOFlow contain BatchNorm.
        2. Normalization and denormalization are not done here, as
            they are done in TOFlow.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network
    Code reference:
        https://github.com/Coldog2333/pytoflow

    Args:
        norm_cfg (dict | None): Config of normalization.
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, norm_cfg, pretrained=None):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [BasicModule(norm_cfg) for _ in range(4)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

    def forward(self, ref, supp):
        """
        Args:
            ref (Tensor): Reference image with shape of (b, 3, h, w).
            supp: The supporting image to be warped: (b, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (b, 2, h, w).
        """

        num_batches, _, h, w = ref.size()
        ref = [ref]
        supp = [supp]

        # generate downsampled frames
        for _ in range(3):
            ref.insert(
                0,
                F.avg_pool2d(
                    input=ref[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.insert(
                0,
                F.avg_pool2d(
                    input=supp[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))

        # flow computation
        flow = ref[0].new_zeros(num_batches, 2, h // 16, w // 16)
        for i in range(4):
            flow_up = F.interpolate(
                input=flow,
                scale_factor=2,
                mode='bilinear',
                align_corners=True) * 2.0
            flow = flow_up + self.basic_module[i](
                torch.cat([
                    ref[i],
                    flow_warp(
                        supp[i],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow


class ResNet(nn.Module):
    """ResNet architecture.

    Three-layers ResNet/ResBlock
    """

    def __init__(self):
        super().__init__()

        self.res_block = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0))

    def forward(self, frames):
        """
        Args:
            frames (Tensor): Tensor with shape of (b, 2, 3, h, w).

        Returns:
            Tensor: Interpolated frame with shape of (b, 3, h, w).
        """

        num_batches, _, _, h, w = frames.size()
        average = frames.mean(dim=1)
        x = frames.view(num_batches, -1, h, w)
        result = self.res_block(x)
        return result + average


@BACKBONES.register_module()
class TOFlowVFINet(nn.Module):
    """PyTorch implementation of TOFlow for video frame interpolation.

    Paper: Xue et al., Video Enhancement with Task-Oriented Flow, IJCV 2018
    Code reference:

    1. https://github.com/anchen1011/toflow
    2. https://github.com/Coldog2333/pytoflow

    Args:
        rgb_mean (list[float]):  Image mean in RGB orders.
            Default: [0.485, 0.456, 0.406]
        rgb_std (list[float]):  Image std in RGB orders.
            Default: [0.229, 0.224, 0.225]
        flow_cfg (dict): Config of SPyNet.
            Default: dict(norm_cfg=None, pretrained=None)
    """

    def __init__(self,
                 rgb_mean=[0.485, 0.456, 0.406],
                 rgb_std=[0.229, 0.224, 0.225],
                 flow_cfg=dict(norm_cfg=None, pretrained=None)):
        super().__init__()

        # The mean and std are for img with range (0, 1)
        self.register_buffer('mean', torch.Tensor(rgb_mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(rgb_std).view(1, -1, 1, 1))

        # flow estimation module
        self.spynet = SPyNet(**flow_cfg)

        # reconstruction module
        self.resnet = ResNet()

    def normalize(self, img):
        """Normalize the input image.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Normalized image.
        """
        return (img - self.mean) / self.std

    def denormalize(self, img):
        """Denormalize the output image.

        Args:
            img (Tensor): Output image.

        Returns:
            Tensor: Denormalized image.
        """
        return img * self.std + self.mean

    def spatial_padding(self, inputs):
        """Apply pdding spatially.

        Since the SPyNet module in TOFlow requires that the resolution is a
        multiple of 16, we apply padding to the input LR images if their
        resolution is not divisible by 16.

        Args:
            inputs (Tensor): Input sequence with shape (n, 2, c, h, w).

        Returns:
            Tensor: Padded sequence with shape (n, 2, c, h_pad, w_pad).
        """
        n, t, c, h, w = inputs.size()

        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        if pad_h != 0 or pad_w != 0:
            # padding
            inputs = inputs.view(-1, c, h, w)
            inputs = F.pad(inputs, [0, pad_w, 0, pad_h], mode='reflect')
            return inputs.view(n, t, c, h + pad_h, w + pad_w)
        else:
            return inputs

    def forward(self, inputs):
        """
        Args:
            inputs: Input frames with shape of (b, 2, 3, h, w).

        Returns:
            Tensor: Interpolated frame with shape of (b, 3, h, w).
        """

        h_ori, w_ori = inputs.shape[-2:]
        inputs = self.spatial_padding(inputs=inputs)
        num_batches, num_frames, c, h, w = inputs.size()

        inputs = self.normalize(inputs.view(-1, c, h, w))
        inputs = inputs.view(num_batches, num_frames, c, h, w)

        flow_10 = self.spynet(inputs[:, 0], inputs[:, 1]).permute(0, 2, 3, 1)
        flow_01 = self.spynet(inputs[:, 1], inputs[:, 0]).permute(0, 2, 3, 1)

        wrap_frame0 = flow_warp(inputs[:, 0], flow_01 / 2)
        wrap_frame1 = flow_warp(inputs[:, 1], flow_10 / 2)

        wrap_frames = torch.stack([wrap_frame0, wrap_frame1], dim=1)
        output = self.resnet(wrap_frames)
        output = self.denormalize(output)

        return output[..., :h_ori, :w_ori]

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
