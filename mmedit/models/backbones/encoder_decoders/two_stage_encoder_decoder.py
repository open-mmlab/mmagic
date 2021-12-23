# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init
from mmcv.runner import auto_fp16, load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmedit.models.builder import build_backbone, build_component
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class DeepFillEncoderDecoder(nn.Module):
    """Two-stage encoder-decoder structure used in DeepFill model.

    The details are in:
    Generative Image Inpainting with Contextual Attention

    Args:
        stage1 (dict): Config dict for building stage1 model. As
            DeepFill model uses Global&Local model as baseline in first stage,
            the stage1 model can be easily built with `GLEncoderDecoder`.
        stage2 (dict): Config dict for building stage2 model.
        return_offset (bool): Whether to return offset feature in contextual
            attention module. Default: False.
    """

    def __init__(self,
                 stage1=dict(
                     type='GLEncoderDecoder',
                     encoder=dict(type='DeepFillEncoder'),
                     decoder=dict(type='DeepFillDecoder', in_channels=128),
                     dilation_neck=dict(
                         type='GLDilationNeck',
                         in_channels=128,
                         act_cfg=dict(type='ELU'))),
                 stage2=dict(type='DeepFillRefiner'),
                 return_offset=False):
        super().__init__()
        self.stage1 = build_backbone(stage1)
        self.stage2 = build_component(stage2)

        self.return_offset = return_offset

        # support fp16
        self.fp16_enabled = False

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): This input tensor has the shape of (n, 5, h, w).
                In channel dimension, we concatenate [masked_img, ones, mask]
                as DeepFillv1 models do.

        Returns:
            tuple[torch.Tensor]: The first two item is the results from first \
                and second stage. If set `return_offset` as True, the offset \
                will be returned as the third item.
        """
        input_x = x.clone()
        masked_img = input_x[:, :3, ...]
        mask = input_x[:, -1:, ...]
        x = self.stage1(x)
        stage1_res = x.clone()
        stage1_img = stage1_res * mask + masked_img * (1. - mask)
        stage2_input = torch.cat([stage1_img, input_x[:, 3:, ...]], dim=1)
        stage2_res, offset = self.stage2(stage2_input, mask)

        if self.return_offset:
            return stage1_res, stage2_res, offset

        return stage1_res, stage2_res

    # TODO: study the effects of init functions
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
                if isinstance(m, nn.Conv2d):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, (_BatchNorm, nn.InstanceNorm2d)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
