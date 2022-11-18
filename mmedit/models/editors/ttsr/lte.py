# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule
from torchvision import models

from mmedit.models.base_archs import ImgNormalize
from mmedit.registry import MODELS


@MODELS.register_module()
class LTE(BaseModule):
    """Learnable Texture Extractor.

    Based on pretrained VGG19. Generate features in 3 levels.

    Args:
        requires_grad (bool): Require grad or not. Default: True.
        pixel_range (float): Pixel range of geature. Default: 1.
        load_pretrained_vgg (bool): Load pretrained VGG from torchvision.
            Default: True.
            Train: must load pretrained VGG.
            Eval: needn't load pretrained VGG, because we will load pretrained
            LTE.
    """

    def __init__(self,
                 requires_grad=True,
                 pixel_range=1.,
                 load_pretrained_vgg=True):
        super().__init__()

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * pixel_range, 0.224 * pixel_range,
                   0.225 * pixel_range)
        self.img_normalize = ImgNormalize(
            pixel_range=pixel_range, img_mean=vgg_mean, img_std=vgg_std)

        # use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(
            pretrained=load_pretrained_vgg).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, 3, h, w).

        Returns:
            Tuple[Tensor]: Forward results in 3 levels.
                x_level3: Forward results in level 3 (n, 256, h/4, w/4).
                x_level2: Forward results in level 2 (n, 128, h/2, w/2).
                x_level1: Forward results in level 1 (n, 64, h, w).
        """

        x = self.img_normalize(x)

        x_level1 = x = self.slice1(x)
        x_level2 = x = self.slice2(x)
        x_level3 = x = self.slice3(x)

        return [x_level3, x_level2, x_level1]
