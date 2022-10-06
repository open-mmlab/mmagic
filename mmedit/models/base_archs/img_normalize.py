# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


class ImgNormalize(nn.Conv2d):
    """Normalize images with the given mean and std value.

    Based on Conv2d layer, can work in GPU.

    Args:
        pixel_range (float): Pixel range of feature.
        img_mean (Tuple[float]): Image mean of each channel.
        img_std (Tuple[float]): Image std of each channel.
        sign (int): Sign of bias. Default -1.
    """

    def __init__(self, pixel_range, img_mean, img_std, sign=-1):

        assert len(img_mean) == len(img_std)
        num_channels = len(img_mean)
        super().__init__(num_channels, num_channels, kernel_size=1)

        std = torch.Tensor(img_std)
        self.weight.data = torch.eye(num_channels).view(
            num_channels, num_channels, 1, 1)
        self.weight.data.div_(std.view(num_channels, 1, 1, 1))
        self.bias.data = sign * pixel_range * torch.Tensor(img_mean)
        self.bias.data.div_(std)

        self.weight.requires_grad = False
        self.bias.requires_grad = False
