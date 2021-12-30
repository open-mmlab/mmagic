# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class SRCNN(nn.Module):
    """SRCNN network structure for image super resolution.

    SRCNN has three conv layers. For each layer, we can define the
    `in_channels`, `out_channels` and `kernel_size`.
    The input image will first be upsampled with a bicubic upsampler, and then
    super-resolved in the HR spatial size.

    Paper: Learning a Deep Convolutional Network for Image Super-Resolution.

    Args:
        channels (tuple[int]): A tuple of channel numbers for each layer
            including channels of input and output . Default: (3, 64, 32, 3).
        kernel_sizes (tuple[int]): A tuple of kernel sizes for each conv layer.
            Default: (9, 1, 5).
        upscale_factor (int): Upsampling factor. Default: 4.
    """

    def __init__(self,
                 channels=(3, 64, 32, 3),
                 kernel_sizes=(9, 1, 5),
                 upscale_factor=4):
        super().__init__()
        assert len(channels) == 4, ('The length of channel tuple should be 4, '
                                    f'but got {len(channels)}')
        assert len(kernel_sizes) == 3, (
            'The length of kernel tuple should be 3, '
            f'but got {len(kernel_sizes)}')
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(
            channels[0],
            channels[1],
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv2d(
            channels[1],
            channels[2],
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv2d(
            channels[2],
            channels[3],
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

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
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
