# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class TTSRDiscriminator(nn.Module):
    """A discriminator for TTSR.

    Args:
        in_channels (int): Channel number of inputs. Default: 3.
        in_size (int): Size of input image. Default: 160.
    """

    def __init__(self, in_channels=3, in_size=160):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 2, 1), nn.LeakyReLU(0.2))

        self.last = nn.Sequential(
            nn.Linear(in_size // 32 * in_size // 32 * 512, 1024),
            nn.LeakyReLU(0.2), nn.Linear(1024, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.last(x)

        return x

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
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
