# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class ResidualBlockWithDropout(nn.Module):
    """Define a Residual Block with dropout layers.

    Ref:
    Deep Residual Learning for Image Recognition

    A residual block is a conv block with skip connections. A dropout layer is
    added between two common conv modules.

    Args:
        channels (int): Number of channels in the conv layer.
        padding_mode (str): The name of padding layer:
            'reflect' | 'replicate' | 'zeros'.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: True.
    """

    def __init__(self,
                 channels,
                 padding_mode,
                 norm_cfg=dict(type='BN'),
                 use_dropout=True):
        super().__init__()
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the residual block with dropout layers.
        # Only for IN, use bias to follow cyclegan's original implementation.
        use_bias = norm_cfg['type'] == 'IN'

        block = [
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode)
        ]

        if use_dropout:
            block += [nn.Dropout(0.5)]

        block += [
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None,
                padding_mode=padding_mode)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward function. Add skip connections without final ReLU.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = x + self.block(x)
        return out


class GANImageBuffer:
    """This class implements an image buffer that stores previously generated
    images.

    This buffer allows us to update the discriminator using a history of
    generated images rather than the ones produced by the latest generator
    to reduce model oscillation.

    Args:
        buffer_size (int): The size of image buffer. If buffer_size = 0,
            no buffer will be created.
        buffer_ratio (float): The chance / possibility  to use the images
            previously stored in the buffer. Default: 0.5.
    """

    def __init__(self, buffer_size, buffer_ratio=0.5):
        self.buffer_size = buffer_size
        # create an empty buffer
        if self.buffer_size > 0:
            self.img_num = 0
            self.image_buffer = []
        self.buffer_ratio = buffer_ratio

    def query(self, images):
        """Query current image batch using a history of generated images.

        Args:
            images (Tensor): Current image batch without history information.
        """
        if self.buffer_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # if the buffer is not full, keep inserting current images
            if self.img_num < self.buffer_size:
                self.img_num = self.img_num + 1
                self.image_buffer.append(image)
                return_images.append(image)
            else:
                use_buffer = np.random.random() < self.buffer_ratio
                # by self.buffer_ratio, the buffer will return a previously
                # stored image, and insert the current image into the buffer
                if use_buffer:
                    random_id = np.random.randint(0, self.buffer_size)
                    image_tmp = self.image_buffer[random_id].clone()
                    self.image_buffer[random_id] = image
                    return_images.append(image_tmp)
                # by (1 - self.buffer_ratio), the buffer will return the
                # current image
                else:
                    return_images.append(image)
        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images
