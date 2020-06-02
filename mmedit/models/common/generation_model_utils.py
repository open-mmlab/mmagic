import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init, xavier_init
from torch.nn import init

from .norm import build_norm_layer


def generation_init_weights(module, init_type='normal', init_gain=0.02):
    """Default initialization of network weights for image generation.

    By default, we use 'normal' init, but xavier and kaiming might work
    better for some applications.

    Args:
        module (nn.Module): Module to be initialized.
        init_type (str): The name of an initialization method:
            normal | xavier | kaiming | orthogonal
        init_gain (float): Scaling factor for normal, xavier and
            orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                normal_init(m, 0.0, init_gain)
            elif init_type == 'xavier':
                xavier_init(m, gain=init_gain, distribution='normal')
            elif init_type == 'kaiming':
                kaiming_init(
                    m,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='normal')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented")
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)

    module.apply(init_func)


class GANImageBuffer(object):
    """This class implements an image buffer that stores previously
    generated images.

    This buffer allows us to update the discriminator using a history of
    generated images rather than the ones produced by the latest generator
    to reduce model oscillation.

    Args:
        buffer_size (int): The size of image buffer. If buffer_size = 0,
            no buffer will be created.
        buffer_ratio (float): The chance / possibility  to use the images
            previously stored in the buffer.
    """

    def __init__(self, buffer_size, buffer_ratio=0.5):
        self.buffer_size = buffer_size
        # create an empty buffer
        if self.buffer_size > 0:
            self.img_num = 0
            self.image_buffer = []
        self.buffer_ratio = buffer_ratio

    def query(self, images):
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


class UnetSkipConnectionBlock(nn.Module):
    """Construct a Unet submodule with skip connections.
    |-- downsampling -- |submodule| -- upsampling --|

    Args:
        outer_channels (int): Number of channels at the outer conv layer.
        inner_channels (int): Number of channels at the inner conv layer.
        in_channels (int): Number of channels in input images/features. If is
            None, equals to `outer_channels`. Default: None.
        submodule (UnetSkipConnectionBlock): Previously constructed submodule.
            Default: None.
        is_outermost (bool): Whether this module is the outermost module.
            Default: False.
        is_innermost (bool): Whether this module is the innermost module.
            Default: False.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
    """

    def __init__(self,
                 outer_channels,
                 inner_channels,
                 in_channels=None,
                 submodule=None,
                 is_outermost=False,
                 is_innermost=False,
                 norm_cfg=dict(type='BN'),
                 use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        # cannot be both outermost and innermost
        assert not (is_outermost and is_innermost), (
            "'is_outermost' and 'is_innermost' cannot be True"
            'at the same time.')
        self.is_outermost = is_outermost
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the unet skip connection block.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        kernel_size = 4
        stride = 2
        padding = 1

        if in_channels is None:
            in_channels = outer_channels
        down_conv = nn.Conv2d(
            in_channels,
            inner_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias)
        down_relu = nn.LeakyReLU(0.2, True)
        _, down_norm = build_norm_layer(norm_cfg, inner_channels)
        up_relu = nn.ReLU(True)
        _, up_norm = build_norm_layer(norm_cfg, outer_channels)

        if is_outermost:
            up_conv = nn.ConvTranspose2d(
                inner_channels * 2,
                outer_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [submodule] + up
        elif is_innermost:
            up_conv = nn.ConvTranspose2d(
                inner_channels,
                outer_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias)
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up
        else:
            up_conv = nn.ConvTranspose2d(
                inner_channels * 2,
                outer_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias)
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.is_outermost:
            return self.model(x)
        else:
            # add skip connections
            return torch.cat([x, self.model(x)], 1)


class ResidualBlockWithDropout(nn.Module):
    """Define a Residual Block with dropout layers.

    Ref:
    Deep Residual Learning for Image Recognition
    A residual block is a conv block with skip connections. The conv block is
    constructed in `build_conv_block` function, skip connection is implemented
    in `forward` function. In `build_conv_block` function, a dropout layer is
    added between two common conv modules.

    Args:
        channels (int): Number of channels in the conv layer.
        padding_mode (str): The name of padding layer:
            'reflect' | 'replicate' | 'zero'.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: True.
    """

    def __init__(self,
                 channels,
                 padding_mode,
                 norm_cfg=dict(type='BN'),
                 use_dropout=True):
        super(ResidualBlockWithDropout, self).__init__()
        self.conv_block = self.build_conv_block(channels, padding_mode,
                                                norm_cfg, use_dropout)

    def build_conv_block(self, channels, padding_mode, norm_cfg, use_dropout):
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the residual block with dropout layers.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        conv_block = []
        padding = 0
        if padding_mode == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_mode == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_mode == 'zero':
            padding = 1
        else:
            raise NotImplementedError(
                f"padding '{padding_mode}' is not implemented")

        _, norm = build_norm_layer(norm_cfg, channels)
        conv_block += [
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                bias=use_bias), norm,
            nn.ReLU(True)
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        padding = 0
        if padding_mode == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_mode == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_mode == 'zero':
            padding = 1
        else:
            raise NotImplementedError(
                f"padding '{padding_mode}' is not implemented")
        _, norm = build_norm_layer(norm_cfg, channels)
        conv_block += [
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                bias=use_bias), norm
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # add skip connections without final ReLU
        out = x + self.conv_block(x)
        return out
