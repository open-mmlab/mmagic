# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, normal_init, update_init_info
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmagic.registry import MODELS


@MODELS.register_module()
class DCGANDiscriminator(BaseModule):
    """Discriminator for DCGAN.

    Implementation Details for DCGAN architecture:

    #. Adopt convolution in the discriminator;
    #. Use batchnorm in the discriminator except for the input and final \
       output layer;
    #. Use LeakyReLU in the discriminator in addition to the output layer.

    Args:
        input_scale (int): The scale of the input image.
        output_scale (int): The final scale of the convolutional feature.
        out_channels (int): The channel number of the final output layer.
        in_channels (int, optional): The channel number of the input image.
            Defaults to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Defaults to 128.
        default_norm_cfg (dict, optional): Norm config for all of layers
            except for the final output layer. Defaults to ``dict(type='BN')``.
        default_act_cfg (dict, optional): Activation config for all of layers
            except for the final output layer. Defaults to
            ``dict(type='ReLU')``.
        out_act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='Tanh')``.
        pretrained (str, optional): Path for the pretrained model. Default to
            ``None``.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self,
                 input_scale,
                 output_scale,
                 out_channels,
                 in_channels=3,
                 base_channels=128,
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='LeakyReLU'),
                 out_act_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.out_channels = out_channels
        self.base_channels = base_channels

        # the number of times for downsampling
        self.num_downsamples = int(np.log2(input_scale // output_scale))

        # build up downsampling backbone (excluding the output layer)
        downsamples = []
        curr_channels = in_channels
        for i in range(self.num_downsamples):
            # remove norm for the first conv
            norm_cfg_ = None if i == 0 else default_norm_cfg
            in_ch = in_channels if i == 0 else base_channels * 2**(i - 1)

            downsamples.append(
                ConvModule(
                    in_ch,
                    base_channels * 2**i,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=norm_cfg_,
                    act_cfg=default_act_cfg))
            curr_channels = base_channels * 2**i

        self.downsamples = nn.Sequential(*downsamples)

        # define output layer
        self.output_layer = ConvModule(
            curr_channels,
            out_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=None,
            act_cfg=out_act_cfg)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Fake or real image tensor.

        Returns:
            torch.Tensor: Prediction for the reality of the input image.
        """

        n = x.shape[0]
        x = self.downsamples(x)
        x = self.output_layer(x)

        # reshape to a flatten feature
        return x.view(n, -1)

    def init_weights(self):
        """Init weights for models.

        We just use the initialization method proposed in the original paper.
        """
        if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
            super().init_weights()
            return
        for m in self.modules():
            module_name = m.__class__.__name__
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                normal_init(m, 0, 0.02)
                init_info = (f'{module_name} belongs to (nn.Conv2d and '
                             'nn.ConvTranspose2d), initialize by normal '
                             'distribution with 0 mean and 0.02 std.')
            elif isinstance(m, _BatchNorm):
                nn.init.normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
                init_info = (f'{module_name} is BatchNorm, initialize weight '
                             'by normal discribution with unit mean and zero '
                             'std, and initialize bias as 0.')
            # save init info
            update_init_info(m, init_info)
