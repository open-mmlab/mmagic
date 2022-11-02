# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmedit.registry import MODULES
from .weight_layer import get_norm_layer


@MODULES.register_module()
class ColorizationNet(BaseModule):
    """Real-Time User-Guided Image Colorization with Learned Deep Priors. The
    backbone used for.

    https://arxiv.org/abs/1705.02999

    Codes adapted from 'https://github.com/ericsujw/InstColorization.git'
    'InstColorization/blob/master/models/networks.py#L108'

    Args:
        input_nc (int): input image channels
        output_nc (int): output image channels
        norm_type (str): instance normalization or batch normalization
        use_tanh (bool): Whether to use nn.Tanh() Default: True.
        classification (bool): backprop trunk using classification,
            otherwise use regression. Default: True
    """

    def __init__(self,
                 input_nc,
                 output_nc,
                 norm_type,
                 use_tanh=True,
                 classification=True):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification

        norm_layer = get_norm_layer(norm_type)

        use_bias = True

        # Conv1
        self.model1 = nn.Sequential(
            nn.Conv2d(
                input_nc,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(64),
        )

        # Conv2
        self.model2 = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(128),
        )

        # Conv3
        self.model3 = nn.Sequential(
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(256),
        )

        # Conv4
        self.model4 = nn.Sequential(
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(512),
        )

        # Conv5
        self.model5 = nn.Sequential(
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
            nn.ReLU(True),
            norm_layer(512),
        )

        # Conv6
        self.model6 = nn.Sequential(
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
            nn.ReLU(True),
            norm_layer(512),
        )

        # Conv7
        self.model7 = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(512),
        )

        # Conv8
        self.model8up = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.model3short8 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias)

        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(256),
        )

        # Conv9
        self.model9up = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.model2short9 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(128),
        )

        # Conv10
        self.model10up = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.model1short10 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias)

        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                bias=use_bias),
            nn.LeakyReLU(negative_slope=.2),
        )

        # classification output
        self.model_class = nn.Conv2d(
            256,
            529,
            kernel_size=1,
            padding=0,
            dilation=1,
            stride=1,
            bias=use_bias)

        # regression output
        model_out = [
            nn.Conv2d(
                128,
                2,
                kernel_size=1,
                padding=0,
                dilation=1,
                stride=1,
                bias=use_bias),
        ]
        if (use_tanh):
            model_out += [nn.Tanh()]
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_A, input_B, mask_B):
        """Forward function.

        Args:
            input_A (tensor): Channel of the image in lab color space
            input_B (tensor): Color patch
            mask_B (tensor): Color patch mask

        Returns:
            out_class (tensor): Classification output
            out_reg (tensor): Regression output
            feature_map (dict): The full-image feature
        """
        conv1_2 = self.model1(torch.cat((input_A, input_B, mask_B), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        if (self.classification):
            out_class = self.model_class(conv8_3)
            conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(
                conv2_2.detach())
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(
                conv1_2.detach())
        else:
            out_class = self.model_class(conv8_3.detach())
            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)

        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        feature_map = {}
        feature_map['conv1_2'] = conv1_2
        feature_map['conv2_2'] = conv2_2
        feature_map['conv3_3'] = conv3_3
        feature_map['conv4_3'] = conv4_3
        feature_map['conv5_3'] = conv5_3
        feature_map['conv6_3'] = conv6_3
        feature_map['conv7_3'] = conv7_3
        feature_map['conv8_up'] = conv8_up
        feature_map['conv8_3'] = conv8_3
        feature_map['conv9_up'] = conv9_up
        feature_map['conv9_3'] = conv9_3
        feature_map['conv10_up'] = conv10_up
        feature_map['conv10_2'] = conv10_2
        feature_map['out_reg'] = out_reg

        return (out_class, out_reg, feature_map)
