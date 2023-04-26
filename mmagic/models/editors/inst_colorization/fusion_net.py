# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmagic.registry import MODELS
from .weight_layer import WeightLayer, get_norm_layer


@MODELS.register_module()
class FusionNet(BaseModule):
    """Instance-aware Image Colorization.

    https://arxiv.org/abs/2005.10825

    Codes adapted from 'https://github.com/ericsujw/InstColorization.git'
    'InstColorization/blob/master/models/networks.py#L314'
    FusionNet: the full image model with weight layer for fusion.

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

        self.weight_layer = WeightLayer(64)

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

        self.weight_layer2 = WeightLayer(128)

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

        self.weight_layer3 = WeightLayer(256)

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

        self.weight_layer4 = WeightLayer(512)

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

        self.weight_layer5 = WeightLayer(512)

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

        self.weight_layer6 = WeightLayer(512)

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

        self.weight_layer7 = WeightLayer(512)

        # Conv8
        self.model8up = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.model3short8 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias)

        self.weight_layer8_1 = WeightLayer(256)

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

        self.weight_layer8_2 = WeightLayer(256)

        # Conv9
        self.model9up = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.model2short9 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias)

        self.weight_layer9_1 = WeightLayer(128)

        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(128),
        )

        self.weight_layer9_2 = WeightLayer(128)

        # Conv10
        self.model10up = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.model1short10 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias)

        self.weight_layer10_1 = WeightLayer(128)

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

        self.weight_layer10_2 = WeightLayer(128)

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

        self.weight_layerout = WeightLayer(2)

        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_A, input_B, mask_B, instance_feature,
                box_info_list):
        """Forward function.

        Args:
            input_A (tensor): Channel of the image in lab color space
            input_B (tensor): Color patch
            mask_B (tensor): Color patch mask
            instance_feature (dict): A bunch of instance features
            box_info_list (list): Bounding box information corresponding
                to the instance

        Returns:
            out_reg (tensor): Regression output
        """
        conv1_2 = self.model1(torch.cat((input_A, input_B, mask_B), dim=1))
        conv1_2 = self.weight_layer(instance_feature['conv1_2'], conv1_2,
                                    box_info_list[0])

        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv2_2 = self.weight_layer2(instance_feature['conv2_2'], conv2_2,
                                     box_info_list[1])

        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv3_3 = self.weight_layer3(instance_feature['conv3_3'], conv3_3,
                                     box_info_list[2])

        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv4_3 = self.weight_layer4(instance_feature['conv4_3'], conv4_3,
                                     box_info_list[3])

        conv5_3 = self.model5(conv4_3)
        conv5_3 = self.weight_layer5(instance_feature['conv5_3'], conv5_3,
                                     box_info_list[3])

        conv6_3 = self.model6(conv5_3)
        conv6_3 = self.weight_layer6(instance_feature['conv6_3'], conv6_3,
                                     box_info_list[3])

        conv7_3 = self.model7(conv6_3)
        conv7_3 = self.weight_layer7(instance_feature['conv7_3'], conv7_3,
                                     box_info_list[3])

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_up = self.weight_layer8_1(instance_feature['conv8_up'], conv8_up,
                                        box_info_list[2])

        conv8_3 = self.model8(conv8_up)
        conv8_3 = self.weight_layer8_2(instance_feature['conv8_3'], conv8_3,
                                       box_info_list[2])

        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_up = self.weight_layer9_1(instance_feature['conv9_up'], conv9_up,
                                        box_info_list[1])

        conv9_3 = self.model9(conv9_up)
        conv9_3 = self.weight_layer9_2(instance_feature['conv9_3'], conv9_3,
                                       box_info_list[1])

        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_up = self.weight_layer10_1(instance_feature['conv10_up'],
                                          conv10_up, box_info_list[0])

        conv10_2 = self.model10(conv10_up)
        conv10_2 = self.weight_layer10_2(instance_feature['conv10_2'],
                                         conv10_2, box_info_list[0])

        out_reg = self.model_out(conv10_2)
        return out_reg
