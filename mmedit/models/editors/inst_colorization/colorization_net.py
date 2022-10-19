# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from .weight_block import get_norm_layer

from mmengine.model import BaseModule
from mmedit.registry import MODULES



@MODULES.register_module()
class ColorizationNet(BaseModule):

    def __init__(self,
                 input_nc,
                 output_nc,
                 norm_type,
                 use_tanh=True,
                 classification=True):
        super(ColorizationNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification

        norm_layer = get_norm_layer(norm_type)

        use_bias = True

        # Conv1
        # model1=[nn.ReflectionPad2d(1),]
        model1 = [
            nn.Conv2d(
                input_nc,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias),
        ]
        # model1+=[norm_layer(64),]
        model1 += [
            nn.ReLU(True),
        ]
        # model1+=[nn.ReflectionPad2d(1),]
        model1 += [
            nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        model1 += [
            nn.ReLU(True),
        ]
        model1 += [
            norm_layer(64),
        ]
        # add a subsampling operation

        # Conv2
        # model2=[nn.ReflectionPad2d(1),]
        model2 = [
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # model2+=[norm_layer(128),]
        model2 += [
            nn.ReLU(True),
        ]
        # model2+=[nn.ReflectionPad2d(1),]
        model2 += [
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        model2 += [
            nn.ReLU(True),
        ]
        model2 += [
            norm_layer(128),
        ]
        # add a subsampling layer operation

        # Conv3
        # model3=[nn.ReflectionPad2d(1),]
        model3 = [
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # model3+=[norm_layer(256),]
        model3 += [
            nn.ReLU(True),
        ]
        # model3+=[nn.ReflectionPad2d(1),]
        model3 += [
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # model3+=[norm_layer(256),]
        model3 += [
            nn.ReLU(True),
        ]
        # model3+=[nn.ReflectionPad2d(1),]
        model3 += [
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            norm_layer(256),
        ]
        # add a subsampling layer operation

        # Conv4
        # model47=[nn.ReflectionPad2d(1),]
        model4 = [
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # model4+=[norm_layer(512),]
        model4 += [
            nn.ReLU(True),
        ]
        # model4+=[nn.ReflectionPad2d(1),]
        model4 += [
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # model4+=[norm_layer(512),]
        model4 += [
            nn.ReLU(True),
        ]
        # model4+=[nn.ReflectionPad2d(1),]
        model4 += [
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            norm_layer(512),
        ]

        # Conv5
        # model47+=[nn.ReflectionPad2d(2),]
        model5 = [
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
        ]
        # model5+=[norm_layer(512),]
        model5 += [
            nn.ReLU(True),
        ]
        # model5+=[nn.ReflectionPad2d(2),]
        model5 += [
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
        ]
        # model5+=[norm_layer(512),]
        model5 += [
            nn.ReLU(True),
        ]
        # model5+=[nn.ReflectionPad2d(2),]
        model5 += [
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            norm_layer(512),
        ]

        # Conv6
        # model6+=[nn.ReflectionPad2d(2),]
        model6 = [
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
        ]
        # model6+=[norm_layer(512),]
        model6 += [
            nn.ReLU(True),
        ]
        # model6+=[nn.ReflectionPad2d(2),]
        model6 += [
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
        ]
        # model6+=[norm_layer(512),]
        model6 += [
            nn.ReLU(True),
        ]
        # model6+=[nn.ReflectionPad2d(2),]
        model6 += [
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                bias=use_bias),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            norm_layer(512),
        ]

        # Conv7
        # model47+=[nn.ReflectionPad2d(1),]
        model7 = [
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # model7+=[norm_layer(512),]
        model7 += [
            nn.ReLU(True),
        ]
        # model7+=[nn.ReflectionPad2d(1),]
        model7 += [
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # model7+=[norm_layer(512),]
        model7 += [
            nn.ReLU(True),
        ]
        # model7+=[nn.ReflectionPad2d(1),]
        model7 += [
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            norm_layer(512),
        ]

        # Conv7
        model8up = [
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)
        ]

        # model3short8=[nn.ReflectionPad2d(1),]
        model3short8 = [
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]

        # model47+=[norm_layer(256),]
        model8 = [
            nn.ReLU(True),
        ]
        # model8+=[nn.ReflectionPad2d(1),]
        model8 += [
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # model8+=[norm_layer(256),]
        model8 += [
            nn.ReLU(True),
        ]
        # model8+=[nn.ReflectionPad2d(1),]
        model8 += [
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        model8 += [
            nn.ReLU(True),
        ]
        model8 += [
            norm_layer(256),
        ]

        # Conv9
        model9up = [
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
        ]

        # model2short9=[nn.ReflectionPad2d(1),]
        model2short9 = [
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # add the two feature maps above

        # model9=[norm_layer(128),]
        model9 = [
            nn.ReLU(True),
        ]
        # model9+=[nn.ReflectionPad2d(1),]
        model9 += [
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        model9 += [
            nn.ReLU(True),
        ]
        model9 += [
            norm_layer(128),
        ]

        # Conv10
        model10up = [
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
        ]

        # model1short10=[nn.ReflectionPad2d(1),]
        model1short10 = [
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        # add the two feature maps above

        # model10=[norm_layer(128),]
        model10 = [
            nn.ReLU(True),
        ]
        # model10+=[nn.ReflectionPad2d(1),]
        model10 += [
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                bias=use_bias),
        ]
        model10 += [
            nn.LeakyReLU(negative_slope=.2),
        ]

        # classification output
        model_class = [
            nn.Conv2d(
                256,
                529,
                kernel_size=1,
                padding=0,
                dilation=1,
                stride=1,
                bias=use_bias),
        ]

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

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[
            nn.Upsample(scale_factor=4, mode='nearest'),
        ])
        self.softmax = nn.Sequential(*[
            nn.Softmax(dim=1),
        ])

    def forward(self, input_A, input_B, mask_B):
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
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
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

        return (out_reg, feature_map)

        # return (out_class, out_reg)
