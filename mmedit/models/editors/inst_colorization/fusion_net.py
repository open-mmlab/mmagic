# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmedit.registry import MODULES
from .weight_block import WeightBlock, get_norm_layer


@MODULES.register_module()
class FusionNet(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 norm_type,
                 use_tanh=True,
                 classification=True):
        super(FusionNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True

        norm_layer = get_norm_layer(norm_type)

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

        self.weight_layer = WeightBlock(64)

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

        self.weight_layer2 = WeightBlock(128)

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

        self.weight_layer3 = WeightBlock(256)

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

        self.weight_layer4 = WeightBlock(512)

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

        self.weight_layer5 = WeightBlock(512)

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

        self.weight_layer6 = WeightBlock(512)

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

        self.weight_layer7 = WeightBlock(512)

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

        self.weight_layer8_1 = WeightBlock(256)

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

        self.weight_layer8_2 = WeightBlock(256)

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

        self.weight_layer9_1 = WeightBlock(128)

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

        self.weight_layer9_2 = WeightBlock(128)

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

        self.weight_layer10_1 = WeightBlock(128)

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

        self.weight_layer10_2 = WeightBlock(128)

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

        self.weight_layerout = WeightBlock(2)

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

    def forward(self, input_A, input_B, mask_B, instance_feature,
                box_info_list):
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
