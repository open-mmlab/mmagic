# Copyright (c) OpenMMLab. All rights reserved.
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

from mmagic.registry import MODELS
from .deblurganv2_util import (get_norm_layer, inceptionresnetv2,
                               se_resnext50_32x4d)

model_list = [
    'FPNInception', 'UNetSEResNext', 'ResnetGenerator', 'FPNMobileNet',
    'FPNDense', 'FPNInceptionSimple'
]


class FPNHead(nn.Module):
    """Head for FPNInception,FPNInceptionSimple and FPNMobilenet."""

    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(
            num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(
            num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPN_inception(nn.Module):

    def __init__(self, norm_layer, num_filter=256, pretrained='imagenet'):
        """Creates an `FPN` instance for feature extraction.

        Args:
          num_filter: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        self.inception = inceptionresnetv2(
            num_classes=1000, pretrained=pretrained)

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,
            self.inception.conv2d_2b,
            self.inception.maxpool_3a,
        )  # 64
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )  # 1088
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        )  # 2080
        self.td1 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1),
            norm_layer(num_filter), nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1),
            norm_layer(num_filter), nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1),
            norm_layer(num_filter), nn.ReLU(inplace=True))
        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, num_filter, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filter, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filter, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filter, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(
            32, num_filter // 2, kernel_size=1, bias=False)

        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0)  # 256

        enc2 = self.enc2(enc1)  # 512

        enc3 = self.enc3(enc2)  # 1024

        enc4 = self.enc4(enc3)  # 2048

        # Lateral connections

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = self.td1(
            lateral3 +
            nn.functional.upsample(map4, scale_factor=2, mode='nearest'))
        map2 = self.td2(
            F.pad(lateral2, pad, 'reflect') +
            nn.functional.upsample(map3, scale_factor=2, mode='nearest'))
        map1 = self.td3(
            lateral1 +
            nn.functional.upsample(map2, scale_factor=2, mode='nearest'))
        return F.pad(lateral0, pad1, 'reflect'), map1, map2, map3, map4


class FPNInception(nn.Module):
    """Feature Pyramid Network (FPN) with four feature maps of resolutions 1/4,
    1/8, 1/16, 1/32 and `num_filter` filters for all feature maps."""

    def __init__(self,
                 norm_layer,
                 output_ch=3,
                 num_filter=128,
                 num_filter_fpn=256):
        super().__init__()

        norm_layer = get_norm_layer(norm_type=norm_layer)
        self.fpn = FPN_inception(
            num_filter=num_filter_fpn, norm_layer=norm_layer)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filter_fpn, num_filter, num_filter)
        self.head2 = FPNHead(num_filter_fpn, num_filter, num_filter)
        self.head3 = FPNHead(num_filter_fpn, num_filter, num_filter)
        self.head4 = FPNHead(num_filter_fpn, num_filter, num_filter)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filter, num_filter, kernel_size=3, padding=1),
            norm_layer(num_filter),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter // 2, kernel_size=3, padding=1),
            norm_layer(num_filter // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(
            num_filter // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(
            self.head4(map4), scale_factor=8, mode='nearest')
        map3 = nn.functional.upsample(
            self.head3(map3), scale_factor=4, mode='nearest')
        map2 = nn.functional.upsample(
            self.head2(map2), scale_factor=2, mode='nearest')
        map1 = nn.functional.upsample(
            self.head1(map1), scale_factor=1, mode='nearest')

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(
            smoothed, scale_factor=2, mode='nearest')
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(
            smoothed, scale_factor=2, mode='nearest')

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1)


class FPN_inceptionsimple(nn.Module):

    def __init__(self, norm_layer, num_filters=256):
        """Creates an `FPN` instance for feature extraction.

        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        self.inception = inceptionresnetv2(
            num_classes=1000, pretrained='imagenet')

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,
            self.inception.conv2d_2b,
            self.inception.maxpool_3a,
        )  # 64
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )  # 1088
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        )  # 2080

        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(
            32, num_filters // 2, kernel_size=1, bias=False)

        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0)  # 256

        enc2 = self.enc2(enc1)  # 512

        enc3 = self.enc3(enc2)  # 1024

        enc4 = self.enc4(enc3)  # 2048

        # Lateral connections

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = lateral3 + nn.functional.upsample(
            map4, scale_factor=2, mode='nearest')
        map2 = F.pad(lateral2, pad, 'reflect') + nn.functional.upsample(
            map3, scale_factor=2, mode='nearest')
        map1 = lateral1 + nn.functional.upsample(
            map2, scale_factor=2, mode='nearest')
        return F.pad(lateral0, pad1, 'reflect'), map1, map2, map3, map4


class FPNInceptionSimple(nn.Module):
    """Feature Pyramid Network (FPN) with four feature maps of resolutions 1/4,
    1/8, 1/16, 1/32 and `num_filter` filters for all feature maps."""

    def __init__(self,
                 norm_layer,
                 output_ch=3,
                 num_filter=128,
                 num_filter_fpn=256):
        super().__init__()

        norm_layer = get_norm_layer(norm_type=norm_layer)
        self.fpn = FPN_inceptionsimple(
            num_filter=num_filter_fpn, norm_layer=norm_layer)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filter_fpn, num_filter, num_filter)
        self.head2 = FPNHead(num_filter_fpn, num_filter, num_filter)
        self.head3 = FPNHead(num_filter_fpn, num_filter, num_filter)
        self.head4 = FPNHead(num_filter_fpn, num_filter, num_filter)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filter, num_filter, kernel_size=3, padding=1),
            norm_layer(num_filter),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter // 2, kernel_size=3, padding=1),
            norm_layer(num_filter // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(
            num_filter // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(
            self.head4(map4), scale_factor=8, mode='nearest')
        map3 = nn.functional.upsample(
            self.head3(map3), scale_factor=4, mode='nearest')
        map2 = nn.functional.upsample(
            self.head2(map2), scale_factor=2, mode='nearest')
        map1 = nn.functional.upsample(
            self.head1(map1), scale_factor=1, mode='nearest')

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(
            smoothed, scale_factor=2, mode='nearest')
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(
            smoothed, scale_factor=2, mode='nearest')

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1)


class FPNSegHead(nn.Module):
    """Head for FPNDense."""

    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(
            num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(
            num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPN_dense(nn.Module):

    def __init__(self, num_filters=256, pretrained=True):
        """Creates an `FPN` instance for feature extraction.

        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        self.features = densenet121(pretrained=pretrained).features

        self.enc0 = nn.Sequential(self.features.conv0, self.features.norm0,
                                  self.features.relu0)
        self.pool0 = self.features.pool0
        self.enc1 = self.features.denseblock1  # 256
        self.enc2 = self.features.denseblock2  # 512
        self.enc3 = self.features.denseblock3  # 1024
        self.enc4 = self.features.denseblock4  # 2048
        self.norm = self.features.norm5  # 2048

        self.tr1 = self.features.transition1  # 256
        self.tr2 = self.features.transition2  # 512
        self.tr3 = self.features.transition3  # 1024

        self.lateral4 = nn.Conv2d(1024, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1024, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(512, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(256, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(
            64, num_filters // 2, kernel_size=1, bias=False)

    def forward(self, x):
        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        pooled = self.pool0(enc0)

        enc1 = self.enc1(pooled)  # 256
        tr1 = self.tr1(enc1)

        enc2 = self.enc2(tr1)  # 512
        tr2 = self.tr2(enc2)

        enc3 = self.enc3(tr2)  # 1024
        tr3 = self.tr3(enc3)

        enc4 = self.enc4(tr3)  # 2048
        enc4 = self.norm(enc4)

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # Top-down pathway

        map4 = lateral4
        map3 = lateral3 + nn.functional.upsample(
            map4, scale_factor=2, mode='nearest')
        map2 = lateral2 + nn.functional.upsample(
            map3, scale_factor=2, mode='nearest')
        map1 = lateral1 + nn.functional.upsample(
            map2, scale_factor=2, mode='nearest')

        return lateral0, map1, map2, map3, map4


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FPNDense(nn.Module):
    """Feature Pyramid Network (FPN) with four feature maps of resolutions 1/4,
    1/8, 1/16, 1/32 and `num_filter` filters for all feature maps."""

    def __init__(self,
                 output_ch=3,
                 num_filter=128,
                 num_filter_fpn=256,
                 pretrained=True):
        super().__init__()

        self.fpn = FPN_dense(num_filter=num_filter_fpn, pretrained=pretrained)

        # The segmentation heads on top of the FPN

        self.head1 = FPNSegHead(num_filter_fpn, num_filter, num_filter)
        self.head2 = FPNSegHead(num_filter_fpn, num_filter, num_filter)
        self.head3 = FPNSegHead(num_filter_fpn, num_filter, num_filter)
        self.head4 = FPNSegHead(num_filter_fpn, num_filter, num_filter)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filter, num_filter, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filter // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(
            num_filter // 2, output_ch, kernel_size=3, padding=1)

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(
            self.head4(map4), scale_factor=8, mode='nearest')
        map3 = nn.functional.upsample(
            self.head3(map3), scale_factor=4, mode='nearest')
        map2 = nn.functional.upsample(
            self.head2(map2), scale_factor=2, mode='nearest')
        map1 = nn.functional.upsample(
            self.head1(map1), scale_factor=1, mode='nearest')

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(
            smoothed, scale_factor=2, mode='nearest')
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(
            smoothed, scale_factor=2, mode='nearest')

        final = self.final(smoothed)
        return torch.tanh(final)

    def unfreeze(self):
        for param in self.fpn.parameters():
            param.requires_grad = True


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(
                            input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(
                            input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class FPN_mobilenet(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=None):
        """Creates an `FPN` instance for feature extraction.

        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        net = MobileNetV2(n_class=1000)

        if pretrained:
            # Load weights into the project directory
            if torch.cuda.is_available():
                state_dict = torch.load(
                    pretrained)  # add map_location='cpu' if no gpu
            else:
                state_dict = torch.load(pretrained, map_location='cpu')
            net.load_state_dict(state_dict)
        self.features = net.features

        self.enc0 = nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[2:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:11])
        self.enc4 = nn.Sequential(*self.features[11:16])

        self.td1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters), nn.ReLU(inplace=True))

        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(
            16, num_filters // 2, kernel_size=1, bias=False)

        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0)  # 256

        enc2 = self.enc2(enc1)  # 512

        enc3 = self.enc3(enc2)  # 1024

        enc4 = self.enc4(enc3)  # 2048

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(
            lateral3 +
            nn.functional.upsample(map4, scale_factor=2, mode='nearest'))
        map2 = self.td2(
            lateral2 +
            nn.functional.upsample(map3, scale_factor=2, mode='nearest'))
        map1 = self.td3(
            lateral1 +
            nn.functional.upsample(map2, scale_factor=2, mode='nearest'))
        return lateral0, map1, map2, map3, map4


class FPNMobileNet(nn.Module):

    def __init__(self,
                 norm_layer,
                 output_ch=3,
                 num_filter=64,
                 num_filter_fpn=128,
                 pretrained=None):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        norm_layer = get_norm_layer(norm_type=norm_layer)
        self.fpn = FPN_mobilenet(
            num_filters=num_filter_fpn,
            norm_layer=norm_layer,
            pretrained=pretrained)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filter_fpn, num_filter, num_filter)
        self.head2 = FPNHead(num_filter_fpn, num_filter, num_filter)
        self.head3 = FPNHead(num_filter_fpn, num_filter, num_filter)
        self.head4 = FPNHead(num_filter_fpn, num_filter, num_filter)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filter, num_filter, kernel_size=3, padding=1),
            norm_layer(num_filter),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter // 2, kernel_size=3, padding=1),
            norm_layer(num_filter // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(
            num_filter // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(
            self.head4(map4), scale_factor=8, mode='nearest')
        map3 = nn.functional.upsample(
            self.head3(map3), scale_factor=4, mode='nearest')
        map2 = nn.functional.upsample(
            self.head2(map2), scale_factor=2, mode='nearest')
        map1 = nn.functional.upsample(
            self.head1(map1), scale_factor=1, mode='nearest')

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(
            smoothed, scale_factor=2, mode='nearest')
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(
            smoothed, scale_factor=2, mode='nearest')

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1)


class ResnetGenerator(nn.Module):
    """Resnet generator for Deblurganv2."""

    def __init__(self,
                 input_nc=3,
                 output_nc=3,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 n_blocks=6,
                 use_parallel=True,
                 learn_residual=True,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias)
            ]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
                         use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ConvRelu(nn.Module):

    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)  # conv3x3
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class UNetSEResNext(nn.Module):
    """UNetSEResNext generator for Deblurganv2."""

    def __init__(self,
                 num_classes=3,
                 num_filters=32,
                 pretrained=True,
                 is_deconv=True):
        super().__init__()
        self.num_classes = num_classes
        pretrain = 'imagenet' if pretrained is True else None
        self.encoder = se_resnext50_32x4d(
            num_classes=1000, pretrained=pretrain)
        bottom_channel_nr = 2048

        self.conv1 = self.encoder.layer0
        # self.se_e1 = SCSEBlock(64)
        self.conv2 = self.encoder.layer1
        # self.se_e2 = SCSEBlock(64 * 4)
        self.conv3 = self.encoder.layer2
        # self.se_e3 = SCSEBlock(128 * 4)
        self.conv4 = self.encoder.layer3
        # self.se_e4 = SCSEBlock(256 * 4)
        self.conv5 = self.encoder.layer4
        # self.se_e5 = SCSEBlock(512 * 4)

        self.center = DecoderCenter(bottom_channel_nr, num_filters * 8 * 2,
                                    num_filters * 8, False)

        self.dec5 = DecoderBlockV(bottom_channel_nr + num_filters * 8,
                                  num_filters * 8 * 2, num_filters * 2,
                                  is_deconv)
        # self.se_d5 = SCSEBlock(num_filters * 2)
        self.dec4 = DecoderBlockV(bottom_channel_nr // 2 + num_filters * 2,
                                  num_filters * 8, num_filters * 2, is_deconv)
        # self.se_d4 = SCSEBlock(num_filters * 2)
        self.dec3 = DecoderBlockV(bottom_channel_nr // 4 + num_filters * 2,
                                  num_filters * 4, num_filters * 2, is_deconv)
        # self.se_d3 = SCSEBlock(num_filters * 2)
        self.dec2 = DecoderBlockV(bottom_channel_nr // 8 + num_filters * 2,
                                  num_filters * 2, num_filters * 2, is_deconv)
        # self.se_d2 = SCSEBlock(num_filters * 2)
        self.dec1 = DecoderBlockV(num_filters * 2, num_filters,
                                  num_filters * 2, is_deconv)
        # self.se_d1 = SCSEBlock(num_filters * 2)
        self.dec0 = ConvRelu(num_filters * 10, num_filters * 2)
        self.final = nn.Conv2d(num_filters * 2, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        # conv1 = self.se_e1(conv1)
        conv2 = self.conv2(conv1)
        # conv2 = self.se_e2(conv2)
        conv3 = self.conv3(conv2)
        # conv3 = self.se_e3(conv3)
        conv4 = self.conv4(conv3)
        # conv4 = self.se_e4(conv4)
        conv5 = self.conv5(conv4)
        # conv5 = self.se_e5(conv5)

        center = self.center(conv5)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        # dec5 = self.se_d5(dec5)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        # dec4 = self.se_d4(dec4)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        # dec3 = self.se_d3(dec3)
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        # dec2 = self.se_d2(dec2)
        dec1 = self.dec1(dec2)
        # dec1 = self.se_d1(dec1)

        f = torch.cat((
            dec1,
            F.upsample(
                dec2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(
                dec3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(
                dec4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(
                dec5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)

        dec0 = self.dec0(f)

        return self.final(dec0)


class DecoderBlockV(nn.Module):

    def __init__(self,
                 in_channels,
                 middle_channels,
                 out_channels,
                 is_deconv=True):
        super(DecoderBlockV, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1), nn.InstanceNorm2d(out_channels, affine=False),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class DecoderCenter(nn.Module):

    def __init__(self,
                 in_channels,
                 middle_channels,
                 out_channels,
                 is_deconv=True):
        super(DecoderCenter, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """Parameters for Deconvolution were chosen to avoid artifacts,
            following link https://distill.pub/2016/deconv-checkerboard/"""

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1), nn.InstanceNorm2d(out_channels, affine=False),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels))

    def forward(self, x):
        return self.block(x)


@MODELS.register_module()
class DeblurGanV2Generator:
    """Defines the generator for DeblurGanv2 with the specified arguments..

    Args:
        model (Str): Type of the generator  model
    """

    def __new__(cls, model, *args, **kwargs):
        if model == 'FPNInception':
            return FPNInception(*args, **kwargs)
        elif model == 'UNetSEResNext':
            return UNetSEResNext(*args, **kwargs)
        elif model == 'ResnetGenerator':
            return ResnetGenerator(*args, **kwargs)
        elif model == 'FPNMobileNet':
            return FPNMobileNet(*args, **kwargs)
        elif model == 'FPNDense':
            return FPNDense(*args, **kwargs)
        elif model == 'FPNInceptionSimple':
            return FPNInceptionSimple(*args, **kwargs)
        else:
            raise Exception('Generator model {} not found, '
                            'Please use the following models: '
                            '{}'.format(model, model_list))
