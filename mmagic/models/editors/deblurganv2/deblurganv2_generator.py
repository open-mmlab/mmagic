# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmagic.registry import MODELS
from .deblurganv2_util import MobileNetV2, get_norm_layer, inceptionresnetv2

backbone_list = ['FPNInception', 'FPNMobileNet', 'FPNInceptionSimple']


class FPNHead(nn.Module):
    """Head for FPNInception,FPNInceptionSimple and FPNMobilenet."""

    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(
            num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(
            num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
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
        """Unfreeze params."""
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
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
        """Unfreeze params."""
        self.fpn.unfreeze()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
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
        """Unfreeze params."""
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
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
        """unfreeze the fpn network."""
        self.fpn.unfreeze()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
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
        """Unfreeze params."""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
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
        """unfreeze the fpn network."""
        self.fpn.unfreeze()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
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


@MODELS.register_module()
class DeblurGanV2Generator:
    """Defines the generator for DeblurGanv2 with the specified arguments..

    Args:
        model (Str): Type of the generator  model
    """

    def __new__(cls, backbone, *args, **kwargs):
        if backbone == 'FPNInception':
            return FPNInception(*args, **kwargs)
        elif backbone == 'FPNMobileNet':
            return FPNMobileNet(*args, **kwargs)
        elif backbone == 'FPNInceptionSimple':
            return FPNInceptionSimple(*args, **kwargs)
        else:
            raise Exception('Generator model {} not found, '
                            'Please use the following models: '
                            '{}'.format(backbone, backbone_list))
