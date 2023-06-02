import torch
import torch.nn as nn

from torchvision.models import densenet121

from mmagic.registry import MODELS


class FPNSegHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


@MODELS.register_module()
class FPNDense(nn.Module):

    def __init__(self, output_ch=3, num_filters=128, num_filters_fpn=256, pretrained=True):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = FPN(num_filters=num_filters_fpn, pretrained=pretrained)

        # The segmentation heads on top of the FPN

        self.head1 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNSegHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        return torch.tanh(final)

    def unfreeze(self):
        for param in self.fpn.parameters():
            param.requires_grad = True


class FPN(nn.Module):

    def __init__(self, num_filters=256, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        self.features = densenet121(pretrained=pretrained).features

        self.enc0 = nn.Sequential(self.features.conv0,
                                  self.features.norm0,
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
        self.lateral0 = nn.Conv2d(64, num_filters // 2, kernel_size=1, bias=False)

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
        map3 = lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest")
        map2 = lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest")
        map1 = lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest")

        return lateral0, map1, map2, map3, map4
