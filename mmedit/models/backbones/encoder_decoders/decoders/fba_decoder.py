import torch
import torch.nn as nn
from mmcv.cnn import ConvWS2d

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class FBADecoder(nn.Module):

    def __init__(self,
                 batch_norm=False,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(FBADecoder, self).__init__()

        # Pyramid Pooling Module
        pool_scales = (1, 2, 3, 6)
        self.batch_norm = batch_norm

        self.ppm = []

        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvWS2d(2048, 256, kernel_size=1, bias=True),
                    norm(256, self.batch_norm), nn.LeakyReLU()))
        self.ppm = nn.ModuleList(self.ppm)

        # Follwed the author's implementation that
        # concatenate conv layers described in the supplementary
        # material between up operations
        self.conv_up1 = nn.Sequential(
            ConvWS2d(
                2048 + len(pool_scales) * 256,
                256,
                kernel_size=3,
                padding=1,
                bias=True), norm(256, self.batch_norm), nn.LeakyReLU(),
            ConvWS2d(256, 256, kernel_size=3, padding=1),
            norm(256, self.batch_norm), nn.LeakyReLU())

        self.conv_up2 = nn.Sequential(
            ConvWS2d(256 + 256, 256, kernel_size=3, padding=1, bias=True),
            norm(256, self.batch_norm), nn.LeakyReLU())

        # Keep the batch_norm here in case we may need to modify something
        if (self.batch_norm):
            d_up3 = 128
        else:
            d_up3 = 64

        self.conv_up3 = nn.Sequential(
            ConvWS2d(256 + d_up3, 64, kernel_size=3, padding=1, bias=True),
            norm(64, self.batch_norm), nn.LeakyReLU())

        self.conv_up4 = nn.Sequential(
            nn.Conv2d(64 + 3 + 3 + 2, 32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 7, kernel_size=1, padding=0, bias=True))

    def init_weights(self, pretrained=None):
        pass

    def forward(self, conv_out, img, two_chan_trimap, indices):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(
                nn.functional.interpolate(
                    pool_scale(conv5), (input_size[2], input_size[3]),
                    mode='bilinear',
                    align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_up1(ppm_out)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-4]), 1)

        x = self.conv_up2(x)
        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-5]), 1)
        x = self.conv_up3(x)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, conv_out[-6][:, :3], img, two_chan_trimap), 1)

        output = self.conv_up4(x)
        alpha = torch.clamp(output[:, 0][:, None], 0, 1)
        F = torch.sigmoid(output[:, 1:4])
        B = torch.sigmoid(output[:, 4:7])

        return alpha, F, B


def norm(dim, bn=False):
    if (bn):
        return nn.BatchNorm2d(dim)
    else:
        return nn.GroupNorm(32, dim)
