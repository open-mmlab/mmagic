import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util


class AdaFMNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, adafm_ksize=None):
        super(AdaFMNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 2, 1, bias=True)

        if adafm_ksize is not None:
            basic_block = functools.partial(ResidualBlock_adafm, nf=nf, adafm_ksize=adafm_ksize)
        else:
            basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        self.LR_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.adafm = nn.Conv2d(nf, nf, kernel_size=adafm_ksize, padding=(adafm_ksize - 1) // 2, groups=nf, bias=True)

        self.upconv = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        arch_util.initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last], 0.1)
        arch_util.initialize_weights([self.adafm], 0.)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        out = self.recon_trunk(fea)
        out = self.LR_conv(out)
        out = self.adafm(out) + out

        out = self.relu(self.pixel_shuffle(self.upconv(out + fea)))
        out = self.conv_last(self.relu(self.HRconv(out)))
        return out


class ResidualBlock_adafm(nn.Module):
    '''Residual block with AdaFM layers
    ---Conv-AdaFM-ReLU-Conv-AdaFM-+-
     |________________|
    '''

    def __init__(self, nf=64, adafm_ksize=1):
        super(ResidualBlock_adafm, self).__init__()
        padding = (adafm_ksize - 1) // 2
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.adafm1 = nn.Conv2d(nf, nf, kernel_size=adafm_ksize, padding=padding, groups=nf, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.adafm2 = nn.Conv2d(nf, nf, kernel_size=adafm_ksize, padding=padding, groups=nf, bias=True)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)
        arch_util.initialize_weights([self.adafm1, self.adafm2], 0.)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = F.relu(self.adafm1(out) + out, inplace=True)
        out = self.conv2(out)
        out = self.adafm2(out) + out
        return identity + out