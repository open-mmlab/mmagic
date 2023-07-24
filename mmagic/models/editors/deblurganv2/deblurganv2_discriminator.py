# Copyright (c) OpenMMLab. All rights reserved.
import functools

import numpy as np
import torch.nn as nn

from mmagic.registry import MODELS
from .deblurganv2_util import get_norm_layer

model_list = ['DoubleGan', 'MultiScale', 'NoGan', 'PatchGan']


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self,
                 input_nc=3,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False,
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(
                ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class DicsriminatorTail(nn.Module):

    def __init__(self,
                 nf_mult,
                 n_layers,
                 ndf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_parallel=True):
        super(DicsriminatorTail, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence = [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(
                ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MultiScaleDiscriminator(nn.Module):

    def __init__(self,
                 input_nc=3,
                 ndf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_parallel=True):
        super(MultiScaleDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.scale_one = nn.Sequential(*sequence)
        self.first_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=3)
        nf_mult_prev = 4
        nf_mult = 8

        self.scale_two = nn.Sequential(
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=2,
                padding=padw,
                bias=use_bias), norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        nf_mult_prev = nf_mult
        self.second_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=4)
        self.scale_three = nn.Sequential(
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=2,
                padding=padw,
                bias=use_bias), norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        self.third_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=5)

    def forward(self, input):
        x = self.scale_one(input)
        x_1 = self.first_tail(x)
        x = self.scale_two(x)
        x_2 = self.second_tail(x)
        x = self.scale_three(x)
        x = self.third_tail(x)
        return [x_1, x_2, x]


def get_fullD(norm_layer):
    model_d = NLayerDiscriminator(
        n_layers=5,
        norm_layer=get_norm_layer(norm_type=norm_layer),
        use_sigmoid=False)
    return model_d


class DoubleGan(nn.Module):

    def __init__(self, norm_layer='instance', d_layers=3):
        super().__init__()
        self.patch_gan = NLayerDiscriminator(
            n_layers=d_layers,
            norm_layer=get_norm_layer(norm_type=norm_layer),
            use_sigmoid=False)
        # patch_gan = nn.DataParallel(patch_gan)
        self.full_gan = get_fullD(norm_layer)
        # full_gan = nn.DataParallel(full_gan)
        # self.model_d = dict(patch=patch_gan, full=full_gan)

    def forward(self, x):
        # d_full_gan = self.model_d['full'](x)
        d_full_gan_output = self.full_gan(x)
        # d_patch_gan = self.model_d['patch'](x)
        d_patch_gan_output = self.patch_gan(x)
        return [d_full_gan_output, d_patch_gan_output]


class PatchGan(nn.Module):

    def __init__(self, norm_layer='instance', d_layers=3):
        super().__init__()
        self.patch_gan = NLayerDiscriminator(
            n_layers=d_layers,
            norm_layer=get_norm_layer(norm_type=norm_layer),
            use_sigmoid=False)

    def forward(self, x):
        d_patch_gan_output = self.patch_gan(x)
        return d_patch_gan_output


class MultiScale(nn.Module):

    def __init__(self, norm_layer='instance', d_layers=3):
        super().__init__()
        self.model_d = MultiScaleDiscriminator(
            norm_layer=get_norm_layer(norm_type=norm_layer))

    def forward(self, x):
        result_d = self.model_d(x)
        return result_d


@MODELS.register_module()
class DeblurGanV2Discriminator:

    def __new__(cls, model, *args, **kwargs):
        if model == 'DoubleGan':
            return DoubleGan(*args, **kwargs)
        elif model == 'NoGan' or model == '':
            return super().__new__(cls)
        elif model == 'PatchGan':
            return PatchGan(*args, **kwargs)
        elif model == 'MultiScale':
            return MultiScale(*args, **kwargs)
        else:
            raise Exception('Discriminator model {} not found, '
                            'Please use the following models: '
                            '{}'.format(model, model_list))
