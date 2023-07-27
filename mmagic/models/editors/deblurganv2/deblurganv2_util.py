# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import absolute_import, division, print_function
import functools
import math
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.utils import model_zoo
from torchvision import models
from torchvision.transforms import transforms

pretrained_settings = {
    'inceptionresnetv2': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/'
            'inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/'
            'inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2))

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(
                128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(
                160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2))

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2))

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2))

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(
                192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(
                224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):
    """Define a inceptionresnetv2 model."""

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        # Special attributes
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(
            32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17),
            Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17),
            Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17),
            Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10), Block17(scale=0.10), Block17(scale=0.10),
            Block17(scale=0.10), Block17(scale=0.10), Block17(scale=0.10),
            Block17(scale=0.10), Block17(scale=0.10), Block17(scale=0.10),
            Block17(scale=0.10), Block17(scale=0.10), Block17(scale=0.10),
            Block17(scale=0.10), Block17(scale=0.10), Block17(scale=0.10),
            Block17(scale=0.10), Block17(scale=0.10), Block17(scale=0.10),
            Block17(scale=0.10), Block17(scale=0.10))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20), Block8(scale=0.20), Block8(scale=0.20),
            Block8(scale=0.20), Block8(scale=0.20), Block8(scale=0.20),
            Block8(scale=0.20), Block8(scale=0.20), Block8(scale=0.20))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        """Get network features.

        Args:
            input (torch.tensor): You can directly input a ``torch.Tensor``.
        """
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        """Get features logits.

        Args:
            features (torch.tensor): You can directly input a ``torch.Tensor``.
        """
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        x = self.features(input)
        x = self.logits(x)
        return x


def inceptionresnetv2(num_classes=1000, pretrained='imagenet'):
    """return a inceptionresnetv2 network."""
    if pretrained:
        settings = pretrained_settings['inceptionresnetv2'][pretrained]
        model = InceptionResNetV2(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']

        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionResNetV2(num_classes=num_classes)
    return model


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
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


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
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
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


def get_norm_layer(norm_type='instance'):
    """Returns a norm layer of the specified type.

    Args:
        norm_type (Str): norm layer type
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer


class ImagePool():

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.sample_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = deque()

    def add(self, images):
        if self.pool_size == 0:
            return images
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
            else:
                self.images.popleft()
                self.images.append(image)

    def query(self):
        if len(self.images) > self.sample_size:
            return_images = list(random.sample(self.images, self.sample_size))
        else:
            return_images = list(self.images)
        return torch.cat(return_images, 0)


class ContentLoss():
    """Defined a criterion to calculator loss."""

    def initialize(self, loss):
        """initialize the criterion."""
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        """Get generator loss."""
        return self.criterion(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        """Get generator loss."""
        return self.get_loss(fakeIm, realIm)


class PerceptualLoss():
    """Defined a criterion to calculator loss."""

    def contentFunc(self):
        """Defined contentFunc."""
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        if torch.cuda.is_available():
            cnn = cnn.cuda()
        else:
            cnn = cnn

        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        """Initialize criterion."""
        with torch.no_grad():
            self.criterion = loss
            self.contentFunc = self.contentFunc()
            self.transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_loss(self, fakeIm, realIm):
        """Get generator loss."""
        fakeIm = (fakeIm + 1) / 2.0
        realIm = (realIm + 1) / 2.0
        fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
        realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        """Get generator loss."""
        return self.get_loss(fakeIm, realIm)


class GANLoss(nn.Module):
    """Defined a GANLoss network to calculator loss."""

    def __init__(self,
                 use_l1=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, input, target_is_real):
        """Get ganloss result."""
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DiscLoss(nn.Module):
    """Defined a criterion to calculator loss."""

    def name(self):
        """return name of criterion."""
        return 'DiscLoss'

    def __init__(self):
        super(DiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_AB_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        """First, G(A) should fake the discriminator."""
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, fakeB, realB):
        """Fake stop backprop to the generator by detaching fake_B Generated
        Image Disc Output should be close to zero."""
        self.pred_fake = net.forward(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net.forward(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        """Get discriminator loss."""
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLoss(nn.Module):
    """Defined a criterion to calculator loss."""

    def name(self):
        """return name of criterion."""
        return 'RelativisticDiscLoss'

    def __init__(self):
        super(RelativisticDiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_pool = ImagePool(
            50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        """First, G(A) should fake the discriminator."""
        self.pred_fake = net.forward(fakeB)

        # Real
        self.pred_real = net.forward(realB)
        errG = (self.criterionGAN(
            self.pred_real - torch.mean(self.fake_pool.query()),
            0) + self.criterionGAN(
                self.pred_fake - torch.mean(self.real_pool.query()), 1)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        """Fake stop backprop to the generator by detaching fake_B Generated
        Image Disc Output should be close to zero."""
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (self.criterionGAN(
            self.pred_real - torch.mean(self.fake_pool.query()),
            1) + self.criterionGAN(
                self.pred_fake - torch.mean(self.real_pool.query()), 0)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        """Get discriminator loss."""
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLossLS(nn.Module):
    """Defined a criterion to calculator loss."""

    def name(self):
        """return name of criterion."""
        return 'RelativisticDiscLossLS'

    def __init__(self):
        super(RelativisticDiscLossLS, self).__init__()

        self.criterionGAN = GANLoss(use_l1=True)
        self.fake_pool = ImagePool(
            50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        """First, G(A) should fake the discriminator."""
        self.pred_fake = net.forward(fakeB)

        # Real
        self.pred_real = net.forward(realB)
        ex_pdata = torch.mean(
            (self.pred_real - torch.mean(self.fake_pool.query()) + 1)**2)
        ez_pz = torch.mean(
            (self.pred_fake - torch.mean(self.real_pool.query()) - 1)**2)
        errG = (ex_pdata + ez_pz) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        """Fake stop backprop to the generator by detaching fake_B Generated
        Image Disc Output should be close to zero."""
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (torch.mean(
            (self.pred_real - torch.mean(self.fake_pool.query()) - 1)**2) +
                       torch.mean(
                           (self.pred_fake -
                            torch.mean(self.real_pool.query()) + 1)**2)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        """Get discriminator loss."""
        return self.get_loss(net, fakeB, realB)


class DiscLossLS(DiscLoss):
    """Defined a criterion to calculator loss."""

    def name(self):
        """return name of criterion."""
        return 'DiscLossLS'

    def __init__(self):
        super(DiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)

    def get_g_loss(self, net, fakeB, realB):
        """First, G(A) should fake the discriminator.

        Get generator loss.
        """
        return DiscLoss.get_g_loss(self, net, fakeB)

    def get_loss(self, net, fakeB, realB):
        """Get discriminator loss."""
        return DiscLoss.get_loss(self, net, fakeB, realB)


class DiscLossWGANGP(DiscLossLS):
    """Defined a criterion to calculator loss."""

    def name(self):
        """return name of criterion."""
        return 'DiscLossWGAN-GP'

    def __init__(self):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = 10

    def get_g_loss(self, net, fakeB, realB):
        """First, G(A) should fake the discriminator.

        Get generator loss.
        """
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        if torch.cuda.is_available():
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if torch.cuda.is_available():
            interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        if torch.cuda.is_available():
            gradients = autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        else:
            gradients = autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size()),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fakeB, realB):
        """Get discriminator loss."""
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB.data,
                                                      fakeB.data)
        return self.loss_D + gradient_penalty


def get_pixel_loss(loss_type):
    """Returns the loss of generator with the specified type.

    Args:
        loss_type (Str): loss type of generator
    """
    if loss_type == 'perceptual':
        content_loss = PerceptualLoss()
        content_loss.initialize(nn.MSELoss())
    elif loss_type == 'l1':
        content_loss = ContentLoss()
        content_loss.initialize(nn.L1Loss())
    else:
        raise ValueError('ContentLoss [%s] not recognized.' % loss_type)
    return content_loss


def get_disc_loss(loss_type):
    """Returns the loss of discriminator with the specified type.

    Args:
        loss_type (Str): loss type of discriminator
    """
    if loss_type == 'wgan-gp':
        disc_loss = DiscLossWGANGP()
    elif loss_type == 'lsgan':
        disc_loss = DiscLossLS()
    elif loss_type == 'gan':
        disc_loss = DiscLoss()
    elif loss_type == 'ragan':
        disc_loss = RelativisticDiscLoss()
    elif loss_type == 'ragan-ls':
        disc_loss = RelativisticDiscLossLS()
    else:
        raise ValueError('GAN Loss [%s] not recognized.' % loss_type)
    return disc_loss
