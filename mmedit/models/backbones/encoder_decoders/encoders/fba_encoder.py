import torch
import torch.nn as nn
from mmcv.cnn import ConvWS2d, build_norm_layer

from mmedit.models.registry import COMPONENTS


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvWS2d(inplanes, planes, stride=stride, kernel_size=3)
        self.bn1 = build_norm_layer(planes, norm_cfg)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvWS2d(planes, planes, stride=stride, kernel_size=3)
        self.bn2 = build_norm_layer(planes, norm_cfg)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_cfg=None):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvWS2d(
            inplanes, planes, stride=1, kernel_size=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = ConvWS2d(
            planes,
            planes,
            stride=stride,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = ConvWS2d(
            planes,
            planes * self.expansion,
            stride=1,
            kernel_size=1,
            bias=False)
        self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.conv1 = ConvWS2d(
            11, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = build_norm_layer(norm_cfg, 64)[1]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ConvWS2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                norm_cfg=self.norm_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg=self.norm_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


@COMPONENTS.register_module()
class FBAResnetDilated(nn.Module):

    def __init__(self,
                 orig_resnet=None,
                 pre_trained=False,
                 dilate_scale=8,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(FBAResnetDilated, self).__init__()
        from functools import partial
        if orig_resnet is None:
            orig_resnet = ResNet(
                Bottleneck, [3, 4, 6, 3], norm_cfg=norm_cfg, act_cfg=act_cfg)
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x, return_feature_maps=False):
        # x cat(image_n, trimap_transformed, two_chan_trimap,img)
        merged_transformed, trimap_transformed, two_chan_trimap, merged = x
        x = torch.cat(
            (merged_transformed, trimap_transformed, two_chan_trimap), 1)
        conv_out = [x]
        x = self.relu(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        return (conv_out, merged, two_chan_trimap, indices)


def l_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
