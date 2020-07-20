import torch.nn as nn
from mmcv.cnn.utils.weight_init import constant_init, xavier_init
from mmcv.runner import load_checkpoint

from mmedit.models.common import ASPP
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class VGG16(nn.Module):
    """Customed VGG16 Encoder.

    A 1x1 conv is added after the original VGG16 conv layers. The indices of
    max pooling layers are returned for unpooling layers in decoders.

    Args:
        in_channels (int): Number of input channels.
        batch_norm (bool, optional): Whether use ``nn.BatchNorm2d``.
            Default to False.
        aspp (bool, optional): Whether use ASPP module after the last conv
            layer. Default to False.
        dilations (list[int], optional): Atrous rates of ASPP module.
            Default to None.
    """

    def __init__(self,
                 in_channels,
                 batch_norm=False,
                 aspp=False,
                 dilations=None):
        super(VGG16, self).__init__()
        self.batch_norm = batch_norm
        self.aspp = aspp
        self.dilations = dilations

        self.layer1 = self._make_layer(in_channels, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 3)
        self.layer4 = self._make_layer(256, 512, 3)
        self.layer5 = self._make_layer(512, 512, 3)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=1)
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        if self.aspp:
            self.aspp = ASPP(512, dilations=self.dilations)
            self.out_channels = 256
        else:
            self.out_channels = 512

    def _make_layer(self, inplanes, planes, convs_layers):
        layers = []
        for _ in range(convs_layers):
            conv2d = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
            if self.batch_norm:
                bn = nn.BatchNorm2d(planes)
                layers += [conv2d, bn, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            inplanes = planes
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, x):
        """Forward function for ASPP module.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            dict: Dict containing output tensor and maxpooling indices.
        """
        out, max_idx_1 = self.layer1(x)
        out, max_idx_2 = self.layer2(out)
        out, max_idx_3 = self.layer3(out)
        out, max_idx_4 = self.layer4(out)
        out, max_idx_5 = self.layer5(out)

        out = self.conv6(out)
        if self.batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        if self.aspp:
            out = self.aspp(out)

        return {
            'out': out,
            'max_idx_1': max_idx_1,
            'max_idx_2': max_idx_2,
            'max_idx_3': max_idx_3,
            'max_idx_4': max_idx_4,
            'max_idx_5': max_idx_5
        }
