import torch
import torch.nn as nn
from mmcv.cnn.weight_init import xavier_init
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module
class PlainRefiner(nn.Module):
    """Simple refiner from Deep Image Matting.

    Args:
        conv_channels (int): Number of channels produced by the three main
            convolutional layer.
        loss_refine (dict): Config of the loss of the refiner. Default: None.
        pretrained (str): Name of pretrained model. Default: None.
    """

    def __init__(self, conv_channels=64, pretrained=None):
        super(PlainRefiner, self).__init__()

        self.refine_conv1 = nn.Conv2d(
            4, conv_channels, kernel_size=3, padding=1)
        self.refine_conv2 = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=3, padding=1)
        self.refine_conv3 = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=3, padding=1)
        self.refine_pred = nn.Conv2d(
            conv_channels, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    def forward(self, x, raw_alpha):
        out = self.relu(self.refine_conv1(x))
        out = self.relu(self.refine_conv2(out))
        out = self.relu(self.refine_conv3(out))
        raw_refine = self.refine_pred(out)
        pred_refine = torch.sigmoid(raw_alpha + raw_refine)
        return pred_refine
