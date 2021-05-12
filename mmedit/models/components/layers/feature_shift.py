import torch
import torch.nn as nn

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class FeatureShift(nn.Conv2d):
    """Modify the statistical information of the feature map,
        including the mean and standard deviation

    Args:
        pixel_range (float): Pixel range of geature.
        feature_mean (Tuple[float]): Feature mean of each channel.
        feature_std (Tuple[float]): Feature std of each channel.
        sign (int): Sign of bias. Default -1.
    """

    def __init__(self, pixel_range, feature_mean, feature_std, sign=-1):
        assert len(feature_mean) == len(feature_std)
        num_channels = len(feature_mean)
        super().__init__(num_channels, num_channels, kernel_size=1)
        std = torch.Tensor(feature_std)
        self.weight.data = torch.eye(num_channels).view(
            num_channels, num_channels, 1, 1)
        self.weight.data.div_(std.view(num_channels, 1, 1, 1))
        self.bias.data = sign * pixel_range * torch.Tensor(feature_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False
