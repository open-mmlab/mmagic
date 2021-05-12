import torch
import torch.nn as nn

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class MeanShift(nn.Conv2d):
    """Mean shift.

    Args:
        pixel_range (float): Pixel range of image.
        rgb_mean (Tuple[float]): Image mean in RGB orders.
        rgb_std (Tuple[float]): Image std in RGB orders.
        sign (int): Sign of bias. Default -1.
    """

    def __init__(self, pixel_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * pixel_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False
