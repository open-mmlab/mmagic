import torch
import torch.nn as nn

from mmedit.models.registry import COMPONENTS


class ResBlock(nn.Module):
    """ResBlock for HourGlass.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-Conv-+-
         |_________Conv________|

        or

        ---Conv-ReLU-Conv-Conv-+-
         |_____________________|

    Args:
        in_channels (int): Number of channels in the input features.
        out_channels (int): Number of channels in the output features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // 2, out_channels // 2, 3, stride=1, padding=1),
            nn.Conv2d(out_channels // 2, out_channels, 1))
        if in_channels == out_channels:
            self.skip_layer = None
        else:
            self.skip_layer = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        residual = self.conv_block(x)
        if self.skip_layer:
            x = self.skip_layer(x)
        return x + residual


class HourGlass(nn.Module):
    """Hour Glass model for face landmark.

    It is a recursive model.

    Args:
        depth (int): Depth of HourGlass, the number of recursions.
        mid_channels (int): Number of channels in the intermediate features.
    """

    def __init__(self, depth, mid_channels):
        super().__init__()
        self.up1 = ResBlock(mid_channels, mid_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = ResBlock(mid_channels, mid_channels)
        if depth == 1:
            self.low2 = ResBlock(mid_channels, mid_channels)
        else:
            self.low2 = HourGlass(depth - 1, mid_channels)
        self.low3 = ResBlock(mid_channels, mid_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        up1 = self.up1(x)
        low1 = self.low1(self.pool(x))
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = nn.functional.interpolate(
            low3, scale_factor=2, mode='bilinear', align_corners=True)
        return up1 + up2


@COMPONENTS.register_module()
class FeedbackHourGlass(nn.Module):
    """Feedback Hour Glass model for face landmark.

    It has a style of:

    ::

        -- preprocessing ----- HourGlass ----->
                           ^               |
                           |_______________|

    Args:
        mid_channels (int): Number of channels in the intermediate features.
        num_keypoints (int): Number of keypoints.
    """

    def __init__(self, mid_channels, num_keypoints):
        super().__init__()
        self.mid_channels = mid_channels
        self.num_keypoints = num_keypoints

        self.pre_conv_block = nn.Sequential(
            nn.Conv2d(3, self.mid_channels // 4, 7, 2, 3),
            nn.ReLU(inplace=True),
            ResBlock(self.mid_channels // 4, self.mid_channels // 2),
            nn.MaxPool2d(2, 2),
            ResBlock(self.mid_channels // 2, self.mid_channels // 2),
            ResBlock(self.mid_channels // 2, self.mid_channels),
        )
        self.first_conv = nn.Conv2d(2 * self.mid_channels,
                                    2 * self.mid_channels, 1)

        self.hg = HourGlass(4, 2 * self.mid_channels)
        self.last = nn.Sequential(
            ResBlock(self.mid_channels, self.mid_channels),
            nn.Conv2d(self.mid_channels, self.mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.num_keypoints, 1))

    def forward(self, x, last_hidden=None):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            last_hidden (Tensor | None): The feedback of FeedbackHourGlass.
                In first step, last_hidden=None. Otherwise, last_hidden is
                the past output of FeedbackHourGlass.
                Default: None.

        Returns:
            heatmap (Tensor): Heatmap of facial landmark.
            feedback (Tensor): Feedback Tensor.
        """

        feature = self.pre_conv_block(x)
        if last_hidden is None:
            feature = self.first_conv(torch.cat((feature, feature), dim=1))
        else:
            feature = self.first_conv(torch.cat((feature, last_hidden), dim=1))
        feature = self.hg(feature)
        heatmap = self.last(feature[:, :self.mid_channels])  # first half
        feedback = feature[:, self.mid_channels:]  # second half
        return heatmap, feedback
