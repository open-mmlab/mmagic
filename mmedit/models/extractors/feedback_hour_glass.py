import torch.nn as nn


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
