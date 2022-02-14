# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmedit.models.registry import COMPONENTS


class ResBlock(nn.Module):
    """ResBlock for Hourglass.

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


class Hourglass(nn.Module):
    """Hourglass model for face landmark.

    It is a recursive model.

    Args:
        depth (int): Depth of Hourglass, the number of recursions.
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
            self.low2 = Hourglass(depth - 1, mid_channels)
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
class FeedbackHourglass(nn.Module):
    """Feedback Hourglass model for face landmark.

    It has a style of:

    ::

        -- preprocessing ----- Hourglass ----->
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

        self.hg = Hourglass(4, 2 * self.mid_channels)
        self.last = nn.Sequential(
            ResBlock(self.mid_channels, self.mid_channels),
            nn.Conv2d(self.mid_channels, self.mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.num_keypoints, 1))

    def forward(self, x, last_hidden=None):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            last_hidden (Tensor | None): The feedback of FeedbackHourglass.
                In first step, last_hidden=None. Otherwise, last_hidden is
                the past output of FeedbackHourglass.
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


def reduce_to_five_heatmaps(ori_heatmap, detach):
    """Reduce facial landmark heatmaps to 5 heatmaps.

    DIC realizes facial SR with the help of key points of the face.
    The number of key points in datasets are different from each other.
    This function reduces the input heatmaps into 5 heatmaps:
        left eye
        right eye
        nose
        mouse
        face silhouette

    Args:
        ori_heatmap (Tensor): Input heatmap tensor. (B, N, 32, 32).
        detach (bool): Detached from the current tensor or not.

    returns:
        Tensor: New heatmap tensor. (B, 5, 32, 32).
    """

    heatmap = ori_heatmap.clone()
    max_heat = heatmap.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    max_heat = max_heat.clamp_min_(0.05)
    heatmap /= max_heat
    if heatmap.size(1) == 5:
        return heatmap.detach() if detach else heatmap
    elif heatmap.size(1) == 68:
        new_heatmap = torch.zeros_like(heatmap[:, :5])
        new_heatmap[:, 0] = heatmap[:, 36:42].sum(1)  # left eye
        new_heatmap[:, 1] = heatmap[:, 42:48].sum(1)  # right eye
        new_heatmap[:, 2] = heatmap[:, 27:36].sum(1)  # nose
        new_heatmap[:, 3] = heatmap[:, 48:68].sum(1)  # mouse
        new_heatmap[:, 4] = heatmap[:, :27].sum(1)  # face silhouette
        return new_heatmap.detach() if detach else new_heatmap
    elif heatmap.size(1) == 194:  # Helen
        new_heatmap = torch.zeros_like(heatmap[:, :5])
        tmp_id = torch.cat((torch.arange(134, 153), torch.arange(174, 193)))
        new_heatmap[:, 0] = heatmap[:, tmp_id].sum(1)  # left eye
        tmp_id = torch.cat((torch.arange(114, 133), torch.arange(154, 173)))
        new_heatmap[:, 1] = heatmap[:, tmp_id].sum(1)  # right eye
        tmp_id = torch.arange(41, 57)
        new_heatmap[:, 2] = heatmap[:, tmp_id].sum(1)  # nose
        tmp_id = torch.arange(58, 113)
        new_heatmap[:, 3] = heatmap[:, tmp_id].sum(1)  # mouse
        tmp_id = torch.arange(0, 40)
        new_heatmap[:, 4] = heatmap[:, tmp_id].sum(1)  # face silhouette
        return new_heatmap.detach() if detach else new_heatmap
    else:
        raise NotImplementedError(
            f'Face landmark number {heatmap.size(1)} not implemented!')
