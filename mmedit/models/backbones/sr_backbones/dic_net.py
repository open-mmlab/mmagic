import torch
import torch.nn as nn

from mmedit.models.common import make_layer


class FeedbackBlock(nn.Module):
    """Feedback Block of DIC

    It has a style of:

    ::

        ----- Module ----->
          ^            |
          |____________|

    Args:
        mid_channels (int): Number of channels in the intermediate features.
        num_blocks (int): Number of blocks.
        upscale_factor (int): upscale factor.
    """

    def __init__(self,
                 mid_channels,
                 num_blocks,
                 upscale_factor,
                 padding=2,
                 prelu_init=0.2):
        super().__init__()

        stride = upscale_factor
        kernel_size = upscale_factor + 4

        self.num_blocks = num_blocks
        self.need_reset = True
        self.last_hidden = None

        self.conv_first = nn.Sequential(
            nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=1),
            nn.PReLU(init=prelu_init))

        self.up_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.lr_blocks = nn.ModuleList()
        self.hr_blocks = nn.ModuleList()

        for idx in range(self.num_blocks):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size,
                                       stride, padding),
                    nn.PReLU(init=prelu_init)))
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, kernel_size, stride,
                              padding), nn.PReLU(init=prelu_init)))
            if idx > 0:
                self.lr_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            mid_channels * (idx + 1),
                            mid_channels,
                            kernel_size=1), nn.PReLU(init=prelu_init)))
                self.hr_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            mid_channels * (idx + 1),
                            mid_channels,
                            kernel_size=1), nn.PReLU(init=prelu_init)))

        self.conv_last = nn.Sequential(
            nn.Conv2d(num_blocks * mid_channels, mid_channels, kernel_size=1),
            nn.PReLU(init=prelu_init))

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.need_reset:
            self.last_hidden = x
            self.need_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.conv_first(x)

        lr_features = [x]
        hr_features = []

        for idx in range(self.num_blocks):
            # when idx == 0, lr_features == [x]
            lr = torch.cat(lr_features, 1)
            if idx > 0:
                lr = self.lr_blocks[idx - 1](lr)
            hr = self.up_blocks[idx](lr)

            hr_features.append(hr)

            hr = torch.cat(hr_features, 1)
            if idx > 0:
                hr = self.hr_blocks[idx - 1](hr)
            lr = self.down_blocks[idx](hr)

            lr_features.append(lr)

        output = torch.cat(lr_features[1:], 1)
        output = self.conv_last(output)

        self.last_hidden = output

        return output


class FeedbackBlockCustom(FeedbackBlock):
    """Custom feedback block, will be used as the first feedback block.

    Args:
        in_channels (int): Number of channels in the input features.
        mid_channels (int): Number of channels in the intermediate features.
        num_blocks (int): Number of blocks.
        upscale_factor (int): upscale factor.
    """

    def __init__(self, in_channels, mid_channels, num_blocks, upscale_factor):
        super().__init__(mid_channels, num_blocks, upscale_factor)

        prelu_init = 0.2
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.PReLU(init=prelu_init))

    def forward(self, x):
        x = self.conv_first(x)

        lr_features = [x]
        hr_features = []

        for idx in range(self.num_blocks):
            # when idx == 0, lr_features == [x]
            lr = torch.cat(lr_features, 1)
            if idx > 0:
                lr = self.lr_blocks[idx - 1](lr)
            hr = self.up_blocks[idx](lr)

            hr_features.append(hr)

            hr = torch.cat(hr_features, 1)
            if idx > 0:
                hr = self.hr_blocks[idx - 1](hr)
            lr = self.down_blocks[idx](hr)

            lr_features.append(lr)

        output = torch.cat(lr_features[1:], 1)
        output = self.conv_last(output)

        return output


class GroupResBlock(nn.Module):
    """ResBlock with Group Conv.
    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        mid_channels (int): Channel number of intermediate features.
        groups (int): Number of blocked connections from input to output.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 groups,
                 res_scale=1.0):
        super().__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, groups=groups),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, groups=groups))
        self.res_scale = res_scale

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        res = self.res(x).mul(self.res_scale)
        return x + res


class FeatureHeatmapFusingBlock(nn.Module):
    """ Fusing Feature and Heatmap.
    Args:
        in_channels (int): Number of channels in the input features.
        num_heatmaps (int): Number of heatmap.
        num_blocks (int): Number of blocks.
        mid_channels (int | None): Number of channels in the intermediate
            features. Default: None
    """

    def __init__(self,
                 in_channels,
                 num_heatmaps,
                 num_blocks,
                 mid_channels=None):
        super().__init__()

        self.num_heatmaps = num_heatmaps
        res_block_channel = in_channels * num_heatmaps
        if mid_channels is None:
            self.mid_channels = num_heatmaps * in_channels
        else:
            self.mid_channels = mid_channels
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, res_block_channel, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body = make_layer(
            GroupResBlock,
            num_blocks,
            in_channels=res_block_channel,
            out_channels=res_block_channel,
            mid_channels=self.mid_channels,
            groups=num_heatmaps)

    def forward(self, feature, heatmap):
        """Forward function.
        Args:
            feature (Tensor): Input feature tensor.
            heatmap (Tensor): Input heatmap tensor.
        Returns:
            Tensor: Forward results.
        """

        assert self.num_heatmaps == heatmap.size(1)
        batch_size = heatmap.size(0)
        w, h = feature.shape[-2:]

        feature = self.conv_first(feature)
        # B * (num_heatmaps*in_channels) * h * w
        feature = self.body(feature)
        attention = nn.functional.softmax(
            heatmap, dim=1)  # B * num_heatmaps * h * w

        feature = feature.view(batch_size, self.num_heatmaps, -1, w,
                               h) * attention.unsqueeze(2)
        feature = feature.sum(1)
        return feature


class FeedbackBlockHeatmapAttention(FeedbackBlock):
    """Feedback block with HeatmapAttention.
    Args:
        in_channels (int): Number of channels in the input features.
        mid_channels (int): Number of channels in the intermediate features.
        num_blocks (int): Number of blocks.
        upscale_factor (int): upscale factor.
    """

    def __init__(self,
                 mid_channels,
                 num_blocks,
                 upscale_factor,
                 num_heatmaps,
                 num_fusion_blocks,
                 padding=2,
                 prelu_init=0.2):

        super().__init__(
            mid_channels,
            num_blocks,
            upscale_factor,
            padding=padding,
            prelu_init=prelu_init)
        self.fusion_block = FeatureHeatmapFusingBlock(mid_channels,
                                                      num_heatmaps,
                                                      num_fusion_blocks)

    def forward(self, x, heatmap):
        """Forward function.
        Args:
            x (Tensor): Input feature tensor.
            heatmap (Tensor): Input heatmap tensor.
        Returns:
            Tensor: Forward results.
        """

        if self.need_reset:
            self.last_hidden = x
            self.need_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.conv_first(x)

        # fusion
        x = self.fusion_block(x, heatmap)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_blocks):
            # when idx == 0, lr_features == [x]
            lr = torch.cat(lr_features, 1)
            if idx > 0:
                lr = self.lr_blocks[idx - 1](lr)
            hr = self.up_blocks[idx](lr)

            hr_features.append(hr)

            hr = torch.cat(hr_features, 1)
            if idx > 0:
                hr = self.hr_blocks[idx - 1](hr)
            lr = self.down_blocks[idx](hr)

            lr_features.append(lr)

        output = torch.cat(lr_features[1:], 1)
        output = self.conv_last(output)

        self.last_hidden = output

        return output
