import torch
import torch.nn as nn


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

    def __init__(self, mid_channels, num_blocks, upscale_factor):
        super().__init__()

        stride = upscale_factor
        padding = 2
        kernel_size = upscale_factor + 4
        prelu_init = 0.2

        self.num_blocks = num_blocks
        self.need_reset = True
        self.last_hidden = None

        self.first = nn.Sequential(
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

        self.last = nn.Sequential(
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
        x = self.first(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_blocks):
            # when idx == 0, lr_features == [x]
            lr = torch.cat(tuple(lr_features), 1)
            if idx > 0:
                lr = self.lr_blocks[idx - 1](lr)
            hr = self.up_blocks[idx](lr)

            hr_features.append(hr)

            hr = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                hr = self.hr_blocks[idx - 1](hr)
            lr = self.down_blocks[idx](hr)

            lr_features.append(lr)

        output = torch.cat(tuple(lr_features[1:]), 1)
        output = self.last(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        """Reset 'self.need_reset' to 'True'.
        """

        self.need_reset = True
