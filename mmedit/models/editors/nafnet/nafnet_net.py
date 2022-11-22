# Copyright (c) 2022 megvii-model. All Rights Reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmedit.registry import MODELS
from .naf_avgpool2d import Local_Base
from .naf_layerNorm2d import LayerNorm2d


@MODELS.register_module()
class NAFNet(BaseModule):
    """NAFNet.

    The original version of NAFNet in "Simple Baseline for Image Restoration".

    Args:
        img_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of intermediate features.
        middle_blk_num (int): Number of middle blocks.
        enc_blk_nums (List of int): Number of blocks for each encoder.
        dec_blk_nums (List of int): Number of blocks for each decoder.
    """

    def __init__(self,
                 img_channel=3,
                 mid_channels=16,
                 middle_blk_num=1,
                 enc_blk_nums=[],
                 dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True)
        self.ending = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = mid_channels
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2**len(self.encoders)

    def forward(self, inp):
        """Forward function.

        args:
            inp: input tensor image with (B, C, H, W) shape
        """
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """Check image size and pad images so that it has enough dimension do
        downsample.

        args:
            x: input tensor image with (B, C, H, W) shape.
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size -
                     h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size -
                     w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


@MODELS.register_module()
class NAFNetLocal(Local_Base, NAFNet):
    """The original version of NAFNetLocal in "Simple Baseline for Image
    Restoration".

    NAFNetLocal uses local average pooling modules than NAFNet.

    Args:
        img_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of intermediate features.
        middle_blk_num (int): Number of middle blocks.
        enc_blk_nums (List of int): Number of blocks for each encoder.
        dec_blk_nums (List of int): Number of blocks for each decoder.
    """

    def __init__(self,
                 *args,
                 train_size=(1, 3, 256, 256),
                 fast_imp=False,
                 **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(
                base_size=base_size, train_size=train_size, fast_imp=fast_imp)


# Components for NAFNet


class NAFBlock(BaseModule):
    """NAFNet's Block in paper.

    Simple gate will shrink the channel to a half.
    To keep the number of channels,
    it expands the channels first.

    Args:
        in_channels (int): number of channels
        DW_Expand (int): channel expansion factor for part 1
        FFN_Expand (int): channel expansion factor for part 2
        drop_out_rate (float): drop out ratio
    """

    def __init__(self,
                 in_channels,
                 DW_Expand=2,
                 FFN_Expand=2,
                 drop_out_rate=0.):
        super().__init__()

        # Part 1

        dw_channel = in_channels * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True),
        )

        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)

        # Part 2

        ffn_channel = FFN_Expand * in_channels
        self.conv4 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)

        # Simple Gate
        self.sg = SimpleGate()

        # Layer Normalization
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)

        # Dropout
        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Feature weight ratio
        self.beta = nn.Parameter(
            torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(
            torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        """Forward Function.

        Args:
            inp: input tensor image
        """
        x = inp
        # part 1
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        # part 2
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        out = y + x * self.gamma

        return out


class SimpleGate(BaseModule):
    """The Simple Gate in "Simple Baseline for Image Restoration".

    Args:
        x: input tensor feature map with (B, 2 * C, H, W)

    Return:
        x1 * x2
        (where x1, x2 are two separate parts by simple split x to [B, C, H, W])
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
