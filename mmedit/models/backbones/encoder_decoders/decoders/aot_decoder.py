import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class AOTDecoder(nn.Module):
    """Decoder used in Global&Local model.

    This implementation follows:
    Globally and locally Consistent Image Completion

    Args:
        in_channels (int): Channel number of input feature.
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        out_act (str): Output activation type, "clip" by default. Noted that
            in our implementation, we clip the output with range [-1, 1].
    """

    def __init__(self,
                 in_channels=256,
                 act_cfg=dict(type='ReLU'),
                 out_act='clip'):
        super().__init__()

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        '''        
        self.dec1 = ConvModule(
            in_channels,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=act_cfg)
        self.dec2 = ConvModule(
            128,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=act_cfg)
        self.dec3 = ConvModule(
            64,
            3,
            kernel_size=3,
            stride=1,
            padding=1, 
            act_cfg=None)

        self.output_act = nn.Tanh()
        '''
        

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        '''
        for i in range(3):
            if i <= 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = getattr(self, f'dec{i + 1}')(x)
        x = self.output_act(x)
        '''
        x = self.decoder(x)
        x = torch.tanh(x)
        
        return x

class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
