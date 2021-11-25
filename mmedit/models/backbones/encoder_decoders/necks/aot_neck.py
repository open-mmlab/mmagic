import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class AOTBlockNeck(nn.Module):
    """Dilation Backbone used in Global&Local model.

    This implementation follows:
    Globally and locally Consistent Image Completion

    Args:
        in_channels (int): Channel number of input feature.
        conv_type (str): The type of conv module. In DeepFillv1 model, the
            `conv_type` should be 'conv'. In DeepFillv2 model, the `conv_type`
            should be 'gated_conv'.
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        kwargs (keyword arguments).
    """

    def __init__(self,
                 in_channels=256,
                 act_cfg=dict(type='ReLU'),
                 dilation_rates='1+2+4+8',
                 num_aotblock=8,
                 **kwargs):
        super().__init__()

        self.dilation_rates = list(map(int, list(dilation_rates.split('+'))))
        
        
        self.num_aotblock = num_aotblock

        self.middle = nn.Sequential(*[AOTBlock(256, self.dilation_rates) for _ in range(num_aotblock)])
        
        '''
        
        self.model = nn.Sequential()
        
        for i in range(self.num_aotblock):
            self.model.add_module(
                f"aotblock{i + 1}", 
                AOTBlock(
                    in_channels=in_channels,
                    dilation_rates=self.dilation_rates,
                    act_cfg=act_cfg,
                )
            )
        '''
        
        
        
    def forward(self, x):
        x = self.middle(x)
        return x


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

'''
class AOTBlock(nn.Module):
    """
    """
    def __init__(self,
                 in_channels,
                 dilation_rates,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        self.dilation_rates = dilation_rates
        
        for i, dilation_rate in enumerate(dilation_rates):
            self.__setattr__(
                f'block{i + 1}',
                nn.Sequential(
                nn.ReflectionPad2d(dilation_rate),
                ConvModule(
                    in_channels,
                    in_channels // 4,
                    kernel_size=3,
                    dilation=dilation_rate,
                    act_cfg=act_cfg)))

        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvModule(in_channels, in_channels, 3, dilation=1, act_cfg=None))

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvModule(in_channels, in_channels, 3, dilation=1, act_cfg=None))

    def norm(self, x):
        mean = x.mean((2, 3), keepdim=True)
        std = x.std((2, 3), keepdim=True) + 1e-9
        x = 2 * (x - mean) / std - 1
        x = 5 * x
        return x

    def forward(self, x):
        dilate_x = [self.__getattr__(f'block{i + 1}')(x) for i in range(len(self.dilation_rates))]
        dilate_x = torch.cat(dilate_x, 1)
        dilate_x = self.fuse(dilate_x)
        mask = self.norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + dilate_x * mask


class AOTBlock(nn.Module):
    """
    """
    def __init__(self,
                 dim,
                 rates,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                f'block{i + 1}',
                nn.Sequential(
                nn.ReflectionPad2d(rate),
                ConvModule(
                    dim,
                    dim // 4,
                    kernel_size=3,
                    padding=0,
                    dilation=rate,
                    norm_cfg=None,
                    act_cfg=act_cfg)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvModule(dim, dim, 3, padding=0, dilation=1, act_cfg=None))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvModule(dim, dim, 3, padding=0, dilation=1, act_cfg=None))

    def norm(self, x):
        mean = x.mean((2, 3), keepdim=True)
        std = x.std((2, 3), keepdim=True) + 1e-9
        x = 2 * (x - mean) / std - 1
        x = 5 * x
        return x

    def forward(self, x):
        out = [self.__getattr__(f'block{i + 1}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = self.norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask
'''