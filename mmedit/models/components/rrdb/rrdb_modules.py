import math
from copy import deepcopy
from functools import partial

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmedit.models.registry import COMPONENTS

@COMPONENTS.register_module()
class DenseBlock5C(nn.Module):
    def __init__(self,
                 in_channels=32,
                 base_channels=32,
                 out_channels=32,
                 padding=1,
                 bias=True,
                 act_cfg=dict(type='LeakyReLU'),
                 norm_cfg=None):
        super(DenseBlock5C, self).__init__()

        self.blocks_0 = ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        self.blocks_1 = ConvModule(
            in_channels=in_channels + base_channels,
            out_channels=base_channels,
            kernel_size=3,
            padding=1,
            bias=bias ,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks_2 = ConvModule(
            in_channels=in_channels + base_channels *2,
            out_channels=base_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks_3 = ConvModule(
            in_channels=in_channels + base_channels *3,
            out_channels=base_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks_4 = ConvModule(
            in_channels=in_channels + base_channels *4,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
    
    def forward(self, x):
        x1 = self.blocks_0(x)
        x2 = self.blocks_1(torch.cat((x, x1), 1))
        x3 = self.blocks_2(torch.cat((x, x1, x2), 1))
        x4 = self.blocks_3(torch.cat((x, x1, x2, x3), 1))
        x5 = self.blocks_4(torch.cat((x, x1, x2, x3, x4), 1))     
        
        return  x5.mul(0.2) + x
        
@COMPONENTS.register_module()
class RrdbBlock(nn.Module):
    def __init__(self,
                 nums_block = 3):
        super(RrdbBlock, self).__init__()

        self.rdb_1 = DenseBlock5C(in_channels=32, 
                                base_channels=32, 
                                out_channels =32,
                                padding=1,
                                bias=True,
                                norm_cfg=None,
                                act_cfg=dict(type='LeakyReLU'))

        self.rdb_2 = DenseBlock5C(in_channels=32, 
                                base_channels=32, 
                                out_channels =32,
                                padding=1,
                                bias=True,
                                norm_cfg=None,
                                act_cfg=dict(type='LeakyReLU'))
        
        self.rdb_3 = DenseBlock5C(in_channels=32, 
                                base_channels=32, 
                                out_channels =32,
                                padding=1,
                                bias=True,
                                norm_cfg=None,
                                act_cfg=dict(type='LeakyReLU'))

    def forward(self, x):
        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        return out.mul(0.2) + x
