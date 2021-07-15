import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import (ResidualBlockWithDropout,
                                  generation_init_weights,
                                  RRDB)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from torch.nn.modules import padding
from mmedit.models.builder import build_component

@BACKBONES.register_module()
class DenseGeneratorFromRRDB(nn.Module):
    """Construct a Bgan generator that consists of residual blocks"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=32,
                 block=dict(type='DenseBlock',
                   in_channels=32,
                   base_channels=32,
                   out_channels=32,
                   use_dropout=False,
                   norm_cfg=dict(type='BN'),
                   act_cfg=dict(type='LeakyReLU')
                   ), 
                 num_blocks=9,                
                 norm_cfg=dict(type='BN'),
                 use_dropout=False,
                 is_skip=True,
                 init_cfg=dict(type='normal', gain=0.02),
                 with_nosie = True,
                 nosie_size = (4,128,128)
                 ):
        super(DenseGeneratorFromRRDB, self).__init__()
        assert num_blocks >= 0, ('Number of residual blocks must be '
                                 f'non-negative, but got {num_blocks}.')
        
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        
        # In this papar,author suggest NO NORM
        self.with_nosie = with_nosie
        self.noise_size = nosie_size

        model = []
        rb_blocks = []
        
        model += [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                )
        ]
        rb_blocks += [ build_component(block) for i in range(num_blocks)]

        if is_skip:
            LR_conv = ConvModule(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=3,
                padding=1,
                )
            rb_skip = self.connectAB(nn.Sequential(*rb_blocks), LR_conv)
            model += [rb_skip]
        else:
            model += rb_blocks
        
        HR_conv0 = ConvModule(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=3,
                padding=1,
                act_cfg = dict(type='LeakyReLU')
                )
                
        HR_conv1 = ConvModule(
                in_channels=base_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                act_cfg = None
                )
        
        model += [HR_conv0, HR_conv1]

        self.model_x = nn.Sequential(*model)

        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')

        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self,x, noise_map = None):
        """Forward function.

        Args:
        x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
        Tensor: Forward results.
        """
        if self.with_nosie:
            input_x = torch.cat((x, noise_map),1)
        else:
            input_x = x
        output = self.model_x(input_x)
        output = torch.clamp(x + output, min = 0, max = 1)

        return output
    
    
    def init_weights(self, pretrained=None, strict=True):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

    def connectAB(self,submodule1,submodule2):
        class SConnet(nn.Module):
            def __init__(self, sub1,sub2):
                super(SConnet, self).__init__()
                self.sub1 = sub1
                self.sub2 = sub2

            def forward(self, x):
                output = x + self.sub2(self.sub1(x))
                return output
        return SConnet(submodule1, submodule2)
    