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


@BACKBONES.register_module()
class BganGenerator(nn.Module):
    """Construct a Bgan generator that consists of residual blocks"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=32,
                 norm_cfg=dict(type='BN'),
                 use_dropout=False,
                 num_blocks=9,
                 init_cfg=dict(type='normal', gain=0.02),
                 is_skippnect=False):
        super(BganGenerator,self).__init__()
        assert num_blocks >= 0, ('Number of residual blocks must be '
                                 f'non-negative, but got {num_blocks}.')
        
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the resnet generator.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        model = []
        rb_blocks = []
        model += [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                act_cfg = None
                )
        ]
        rb_blocks += [
                RRDB(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    kernel_size=3,
                    stride=1,
                    bias=use_bias,
                    act_cfg=dict(type='LeakyReLU'),
                    mode='CNA')
                    for i in range(num_blocks)]

        if is_skippnect:
            LR_conv = ConvModule(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=3,
                padding=1,
                act_cfg = None
                )
            rb_ = self.skip_connect(nn.Sequential(*rb_blocks) ,  LR_conv)
            model += [rb_]
        
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



        self.model = nn.Sequential(*model)
       
        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')

        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        output = self.model(x)
        output = torch.clamp(x[:,0:3,:,:] + output, min = 0, max = 1)

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

    def skip_connect(self, submodule1, submodule2):
        class skipConnet(nn.modules):
            def __init__(self,sub1,sub2):
                super(skipConnet, self).__init__()
                self.sub1 = sub1
                self.sub2 = sub2

            def forward(self, x ):
                output = x + self.sub(self.sub1(x))
                return output
        return skipConnet(submodule1, submodule2)