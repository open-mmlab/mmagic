import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class AOTEncoder(nn.Module):
    """Encoder used in AOT-GAN model.

    This implementation follows:
    Aggregated contextual transformations for high-resolution image inpainting 

    Args:
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
    """

    def __init__(self, 
                 in_channels=4,
                 act_cfg=dict(type='ReLU')):
        super().__init__() 

        '''
       
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        '''
        self.reflectionpad2d = nn.ReflectionPad2d(3)
        
        self.enc1 = ConvModule(
            in_channels, 
            64,
            kernel_size=7,
            stride=1,
            act_cfg=act_cfg)
            
        self.enc2 = ConvModule(
            64, 
            128,
            kernel_size=4,
            stride=2,
            padding=1,
            act_cfg=act_cfg)
            
        self.enc3 = ConvModule(
            128, 
            256,
            kernel_size=4,
            stride=2,
            padding=1,
            act_cfg=act_cfg)
        
        # self.init_weights()
    """
    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
    """
        

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """

        
        x = self.reflectionpad2d(x)
        for i in range(3):
            x = getattr(self, f'enc{i + 1}')(x)
        '''
        x = self.encoder(x)
        '''
        return x
