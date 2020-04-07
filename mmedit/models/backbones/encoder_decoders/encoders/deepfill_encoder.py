import torch.nn as nn
from mmedit.models.common import ConvModule
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module
class DeepFillEncoder(nn.Module):
    """Encoder used in DeepFill model.

    This implementation follows:
    Generative Image Inpainting with Contextual Attention

    Args:
        in_channels (int): The number of input channels. Default: 5.
        norm_cfg (dict): Config dict to build norm layer. Default: None.
        act_cfg (dict): Config dict for activation layer, "elu" by default.
        encoder_type (str): Type of the encoder. Should be one of ['stage1',
            'stage2_conv', 'stage2_attention']. Default: 'stage1'.
    """

    def __init__(self,
                 in_channels=5,
                 norm_cfg=None,
                 act_cfg=dict(type='ELU'),
                 encoder_type='stage1'):
        super(DeepFillEncoder, self).__init__()
        channel_list_dict = dict(
            stage1=[32, 64, 64, 128, 128, 128],
            stage2_conv=[32, 32, 64, 64, 128, 128],
            stage2_attention=[32, 32, 64, 128, 128, 128])
        channel_list = channel_list_dict[encoder_type]
        kernel_size_list = [5, 3, 3, 3, 3, 3]
        stride_list = [1, 2, 1, 2, 1, 1]
        for i in range(6):
            ks = kernel_size_list[i]
            padding = (ks - 1) // 2
            self.add_module(
                f'enc{i + 1}',
                ConvModule(
                    in_channels,
                    channel_list[i],
                    kernel_size=ks,
                    stride=stride_list[i],
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channels = channel_list[i]

    def forward(self, x):
        for i in range(6):
            x = getattr(self, f'enc{i + 1}')(x)
        outputs = dict(out=x)
        return outputs
