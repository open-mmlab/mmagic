# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule


class MaskConvModule(ConvModule):
    """Mask convolution module.

    This is a simple wrapper for mask convolution like: 'partial conv'.
    Convolutions in this module always need a mask as extra input.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        inplace (bool): Whether to use inplace mode for activation.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in Pytorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """
    supported_conv_list = ['PConv']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.conv_cfg['type'] in self.supported_conv_list

        self.init_weights()

    def forward(self,
                x,
                mask=None,
                activate=True,
                norm=True,
                return_mask=True):
        """Forward function for partial conv2d.

        Args:
            input (torch.Tensor): Tensor with shape of (n, c, h, w).
            mask (torch.Tensor): Tensor with shape of (n, c, h, w) or
                (n, 1, h, w). If mask is not given, the function will
                work as standard conv2d. Default: None.
            activate (bool): Whether use activation layer.
            norm (bool): Whether use norm layer.
            return_mask (bool): If True and mask is not None, the updated
                mask will be returned. Default: True.

        Returns:
            Tensor or tuple: Result Tensor or 2-tuple of

                ``Tensor``: Results after partial conv.

                ``Tensor``: Updated mask will be returned if mask is given \
                    and `return_mask` is True.
        """
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                    mask = self.padding_layer(mask)
                if return_mask:
                    x, updated_mask = self.conv(
                        x, mask, return_mask=return_mask)
                else:
                    x = self.conv(x, mask, return_mask=False)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)

        if return_mask:
            return x, updated_mask

        return x
