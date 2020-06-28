import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import CONV_LAYERS


@CONV_LAYERS.register_module(name='PConv')
class PartialConv2d(nn.Conv2d):
    """Implementation for partial convolution.

    Image Inpainting for Irregular Holes Using Partial Convolutions
    [https://arxiv.org/abs/1804.07723]

    Args:
        multi_channel (bool): If True, the mask is multi-channle. Otherwise,
            the mask is single-channel.
        eps (float): Need to be changed for mixed precision training.
            For mixed precision training, you need change 1e-8 to 1e-6.
    """

    def __init__(self, *args, multi_channel=False, eps=1e-8, **kwargs):
        super(PartialConv2d, self).__init__(*args, **kwargs)

        # whether the mask is multi-channel or not
        self.multi_channel = multi_channel
        self.eps = eps

        if self.multi_channel:
            out_channels, in_channels = self.out_channels, self.in_channels
        else:
            out_channels, in_channels = 1, 1

        self.register_buffer(
            'weight_mask_updater',
            torch.ones(out_channels, in_channels, self.kernel_size[0],
                       self.kernel_size[1]))

        self.mask_kernel_numel = np.prod(self.weight_mask_updater.shape[1:4])
        self.mask_kernel_numel = np.asscalar(self.mask_kernel_numel)

    def forward(self, input, mask=None, return_mask=True):
        """Forward function for partial conv2d.

        Args:
            input (torch.Tensor): Tensor with shape of (n, c, h, w).
            mask (torch.Tensor): Tensor with shape of (n, c, h, w) or
                (n, 1, h, w). If mask is not given, the function will
                work as standard conv2d. Default: None.
            return_mask (bool): If True and mask is not None, the updated
                mask will be returned. Default: True.

        Returns:
            torch.Tensor : Results after partial conv.\
            torch.Tensor : Updated mask will be returned if mask is given and \
                ``return_mask`` is True.
        """
        assert input.dim() == 4
        if mask is not None:
            assert mask.dim() == 4
            if self.multi_channel:
                assert mask.shape[1] == input.shape[1]
            else:
                assert mask.shape[1] == 1

        # update mask and compute mask ratio
        if mask is not None:
            with torch.no_grad():

                updated_mask = F.conv2d(
                    mask,
                    self.weight_mask_updater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation)
                mask_ratio = self.mask_kernel_numel / (updated_mask + self.eps)

                updated_mask = torch.clamp(updated_mask, 0, 1)
                mask_ratio = mask_ratio * updated_mask

        # standard conv2d
        if mask is not None:
            input = input * mask
        raw_out = super(PartialConv2d, self).forward(input)

        if mask is not None:
            if self.bias is None:
                output = raw_out * mask_ratio
            else:
                # compute new bias when mask is given
                bias_view = self.bias.view(1, self.out_channels, 1, 1)
                output = (raw_out - bias_view) * mask_ratio + bias_view
                output = output * updated_mask
        else:
            output = raw_out

        if return_mask and mask is not None:
            return output, updated_mask
        else:
            return output
