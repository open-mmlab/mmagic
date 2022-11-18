# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import Parameter
from torch.nn.modules.batchnorm import SyncBatchNorm
from torch.nn.utils import spectral_norm

from mmedit.registry import MODELS, MODULES
from .biggan_snmodule import SNConv2d, SNLinear


class SNConvModule(ConvModule):
    """Spectral Normalization ConvModule.

    In this module, we inherit default ``mmcv.cnn.ConvModule`` and adopt
    spectral normalization. The spectral normalization is proposed in:
    Spectral Normalization for Generative Adversarial Networks.

    Args:
        with_spectral_norm (bool, optional): Whether to use Spectral
            Normalization. Defaults to False.
        spectral_norm_cfg (dict, optional): Config for Spectral Normalization.
            Defaults to None.
    """

    def __init__(self,
                 *args,
                 with_spectral_norm=False,
                 spectral_norm_cfg=None,
                 **kwargs):
        super().__init__(*args, with_spectral_norm=False, **kwargs)
        self.with_spectral_norm = with_spectral_norm
        self.spectral_norm_cfg = deepcopy(
            spectral_norm_cfg) if spectral_norm_cfg else dict()

        self.sn_eps = self.spectral_norm_cfg.get('eps', 1e-6)
        self.sn_style = self.spectral_norm_cfg.get('sn_style', 'torch')

        if self.with_spectral_norm:
            if self.sn_style == 'torch':
                self.conv = spectral_norm(self.conv, eps=self.sn_eps)
            elif self.sn_style == 'ajbrock':
                self.snconv_kwargs = deepcopy(kwargs) if kwargs else dict()
                if 'act_cfg' in self.snconv_kwargs.keys():
                    self.snconv_kwargs.pop('act_cfg')
                if 'norm_cfg' in self.snconv_kwargs.keys():
                    self.snconv_kwargs.pop('norm_cfg')
                if 'order' in self.snconv_kwargs.keys():
                    self.snconv_kwargs.pop('order')
                self.conv = SNConv2d(
                    *args, **self.snconv_kwargs, eps=self.sn_eps)
            else:
                raise NotImplementedError(
                    f'{self.sn_style} style spectral Norm is not supported yet'
                )


@MODULES.register_module()
class BigGANGenResBlock(nn.Module):
    """Residual block used in BigGAN's generator.

    Args:
        in_channels (int): The channel number of the input feature map.
        out_channels (int): The channel number of the output feature map.
        dim_after_concat (int): The channel number of the noise concatenated
            with the class vector.
        act_cfg (dict, optional): Config for the activation layer. Defaults to
            dict(type='ReLU').
        upsample_cfg (dict, optional): Config for the upsampling operation.
            Defaults to dict(type='nearest', scale_factor=2).
        sn_eps (float, optional): Epsilon value for spectral normalization.
            Defaults to 1e-6.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `ajbrock`.
        with_spectral_norm (bool, optional): Whether to use spectral
            normalization in this block. Defaults to True.
        input_is_label (bool, optional): Whether the input of BNs' linear layer
            is raw label instead of class vector. Defaults to False.
        auto_sync_bn (bool, optional): Whether to use synchronized batch
            normalization. Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dim_after_concat,
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 sn_eps=1e-6,
                 sn_style='ajbrock',
                 with_spectral_norm=True,
                 input_is_label=False,
                 auto_sync_bn=True):
        super().__init__()
        self.activation = MODELS.build(act_cfg)
        self.upsample_cfg = deepcopy(upsample_cfg)
        self.with_upsample = upsample_cfg is not None
        if self.with_upsample:
            self.upsample_layer = MODELS.build(self.upsample_cfg)
        self.learnable_sc = in_channels != out_channels or self.with_upsample
        if self.learnable_sc:
            self.shortcut = SNConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm,
                spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))
        # Here in_channels of BigGANGenResBlock equal to num_features of
        # BigGANConditionBN
        self.bn1 = BigGANConditionBN(
            in_channels,
            dim_after_concat,
            sn_eps=sn_eps,
            sn_style=sn_style,
            input_is_label=input_is_label,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)
        # Here out_channels of BigGANGenResBlock equal to num_features of
        # BigGANConditionBN
        self.bn2 = BigGANConditionBN(
            out_channels,
            dim_after_concat,
            sn_eps=sn_eps,
            sn_style=sn_style,
            input_is_label=input_is_label,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)

        self.conv1 = SNConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

        self.conv2 = SNConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

    def forward(self, x, y):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.
            y (torch.Tensor): Label tensor or class embedding concatenated with
                noise tensor.

        Returns:
            torch.Tensor: Output feature map tensor.
        """
        x0 = self.bn1(x, y)
        x0 = self.activation(x0)
        if self.with_upsample:
            x0 = self.upsample_layer(x0)
            x = self.upsample_layer(x)
        x0 = self.conv1(x0)
        x0 = self.bn2(x0, y)
        x0 = self.activation(x0)
        x0 = self.conv2(x0)
        if self.learnable_sc:
            x = self.shortcut(x)
        return x0 + x


@MODULES.register_module()
class BigGANConditionBN(nn.Module):
    """Conditional Batch Normalization used in BigGAN.

    Args:
        num_features (int): The channel number of the input feature map tensor.
        linear_input_channels (int): The channel number of the linear layers'
            input tensor.
        bn_eps (float, optional): Epsilon value for batch normalization.
            Defaults to 1e-5.
        sn_eps (float, optional): Epsilon value for spectral normalization.
            Defaults to 1e-6.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `ajbrock`.
        momentum (float, optional): The value used for the running_mean and
            running_var computation. Defaults to 0.1.
        input_is_label (bool, optional): Whether the input of BNs' linear layer
            is raw label instead of class vector. Defaults to False.
        with_spectral_norm (bool, optional): Whether to use spectral
            normalization. Defaults to True.
        auto_sync_bn (bool, optional): Whether to use synchronized batch
            normalization. Defaults to True.
    """

    def __init__(self,
                 num_features,
                 linear_input_channels,
                 bn_eps=1e-5,
                 sn_eps=1e-6,
                 sn_style='ajbrock',
                 momentum=0.1,
                 input_is_label=False,
                 with_spectral_norm=True,
                 auto_sync_bn=True):
        super().__init__()
        assert num_features > 0
        if linear_input_channels > 0:
            self.use_cbn = True
        else:
            self.use_cbn = False
        # Prepare gain and bias layers
        if self.use_cbn:
            if not input_is_label:
                self.gain = nn.Linear(
                    linear_input_channels, num_features, bias=False)
                self.bias = nn.Linear(
                    linear_input_channels, num_features, bias=False)
                # please pay attention if shared_embedding is False
                if with_spectral_norm:
                    if sn_style == 'torch':
                        self.gain = spectral_norm(self.gain, eps=sn_eps)
                        self.bias = spectral_norm(self.bias, eps=sn_eps)
                    elif sn_style == 'ajbrock':
                        self.gain = SNLinear(
                            linear_input_channels,
                            num_features,
                            bias=False,
                            eps=sn_eps)
                        self.bias = SNLinear(
                            linear_input_channels,
                            num_features,
                            bias=False,
                            eps=sn_eps)
                    else:
                        raise NotImplementedError('sn style')
            else:
                self.gain = nn.Embedding(linear_input_channels, num_features)
                self.bias = nn.Embedding(linear_input_channels, num_features)

        self.bn = nn.BatchNorm2d(
            num_features,
            eps=bn_eps,
            momentum=momentum,
            affine=not self.use_cbn)

        if auto_sync_bn and dist.is_initialized():
            self.bn = SyncBatchNorm.convert_sync_batchnorm(self.bn)

    def forward(self, x, y):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.
            y (torch.Tensor): Label tensor or class embedding concatenated with
                noise tensor.

        Returns:
            torch.Tensor: Output feature map tensor.
        """
        if self.use_cbn:
            # Calculate class-conditional gains and biases
            gain = (1. + self.gain(y)).view(y.size(0), -1, 1, 1)
            bias = self.bias(y).view(y.size(0), -1, 1, 1)
            out = self.bn(x)
            out = out * gain + bias
        else:
            out = self.bn(x)
        return out


@MODULES.register_module()
class SelfAttentionBlock(nn.Module):
    """Self-Attention block used in BigGAN.

    Args:
        in_channels (int): The channel number of the input feature map.
        with_spectral_norm (bool, optional): Whether to use spectral
            normalization. Defaults to True.
        sn_eps (float, optional): Epsilon value for spectral normalization.
            Defaults to 1e-6.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `ajbrock`.
    """

    def __init__(self,
                 in_channels,
                 with_spectral_norm=True,
                 sn_eps=1e-6,
                 sn_style='ajbrock'):
        super(SelfAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.theta = SNConvModule(
            self.in_channels,
            self.in_channels // 8,
            kernel_size=1,
            padding=0,
            bias=False,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))
        self.phi = SNConvModule(
            self.in_channels,
            self.in_channels // 8,
            kernel_size=1,
            padding=0,
            bias=False,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))
        self.g = SNConvModule(
            self.in_channels,
            self.in_channels // 2,
            kernel_size=1,
            padding=0,
            bias=False,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))
        self.o = SNConvModule(
            self.in_channels // 2,
            self.in_channels,
            kernel_size=1,
            padding=0,
            bias=False,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))
        # Learnable gain parameter
        self.gamma = Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.

        Returns:
            torch.Tensor: Output feature map tensor.
        """
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.in_channels // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(-1, self.in_channels // 2,
                                                    x.shape[2], x.shape[3]))
        return self.gamma * o + x


@MODULES.register_module()
class BigGANDiscResBlock(nn.Module):
    """Residual block used in BigGAN's discriminator.

    Args:
        in_channels (int): The channel number of the input tensor.
        out_channels (int): The channel number of the output tensor.
        act_cfg (dict, optional): Config for the activation layer. Defaults to
            dict(type='ReLU', inplace=False).
        sn_eps (float, optional): Epsilon value for spectral normalization.
            Defaults to 1e-6.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `ajbrock`.
        with_downsample (bool, optional): Whether to use downsampling in this
            block. Defaults to True.
        with_spectral_norm (bool, optional): Whether to use spectral
            normalization. Defaults to True.
        is_head_block (bool, optional): Whether this block is the first block
            of BigGAN. Defaults to False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='ReLU', inplace=False),
                 sn_eps=1e-6,
                 sn_style='ajbrock',
                 with_downsample=True,
                 with_spectral_norm=True,
                 is_head_block=False):
        super().__init__()
        self.activation = MODELS.build(act_cfg)
        self.with_downsample = with_downsample
        self.is_head_block = is_head_block
        if self.with_downsample:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.learnable_sc = in_channels != out_channels or self.with_downsample
        if self.learnable_sc:
            self.shortcut = SNConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm,
                spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

        self.conv1 = SNConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

        self.conv2 = SNConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

    def forward_sc(self, x):
        """Forward function of shortcut.

        Args:
            x (torch.Tensor): Input feature map tensor.

        Returns:
            torch.Tensor: Output tensor of shortcut.
        """
        if self.is_head_block:
            if self.with_downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.shortcut(x)
        else:
            if self.learnable_sc:
                x = self.shortcut(x)
            if self.with_downsample:
                x = self.downsample(x)
        return x

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.

        Returns:
            torch.Tensor: Output feature map tensor.
        """
        if self.is_head_block:
            x0 = x
        else:
            x0 = self.activation(x)
        x0 = self.conv1(x0)
        x0 = self.activation(x0)
        x0 = self.conv2(x0)
        if self.with_downsample:
            x0 = self.downsample(x0)
        x1 = self.forward_sc(x)
        return x0 + x1


@MODULES.register_module()
class BigGANDeepGenResBlock(nn.Module):
    """Residual block used in BigGAN-Deep's generator.

    Args:
        in_channels (int): The channel number of the input feature map.
        out_channels (int): The channel number of the output feature map.
        dim_after_concat (int): The channel number of the noise concatenated
            with the class vector.
        act_cfg (dict, optional): Config for the activation layer. Defaults to
            dict(type='ReLU').
        upsample_cfg (dict, optional): Config for the upsampling operation.
            Defaults to dict(type='nearest', scale_factor=2).
        sn_eps (float, optional): Epsilon value for spectral normalization.
            Defaults to 1e-6.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `ajbrock`.
        bn_eps (float, optional): Epsilon value for batch normalization.
            Defaults to 1e-5.
        with_spectral_norm (bool, optional): Whether to use spectral
            normalization in this block. Defaults to True.
        input_is_label (bool, optional): Whether the input of BNs' linear layer
            is raw label instead of class vector. Defaults to False.
        auto_sync_bn (bool, optional): Whether to use synchronized batch
            normalization. Defaults to True.
        channel_ratio (int, optional): The ratio of the input channels' number
            to the hidden channels' number. Defaults to 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dim_after_concat,
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 sn_eps=1e-6,
                 sn_style='ajbrock',
                 bn_eps=1e-5,
                 with_spectral_norm=True,
                 input_is_label=False,
                 auto_sync_bn=True,
                 channel_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = self.in_channels // channel_ratio
        self.activation = MODELS.build(act_cfg)
        self.upsample_cfg = deepcopy(upsample_cfg)
        self.with_upsample = upsample_cfg is not None
        if self.with_upsample:
            self.upsample_layer = MODELS.build(self.upsample_cfg)
        # Here in_channels of BigGANGenResBlock equal to num_features of
        # BigGANConditionBN
        self.bn1 = BigGANConditionBN(
            in_channels,
            dim_after_concat,
            sn_eps=sn_eps,
            sn_style=sn_style,
            bn_eps=bn_eps,
            input_is_label=input_is_label,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)
        # Here out_channels of BigGANGenResBlock equal to num_features of
        # BigGANConditionBN
        self.bn2 = BigGANConditionBN(
            self.hidden_channels,
            dim_after_concat,
            sn_eps=sn_eps,
            sn_style=sn_style,
            bn_eps=bn_eps,
            input_is_label=input_is_label,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)

        self.bn3 = BigGANConditionBN(
            self.hidden_channels,
            dim_after_concat,
            sn_eps=sn_eps,
            sn_style=sn_style,
            bn_eps=bn_eps,
            input_is_label=input_is_label,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)

        self.bn4 = BigGANConditionBN(
            self.hidden_channels,
            dim_after_concat,
            sn_eps=sn_eps,
            sn_style=sn_style,
            bn_eps=bn_eps,
            input_is_label=input_is_label,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)

        self.conv1 = SNConvModule(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

        self.conv2 = SNConvModule(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

        self.conv3 = SNConvModule(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

        self.conv4 = SNConvModule(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

    def forward(self, x, y):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.
            y (torch.Tensor): Label tensor or class embedding concatenated with
                noise tensor.

        Returns:
            torch.Tensor: Output feature map tensor.
        """
        x0 = self.bn1(x, y)
        x0 = self.activation(x0)
        x0 = self.conv1(x0)

        x0 = self.bn2(x0, y)
        x0 = self.activation(x0)
        # Drop channels in x  if necessary
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]
        # unsample both h and x at this point
        if self.with_upsample:
            x0 = self.upsample_layer(x0)
            x = self.upsample_layer(x)
        x0 = self.conv2(x0)

        x0 = self.bn3(x0, y)
        x0 = self.activation(x0)
        x0 = self.conv3(x0)

        x0 = self.bn4(x0, y)
        x0 = self.activation(x0)
        x0 = self.conv4(x0)
        return x0 + x


@MODULES.register_module()
class BigGANDeepDiscResBlock(nn.Module):
    """Residual block used in BigGAN-Deep's discriminator.

    Args:
        in_channels (int): The channel number of the input tensor.
        out_channels (int): The channel number of the output tensor.
        channel_ratio (int, optional): The ratio of the input channels' number
            to the hidden channels' number. Defaults to 4.
        act_cfg (dict, optional): Config for the activation layer. Defaults to
            dict(type='ReLU', inplace=False).
        sn_eps (float, optional): Epsilon value for spectral normalization.
            Defaults to 1e-6.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `ajbrock`.
        with_downsample (bool, optional): Whether to use downsampling in this
            block. Defaults to True.
        with_spectral_norm (bool, optional): Whether to use spectral
            normalization. Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 channel_ratio=4,
                 act_cfg=dict(type='ReLU', inplace=False),
                 sn_eps=1e-6,
                 sn_style='ajbrock',
                 with_downsample=True,
                 with_spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = self.out_channels // channel_ratio
        self.activation = MODELS.build(act_cfg)
        self.with_downsample = with_downsample

        if self.with_downsample:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.learnable_sc = (in_channels != out_channels)
        if self.learnable_sc:
            self.shortcut = SNConvModule(
                in_channels=in_channels,
                out_channels=out_channels - in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm,
                spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

        self.conv1 = SNConvModule(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style),
            order=('act', 'conv', 'norm'))

        self.conv2 = SNConvModule(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style),
            order=('act', 'conv', 'norm'))

        self.conv3 = SNConvModule(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style),
            order=('act', 'conv', 'norm'))

        self.conv4 = SNConvModule(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style))

    def forward_sc(self, x):
        """Forward function of shortcut.

        Args:
            x (torch.Tensor): Input feature map tensor.

        Returns:
            torch.Tensor: Output tensor of shortcut.
        """
        if self.with_downsample:
            x = self.downsample(x)
        if self.learnable_sc:
            x0 = self.shortcut(x)
            x = torch.cat([x, x0], dim=1)
        return x

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.

        Returns:
            torch.Tensor: Output feature map tensor.
        """

        x0 = self.conv1(x)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = self.activation(x0)
        # downsample
        if self.with_downsample:
            x0 = self.downsample(x0)
        x0 = self.conv4(x0)
        x1 = self.forward_sc(x)
        return x0 + x1
