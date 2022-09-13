# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.utils.weight_init import xavier_init
from torch.autograd import Function
from torch.nn.modules.pooling import _MaxUnpoolNd
from torch.nn.modules.utils import _pair

from mmedit.models.registry import COMPONENTS


class MaxUnpool2dop(Function):
    """We warp the `torch.nn.functional.max_unpool2d` with an extra `symbolic`
    method, which is needed while exporting to ONNX.

    Users should not call this function directly.
    """

    @staticmethod
    def forward(ctx, input, indices, kernel_size, stride, padding,
                output_size):
        """Forward function of MaxUnpool2dop.

        Args:
            input (Tensor): Tensor needed to upsample.
            indices (Tensor): Indices output of the previous MaxPool.
            kernel_size (Tuple): Size of the max pooling window.
            stride (Tuple): Stride of the max pooling window.
            padding (Tuple): Padding that was added to the input.
            output_size (List or Tuple): The shape of output tensor.
        Returns:
            Tensor: Output tensor.
        """
        return F.max_unpool2d(input, indices, kernel_size, stride, padding,
                              output_size)

    @staticmethod
    def symbolic(g, input, indices, kernel_size, stride, padding, output_size):
        # get shape
        input_shape = g.op('Shape', input)
        const_0 = g.op('Constant', value_t=torch.tensor(0))
        const_1 = g.op('Constant', value_t=torch.tensor(1))
        batch_size = g.op('Gather', input_shape, const_0, axis_i=0)
        channel = g.op('Gather', input_shape, const_1, axis_i=0)

        # height = (height - 1) * stride + kernel_size
        height = g.op(
            'Gather',
            input_shape,
            g.op('Constant', value_t=torch.tensor(2)),
            axis_i=0)
        height = g.op('Sub', height, const_1)
        height = g.op('Mul', height,
                      g.op('Constant', value_t=torch.tensor(stride[1])))
        height = g.op('Add', height,
                      g.op('Constant', value_t=torch.tensor(kernel_size[1])))

        # width = (width - 1) * stride + kernel_size
        width = g.op(
            'Gather',
            input_shape,
            g.op('Constant', value_t=torch.tensor(3)),
            axis_i=0)
        width = g.op('Sub', width, const_1)
        width = g.op('Mul', width,
                     g.op('Constant', value_t=torch.tensor(stride[0])))
        width = g.op('Add', width,
                     g.op('Constant', value_t=torch.tensor(kernel_size[0])))

        # step of channel
        channel_step = g.op('Mul', height, width)
        # step of batch
        batch_step = g.op('Mul', channel_step, channel)

        # channel offset
        range_channel = g.op('Range', const_0, channel, const_1)
        range_channel = g.op(
            'Reshape', range_channel,
            g.op('Constant', value_t=torch.tensor([1, -1, 1, 1])))
        range_channel = g.op('Mul', range_channel, channel_step)
        range_channel = g.op('Cast', range_channel, to_i=7)  # 7 is int64

        # batch offset
        range_batch = g.op('Range', const_0, batch_size, const_1)
        range_batch = g.op(
            'Reshape', range_batch,
            g.op('Constant', value_t=torch.tensor([-1, 1, 1, 1])))
        range_batch = g.op('Mul', range_batch, batch_step)
        range_batch = g.op('Cast', range_batch, to_i=7)  # 7 is int64

        # update indices
        indices = g.op('Add', indices, range_channel)
        indices = g.op('Add', indices, range_batch)

        return g.op(
            'MaxUnpool',
            input,
            indices,
            kernel_shape_i=kernel_size,
            strides_i=stride)


class MaxUnpool2d(_MaxUnpoolNd):
    """This module is modified from Pytorch `MaxUnpool2d` module.

    Args:
      kernel_size (int or tuple): Size of the max pooling window.
      stride (int or tuple): Stride of the max pooling window.
          Default: None (It is set to `kernel_size` by default).
      padding (int or tuple): Padding that is added to the input.
          Default: 0.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride or kernel_size)
        self.padding = _pair(padding)

    def forward(self, input, indices, output_size=None):
        """Forward function of MaxUnpool2d.

        Args:
            input (Tensor): Tensor needed to upsample.
            indices (Tensor): Indices output of the previous MaxPool.
            output_size (List or Tuple): The shape of output tensor.
                Default: None.
        Returns:
            Tensor: Output tensor.
        """
        return MaxUnpool2dop.apply(input, indices, self.kernel_size,
                                   self.stride, self.padding, output_size)


@COMPONENTS.register_module()
class PlainDecoder(nn.Module):
    """Simple decoder from Deep Image Matting.

    Args:
        in_channels (int): Channel num of input features.
    """

    def __init__(self, in_channels):
        super().__init__()

        self.deconv6_1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.deconv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2)
        self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2)
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)
        self.max_unpool2d_for_onnx = MaxUnpool2d(kernel_size=2, stride=2)
        self.max_unpool2d = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def init_weights(self):
        """Init weights for the module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    def forward(self, inputs):
        """Forward function of PlainDecoder.

        Args:
            inputs (dict): Output dictionary of the VGG encoder containing:

              - out (Tensor): Output of the VGG encoder.
              - max_idx_1 (Tensor): Index of the first maxpooling layer in the
                VGG encoder.
              - max_idx_2 (Tensor): Index of the second maxpooling layer in the
                VGG encoder.
              - max_idx_3 (Tensor): Index of the third maxpooling layer in the
                VGG encoder.
              - max_idx_4 (Tensor): Index of the fourth maxpooling layer in the
                VGG encoder.
              - max_idx_5 (Tensor): Index of the fifth maxpooling layer in the
                VGG encoder.

        Returns:
            Tensor: Output tensor.
        """
        max_idx_1 = inputs['max_idx_1']
        max_idx_2 = inputs['max_idx_2']
        max_idx_3 = inputs['max_idx_3']
        max_idx_4 = inputs['max_idx_4']
        max_idx_5 = inputs['max_idx_5']
        x = inputs['out']

        max_unpool2d = self.max_unpool2d
        if torch.onnx.is_in_onnx_export():
            max_unpool2d = self.max_unpool2d_for_onnx

        out = self.relu(self.deconv6_1(x))
        out = max_unpool2d(out, max_idx_5)

        out = self.relu(self.deconv5_1(out))
        out = max_unpool2d(out, max_idx_4)

        out = self.relu(self.deconv4_1(out))
        out = max_unpool2d(out, max_idx_3)

        out = self.relu(self.deconv3_1(out))
        out = max_unpool2d(out, max_idx_2)

        out = self.relu(self.deconv2_1(out))
        out = max_unpool2d(out, max_idx_1)

        out = self.relu(self.deconv1_1(out))
        raw_alpha = self.deconv1(out)
        return raw_alpha
