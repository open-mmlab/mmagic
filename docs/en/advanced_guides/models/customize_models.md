# Design Your Own Models

MMEditing is built upon MMEngine and MMCV, which enables users to design new models quickly, train and evaluate them easily.
In this section, you will learn how to design your own models.
Here, we take the implementation of the classical image super-resolution model, SRCNN \[1\], as an example.

To implement a classical image super-resolution model, SRCNN, you need to follow these steps:

- [Step 1: Define your own network architectures](#step-1-define-your-own-network-architectures)
- [Step 2: Define the for step](#step-2-define-the-forward-loop-of-your-model)
- [Step 3: Start training](#step-3-start-training)

## Step 1: Define your own network architectures

SRCNN is the first deep learning method for single image super-resolution \[1\].
To implement the network architecture of SRCNN,
we need to create a new file `mmedit/models/editors/srgan/sr_resnet.py` and implement `class MSRResNet`.

In this step, we implement `class MSRResNet` by inheriting from `mmengine.models.BaseModule` and define the network architecture in `__init__` function.
In particular, we need to use `@MODELS.register_module()` to add the implementation of `class MSRResNet` into the registration of MMEditing.

```python
import torch.nn as nn
from mmengine.model import BaseModule
from mmedit.registry import MODELS

from mmedit.models.utils import (PixelShufflePack, ResidualBlockNoBN,
                                 default_init_weights, make_layer)


@MODELS.register_module()
class MSRResNet(BaseModule):
    """Modified SRResNet.

    A compacted version modified from SRResNet in "Photo-Realistic Single
    Image Super-Resolution Using a Generative Adversarial Network".

    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 16.
        upscale_factor (int): Upsampling factor. Support x2, x3 and x4.
            Default: 4.
    """
    _supported_upscale_factors = [2, 3, 4]

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=16,
                 upscale_factor=4):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor

        self.conv_first = nn.Conv2d(
            in_channels, mid_channels, 3, 1, 1, bias=True)
        self.trunk_net = make_layer(
            ResidualBlockNoBN, num_blocks, mid_channels=mid_channels)

        # upsampling
        if self.upscale_factor in [2, 3]:
            self.upsample1 = PixelShufflePack(
                mid_channels,
                mid_channels,
                self.upscale_factor,
                upsample_kernel=3)
        elif self.upscale_factor == 4:
            self.upsample1 = PixelShufflePack(
                mid_channels, mid_channels, 2, upsample_kernel=3)
            self.upsample2 = PixelShufflePack(
                mid_channels, mid_channels, 2, upsample_kernel=3)
        else:
            raise ValueError(
                f'Unsupported scale factor {self.upscale_factor}. '
                f'Currently supported ones are '
                f'{self._supported_upscale_factors}.')

        self.conv_hr = nn.Conv2d(
            mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(
            mid_channels, out_channels, 3, 1, 1, bias=True)

        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.init_weights()

    def init_weights(self):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """

        for m in [self.conv_first, self.conv_hr, self.conv_last]:
            default_init_weights(m, 0.1)

```

Then, we implement the `forward` function of  `class MSRResNet`, which takes as input tensor and then returns the results from `MSRResNet`.

```python
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat = self.lrelu(self.conv_first(x))
        out = self.trunk_net(feat)

        if self.upscale_factor in [2, 3]:
            out = self.upsample1(out)
        elif self.upscale_factor == 4:
            out = self.upsample1(out)
            out = self.upsample2(out)

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        upsampled_img = self.img_upsampler(x)
        out += upsampled_img
        return out
```

After the implementation of `class MSRResNet`, we need to update the model list in `mmedit/models/editors/__init__.py`, so that we can import and use `class MSRResNet` by `mmedit.models.editors`.

```python
from .srgan.sr_resnet import MSRResNet
```

## Step 2: Define the forward function of your model

After the implementation of the network architecture,
we need to define our model `class BaseEditModel` and implement the forward loop of `class BaseEditModel`.

To implement `class BaseEditModel`,
we create a new file `mmedit/models/base_models/base_edit_model.py`.
Specifically, `class BaseEditModel` inherits from `mmengine.model.BaseModel`.
In the `__init__` function, we define the loss functions, training and testing configurations, networks of `class BaseEditModel`.

```python
from typing import List, Optional

import torch
from mmengine.model import BaseModel

from mmedit.registry import MODELS
from mmedit.structures import EditDataSample, PixelData


@MODELS.register_module()
class BaseEditModel(BaseModel):
    """Base model for image and video editing.

    It must contain a generator that takes frames as inputs and outputs an
    interpolated frame. It also has a pixel-wise loss for training.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.

    Attributes:
        init_cfg (dict, optional): Initialization config dict.
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = MODELS.build(generator)

        # loss
        self.pixel_loss = MODELS.build(pixel_loss)
```

Since `mmengine.model.BaseModel` provides the basic functions of the algorithmic model,
such as weights initialize, batch inputs preprocess, parse losses, and update model parameters.
Therefore, the subclasses inherit from BaseModel, i.e., `class BaseEditModel` in this example,
only need to implement the forward method,
which implements the logic to calculate loss and predictions.

Specifically, the implemented `forward` function of `class BaseEditModel` takes as input `batch_inputs` and `data_samples` and return results according to mode arguments.

```python
    def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: Optional[List[EditDataSample]] = None,
                mode: str = 'tensor',
                **kwargs):
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``batch_inputs`` and ``data_samples`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.val_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            batch_inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            ForwardResults:

                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of
                  :obj:`BaseDataElement` for computing metric
                  and getting inference result.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict or tensor for custom use.
        """

        if mode == 'tensor':
            return self.forward_tensor(batch_inputs, data_samples, **kwargs)

        elif mode == 'predict':
            return self.forward_inference(batch_inputs, data_samples, **kwargs)

        elif mode == 'loss':
            return self.forward_train(batch_inputs, data_samples, **kwargs)
```

Specifically, in `forward_tensor`, `class BaseEditModel` returns the forward tensors of the network directly.

```python
    def forward_tensor(self, batch_inputs, data_samples=None, **kwargs):
        """Forward tensor.
            Returns result of simple forward.

        Args:
            batch_inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """

        feats = self.generator(batch_inputs, **kwargs)

        return feats
```

In `forward_inference` function, `class BaseEditModel` first converts the forward tensors to images and then returns the images as output.

```python
    def forward_inference(self, batch_inputs, data_samples=None, **kwargs):
        """Forward inference.
            Returns predictions of validation, testing, and simple inference.

        Args:
            batch_inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            List[EditDataSample]: predictions.
        """

        feats = self.forward_tensor(batch_inputs, data_samples, **kwargs)
        feats = self.data_preprocessor.destructor(feats)
        predictions = []
        for idx in range(feats.shape[0]):
            predictions.append(
                EditDataSample(
                    pred_img=PixelData(data=feats[idx].to('cpu')),
                    metainfo=data_samples[idx].metainfo))

        return predictions
```

In `forward_train`, `class BaseEditModel` calculate the loss function and returns a dictionary contains the losses as output.

```python
    def forward_train(self, batch_inputs, data_samples=None, **kwargs):
        """Forward training.
            Returns dict of losses of training.

        Args:
            batch_inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """

        feats = self.forward_tensor(batch_inputs, data_samples, **kwargs)
        gt_imgs = [data_sample.gt_img.data for data_sample in data_samples]
        batch_gt_data = torch.stack(gt_imgs)

        loss = self.pixel_loss(feats, batch_gt_data)

        return dict(loss=loss)

```

After the implementation of `class BaseEditModel`,
we need to update the model list in `mmedit/models/__init__.py`,
so that we can import and use `class BaseEditModel` by `mmedit.models`.

```python
from .base_models.base_edit_model import BaseEditModel
```

## Step 3: Start training

After implementing the network architecture and the forward loop of SRCNN,
now we can create a new file `configs/srcnn/srcnn_x4k915_g1_1000k_div2k.py`
to set the configurations needed by training SRCNN.

In the configuration file, we need to specify the parameters of our model, `class BaseEditModel`, including the generator network architecture, loss function, additional training and testing configuration, and data preprocessor of input tensors.

```python
# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='SRCNNNet',
        channels=(3, 64, 32, 3),
        kernel_sizes=(9, 1, 5),
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR'], crop_border=scale),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))
```

We also need to specify the training dataloader and testing dataloader according to [create your own dataloader](advanced_tutorial/dataset.md).
Finally we can start training our own model by：

```python
python train.py configs/srcnn/srcnn_x4k915_g1_1000k_div2k.py
```

## References

1. Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou. Image Super-Resolution Using Deep Convolutional Networks\[J\]. IEEE transactions on pattern analysis and machine intelligence, 2015.
