# How to design your own models

MMagic is built upon MMEngine and MMCV, which enables users to design new models quickly, train and evaluate them easily.
In this section, you will learn how to design your own models.

The structure of this guide are as follows:

- [How to design your own models](#how-to-design-your-own-models)
  - [Overview of models in MMagic](#overview-of-models-in-mmagic)
  - [An example of SRCNN](#an-example-of-srcnn)
    - [Step 1: Define the network of SRCNN](#step-1-define-the-network-of-srcnn)
    - [Step 2: Define the model of SRCNN](#step-2-define-the-model-of-srcnn)
    - [Step 3: Start training SRCNN](#step-3-start-training-srcnn)
  - [An example of DCGAN](#an-example-of-dcgan)
    - [Step 1: Define the network of DCGAN](#step-1-define-the-network-of-dcgan)
    - [Step 2: Design the model of DCGAN](#step-2-design-the-model-of-dcgan)
    - [Step 3: Start training DCGAN](#step-3-start-training-dcgan)
  - [References](#references)

## Overview of models in MMagic

In MMagic, one algorithm can be splited two compents: **Model** and **Module**.

- **Model** are topmost wrappers and always inherint from `BaseModel` provided in MMEngine. **Model** is responsible to network forward, loss calculation and backward, parameters updating, etc. In MMagic, **Model** should be registered as `MODELS`.
- **Module** includes the neural network **architectures** to train or inference, pre-defined **loss classes**, and **data preprocessors** to preprocess the input data batch. **Module** always present as elements of **Model**. In MMagic, **Module** should be registered as **MODULES**.

Take DCGAN model as an example, [generator](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/editors/dcgan/dcgan_generator.py) and [discriminator](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/editors/dcgan/dcgan_discriminator.py) are the **Module**, which generate images and discriminate real or fake images. [`DCGAN`](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/editors/dcgan/dcgan.py) is the **Model**, which take data from dataloader and train generator and discriminator alternatively.

You can find the implementation of **Model** and **Module** by the following link.

- **Model**:
  - [Editors](https://github.com/open-mmlab/mmagic/tree/main/mmagic/models/editors)
- **Module**:
  - [Layers](https://github.com/open-mmlab/mmagic/tree/main/mmagic/models/layers)
  - [Losses](https://github.com/open-mmlab/mmagic/tree/main/mmagic/models/losses)
  - [Data Preprocessor](https://github.com/open-mmlab/mmagic/tree/main/mmagic/models/data_preprocessors)

## An example of SRCNN

Here, we take the implementation of the classical image super-resolution model, SRCNN \[1\], as an example.

### Step 1: Define the network of SRCNN

SRCNN is the first deep learning method for single image super-resolution \[1\].
To implement the network architecture of SRCNN,
we need to create a new file `mmagic/models/editors/srgan/sr_resnet.py` and implement `class MSRResNet`.

In this step, we implement `class MSRResNet` by inheriting from `mmengine.models.BaseModule` and define the network architecture in `__init__` function.
In particular, we need to use `@MODELS.register_module()` to add the implementation of `class MSRResNet` into the registration of MMagic.

```python
import torch.nn as nn
from mmengine.model import BaseModule
from mmagic.registry import MODELS

from mmagic.models.utils import (PixelShufflePack, ResidualBlockNoBN,
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

After the implementation of `class MSRResNet`, we need to update the model list in `mmagic/models/editors/__init__.py`, so that we can import and use `class MSRResNet` by `mmagic.models.editors`.

```python
from .srgan.sr_resnet import MSRResNet
```

### Step 2: Define the model of SRCNN

After the implementation of the network architecture,
we need to define our model `class BaseEditModel` and implement the forward loop of `class BaseEditModel`.

To implement `class BaseEditModel`,
we create a new file `mmagic/models/base_models/base_edit_model.py`.
Specifically, `class BaseEditModel` inherits from `mmengine.model.BaseModel`.
In the `__init__` function, we define the loss functions, training and testing configurations, networks of `class BaseEditModel`.

```python
from typing import List, Optional

import torch
from mmengine.model import BaseModel

from mmagic.registry import MODELS
from mmagic.structures import DataSample


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
                data_samples: Optional[List[DataSample]] = None,
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
            List[DataSample]: predictions.
        """

        feats = self.forward_tensor(batch_inputs, data_samples, **kwargs)
        feats = self.data_preprocessor.destructor(feats)
        predictions = []
        for idx in range(feats.shape[0]):
            predictions.append(
                DataSample(
                    pred_img=feats[idx].to('cpu'),
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
we need to update the model list in `mmagic/models/__init__.py`,
so that we can import and use `class BaseEditModel` by `mmagic.models`.

```python
from .base_models.base_edit_model import BaseEditModel
```

### Step 3: Start training SRCNN

After implementing the network architecture and the forward loop of SRCNN,
now we can create a new file `configs/srcnn/srcnn_x4k915_g1_1000k_div2k.py`
to set the configurations needed by training SRCNN.

In the configuration file, we need to specify the parameters of our model, `class BaseEditModel`, including the generator network architecture, loss function, additional training and testing configuration, and data preprocessor of input tensors. Please refer to the [Introduction to the loss in MMagic](./losses.md) for more details of losses in MMagic.

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
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))
```

We also need to specify the training dataloader and testing dataloader according to create your own dataloader.
Finally we can start training our own model by：

```python
python tools/train.py configs/srcnn/srcnn_x4k915_g1_1000k_div2k.py
```

## An example of DCGAN

Here, we take the implementation of the classical gan model, DCGAN \[2\], as an example.

### Step 1: Define the network of DCGAN

DCGAN is a classical image generative adversarial network \[2\]. To implement the network architecture of DCGAN, we need to create tow new files `mmagic/models/editors/dcgan/dcgan_generator.py` and `mmagic/models/editors/dcgan/dcgan_discriminator.py`, and implement generator (`class DCGANGenerator`) and discriminator (`class DCGANDiscriminator`).

In this step, we implement `class DCGANGenerator`, `class DCGANDiscriminator` and define the network architecture in `__init__` function.
In particular, we need to use `@MODULES.register_module()` to add the generator and discriminator into the registration of MMagic.

Take the following code as example:

```python
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmengine.logging import MMLogger
from mmengine.model.utils import normal_init

from mmagic.models.builder import MODULES
from ..common import get_module_device


@MODULES.register_module()
class DCGANGenerator(nn.Module):
    def __init__(self,
                 output_scale,
                 out_channels=3,
                 base_channels=1024,
                 input_scale=4,
                 noise_size=100,
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='Tanh'),
                 pretrained=None):
        super().__init__()
        self.output_scale = output_scale
        self.base_channels = base_channels
        self.input_scale = input_scale
        self.noise_size = noise_size

        # the number of times for upsampling
        self.num_upsamples = int(np.log2(output_scale // input_scale))

        # output 4x4 feature map
        self.noise2feat = ConvModule(
            noise_size,
            base_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=default_norm_cfg,
            act_cfg=default_act_cfg)

        # build up upsampling backbone (excluding the output layer)
        upsampling = []
        curr_channel = base_channels
        for _ in range(self.num_upsamples - 1):
            upsampling.append(
                ConvModule(
                    curr_channel,
                    curr_channel // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='ConvTranspose2d'),
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg))

            curr_channel //= 2

        self.upsampling = nn.Sequential(*upsampling)

        # output layer
        self.output_layer = ConvModule(
            curr_channel,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=None,
            act_cfg=out_act_cfg)
```

Then, we implement the `forward` function of  `DCGANGenerator`, which takes as `noise` tensor or `num_batches` and then returns the results from `DCGANGenerator`.

```python
    def forward(self, noise, num_batches=0, return_noise=False):
        noise_batch = noise_batch.to(get_module_device(self))
        x = self.noise2feat(noise_batch)
        x = self.upsampling(x)
        x = self.output_layer(x)
        return x
```

If you want to implement specific weights initialization method for you network, you need add `init_weights` function by yourself.

```python
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, _BatchNorm):
                    nn.init.normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
```

After the implementation of class `DCGANGenerator`, we need to update the model list in `mmagic/models/editors/__init__.py`, so that we can import and use class `DCGANGenerator` by `mmagic.models.editors`.

Implementation of Class `DCGANDiscriminator` follows the similar logic, and you can find the implementation [here](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/editors/dcgan/dcgan_discriminator.py).

### Step 2: Design the model of DCGAN

After the implementation of the network **Module**, we need to define our **Model** class `DCGAN`.

Your **Model** should inherit from [`BaseModel`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/base_model.py#L16) provided by MMEngine and implement three functions, `train_step`, `val_step` and `test_step`.

- `train_step`: This function is responsible to update the parameters of the network and called by MMEngine's Loop ([`IterBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L183) or [`EpochBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L18)). `train_step` take data batch and [`OptimWrapper`](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/optim_wrapper.md) as input and return a dict of log.
- `val_step`: This function is responsible for getting output for validation during the training process. and is called by [`MultiValLoop`](https://github.com/open-mmlab/mmagic/blob/main/mmagic/engine/runner/multi_loops.py#L19).
- `test_step`: This function is responsible for getting output in test process and is called by [`MultiTestLoop`](https://github.com/open-mmlab/mmagic/blob/main/mmagic/engine/runner/multi_loops.py#L274).

> Note that, in `train_step`, `val_step` and `test_step`, `DataPreprocessor` is called to preprocess the input data batch before feed them to the neural network. To know more about `DataPreprocessor` please refer to this [file](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/data_preprocessors/gen_preprocessor.py) and this [tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/model.md#%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E5%99%A8datapreprocessor).

For simplify using, we provide [`BaseGAN`](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/base_models/base_gan.py) class in MMagic, which implements generic `train_step`, `val_step` and `test_step` function for GAN models. With `BaseGAN` as base class, each specific GAN algorithm only need to implement `train_generator` and `train_discriminator`.

In `train_step`, we support data preprocessing, gradient accumulation (realized by [`OptimWrapper`](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/optim_wrapper.md)) and expontial moving averate (EMA) realized by [(`ExponentialMovingAverage`)](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/base_models/average_model.py#L19). With `BaseGAN.train_step`, each specific GAN algorithm only need to implement `train_generator` and `train_discriminator`.

```python
    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        data = self.data_preprocessor(data, True)
        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        # train discriminator, use context manager provided by MMEngine
        with disc_optimizer_wrapper.optim_context(self.discriminator):
            # train_discriminator should be implemented!
            log_vars = self.train_discriminator(
                **data, optimizer_wrapper=disc_optimizer_wrapper)

        # add 1 to `curr_iter` because iter is updated in train loop.
        # Whether to update the generator. We update generator with
        # discriminator is fully updated for `self.n_discriminator_steps`
        # iterations. And one full updating for discriminator contains
        # `disc_accu_counts` times of grad accumulations.
        if (curr_iter + 1) % (self.discriminator_steps * disc_accu_iters) == 0:
            set_requires_grad(self.discriminator, False)
            gen_optimizer_wrapper = optim_wrapper['generator']
            gen_accu_iters = gen_optimizer_wrapper._accumulative_counts

            log_vars_gen_list = []
            # init optimizer wrapper status for generator manually
            gen_optimizer_wrapper.initialize_count_status(
                self.generator, 0, self.generator_steps * gen_accu_iters)
            # update generator, use context manager provided by MMEngine
            for _ in range(self.generator_steps * gen_accu_iters):
                with gen_optimizer_wrapper.optim_context(self.generator):
                    # train_generator should be implemented!
                    log_vars_gen = self.train_generator(
                        **data, optimizer_wrapper=gen_optimizer_wrapper)

                log_vars_gen_list.append(log_vars_gen)
            log_vars_gen = gather_log_vars(log_vars_gen_list)
            log_vars_gen.pop('loss', None)  # remove 'loss' from gen logs

            set_requires_grad(self.discriminator, True)

            # only do ema after generator update
            if self.with_ema_gen and (curr_iter + 1) >= (
                    self.ema_start * self.discriminator_steps *
                    disc_accu_iters):
                self.generator_ema.update_parameters(
                    self.generator.module
                    if is_model_wrapper(self.generator) else self.generator)

            log_vars.update(log_vars_gen)

        # return the log dict
        return log_vars
```

In `val_step` and `test_step`, we call data preprocessing and `BaseGAN.forward` progressively.

```python
    def val_step(self, data: dict) -> SampleList:
        data = self.data_preprocessor(data)
        # call `forward`
        outputs = self(**data)
        return outputs

    def test_step(self, data: dict) -> SampleList:
        data = self.data_preprocessor(data)
        # call `orward`
        outputs = self(**data)
        return outputs
```

Then, we implement `train_generator` and `train_discriminator` in `DCGAN` class.

```python
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmagic.registry import MODELS
from .base_gan import BaseGAN


@MODELS.register_module()
class DCGAN(BaseGAN):
    def disc_loss(self, disc_pred_fake: Tensor,
                  disc_pred_real: Tensor) -> Tuple:
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.binary_cross_entropy_with_logits(
            disc_pred_fake, 0. * torch.ones_like(disc_pred_fake))
        losses_dict['loss_disc_real'] = F.binary_cross_entropy_with_logits(
            disc_pred_real, 1. * torch.ones_like(disc_pred_real))

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake: Tensor) -> Tuple:
        losses_dict = dict()
        losses_dict['loss_gen'] = F.binary_cross_entropy_with_logits(
            disc_pred_fake, 1. * torch.ones_like(disc_pred_fake))
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def train_discriminator(
            self, inputs, data_sample,
            optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        real_imgs = inputs['img']

        num_batches = real_imgs.shape[0]

        noise_batch = self.noise_fn(num_batches=num_batches)
        with torch.no_grad():
            fake_imgs = self.generator(noise=noise_batch, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real)
        optimizer_wrapper.update_params(parsed_losses)
        return log_vars

    def train_generator(self, inputs, data_sample,
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        num_batches = inputs['img'].shape[0]

        noise = self.noise_fn(num_batches=num_batches)
        fake_imgs = self.generator(noise=noise, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars
```

After the implementation of `class DCGAN`, we need to update the model list in `mmagic/models/__init__.py`, so that we can import and use `class DCGAN` by `mmagic.models`.

### Step 3: Start training DCGAN

After implementing the network **Module** and the **Model** of DCGAN,
now we can create a new file `configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py`
to set the configurations needed by training DCGAN.

In the configuration file, we need to specify the parameters of our model, `class DCGAN`, including the generator network architecture and data preprocessor of input tensors.

```python
# model settings
model = dict(
    type='DCGAN',
    noise_size=100,
    data_preprocessor=dict(type='GANDataPreprocessor'),
    generator=dict(type='DCGANGenerator', output_scale=64, base_channels=1024),
    discriminator=dict(
        type='DCGANDiscriminator',
        input_scale=64,
        output_scale=4,
        out_channels=1))
```

We also need to specify the training dataloader and testing dataloader according to [create your own dataloader](dataset.md).
Finally we can start training our own model by：

```python
python tools/train.py configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py
```

## References

1. Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou. Image Super-Resolution Using Deep Convolutional Networks\[J\]. IEEE transactions on pattern analysis and machine intelligence, 2015.

2. Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
