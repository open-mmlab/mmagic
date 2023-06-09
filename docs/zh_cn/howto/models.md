# 如何设计自己的模型

MMagic建立在MMEngine和MMCV的基础上，使用户能够快速地设计新模型，轻松地地训练和评估它们。
在本节中，您将学习如何设计自己的模型。

本指南的结构如下:

- [如何设计自己的模型](#如何设计自己的模型)
  - [MMagic中的模型概述](#mmagic中的模型概述)
  - [一个SRCNN的例子](#一个srcnn的例子)
    - [Step 1: 定义SRCNN网络](#step-1-定义srcnn网络)
    - [Step 2: 定义SRCNN的模型](#step-2-定义srcnn的模型)
    - [Step 3: 开始训练SRCNN](#step-3-开始训练srcnn)
  - [一个DCGAN的例子](#一个dcgan的例子)
    - [Step 1: 定义DCGAN的网络](#step-1-定义dcgan的网络)
    - [Step 2: 设计DCGAN的模型](#step-2-设计dcgan的模型)
    - [Step 3: 开始训练DCGAN](#step-3-开始训练dcgan)
  - [参考文献](#参考文献)

## MMagic中的模型概述

在MMagic中，一个算法可以分为两部分: **Model** 和 **Module**.

- **Model** 是最顶层的包装，并且总是继承自MMEngine中提供的 `BaseModel` 。 **Model** 负责网络前向、损耗计算、反向、参数更新等. 在MMagic中, **Model** 应该注册为 `MODELS`.
- **Module** 模块包括用于训练或推理的 **architectures** , 预定义的 **loss classes**, 以及对批量输入数据预处理的 **data preprocessors** 。 **Module** 总是作为**Model**的元素呈现。 在MMagic中, **Module** 应该注册为 **MODULES**。

以DCGAN model 模型为例，[生成器](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/editors/dcgan/dcgan_generator.py) 和 [判别器](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/editors/dcgan/dcgan_discriminator.py) 是 **Module**, 分别用于生成图像和鉴别图像真伪。 [`DCGAN`](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/editors/dcgan/dcgan.py) 是 **Model**, 它从dataloader中获取数据，交替训练生成器和鉴别器。

您可以通过以下链接找到 **Model** 和 **Module** 的实现。

- **Model**:
  - [Editors](https://github.com/open-mmlab/mmagic/tree/main/mmagic/models/editors)
- **Module**:
  - [Layers](https://github.com/open-mmlab/mmagic/tree/main/mmagic/models/layers)
  - [Losses](https://github.com/open-mmlab/mmagic/tree/main/mmagic/models/losses)
  - [Data Preprocessor](https://github.com/open-mmlab/mmagic/tree/main/mmagic/models/data_preprocessors)

## 一个SRCNN的例子

这里，我们以经典图像超分辨率模型SRCNN\[1\]的实现为例。

### Step 1: 定义SRCNN网络

SRCNN 是第一个用于单幅图像超分辨率\[1\]的深度学习方法。为了实现SRCNN的网络架构，我们需要创建一个新文件 `mmagic/models/editors/srgan/sr_resnet.py` 并执行 `class MSRResNet`。

在这一步中，我们通过继承`mmengine.models.BaseModule`来实现 `class MSRResNet`，并在`__init__`函数中定义网络架构。
特别地，我们需要使用`@MODELS.register_module()`将`class MSRResNet`的实现添加到MMagic的注册中。

```python
import torch.nn as nn
from mmengine.model import BaseModule
from mmagic.registry import MODELS

from mmagic.models.utils import (PixelShufflePack, ResidualBlockNoBN,
                                 default_init_weights, make_layer)


@MODELS.register_module()
class MSRResNet(BaseModule):
    """修改后的SRResNet。

    由 "使用生成对抗网络的照片-现实的单幅图像超级分辨率 "中的SRResNet修改而来的压缩版本。

    它使用无BN的残差块，类似于EDSR。
    目前支持x2、x3和x4上采样比例因子。

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

然后，我们实现了`class MSRResNet`的`forward` 函数, 该函数将输入张量作为输入张量，然后返回`MSRResNet`的结果。

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

在`class MSRResNet`实现后，我们需要更新`mmagic/models/editors/__init__.py`中的模型列表，以便我们可以通过`mmagic.models.editors`导入和使用`class MSRResNet`。

```python
from .srgan.sr_resnet import MSRResNet
```

### Step 2: 定义SRCNN的模型

网络架构实现后, 我们需要定义我们的模型`class BaseEditModel` 并实现`class BaseEditModel`的前向循环。

为了实现`class BaseEditModel`,
我们创建一个新文件`mmagic/models/base_models/base_edit_model.py`。
具体来说，`class BaseEditModel`继承自`mmengine.model.BaseModel`.
在`__init__`函数中，我们定义了`class BaseEditModel`的损失函数，训练, 测试配置和网络。

```python
from typing import List, Optional

import torch
from mmengine.model import BaseModel

from mmagic.registry import MODELS
from mmagic.structures import DataSample


@MODELS.register_module()
class BaseEditModel(BaseModel):
    """用于图像和视频编辑的基本模型。

    它必须包含一个生成器，将帧作为输入并输出插值帧。它也有一个用于训练的pixel-wise损失。

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

因为`mmengine.model.BaseModel`提供了算法模型的基本功能，例如权重初始化、批量输入预处理、解析损失和更新模型参数。
因此，子类继承自BaseModel，即本例中的`class BaseEditModel`,
只需要实现forward方法，该方法实现了计算损失和预测的逻辑。

具体来说，`class BaseEditModel`实现的`forward`函数将`batch_inputs`和`data_samples`作为输入，并根据模式参数返回结果。

```python
    def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs):
        """返回训练、验证、测试和简单推理过程的损失或预测。

        BaseModel的``forward``方法是一个抽象方法，它的子类必须实现这个方法。

        接受由:attr:`data_preprocessor`处理的``batch_inputs`` 和 ``data_samples``, 并根据模式参数返回结果。.

        在非分布式训练、验证和测试过程中，
        ``forward``将被``BaseModel.train_step``,
        ``BaseModel.val_step``和``BaseModel.val_step``直接调用。

        在分布式数据并行训练过程中,``MMSeparateDistributedDataParallel.train_step``将首先调用``DistributedDataParallel.forward``以启用自动梯度同步，然后调用``forward``获得训练损失。

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

具体来说，在`forward_tensor`中, `class BaseEditModel`直接返回网络的前向张量。

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

在`forward_inference`函数中，`class BaseEditModel`首先将前向张量转换为图像，然后返回该图像作为输出。

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

在`forward_train`中, `class BaseEditModel`计算损失函数，并返回一个包含损失的字典作为输出。

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

在实现了`class BaseEditModel`之后，我们需要更新
`mmagic/models/__init__.py`中的模型列表，这样我们就可以通过`mmagic.models`导入和使用`class BaseEditModel`。

```python
from .base_models.base_edit_model import BaseEditModel
```

### Step 3: 开始训练SRCNN

在实现了网络结构和SRCNN的前向循环后、 现在我们可以创建一个新的文件`configs/srcnn/srcnn_x4k915_g1_1000k_div2k.py`
来设置训练SRCNN所需的配置。

在配置文件中，我们需要指定我们的模型`class BaseEditModel`的参数，包括生成器网络结构、损失函数、额外的训练和测试配置，以及输入张量的数据预处理器。请参考[MMagic中的损失函数介绍](./losses.md)了解MMagic中损失函数的更多细节。

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

我们还需要根据创建自己的数据加载器来指定训练数据加载器和测试数据加载器。
最后，我们可以开始训练我们自己的模型:

```python
python tools/train.py configs/srcnn/srcnn_x4k915_g1_1000k_div2k.py
```

## 一个DCGAN的例子

这里，我们以经典gan模型DCGAN\[2\]的实现为例。

### Step 1: 定义DCGAN的网络

DCGAN是一种经典的图像生成对抗网络\[2\]。为了实现DCGAN的网络架构，我们需要创建两个新文件`mmagic/models/editors/dcgan/dcgan_generator.py`和`mmagic/models/editors/dcgan/dcgan_discriminator.py`，并实现生成器(`class DCGANGenerator`) 和鉴别器(`class DCGANDiscriminator`)。

在这一步中，我们实现了`class DCGANGenerator`, `class DCGANDiscriminator` 并在`__init__`函数中定义了网络架构。
特别地，我们需要使用`@MODULES.register_module()`来将生成器和鉴别器添加到MMagic的注册中。

以下面的代码为例:

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

        # 上采样的次数
        self.num_upsamples = int(np.log2(output_scale // input_scale))

        # 输出4x4的特征图
        self.noise2feat = ConvModule(
            noise_size,
            base_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=default_norm_cfg,
            act_cfg=default_act_cfg)

        # 建立上采样骨干（不包括输出层）
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

        # 输出层
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

然后，我们实现了`DCGANGenerator`的`forward`函数，该函数接受 `noise`张量或`num_batches`，然后返回`DCGANGenerator`的结果。

```python
    def forward(self, noise, num_batches=0, return_noise=False):
        noise_batch = noise_batch.to(get_module_device(self))
        x = self.noise2feat(noise_batch)
        x = self.upsampling(x)
        x = self.output_layer(x)
        return x
```

如果你想为你的网络实现特定的权重初始化方法，你需要自己添加`init_weights`函数。

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

在实现`DCGANGenerator`类之后，我们需要更新`mmagic/models/editors/__init__.py`中的模型列表，以便我们可以通过`mmagic.models.editors`导入和使用`DCGANGenerator`类。

类`DCGANDiscriminator`的实现遵循类似的逻辑，你可以在[这里](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/editors/dcgan/dcgan_discriminator.py)找到实现。

### Step 2: 设计DCGAN的模型

在实现了网络**Module**之后，我们需要定义我们的**Model**类 `DCGAN`。

你的**Model**应该继承自MMEngine提供的[`BaseModel`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/base_model.py#L16)，并实现三个函数，`train_step`, `val_step`和`test_step`。

- `train_step`:  这个函数负责更新网络的参数，由MMEngine的Loop ([`IterBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L183)或 [`EpochBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L18))调用。 `train_step`将数据批处理和[`OptimWrapper`](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/optim_wrapper.md)作为输入并返回一个日志字典。
- `val_step`: 该函数负责在训练过程中获取用于验证的输出，由 [`MultiValLoop`](https://github.com/open-mmlab/mmagic/blob/main/mmagic/engine/runner/multi_loops.py#L19)调用。
- `test_step`: 该函数负责在测试过程中获取输出，由[`MultiTestLoop`](https://github.com/open-mmlab/mmagic/blob/main/mmagic/engine/runner/multi_loops.py#L274)调用。

> 请注意，在`train_step`, `val_step`和`test_step`中，调用`DataPreprocessor`对输入数据进行预处理，然后再将它们提供给神经网络。要了解有关`DataPreprocessor`的更多信息，请参阅此[文件](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/data_preprocessors/gen_preprocessor.py) and 和本[教程](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/model.md#%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E5%99%A8datapreprocessor)。

为了简化使用，我们在MMagic中提供了[`BaseGAN`](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/base_models/base_gan.py)类，它为GAN模型实现了通用的`train_step`, `val_step`和`test_step`函数。使用`BaseGAN`作为基类，每个特定的GAN算法只需要实现`train_generator` and `train_discriminator`.

在`train_step`中，我们支持数据预处理、梯度累积(由[`OptimWrapper`](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/optim_wrapper.md)实现)和指数滑动平均(EMA)通过[(`ExponentialMovingAverage`)](https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/base_models/average_model.py#L19)实现。使用`BaseGAN.train_step`，每个特定的GAN算法只需要实现`train_generator`和`train_discriminator`。

```python
    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        data = self.data_preprocessor(data, True)
        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        # 训练判别器，使用MMEngine提供的上下文管理器
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

在`val_step`和`test_step`，我们渐进地调用数据预处理和`BaseGAN.forward`。

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

然后，我们在`DCGAN`类中实现`train_generator`和`train_discriminator`。

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

在实现了`class DCGAN`之后，我们需要更新`mmagic/models/__init__.py`中的模型列表，以便我们可以通过`mmagic.models`导入和使用`class DCGAN`。

### Step 3: 开始训练DCGAN

在实现了网络**Module**和DCGAN的**Model**之后，现在我们可以创建一个新文件`configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py`
来设置训练DCGAN所需的配置。

在配置文件中，我们需要指定模型的参数，`class DCGAN`，包括生成器网络架构和输入张量的数据预处理器。

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

我们还需要根据[创建自己的数据加载器](dataset.md)指定训练数据加载器和测试数据加载器。
最后，我们可以开始训练我们自己的模型:

```python
python tools/train.py configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py
```

## 参考文献

1. Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou. Image Super-Resolution Using Deep Convolutional Networks\[J\]. IEEE transactions on pattern analysis and machine intelligence, 2015.

2. Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
