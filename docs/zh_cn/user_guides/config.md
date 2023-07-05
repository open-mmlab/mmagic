# 教程1 了解MMagic的配置文件

我们在我们的配置系统中采用了模块化和继承设计，方便进行各种实验。
如果您希望查看配置文件，您可以运行 `python tools/misc/print_config.py /PATH/TO/CONFIG` 来查看完整的配置。

您可以根据以下教程了解我们配置系统的使用方法。

- [教程1：了解MMagic中的配置](#教程1-了解MMagic的配置文件)
  - [通过脚本参数修改配置](#通过脚本参数修改配置)
  - [配置文件结构](#配置文件结构)
  - [配置文件命名风格](#配置文件命名风格)
  - [EDSR的示例](#EDSR的示例)
    - [模型配置](#模型配置)
    - [数据配置](#数据配置)
      - [数据流程](#数据流程)
      - [数据加载器](#数据加载器)
    - [评估配置](#评估配置)
    - [训练和测试配置](#训练和测试配置)
    - [优化配置](#优化配置)
    - [钩子配置](#钩子配置)
    - [运行时配置](#运行时配置)
  - [StyleGAN2的示例](#StyleGAN2的示例)
    - [模型配置](#模型配置)
    - [数据集和评估器配置](#数据集和评估器配置)
    - [训练和测试配置](#训练和测试配置-1)
    - [优化配置](#优化配置-1)
    - [钩子配置](#钩子配置-1)
    - [运行时配置](#运行时配置-1)
  - [其他示例](#其他示例)
    - [修复任务的配置示例](#修复任务的配置示例)
    - [抠图任务的配置示例](#抠图任务的配置示例)
    - [恢复任务的配置示例](#恢复任务的配置示例)

## 通过脚本参数修改配置

使用 `tools/train.py`或 `tools/test.py` 来运行时，您可以通过指定 `--cfg-options` 来临时修改配置。

- 更新字典链中的配置键

  可以按照原始配置中字典键的顺序指定配置选项。例如，`--cfg-options test_cfg.use_ema=False`
  将默认的采样模型更改为原始生成器，`--cfg-options train_dataloader.batch_size=8` 将训练数据加载器的批大小更改为8。

- 更新配置列表中的键

  您的配置中有些配置字典是作为列表组成的。例如，训练流程 `train_dataloader.dataset.pipeline`
  通常是一个列表，例如 `[dict(type='LoadImageFromFile'), ...]`。如果您想要在流程中将 `'LoadImageFromFile'`
  更改为 `'LoadImageFromWebcam'`，可以指定 `--cfg-options train_dataloader.dataset.pipeline.0.type=LoadImageFromWebcam`
  。训练流程 `train_pipeline` 通常也是一个列表，例如 `[dict(type='LoadImageFromFile'), ...]`
  。如果您想要将 `'LoadImageFromFile'` 更改为 `'LoadMask'`，可以指定 `--cfg-options train_pipeline.0.type=LoadMask`。

- 更新列表/元组的值

  如果要更新的值是列表或元组，您可以设置 `--cfg-options key="[a,b]"` 或 `--cfg-options key=a,b`
  。它还允许嵌套的列表/元组值，例如 `--cfg-options key="[(a,b),(c,d)]"`。请注意，为了支持列表/元组数据类型，引号 `"`
  是必需的，并且在指定的值内引号之间不允许有空格。

## 配置文件结构

在`config/_base_`
目录下有三种基本组件类型：数据集（datasets）、模型（models）和默认运行时（default_runtime）。许多方法都可以通过使用其中的每种组件之一进行简单构建，例如AOT-GAN、EDVR、GLEAN、StyleGAN2、CycleGAN、SinGAN等。由`_base_`
组件组成的配置被称为原始配置。

对于同一文件夹下的所有配置文件，建议只有**一个**原始配置。所有其他配置文件都应该继承自原始配置。同时，最大的继承层级为3。

为了便于理解，我们建议贡献者从现有方法中继承。例如，如果基于BasicVSR进行了某些修改，用户可以通过在配置文件中指定`_base_ = ../basicvsr/basicvsr_reds4.py`
来首先继承基本的BasicVSR结构，然后修改配置文件中的必要字段。如果基于StyleGAN2进行了某些修改，用户可以通过在配置文件中指定`_base_ = ../styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py`
来首先继承基本的StyleGAN2结构，然后修改配置文件中的必要字段。

如果您正在构建一种完全不与任何现有方法共享结构的全新方法，您可以在`configs`目录下创建一个名为`xxx`的文件夹。

详细的文档请参考[MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/config.md)。

## 配置文件命名风格

配置文件按照下面的风格命名。我们建议社区贡献者使用同样的风格。

```
{model}_[module setting]_{training schedule}_{dataset}
```

`{xxx}` 是必填字段，`[yyy]` 是可选字段。

- `{model}`：模型类型，如 `stylegan`、`dcgan`、`basicvsr`、`dim`
  等。原始论文中提到的设置也包含在此字段中（例如 `Stylegan2-config-f`、`edvrm` 的 `edvrm_8xb4-600k_reds`）。
- `[module setting]`：某些模块的具体设置，包括 Encoder、Decoder、Generator、Discriminator、Normalization、loss、Activation
  等。例如 `c64n7` 的 `basicvsr-pp_c64n7_8xb1-600k_reds4`，dcgan 的学习率 `Glr4e-4_Dlr1e-4`，stylegan3 的 `gamma32.8`，sagan
  中的 `woReLUInplace`。在这个部分，来自不同子模块（例如 generator 和 discriminator）的信息用 `_` 连接起来。
- `{training_scheduler}`：训练的特定设置，包括批量大小、训练计划等。例如，学习率（例如 `lr1e-3`），使用的 GPU
  数量和批量大小（例如 `8xb32`），总迭代次数（例如 `160kiter`）或在 discriminator 中显示的图像数量（例如 `12Mimgs`）。
- `{dataset}`：数据集名称和数据大小信息，例如 `deepfillv1_4xb4_celeba-256x256` 的 `celeba-256x256`，`basicvsr_2xb4_reds4`
  的 `reds4`，`ffhq`，`lsun-car`，`celeba-hq`。

## EDSR的示例

为了帮助用户对完整的配置文件有一个基本的了解，我们对我们实现的[EDSR模型的配置文件](https://github.com/open-mmlab/mmagic/blob/main/configs/edsr/edsr_x2c64b16_g1_300k_div2k.py)
进行简要说明，如下所示。关于每个模块的更详细用法和相应的替代方案，请参考API文档和[MMEngine中的教程](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/config.md)。

### 模型配置

在MMagic的配置文件中，我们使用 `model` 字段来设置模型。

```python
model = dict(
    type='BaseEditModel',  # 模型的名称
    generator=dict(  # 生成器的配置
        type='EDSRNet',  # 生成器的类型
        in_channels=3,  # 输入的通道数
        out_channels=3,  # 输出的通道数
        mid_channels=64,  # 中间特征的通道数
        num_blocks=16,  # 主干网络中的块数
        upscale_factor=scale,  # 上采样因子
        res_scale=1,  # 用于缩放残差块中的残差
        rgb_mean=(0.4488, 0.4371, 0.4040),  # RGB图像的均值
        rgb_std=(1.0, 1.0, 1.0)),  # RGB图像的标准差
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),  # 配置像素损失
    train_cfg=dict(),  # 训练模型的配置
    test_cfg=dict(),  # 测试模型的配置
    data_preprocessor=dict(  # 数据预处理器的配置
        type='DataPreprocessor', mean=[0., 0., 0.], std=[255., 255., 255.])
)
```

### 数据配置

训练、验证和测试[运行器(runner)](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html)
都需要使用[数据加载器(Dataloader)](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader)。
为了构建数据加载器，需要设置数据集(Dataset)和数据处理流程(data pipeline)。
由于这部分的复杂性，我们使用中间变量来简化数据加载器配置的编写。

#### 数据流程

```python
train_pipeline = [  # 训练数据处理流程
    dict(type='LoadImageFromFile',  # 从文件中加载图像
         key='img',  # 在结果中查找对应路径的键名
         color_type='color',  # 图像的颜色类型
         channel_order='rgb',  # 图像的通道顺序
         imdecode_backend='cv2'),  # 解码后端
    dict(type='LoadImageFromFile',  # 从文件中加载图像
         key='gt',  # 在结果中查找对应路径的键名
         color_type='color',  # 图像的颜色类型
         channel_order='rgb',  # 图像的通道顺序
         imdecode_backend='cv2'),  # 解码后端
    dict(type='SetValues', dictionary=dict(scale=scale)),  # 将值设置给目标键名
    dict(type='PairedRandomCrop', gt_patch_size=96),  # 随机裁剪成配对图像
    dict(type='Flip',  # 翻转图像
         keys=['lq', 'gt'],  # 需要翻转的图像键名
         flip_ratio=0.5,  # 翻转比例
         direction='horizontal'),  # 翻转方向
    dict(type='Flip',  # 翻转图像
         keys=['lq', 'gt'],  # 需要翻转的图像键名
         flip_ratio=0.5,  # 翻转比例
         direction='vertical'),  # 翻转方向
    dict(type='RandomTransposeHW',  # 随机交换图像的宽高
         keys=['lq', 'gt'],  # 需要交换的图像键名
         transpose_ratio=0.5  # 交换比例
         ),
    dict(type='PackInputs')  # 从当前处理流程中收集数据的配置
]

test_pipeline = [  # 测试数据处理流程
    dict(type='LoadImageFromFile',  # 从文件中加载图像
         key='img',  # 在结果中查找对应路径的键名
         color_type='color',  # 图像的颜色类型
         channel_order='rgb',  # 图像的通道顺序
         imdecode_backend='cv2'),  # 解码后端
    dict(type='LoadImageFromFile',  # 从文件中加载图像
         key='gt',  # 在结果中查找对应路径的键名
         color_type='color',  # 图像的颜色类型
         channel_order='rgb',  # 图像的通道顺序
         imdecode_backend='cv2'),  # 解码后端
    dict(type='PackInputs')  # 从当前处理流程中收集数据的配置
]
```

#### 数据加载器

```python
dataset_type = 'BasicImageDataset'  # 数据集的类型
data_root = 'data'  # 数据的根路径

train_dataloader = dict(
    num_workers=4,  # 每个 GPU 预取数据的工作进程数
    persistent_workers=False,  # 是否保持工作进程中的数据集实例处于活动状态
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 数据采样器的类型
    dataset=dict(  # 训练数据集配置
        type=dataset_type,  # 数据集的类型
        ann_file='meta_info_DIV2K800sub_GT.txt',  # 注释文件的路径
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DIV2K',  # 数据的根路径
        data_prefix=dict(  # 图像路径的前缀
            img='DIV2K_train_LR_bicubic/X2_sub', gt='DIV2K_train_HR_sub'),
        filename_tmpl=dict(img='{}', gt='{}'),  # 文件名模板
        pipeline=train_pipeline)
)

val_dataloader = dict(
    num_workers=4,  # 每个 GPU 预取数据的工作进程数
    persistent_workers=False,  # 是否保持工作进程中的数据集实例处于活动状态
    drop_last=False,  # 是否丢弃最后一个不完整的批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 数据采样器的类型
    dataset=dict(  # 验证数据集配置
        type=dataset_type,  # 数据集的类型
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root=data_root + '/Set5',  # 数据的根路径
        data_prefix=dict(img='LRbicx2', gt='GTmod12'),  # 图像路径的前缀
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader
```

### 评估配置

[评估器](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html)用于计算在验证集和测试集上训练模型的指标。
评估器的配置包括一个或多个指标配置：

```python
val_evaluator = [
    dict(type='MAE'),  # 要评估的指标名称
    dict(type='PSNR', crop_border=scale),  # 要评估的指标名称
    dict(type='SSIM', crop_border=scale),  # 要评估的指标名称
]

test_evaluator = val_evaluator  # 测试评估器的配置与验证评估器相同

```

### 训练和测试配置

MMEngine的运行器使用Loop来控制训练、验证和测试过程。
用户可以使用这些字段设置最大训练迭代次数和验证间隔。

```python
train_cfg = dict(
    type='IterBasedTrainLoop',  # 训练循环类型的名称
    max_iters=300000,  # 总迭代次数
    val_interval=5000,  # 验证间隔迭代次数
)
val_cfg = dict(type='ValLoop')  # 验证循环类型的名称
test_cfg = dict(type='TestLoop')  # 测试循环类型的名称
```

### 优化配置

`optim_wrapper`是用于配置优化相关设置的字段。
优化器包装器不仅提供优化器的功能，还支持梯度裁剪、混合精度训练等功能。在[optimizer wrapper教程](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html)
中可以了解更多信息。

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.00001),
)  # 用于构建优化器的配置，支持所有与PyTorch中参数相同的优化器。
```

`param_scheduler`是一个配置优化超参数（如学习率和动量）调整方法的字段。
用户可以结合多个调度器来创建所需的参数调整策略。
在[parameter scheduler教程](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html)中可以了解更多信息。

```python
param_scheduler = dict(  # 学习策略的配置
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5)
```

### 钩子配置

用户可以将钩子（hooks）附加到训练、验证和测试循环中，在运行过程中插入一些操作。有两个不同的钩子字段，一个是`default_hooks`
，另一个是`custom_hooks`。

`default_hooks`是一个包含钩子配置的字典。`default_hooks`
是运行时必需的钩子，它们具有默认的优先级，不应修改。如果未设置，默认值将被使用。要禁用默认钩子，用户可以将其配置设置为`None`。

```python
default_hooks = dict(  # 用于构建默认钩子的配置
    checkpoint=dict(  # 配置检查点钩子
        type='CheckpointHook',
        interval=5000,  # 保存间隔为5000次迭代
        save_optimizer=True,
        by_epoch=False,  # 以迭代次数计数
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),  # 配置注册日志钩子
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
```

`custom_hooks`是一个钩子配置的列表。用户可以开发自己的钩子并将其插入到该字段中。

```python
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]  # 可视化钩子的配置
```

### 运行时配置

```python
default_scope = 'mmagic'  # 用于设置注册表位置
env_cfg = dict(  # 设置分布式训练的参数，端口也可以设置
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'  # 日志记录的级别
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)  # 用于构建日志处理器
load_from = None  # 从给定路径加载模型作为预训练模型。这不会恢复训练。
resume = False  # 从给定路径恢复检查点，训练将从检查点保存的时期继续。
```

## StyleGAN2的示例

以[Stylegan2 在 1024x1024 分辨率上的配置文件](https://github.com/open-mmlab/mmagic/blob/main/configs//styleganv2/stylegan2_c2_8xb4-fp16-global-800kiters_quicktest-ffhq-256x256.py)
为例，我们根据不同的功能模块介绍配置中的各个字段。

### 模型配置

除了包括生成器、鉴别器等神经网络组件之外，还需要`data_preprocessor`、`loss_config`等字段，其中一些还包含`ema_config`。

`data_preprocessor`负责处理数据加载器输出的一个批次数据。

`loss_config`负责设置损失项的权重。

`ema_config`负责为生成器执行指数移动平均（EMA）操作。

```python
model = dict(
    type='StyleGAN2',  # 模型的名称
    data_preprocessor=dict(type='DataPreprocessor'),  # 数据预处理器的配置，通常包括图像归一化和填充
    generator=dict(  # 生成器的配置
        type='StyleGANv2Generator',  # 生成器的名称
        out_size=1024,  # 生成器的输出分辨率
        style_channels=512),  # 生成器的风格通道数
    discriminator=dict(  # 鉴别器的配置
        type='StyleGAN2Discriminator',  # 鉴别器的名称
        in_size=1024),  # 鉴别器的输入分辨率
    ema_config=dict(  # EMA的配置
        type='ExponentialMovingAverage',  # 平均模型的具体类型
        interval=1,  # EMA操作的间隔
        momentum=0.9977843871238888),  # EMA操作的动量
    loss_config=dict(  # 损失项的配置
        r1_loss_weight=80.0,  # r1梯度惩罚的权重
        r1_interval=16,  # r1梯度惩罚的间隔
        norm_mode='HWC',  # r1梯度惩罚的归一化模式
        g_reg_interval=4,  # 生成器的正则化间隔
        g_reg_weight=8.0,  # 生成器的正则化权重
        pl_batch_shrink=2))  # 路径长度正则化中缩减批次大小的因子
```

### 数据集和评估器配置

训练、验证和测试[runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html)
需要使用数据加载器[Dataloaders](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader)。
需要设置数据集和数据处理流程来构建数据加载器。由于这部分的复杂性，我们使用中间变量来简化数据加载器配置的编写。

```python
dataset_type = 'BasicImageDataset'  # 数据集类型，将用于定义数据集
data_root = './data/ffhq/'  # 数据的根目录

train_pipeline = [  # 训练数据处理流程
    dict(type='LoadImageFromFile', key='img'),  # 第一个处理流程，从文件路径加载图像
    dict(type='Flip', keys=['img'], direction='horizontal'),  # 图像翻转的数据增强处理流程
    dict(type='PackInputs', keys=['img'])  # 最后一个处理流程，格式化注释数据（如果有）并决定哪些键应该打包到data_samples中
]
val_pipeline = [
    dict(type='LoadImageFromFile', key='img'),  # 第一个处理流程，从文件路径加载图像
    dict(type='PackInputs', keys=['img'])  # 最后一个处理流程，格式化注释数据（如果有）并决定哪些键应该打包到data_samples中
]
train_dataloader = dict(  # 训练数据加载器的配置
    batch_size=4,  # 单个GPU的批次大小
    num_workers=8,  # 每个单个GPU的数据预取工作线程数
    persistent_workers=True,  # 如果为True，则数据加载器将在一个epoch结束后不会关闭工作进程，这可以加速训练速度。
    sampler=dict(  # 训练数据采样器的配置
        type='InfiniteSampler',
        # 用于迭代训练的InfiniteSampler。参考 https://github.com/open-mmlab/mmengine/blob/fe0eb0a5bbc8bf816d5649bfdd34908c258eb245/mmengine/dataset/sampler.py#L107
        shuffle=True),  # 是否随机打乱训练数据
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline))
val_dataloader = dict(  # 验证数据加载器的配置
    batch_size=4,  # 单个GPU的批次大小
    num_workers=8,  # 每个单个GPU的数据预取工作线程数
    dataset=dict(  # 验证数据集的配置
        type=dataset_type,
        data_root=data_root,
        pipeline=val_pipeline),
    sampler=dict(  # 验证数据采样器的配置
        type='DefaultSampler',
        # 支持分布式和非分布式训练的DefaultSampler。参考 https://github.com/open-mmlab/mmengine/blob/fe0eb0a5bbc8bf816d5649bfdd34908c258eb245/mmengine/dataset/sampler.py#L14
        shuffle=False),  # 是否随机打乱验证数据
    persistent_workers=True)
test_dataloader = val_dataloader  # 测试数据加载器的配置与验证数据加载器相同
```

[评估器](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html)用于计算在验证和测试数据集上训练模型的指标。
评估器的配置由一个或多个指标配置组成：

```python
val_evaluator = dict(  # 验证评估器的配置
    type='Evaluator',  # 评估类型
    metrics=[  # 指标的配置
        dict(
            type='FrechetInceptionDistance',
            prefix='FID-Full-50k',
            fake_nums=50000,
            inception_style='StyleGAN',
            sample_model='ema'),
        dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K'),
        dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
    ])
test_evaluator = val_evaluator  # 测试评估器的配置与验证评估器相同
```

### 训练和测试配置

MMEngine的runner使用Loop来控制训练、验证和测试过程。
用户可以使用以下字段设置最大训练迭代次数和验证间隔：

```python
train_cfg = dict(  # 训练配置
    by_epoch=False,  # 设置`by_epoch`为False以使用基于迭代的训练
    val_begin=1,  # 开始验证的迭代次数
    val_interval=10000,  # 验证间隔
    max_iters=800002)  # 最大训练迭代次数
val_cfg = dict(type='MultiValLoop')  # 验证循环类型
test_cfg = dict(type='MultiTestLoop')  # 测试循环类型
```

### 优化配置

`optim_wrapper`是配置优化相关设置的字段。
优化器包装器不仅提供优化器的功能，还支持梯度裁剪、混合精度训练等功能。在[optimizer wrapper tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html)
中可以找到更多信息。

```python
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        optimizer=dict(type='Adam', lr=0.0016, betas=(0, 0.9919919678228657))),
    discriminator=dict(
        optimizer=dict(
            type='Adam',
            lr=0.0018823529411764706,
            betas=(0, 0.9905854573074332))))
```

`param_scheduler`是一个配置优化超参数（如学习率和动量）调整方法的字段。
用户可以组合多个调度器来创建所需的参数调整策略。
在[parameter scheduler tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html)中可以找到更多信息。
由于StyleGAN2不使用参数调度器，我们以[CycleGAN](https://github.com/open-mmlab/mmagic/blob/main/configs/cyclegan/cyclegan_lsgan-id0-resnet-in_1xb1-250kiters_summer2winter.py)
的配置作为示例：

```python
# CycleGAN配置中的参数调度器
param_scheduler = dict(
    type='LinearLrInterval',  # 调度器的类型
    interval=400,  # 更新学习率的间隔
    by_epoch=False,  # 调度器按迭代调用
    start_factor=0.0002,  # 在第一次迭代中乘以参数值的数值
    end_factor=0,  # 在线性变化过程结束时乘以参数值的数值
    begin=40000,  # 调度器的起始迭代次数
    end=80000)  # 调度器的结束迭代次数
```

### 钩子配置

用户可以在训练、验证和测试循环中附加钩子，以在运行过程中插入一些操作。这里有两个不同的钩子字段，一个是`default_hooks`
，另一个是`custom_hooks`。

`default_hooks`是一个钩子配置的字典。`default_hooks`
是在运行时必须的钩子。它们具有默认的优先级，不应该被修改。如果没有设置，运行器将使用默认值。要禁用一个默认钩子，用户可以将其配置设置为`None`。

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        less_keys=['FID-Full-50k/fid'],
        greater_keys=['IS-50k/is'],
        save_optimizer=True,
        save_best='FID-Full-50k/fid'))
```

`custom_hooks` 是一个钩子配置的列表。用户可以开发自己的钩子并将它们插入到这个字段中。

```python
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]
```

### 运行时配置

```python
default_scope = 'mmagic'  # 默认的注册表作用域，用于查找模块。参考 https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

# 环境配置
env_cfg = dict(
    cudnn_benchmark=True,  # 是否启用cudnn基准测试
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # 设置多进程参数
    dist_cfg=dict(backend='nccl')  # 设置分布式参数
)

log_level = 'INFO'  # 日志级别
log_processor = dict(
    type='LogProcessor',  # 日志处理器，用于处理运行时日志
    by_epoch=False)  # 按迭代打印日志
load_from = None  # 从给定路径加载模型检查点作为预训练模型
resume = False  # 是否从`load_from`定义的检查点恢复训练。如果`load_from`为`None`，将恢复`work_dir`中的最新检查点
```

## 其他示例

### 修复任务的配置示例

为了帮助用户对修复系统的完整配置和模块有一个基本的了解，我们对全局和局部修复的配置进行简要注释，如下所示。有关更详细的用法和每个模块的替代选项，请参考API文档。

```python
model = dict(
    type='GLInpaintor',  # 修复模型的名称
    data_preprocessor=dict(
        type='DataPreprocessor',  # 数据预处理器的名称
        mean=[127.5],  # 数据归一化时使用的均值
        std=[127.5],  # 数据归一化时使用的标准差
    ),
    encdec=dict(
        type='GLEncoderDecoder',  # 编码器-解码器的名称
        encoder=dict(type='GLEncoder', norm_cfg=dict(type='SyncBN')),  # 编码器的配置
        decoder=dict(type='GLDecoder', norm_cfg=dict(type='SyncBN')),  # 解码器的配置
        dilation_neck=dict(
            type='GLDilationNeck', norm_cfg=dict(type='SyncBN'))),  # 膨胀模块的配置
    disc=dict(
        type='GLDiscs',  # 判别器的名称
        global_disc_cfg=dict(
            in_channels=3,  # 判别器的输入通道数
            max_channels=512,  # 判别器中间通道的最大数量
            fc_in_channels=512 * 4 * 4,  # 最后一个全连接层的输入通道数
            fc_out_channels=1024,  # 最后一个全连接层的输出通道数
            num_convs=6,  # 判别器中使用的卷积层数量
            norm_cfg=dict(type='SyncBN')  # 归一化层的配置
        ),
        local_disc_cfg=dict(
            in_channels=3,  # 判别器的输入通道数
            max_channels=512,  # 判别器中间通道的最大数量
            fc_in_channels=512 * 4 * 4,  # 最后一个全连接层的输入通道数
            fc_out_channels=1024,  # 最后一个全连接层的输出通道数
            num_convs=5,  # 判别器中使用的卷积层数量
            norm_cfg=dict(type='SyncBN')  # 归一化层的配置
        ),
    ),
    loss_gan=dict(
        type='GANLoss',  # GAN损失的名称
        gan_type='vanilla',  # GAN损失的类型
        loss_weight=0.001  # GAN损失的权重
    ),
    loss_l1_hole=dict(
        type='L1Loss',  # L1损失的类型
        loss_weight=1.0  # L1损失的权重
    )
)

train_cfg = dict(
    type='IterBasedTrainLoop',  # 训练循环的类型
    max_iters=500002,  # 总迭代次数
    val_interval=50000  # 验证间隔的迭代次数
)
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型

val_evaluator = [
    dict(type='MAE', mask_key='mask', scaling=100),  # 用于评估的指标名称
    dict(type='PSNR'),  # 用于评估的指标名称
    dict(type='SSIM'),  # 用于评估的指标名称
]
test_evaluator = val_evaluator

input_shape = (256, 256)  # 输入图像的形状

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),  # 加载图像的配置
    dict(
        type='LoadMask',  # 加载掩膜的类型
        mask_mode='bbox',  # 掩膜的类型
        mask_config=dict(
            max_bbox_shape=(128, 128),  # 边界框的形状
            max_bbox_delta=40,  # 边界框高度和宽度的变化范围
            min_margin=20,  # 边界框与图像边界的最小间距
            img_shape=input_shape)),  # 输入图像的形状
    dict(
        type='Crop',  # 裁剪的类型
        keys=['gt'],  # 需要裁剪的图像的键
        crop_size=(384, 384),  # 裁剪后的大小
        random_crop=True,  # 是否随机裁剪
    ),
    dict(
        type='Resize',  # 调整大小的类型
        keys=['gt'],  # 需要调整大小的图像的键
        scale=input_shape,  # 调整大小的比例
        keep_ratio=False,  # 是否保持比例
    ),
    dict(
        type='Normalize',  # 归一化的类型
        keys=['gt_img'],  # 需要归一化的图像的键
        mean=[127.5] * 3,  # 归一化时使用的均值
        std=[127.5] * 3,  # 归一化时使用的标准差
        to_rgb=False),  # 是否将图像通道转换为RGB
    dict(type='GetMaskedImage'),  # 获取掩膜图像的配置
    dict(type='PackInputs'),  # 收集当前流水线中的数据的配置
]

test_pipeline = train_pipeline  # 构建测试/验证流水线

dataset_type = 'BasicImageDataset'  # 数据集的类型
data_root = 'data/places'  # 数据的根路径

train_dataloader = dict(
    batch_size=12,  # 单个GPU的批处理大小
    num_workers=4,  # 每个单个GPU预取数据的工作线程数
    persistent_workers=False,  # 是否保持工作线程的数据集实例
    sampler=dict(type='InfiniteSampler', shuffle=False),  # 数据采样器的类型
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,  # 数据集的类型
        data_root=data_root,  # 数据的根路径
        data_prefix=dict(gt='data_large'),  # 图像路径的前缀
        ann_file='meta/places365_train_challenge.txt',  # 注释文件的路径
        test_mode=False,
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,  # 单个GPU的批处理大小
    num_workers=4,  # 每个单个GPU预取数据的工作线程数
    persistent_workers=False,  # 是否保持工作线程的数据集实例
    drop_last=False,  # 是否丢弃最后一个不完整的批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 数据采样器的类型
    dataset=dict(  # 验证数据集的配置
        type=dataset_type,  # 数据集的类型
        data_root=data_root,  # 数据的根路径
        data_prefix=dict(gt='val_large'),  # 图像路径的前缀
        ann_file='meta/places365_val.txt',  # 注释文件的路径
        test_mode=True,
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader

model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')  # 模型包装器的名称

optim_wrapper = dict(  # 用于构建优化器的配置，支持PyTorch中的所有优化器，其参数与PyTorch中的优化器的参数相同
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper', optimizer=dict(type='Adam', lr=0.0004)),
    disc=dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=0.0004)))

default_scope = 'mmagic'  # 用于设置注册表位置
save_dir = './work_dirs'  # 保存模型检查点和日志的目录
exp_name = 'gl_places'  # 实验名称

default_hooks = dict(  # 用于构建默认挂钩
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),  # 注册记录器挂钩的配置
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(  # 设置检查点挂钩的配置
        type='CheckpointHook',
        interval=50000,
        by_epoch=False,
        out_dir=save_dir),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(  # 用于设置分布式训练的参数，也可以设置端口
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]  # 可视化后端的名称
visualizer = dict(  # 用于构建可视化器的配置
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]  # 用于构建自定义挂钩

log_level = 'INFO'  # 记录级别
log_processor = dict(type='LogProcessor', by_epoch=False)  # 用于构建日志处理器

load_from = None  # 从给定路径加载预训练模型
resume = False  # 从给定路径恢复检查点

find_unused_parameters = False  # 是否在DDP中设置未使用的参数
```

### 抠图任务的配置示例

为了帮助用户对完整的配置有一个基本的了解，我们对我们实现的原始DIM（Deep Image
Matting）模型的配置进行了简要的注释，如下所示。有关每个模块的更详细用法和相应的替代方案，请参阅API文档。

```python
# 模型设置
model = dict(
    type='DIM',  # 模型的名称（我们称之为mattor）。
    data_preprocessor=dict(  # 数据预处理器的配置
        type='DataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        proc_inputs='normalize',
        proc_trimap='rescale_to_zero_one',
        proc_gt='rescale_to_zero_one',
    ),
    backbone=dict(  # 骨干网络的配置。
        type='SimpleEncoderDecoder',  # 骨干网络的类型。
        encoder=dict(  # 编码器的配置。
            type='VGG16'),  # 编码器的类型。
        decoder=dict(  # 解码器的配置。
            type='PlainDecoder')),  # 解码器的类型。
    pretrained='./weights/vgg_state_dict.pth',  # 要加载的编码器的预训练权重。
    loss_alpha=dict(  # alpha损失的配置。
        type='CharbonnierLoss',  # 预测的alpha融合图像的损失类型。
        loss_weight=0.5),  # alpha损失的权重。
    loss_comp=dict(  # 合成损失的配置。
        type='CharbonnierCompLoss',  # 合成损失的类型。
        loss_weight=0.5),  # 合成损失的权重。
    train_cfg=dict(  # DIM模型的训练配置。
        train_backbone=True,  # 在DIM阶段1中，训练骨干网络。
        train_refiner=False),  # 在DIM阶段1中，不训练refiner。
    test_cfg=dict(  # DIM模型的测试配置。
        refine=False,  # 是否使用refiner输出作为输出，在阶段1中我们不使用它。
        resize_method='pad',
        resize_mode='reflect',
        size_divisor=32,
    ),
)

# 数据设置
dataset_type = 'AdobeComp1kDataset'  # 数据集类型，用于定义数据集。
data_root = 'data/adobe_composition-1k'  # 数据的根路径。

train_pipeline = [  # 训练数据处理流程。
    dict(
        type='LoadImageFromFile',  # 从文件加载alpha融合图像。
        key='alpha',  # 注释文件中alpha融合图像的键。该流程将从路径`alpha_path`读取alpha融合图像。
        color_type='grayscale'),  # 加载为灰度图像，具有形状（高度，宽度）。
    dict(
        type='LoadImageFromFile',  # 从文件加载图像。
        key='fg'),  # 要加载的图像的键。该流程将从路径`fg_path`读取前景图像。
    dict(
        type='LoadImageFromFile',  # 从文件加载图像。
        key='bg'),  # 要加载的图像的键。该流程将从路径`bg_path`读取背景图像。
    dict(
        type='LoadImageFromFile',  # 从文件加载图像。
        key='merged'),  # 要加载的图像的键。该流程将从路径`merged_path`读取合并图像。
    dict(
        type='CropAroundUnknown',  # 在未知区域（半透明区域）周围裁剪图像。
        keys=['alpha', 'merged', 'fg', 'bg'],  # 要裁剪的图像。
        crop_sizes=[320, 480, 640]),  # 候选裁剪大小。
    dict(
        type='Flip',  # 翻转图像的增强流程。
        keys=['alpha', 'merged', 'fg', 'bg']),  # 要翻转的图像。
    dict(
        type='Resize',  # 调整图像大小的增强流程。
        keys=['alpha', 'merged', 'fg', 'bg'],  # 要调整大小的图像。
        scale=(320, 320),  # 目标大小。
        keep_ratio=False),  # 是否保持高度和宽度之间的比例。
    dict(
        type='GenerateTrimap',  # 从alpha融合图像生成trimap。
        kernel_size=(1, 30)),  # 腐蚀/膨胀内核的大小范围。
    dict(type='PackInputs'),  # 从当前流程中收集数据的配置
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',  # 加载alpha融合图像。
        key='alpha',  # 注释文件中alpha融合图像的键。该流程将从路径`alpha_path`读取alpha融合图像。
        color_type='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',  # 从文件加载图像。
        key='trimap',  # 要加载的图像的键。该流程将从路径`trimap_path`读取trimap。
        color_type='grayscale',  # 加载为灰度图像，具有形状（高度，宽度）。
        save_original_img=True),  # 保存trimap的副本用于计算指标。它将以键`ori_trimap`保存。
    dict(
        type='LoadImageFromFile',  # 从文件加载图像。
        key='merged'),  # 要加载的图像的键。该流程将从路径`merged_path`读取合并图像。
    dict(type='PackInputs'),  # 从当前流程中收集数据的配置
]

train_dataloader = dict(
    batch_size=1,  # 单个GPU的批处理大小
    num_workers=4,  # 每个单个GPU预提取数据的工作线程数
    persistent_workers=False,  # 是否保持工作线程Dataset实例处于活动状态
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 数据采样器的类型
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,  # 数据集的类型
        data_root=data_root,  # 数据的根路径
        ann_file='training_list.json',  # 注释文件的路径
        test_mode=False,
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,  # 单个GPU的批处理大小
    num_workers=4,  # 每个单个GPU预提取数据的工作线程数
    persistent_workers=False,  # 是否保持工作线程Dataset实例处于活动状态
    drop_last=False,  # 是否丢弃最后一个不完整的批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 数据采样器的类型
    dataset=dict(  # 验证数据集的配置
        type=dataset_type,  # 数据集的类型
        data_root=data_root,  # 数据的根路径
        ann_file='test_list.json',  # 注释文件的路径
        test_mode=True,
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='SAD'),  # 要评估的指标名称
    dict(type='MattingMSE'),  # 要评估的指标名称
    dict(type='GradientError'),  # 要评估的指标名称
    dict(type='ConnectivityError'),  # 要评估的指标名称
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop',  # 训练循环类型的名称
    max_iters=1_000_000,  # 总迭代次数
    val_interval=40000,  # 验证间隔迭代次数
)
val_cfg = dict(type='ValLoop')  # 验证循环类型的名称
test_cfg = dict(type='TestLoop')  # 测试循环类型的名称

# 优化器
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.00001),
    )
)  # 用于构建优化器的配置，支持PyTorch中所有优化器，其参数也与PyTorch中的参数相同。

default_scope = 'mmagic'  # 用于设置注册表位置
save_dir = './work_dirs'  # 保存当前实验的模型检查点和日志的目录。

default_hooks = dict(  # 用于构建默认钩子
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),  # 注册日志记录器钩子的配置
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(  # 配置检查点钩子
        type='CheckpointHook',
        interval=40000,  # 保存间隔为40000次迭代。
        by_epoch=False,  # 按迭代计数。
        out_dir=save_dir),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(  # 设置分布式训练的参数，也可以设置端口
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'  # 日志级别
log_processor = dict(type='LogProcessor', by_epoch=False)  # 用于构建日志处理器的配置。

# 导入模块并设置MMDetection的配置
load_from = None  # 从给定路径加载模型作为预训练模型，这不会恢复训练。
resume = False  # 从给定路径恢复检查点，训练将从检查点保存的时期恢复。
```

### 恢复任务的配置示例

为了帮助用户对完整配置有一个基本的理解，我们对我们实现的EDSR模型的配置进行了简要的注释，如下所示。有关更详细的用法和每个模块的相应替代方案，请参阅API文档。

```python
exp_name = 'edsr_x2c64b16_1x16_300k_div2k'  # 实验名称
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

load_from = None  # 基于预训练的x2模型

scale = 2  # 上采样的比例
# 模型设置
model = dict(
    type='BaseEditModel',  # 模型名称
    generator=dict(  # 生成器的配置
        type='EDSRNet',  # 生成器的类型
        in_channels=3,  # 输入的通道数
        out_channels=3,  # 输出的通道数
        mid_channels=64,  # 中间特征的通道数
        num_blocks=16,  # 主干网络中的块数
        upscale_factor=scale,  # 上采样因子
        res_scale=1,  # 用于缩放残差块中的残差
        rgb_mean=(0.4488, 0.4371, 0.4040),  # 图像的RGB均值
        rgb_std=(1.0, 1.0, 1.0)),  # 图像的RGB标准差
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean')  # 像素损失的配置
train_cfg = dict(),  # 训练模型的配置
test_cfg = dict(),  # 测试模型的配置
data_preprocessor = dict(  # 数据预处理器的配置
    type='DataPreprocessor', mean=[0., 0., 0.], std=[255., 255., 255.]))

train_pipeline = [  # 训练数据处理的流程
    dict(type='LoadImageFromFile',  # 从文件加载图像
         key='img',  # 结果中寻找对应路径的关键字
         color_type='color',  # 图像的颜色类型
         channel_order='rgb',  # 图像的通道顺序
         imdecode_backend='cv2'),  # 解码后端
    dict(type='LoadImageFromFile',  # 从文件加载图像
         key='gt',  # 结果中寻找对应路径的关键字
         color_type='color',  # 图像的颜色类型
         channel_order='rgb',  # 图像的通道顺序
         imdecode_backend='cv2'),  # 解码后端
    dict(type='SetValues', dictionary=dict(scale=scale)),  # 设置目标关键字的值
    dict(type='PairedRandomCrop', gt_patch_size=96),  # 随机裁剪配对图像
    dict(type='Flip',  # 翻转图像
         keys=['lq', 'gt'],  # 需要翻转的图像
         flip_ratio=0.5,  # 翻转的比例
         direction='horizontal'),  # 翻转的方向
    dict(type='Flip',  # 翻转图像
         keys=['lq', 'gt'],  # 需要翻转的图像
         flip_ratio=0.5,  # 翻转的比例
         direction='vertical'),  # 翻转的方向
    dict(type='RandomTransposeHW',  # 随机转置图像的高度和宽度
         keys=['lq', 'gt'],  # 需要转置的图像
         transpose_ratio=0.5  # 转置的比例
         ),
    dict(type='PackInputs')  # 收集当前流程中的数据的配置
]
test_pipeline = [  # 测试流程
    dict(type='LoadImageFromFile',  # 从文件加载图像
         key='img',  # 结果中寻找对应路径的关键字
         color_type='color',  # 图像的颜色类型
         channel_order='rgb',  # 图像的通道顺序
         imdecode_backend='cv2'),  # 解码后端
    dict(type='LoadImageFromFile',  # 从文件加载图像
         key='gt',  # 结果中寻找对应路径的关键字
         color_type='color',  # 图像的颜色类型
         channel_order='rgb',  # 图像的通道顺序
         imdecode_backend='cv2'),  # 解码后端
    dict(type='ToTensor', keys=['img', 'gt']),  # 将图像转换为张量
    dict(type='PackInputs')  # 收集当前流程中的数据的配置
]

# 数据集设置
dataset_type = 'BasicImageDataset'  # 数据集的类型
data_root = 'data'  # 数据的根路径

train_dataloader = dict(
    num_workers=4,  # 每个GPU预提取数据的工作进程数
    persistent_workers=False,  # 是否保持工作进程中的数据集实例处于活动状态
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 数据采样器的类型
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,  # 数据集的类型
        ann_file='meta_info_DIV2K800sub_GT.txt',  # 注释文件的路径
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DIV2K',  # 数据的根路径
        data_prefix=dict(  # 图像路径的前缀
            img='DIV2K_train_LR_bicubic/X2_sub', gt='DIV2K_train_HR_sub'),
        filename_tmpl=dict(img='{}', gt='{}'),  # 文件名模板
        pipeline=train_pipeline))
val_dataloader = dict(
    num_workers=4,  # 每个GPU预提取数据的工作进程数
    persistent_workers=False,  # 是否保持工作进程中的数据集实例处于活动状态
    drop_last=False,  # 是否丢弃最后一个不完整的批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 数据采样器的类型
    dataset=dict(  # 验证数据集的配置
        type=dataset_type,  # 数据集的类型
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root=data_root + '/Set5',  # 数据的根路径
        data_prefix=dict(img='LRbicx2', gt='GTmod12'),  # 图像路径的前缀
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),  # 用于评估的指标的名称
    dict(type='PSNR', crop_border=scale),  # 用于评估的指标的名称
    dict(type='SSIM', crop_border=scale),  # 用于评估的指标的名称
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=5000)  # 训练循环类型的配置
val_cfg = dict(type='ValLoop')  # 验证循环类型的名称
test_cfg = dict(type='TestLoop')  # 测试循环类型的名称

# 优化器
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.00001),
    )
)  # 用于构建优化器的配置，支持PyTorch中所有优化器，参数与PyTorch中的相同。

param_scheduler = dict(  # 学习策略的配置
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5)

default_hooks = dict(  # 用于构建默认钩子
    checkpoint=dict(  # 配置保存检查点的钩子
        type='CheckpointHook',
        interval=5000,  # 保存间隔为5000次迭代
        save_optimizer=True,
        by_epoch=False,  # 以迭代计数
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),  # 注册记录器钩子的配置
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

default_scope = 'mmagic'  # 用于设置注册表位置
save_dir = './work_dirs'  # 保存当前实验的模型检查点和日志的目录。

env_cfg = dict(  # 设置分布式训练的参数，端口也可以设置
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'  # 记录的级别
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)  # 用于构建日志处理器

load_from = None  # 从给定路径加载模型作为预训练模型，这不会恢复训练。
resume = False  # 从给定路径恢复检查点，训练将从检查点保存的时期恢复。
```
