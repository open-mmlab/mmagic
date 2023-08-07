# 教程 6：可视化

图像的可视化是衡量图像处理、编辑和合成质量的重要手段。
在配置文件中使用 `visualizer` 可以在训练或测试时保存可视化结果。您可以跟随[MMEngine文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/visualization.md)学习可视化的用法。MMagic提供了一套丰富的可视化功能。
在本教程中，我们将介绍MMagic提供的可视化函数的用法。

- [教程6:可视化](#教程6:可视化)
  - [概述](#概述)
    - [GAN的可视化配置](#gan的可视化配置)
    - [图像翻译模型的可视化配置](#图像翻译模型的可视化配置)
    - [扩散模型的可视化配置](#扩散模型的可视化配置)
    - [图像补全模型的可视化配置](#图像补全模型的可视化配置)
    - [图像抠图模型的可视化配置](#图像抠图模型的可视化配置)
    - [SISR/VSR/VFI等模型的可视化配置](#sisrvsrvfi等模型的可视化配置)
  - [可视化钩子](#可视化钩子)
  - [可视化器](#可视化器)
  - [可视化后端](#可视化后端)
    - [在不同的存储后端可视化](#在不同的存储后端可视化)

## 概述

建议先学习 [设计文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/design/visualization.md) 里关于可视化的基本概念。

在MMagic中，训练或测试过程的可视化需要配置三个组件:`VisualizationHook`、`Visualizer`和`VisBackend`, 如下图表展示了 `Visualizer` 和 `VisBackend` 的关系。

<div align="center">
<img src="https://user-images.githubusercontent.com/17425982/163327736-f7cb3b16-ef07-46bc-982a-3cc7495e6c82.png" width="800" />
</div>

**VisualizationHook** 在训练期间以固定的间隔获取模型输出的可视化结果，并将其传递给**Visualizer**。
**Visualizer** 负责将原始可视化结果转换为所需的类型(png, gif等)，然后将其传输到**VisBackend**进行存储或显示。

### GAN的可视化配置

对于像`StyleGAN`和`SAGAN`这样的GAN模型，通常的配置如下所示:

```python
# 可视化钩子
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,  # 配置可视化勾子的间隔数
        fixed_input=True,  # 是否固定噪声输入生成图像
        vis_kwargs_list=dict(type='GAN', name='fake_img')  # 对于GAN模型预先定义可视化参数
    )
]
# 可视化后端
vis_backends = [
    dict(type='VisBackend'),  # 可视化后端用于存储。
    dict(type='WandbVisBackend',  # 可以上传至Wandb的可视化后端
        init_kwargs=dict(
            project='MMagic',   # Wandb项目名
            name='GAN-Visualization-Demo'  # Wandb实验命名
        ))
]
# 可视化器
visualizer = dict(type='Visualizer', vis_backends=vis_backends)
```

如果您将指数移动平均(EMA)应用于生成器，并希望可视化EMA模型，您可以修改`VisualizationHook`的配置，如下所示:

```python
custom_hooks = [
   dict(
       type='VisualizationHook',
       interval=5000,
       fixed_input=True,
       # 同时在`fake_img`中可视化ema以及orig
       vis_kwargs_list=dict(
           type='Noise',
           name='fake_img',  # 使用`fake_img`保存图片
           sample_model='ema/orig',  # 对于`NoiseSampler`特别定义参数
           target_keys=['ema.fake_img', 'orig.fake_img']  # 指定的可视化的键值
       ))
]
```

### 图像翻译模型的可视化配置

对于`CycleGAN`、`Pix2Pix`等翻译模型，可以形成如下可视化配置:

```python
# 可视化钩子
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=[
            dict(
                type='Translation',  # 在训练集可视化结果
                name='trans'),  #  保存`trans`字段的图像
            dict(
                type='Translationval',  # 在验证集可视化结果
                name='trans_val'),  #  保存`trans_val`字段的图像
        ])
]
# 可视化后端
vis_backends = [
    dict(type='VisBackend'),  # 可视化后端用于存储。
    dict(type='WandbVisBackend',  # 可以上传至Wandb的可视化后端
        init_kwargs=dict(
            project='MMagic',   # Wandb项目名
            name='Translation-Visualization-Demo'  # Wandb实验命名
        ))
]
# 可视化器
visualizer = dict(type='Visualizer', vis_backends=vis_backends)
```

### 扩散模型的可视化配置

对于扩散模型，例如`Improved-DDPM`，我们可以使用以下配置通过gif来可视化去噪过程:

```python
# 可视化钩子
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='DDPMDenoising'))  # 对于DDPM模型预先定义可视化参数
]
# 可视化后端
vis_backends = [
    dict(type='VisBackend'),  # 可视化后端用于存储。
    dict(type='WandbVisBackend',  # 可以上传至Wandb的可视化后端
        init_kwargs=dict(
            project='MMagic',   # Wandb项目名
            name='Diffusion-Visualization-Demo'  # Wandb实验命名
        ))
]
# 可视化器
visualizer = dict(type='Visualizer', vis_backends=vis_backends)
```

### 图像补全模型的可视化配置

对于图像补全模型，如`AOT-GAN`和`Global&Local`，通常的配置如下所示:

```python
# 可视化后端
vis_backends = [dict(type='LocalVisBackend')]
# 可视化器
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
# 可视化钩子
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
```

### 图像抠图模型的可视化配置

对于`DIM`和`GCA`等图像抠图模型，通常的配置如下所示:

```python
# 可视化后端
vis_backends = [dict(type='LocalVisBackend')]
# 可视化器
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='trimap_path',
    img_keys=['pred_alpha', 'trimap', 'gt_merged', 'gt_alpha'],
    bgr2rgb=True)
# 可视化钩子
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
```

### SISR/VSR/VFI等模型的可视化配置

对于SISR/VSR/VFI等模型，如`EDSR`, `EDVR`和`CAIN`，通常的配置如下所示:

```python
# 可视化后端
vis_backends = [dict(type='LocalVisBackend')]
# 可视化器
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=False)
# 可视化钩子
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
```

可视化钩子、可视化器和可视化后端组件的具体配置如下所述.

## 可视化钩子

在MMagic中，我们使用`BasicVisualizationHook`和`VisualizationHook`作为可视化钩子。
可视化钩子支持以下三种情况。

(1) 修改`vis_kwargs_list`，实现特定输入下模型输出的可视化，适用于特定数据输入下GAN生成结果和图像翻译模型的翻译结果的可视化等。下面是两个典型的例子:

```python
# input as dict
vis_kwargs_list = dict(
    type='Noise',  # 使用'Noise'采样生成模型输入
    name='fake_img',  # 定义保存图像的命名
)

# input as list of dict
vis_kwargs_list = [
    dict(type='Arguments',  # 使用'Arguments'采样生成模型输入
         name='arg_output',  # 定义保存图像的命名
         vis_mode='gif',  # 通过gif来可视化
         forward_kwargs=dict(forward_mode='sampling', sample_kwargs=dict(show_pbar=True))  # 为'Arguments'采样定义参数
    ),
    dict(type='Data',  # 在dataloader使用'Data'采样提供数据作为可视化输入
         n_samples=36,  # 定义多少采样生成可视化结果
         fixed_input=False,  # 定义对于可视化过程不固定输入
    )
]
```

`vis_kwargs_list`接受字典或字典的列表作为输入。每个字典必须包含一个`type`字段，指示用于生成模型输入的采样器类型，并且每个字典还必须包含采样器所需的关键字字段(例如:`ArgumentSampler`要求参数字典包含`forward_kwargs`)。

> 需要注意的是，此内容由相应的采样器检查，不受`BasicVisualizationHook`的限制。

此外，其他字段是通用字段(例如:`n_samples`、`n_row`,`name`,`fixed_input`等等)。
如果没有传入，则使用`BasicVisualizationHook`初始化的默认值。

为了方便用户使用，MMagic为**GAN**、**Translation models**、**SinGAN**和**Diffusion models**预置了可视化参数，用户可以通过以下配置直接使用预定义的可视化方法:

```python
vis_kwargs_list = dict(type='GAN')
vis_kwargs_list = dict(type='SinGAN')
vis_kwargs_list = dict(type='Translation')
vis_kwargs_list = dict(type='TranslationVal')
vis_kwargs_list = dict(type='TranslationTest')
vis_kwargs_list = dict(type='DDPMDenoising')
```

## 可视化器

在MMagic中，我们实现了`ConcatImageVisualizer`和`Visualizer`，它们继承自`mmengine.Visualizer`。
`Visualizer`的基类是`ManagerMixin`，这使得`Visualizer`成为一个全局唯一的对象。
在实例化之后，`Visualizer`可以在代码的任何地方通过`Visualizer.get_current_instance()`调用，如下所示:

```python
# configs
vis_backends = [dict(type='VisBackend')]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# `get_instance()` 是为全局唯一实例化调用
VISUALIZERS.build(cfg.visualizer)

# 通过上述代码实例化后，您可以在任何位置调用`get_current_instance`方法来获取可视化器
visualizer = Visualizer.get_current_instance()
```

`Visualizer`的核心接口是`add_datasample`。
通过这个界面，该接口将根据相应的`vis_mode`调用相应的绘图函数，以获得`np.ndarray`类型的可视化结果。
然后调用`show`或`add_image`来直接显示结果或将可视化结果传递给预定义的`vis_backend`。

## 可视化后端

- MMEngine的基本VisBackend包括`LocalVisBackend`、`TensorboardVisBackend`和`WandbVisBackend`。您可以关注[MMEngine Documents](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/visualization.md)了解更多有关它们的信息。
- `VisBackend`: **File System**的后端。将可视化结果保存到相应位置。
- `TensorboardVisBackend`: **Tensorboard**的后端。将可视化结果发送到Tensorboard。
- `WandbVisBackend`: **Wandb**的后端。将可视化结果发送到Tensorboard。

一个`Visualizer`对象可以访问任意数量的`visbackend`，用户可以在代码中通过类名访问后端。

```python
# 配置文件
vis_backends = [dict(type='Visualizer'), dict(type='WandbVisBackend')]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# 代码内
VISUALIZERS.build(cfg.visualizer)
visualizer = Visualizer.get_current_instance()

# 通过类名访问后端
gen_vis_backend = visualizer.get_backend('VisBackend')
gen_wandb_vis_backend = visualizer.get_backend('GenWandbVisBackend')
```

当有多个`VisBackend`具有相同的类名时，用户必须为每个`VisBackend`指定名称。

```python
# 配置文件
vis_backends = [
    dict(type='VisBackend', name='gen_vis_backend_1'),
    dict(type='VisBackend', name='gen_vis_backend_2')
]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# 代码内
VISUALIZERS.build(cfg.visualizer)
visualizer = Visualizer.get_current_instance()

local_vis_backend_1 = visualizer.get_backend('gen_vis_backend_1')
local_vis_backend_2 = visualizer.get_backend('gen_vis_backend_2')
```

### 在不同的存储后端可视化

如果想用不同的存储后端( Wandb, Tensorboard, 或者远程窗口里常规的后端)，像以下这样改配置文件的 `vis_backends` 就行了：

**Local**

```python
vis_backends = [dict(type='LocalVisBackend')]
```

**Tensorboard**

```python
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer', vis_backends=vis_backends, name='visualizer')
```

**Wandb**

```python
vis_backends = [dict(type='WandbVisBackend', init_kwargs=dict(project={PROJECTS}, name={EXPNAME}))]
visualizer = dict(
    type='ConcatImageVisualizer', vis_backends=vis_backends, name='visualizer')
```
