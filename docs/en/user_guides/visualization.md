# Tutorial 6: Visualization

The visualization of images is an important way to measure the quality of image processing, editing and synthesis.
Using `visualizer` in config file can save visual results when training or testing. You can follow [MMEngine Documents](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md) to learn the usage of visualization. MMagic provides a rich set of visualization functions.
In this tutorial, we introduce the usage of the visualization functions provided by MMagic.

- [Tutorial 6: Visualization](#tutorial-6-visualization)
  - [Overview](#overview)
    - [Visualization configuration of GANs](#visualization-configuration-of-gans)
    - [Visualization configuration of image translation models](#visualization-configuration-of-image-translation-models)
    - [Visualization configuration of diffusion models](#visualization-configuration-of-diffusion-models)
    - [Visualization configuration of inpainting models](#visualization-configuration-of-inpainting-models)
    - [Visualization configuration of matting models](#visualization-configuration-of-matting-models)
    - [Visualization configuration of SISR/VSR/VFI models](#visualization-configuration-of-sisrvsrvfi-models)
  - [Visualization Hook](#visualization-hook)
  - [Visualizer](#visualizer)
  - [VisBackend](#visbackend)
    - [Use Different Storage Backends](#use-different-storage-backends)

## Overview

It is recommended to learn the basic concept of visualization in design documentation.

In MMagic, the visualization of the training or testing process requires the configuration of three components: `VisualizationHook`, `Visualizer`, and `VisBackend`, The diagram below shows the relationship between Visualizer and VisBackend,

<div align="center">
<img src="https://user-images.githubusercontent.com/17425982/163327736-f7cb3b16-ef07-46bc-982a-3cc7495e6c82.png" width="800" />
</div>

**VisualizationHook** fetches the visualization results of the model output in fixed intervals during training and passes them to Visualizer.
**Visualizer** is responsible for converting the original visualization results into the desired type (png, gif, etc.) and then transferring them to **VisBackend** for storage or display.

### Visualization configuration of GANs

For GAN models, such as StyleGAN and SAGAN, a usual configuration is shown below:

```python
# VisualizationHook
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,  # visualization interval
        fixed_input=True,  # whether use fixed noise input to generate images
        vis_kwargs_list=dict(type='GAN', name='fake_img')  # pre-defined visualization arguments for GAN models
    )
]
# VisBackend
vis_backends = [
    dict(type='VisBackend'),  # vis_backend for saving images to file system
    dict(type='WandbVisBackend',  # vis_backend for uploading images to Wandb
        init_kwargs=dict(
            project='MMagic',   # project name for Wandb
            name='GAN-Visualization-Demo'  # name of the experiment for Wandb
        ))
]
# Visualizer
visualizer = dict(type='Visualizer', vis_backends=vis_backends)
```

If you apply Exponential Moving Average (EMA) to a generator and want to visualize the EMA model, you can modify config of `VisualizationHook` as below:

```python
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        # vis ema and orig in `fake_img` at the same time
        vis_kwargs_list=dict(
            type='Noise',
            name='fake_img',  # save images with prefix `fake_img`
            sample_model='ema/orig',  # specified kwargs for `NoiseSampler`
            target_keys=['ema.fake_img', 'orig.fake_img']  # specific key to visualization
        ))
]
```

### Visualization configuration of image translation models

For Translation models, such as CycleGAN and Pix2Pix, visualization configs can be formed as below:

```python
# VisualizationHook
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=[
            dict(
                type='Translation',  # Visualize results on the training set
                name='trans'),  #  save images with prefix `trans`
            dict(
                type='Translationval',  # Visualize results on the validation set
                name='trans_val'),  #  save images with prefix `trans_val`
        ])
]
# VisBackend
vis_backends = [
    dict(type='VisBackend'),  # vis_backend for saving images to file system
    dict(type='WandbVisBackend',  # vis_backend for uploading images to Wandb
        init_kwargs=dict(
            project='MMagic',   # project name for Wandb
            name='Translation-Visualization-Demo'  # name of the experiment for Wandb
        ))
]
# Visualizer
visualizer = dict(type='Visualizer', vis_backends=vis_backends)
```

### Visualization configuration of diffusion models

For Diffusion models, such as Improved-DDPM, we can use the following configuration to visualize the denoising process through a gif:

```python
# VisualizationHook
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='DDPMDenoising'))  # pre-defined visualization argument for DDPM models
]
# VisBackend
vis_backends = [
    dict(type='VisBackend'),  # vis_backend for saving images to file system
    dict(type='WandbVisBackend',  # vis_backend for uploading images to Wandb
        init_kwargs=dict(
            project='MMagic',   # project name for Wandb
            name='Diffusion-Visualization-Demo'  # name of the experiment for Wandb
        ))
]
# Visualizer
visualizer = dict(type='Visualizer', vis_backends=vis_backends)
```

### Visualization configuration of inpainting models

For inpainting models, such as AOT-GAN and Global&Local, a usual configuration is shown below:

```python
# VisBackend
vis_backends = [dict(type='LocalVisBackend')]
# Visualizer
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
# VisualizationHook
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
```

### Visualization configuration of matting models

For matting models, such as DIM and GCA, a usual configuration is shown below:

```python
# VisBackend
vis_backends = [dict(type='LocalVisBackend')]
# Visualizer
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='trimap_path',
    img_keys=['pred_alpha', 'trimap', 'gt_merged', 'gt_alpha'],
    bgr2rgb=True)
# VisualizationHook
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
```

### Visualization configuration of SISR/VSR/VFI models

For SISR/VSR/VFI models, such as EDSR, EDVR and CAIN, a usual configuration is shown below:

```python
# VisBackend
vis_backends = [dict(type='LocalVisBackend')]
# Visualizer
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=False)
# VisualizationHook
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
```

The specific configuration of the `VisualizationHook`, `Visualizer` and `VisBackend` components are described below

## Visualization Hook

In MMagic, we use `BasicVisualizationHook` and `VisualizationHook` as `VisualizationHook`.
`VisualizationHook` supports three following cases.

(1) Modify `vis_kwargs_list` to visualize the output of the model under specific inputs , which is suitable for visualization of the generated results of GAN and translation results of Image-to-Image-Translation models under specific data input, etc. Below are two typical examples:

```python
# input as dict
vis_kwargs_list = dict(
    type='Noise',  # use 'Noise' sampler to generate model input
    name='fake_img',  # define prefix of saved images
)

# input as list of dict
vis_kwargs_list = [
    dict(type='Arguments',  # use `Arguments` sampler to generate model input
         name='arg_output',  # define prefix of saved images
         vis_mode='gif',  # specific visualization mode as GIF
         forward_kwargs=dict(forward_mode='sampling', sample_kwargs=dict(show_pbar=True))  # specific kwargs for `Arguments` sampler
    ),
    dict(type='Data',  # use `Data` sampler to feed data in dataloader to model as input
         n_samples=36,  # specific how many samples want to generate
         fixed_input=False,  # specific do not use fixed input for each visualization process
    )
]
```

`vis_kwargs_list` takes dict or list of dict as input. Each of dict must contain a `type` field indicating the **type of sampler** used to generate the model input, and each of the dict must also contain the keyword fields necessary for the sampler (e.g. `ArgumentSampler` requires that the argument dictionary contain `forward_kwargs`).

> To be noted that, this content is checked by the corresponding sampler and is not restricted by `BasicVisualizationHook`.

In addition, the other fields are generic fields (e.g. `n_samples`, `n_row`, `name`, `fixed_input`, etc.).
If not passed in, the default values from the BasicVisualizationHook initialization will be used.

For the convenience of users, MMagic has pre-defined visualization parameters for **GAN**, **Translation models**, **SinGAN** and **Diffusion models**, and users can directly use the predefined visualization methods by using the following configuration:

```python
vis_kwargs_list = dict(type='GAN')
vis_kwargs_list = dict(type='SinGAN')
vis_kwargs_list = dict(type='Translation')
vis_kwargs_list = dict(type='TranslationVal')
vis_kwargs_list = dict(type='TranslationTest')
vis_kwargs_list = dict(type='DDPMDenoising')
```

## Visualizer

In MMagic, we implement `ConcatImageVisualizer` and `Visualizer`, which inherit from `mmengine.Visualizer`.
The base class of `Visualizer` is `ManagerMixin` and this makes `Visualizer` a globally unique object.
After being instantiated, `Visualizer` can be called at anywhere of the code by `Visualizer.get_current_instance()`, as shown below:

```python
# configs
vis_backends = [dict(type='VisBackend')]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# `get_instance()` is called for globally unique instantiation
VISUALIZERS.build(cfg.visualizer)

# Once instantiated by the above code, you can call the `get_current_instance` method at any location to get the visualizer
visualizer = Visualizer.get_current_instance()
```

The core interface of `Visualizer` is `add_datasample`.
Through this interface,
This interface will call the corresponding drawing function according to the corresponding `vis_mode` to obtain the visualization result in `np.ndarray` type.
Then `show` or `add_image` will be called to directly show the results or pass the visualization result to the predefined vis_backend.

## VisBackend

In general, users do not need to manipulate `VisBackend` objects, only when the current visualization storage can not meet the needs, users will want to manipulate the storage backend directly.
MMagic supports a variety of different visualization backends, including:

- Basic VisBackend of MMEngine: including LocalVisBackend, TensorboardVisBackend and WandbVisBackend. You can follow [MMEngine Documents](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md) to learn more about them
- VisBackend: Backend for **File System**. Save the visualization results to the corresponding position.
- TensorboardVisBackend: Backend for **Tensorboard**. Send the visualization results to Tensorboard.
- WandbVisBackend: Backend for **Wandb**. Send the visualization results to Tensorboard.

One `Visualizer` object can have access to any number of VisBackends and users can access to the backend by their class name in their code.

```python
# configs
vis_backends = [dict(type='Visualizer'), dict(type='WandbVisBackend')]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# code
VISUALIZERS.build(cfg.visualizer)
visualizer = Visualizer.get_current_instance()

# access to the backend by class name
gen_vis_backend = visualizer.get_backend('VisBackend')
gen_wandb_vis_backend = visualizer.get_backend('GenWandbVisBackend')
```

When there are multiply VisBackend with the same class name, user must specific name for each VisBackend.

```python
# configs
vis_backends = [
    dict(type='VisBackend', name='gen_vis_backend_1'),
    dict(type='VisBackend', name='gen_vis_backend_2')
]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# code
VISUALIZERS.build(cfg.visualizer)
visualizer = Visualizer.get_current_instance()

local_vis_backend_1 = visualizer.get_backend('gen_vis_backend_1')
local_vis_backend_2 = visualizer.get_backend('gen_vis_backend_2')
```

### Visualize by Different Storage Backends

If you want to use a different backend (Wandb, Tensorboard, or a custom backend with a remote window), just change the `vis_backends` in the config, as follows:

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

```python
vis_backends = [dict(type='WandbVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer', vis_backends=vis_backends, name='visualizer')
```
