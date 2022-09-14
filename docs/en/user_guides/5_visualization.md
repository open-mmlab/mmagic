# Tutorial 5: Visualization

The visualization of images is an important way to measure the quality of image processing, editing and synthesis.
MMEditing provides a rich set of visualization functions.
In this tutorial, we introduce the usage of the visualization functions provided by MMEditing.

- [Overview](#overview)
- [Visualization hook](#visualization-hook)
- [Visualizer](#visualizer)
- [VisBackend](#visbackend)
  - [LocalVisBackend](#localvisbackend)
  - [TensorboardVisBackend](#tensorboardvisbackend)

## Overview

In MMEditing, the visualization of the training or testing process requires the configuration of three components: VisualizationHook, Visualizer, and VisBackend.

**VisualizationHook** fetches the visualization results of the model output in fixed intervals during training and passes them to Visualizer.
**Visualizer** is responsible for converting the original visualization results into the desired type (png, gif, etc.) and then transferring them to **VisBackend** for storage or display.

### Visualization configuration of GANs

For GAN models, such as StyleGAN and SAGAN, a usual configuration is shown below:

```python
# VisualizationHook
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,  # visualization interval
        fixed_input=True,  # whether use fixed noise input to generate images
        vis_kwargs_list=dict(type='GAN', name='fake_img')  # pre-defined visualization arguments for GAN models
    )
]
# VisBackend
vis_backends = [
    dict(type='GenVisBackend'),  # vis_backend for saving images to file system
    dict(type='WandbGenVisBackend',  # vis_backend for uploading images to Wandb
        init_kwargs=dict(
            project='MMGeneration',   # project name for Wandb
            name='GAN-Visualization-Demo'  # name of experiment for Wandb
        ))
]
# Visualizer
visualizer = dict(type='GenVisualizer', vis_backends=vis_backends)
```

If you apply Exponential Moving Average (EMA) to generator and want to visualize the EMA model, you can modify config of `VisualizationHook` as below:

```python
custom_hooks = [
    dict(
        type='GenVisualizationHook',
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
        type='GenVisualizationHook',
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
    dict(type='GenVisBackend'),  # vis_backend for saving images to file system
    dict(type='WandbGenVisBackend',  # vis_backend for uploading images to Wandb
        init_kwargs=dict(
            project='MMGeneration',   # project name for Wandb
            name='Translation-Visualization-Demo'  # name of experiment for Wandb
        ))
]
# Visualizer
visualizer = dict(type='GenVisualizer', vis_backends=vis_backends)
```

### Visualization configuration of diffusion models

For Diffusion models, such as Improved-DDPM, we can use the following configuration to visualize the denoising process through a gif:

```python
# VisualizationHook
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='DDPMDenoising'))  # pre-defined visualization argument for DDPM models
]
# VisBackend
vis_backends = [
    dict(type='GenVisBackend'),  # vis_backend for saving images to file system
    dict(type='WandbGenVisBackend',  # vis_backend for uploading images to Wandb
        init_kwargs=dict(
            project='MMGeneration',   # project name for Wandb
            name='Diffusion-Visualization-Demo'  # name of experiment for Wandb
        ))
]
# Visualizer
visualizer = dict(type='GenVisualizer', vis_backends=vis_backends)
```

The specific configuration of the `VisualizationHook`, `Visualizer` and `GenVisBackend` components are described below

## Visualization Hook

In MMEditing, we use `GenVisualizationHook` as `VisualizationHook`. `GenVisualizationHook` support three following cases.

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

> To be noted that, this content is checked by the corresponding sampler and is not restricted by `GenVisHook`.

In addition, the other fields are generic fields (e.g. `n_samples`, `n_row`, `name`, `fixed_input`, etc.).
If not passed in, the default values from the GenVisHook initialization will be used.

For the convenience of users, MMGeneration has pre-defined visualization parameters for **GAN**, **Translation models**, **SinGAN** and **Diffusion models**, and users can directly use the predefined visualization methods by using the following configuration:

```python
vis_kwargs_list = dict(type='GAN')
vis_kwargs_list = dict(type='SinGAN')
vis_kwargs_list = dict(type='Translation')
vis_kwargs_list = dict(type='TranslationVal')
vis_kwargs_list = dict(type='TranslationTest')
vis_kwargs_list = dict(type='DDPMDenoising')
```

## Visualizer

In MMGeneration, we implement `GenVisualizer`, which inherits from `mmengine.Visualizer`.
The base class of `GenVisualizer` is `ManagerMixin` and this make `GenVisualizer` a globally unique object.
After be instantiated, `GenVisualizer` can be called at anywhere of the code by `Visualizer.get_current_instance()`, as shown below:

```python
# configs
vis_backends = [dict(type='GenVisBackend')]
visualizer = dict(
    type='GenVisualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# `get_instance()` is called for globally unique instantiation
VISUALIZERS.build(cfg.visualizer)

# Once instantiated by the above code, you can call the `get_current_instance` method at any location to get the visualizer
visualizer = Visualizer.get_current_instance()
```

The core interface of `GenVisualizer` is `add_datasample`.
By this interface,
This interface will call the corresponding drawing function according to the corresponding `vis_mode` to obtain the visualization result in `np.ndarray` type.
Then `show` or `add_image` will be called to directly show the results or pass the visualization result to the predefined vis_backend.

## VisBackend

In general, users do not need to manipulate `VisBackend` objects, only when the current visualization storage can not meet the needs, users will want to manipulate the storage backend directly.
MMGeneration supports a variety of different visualization backends, including:

- GenVisBackend: Backend for **File System**. Save the visualization results to corresponding position.
- TensorboardGenVisBackend: Backend for **Tensorboard**. Send the visualization results to Tensorboard.
- PaviGenVisBackend: Backend for **Pavi**. Send the visualization results to Tensorboard.
- WandbGenVisBackend: Backend for **Wandb**. Send the visualization results to Tensorboard.

One `GenVisualizer` object can have access to any number of VisBackends and users can access to the backend by their class name in their code.

```python
# configs
vis_backends = [dict(type='GenVisualizer'), dict(type='WandbVisBackend')]
visualizer = dict(
    type='GenVisualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# code
VISUALIZERS.build(cfg.visualizer)
visualizer = Visualizer.get_current_instance()

# access to the backend by class name
gen_vis_backend = visualizer.get_backend('GenVisBackend')
gen_wandb_vis_backend = visualizer.get_backend('GenWandbVisBackend')
```

When there are multiply VisBackend with the same class name, user must specific name for each VisBackend.

```python
# configs
vis_backends = [
    dict(type='GenVisBackend', name='gen_vis_backend_1'),
    dict(type='GenVisBackend', name='gen_vis_backend_2')
]
visualizer = dict(
    type='GenVisualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# code
VISUALIZERS.build(cfg.visualizer)
visualizer = Visualizer.get_current_instance()

local_vis_backend_1 = visualizer.get_backend('gen_vis_backend_1')
local_vis_backend_2 = visualizer.get_backend('gen_vis_backend_2')
```

The structure of this guide are as follows:

Using `visualizer` in config file can save visual results when training or testing. You can follow [MMEngine Documents](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/visualization.md) to learn the usage of visualization.
MMEditing supports `Local` visual backend and `Tensorboard ` visual backend now.

### LocalVisBackend

Here are configs using `LocalVisBackend` to save visual results as an example.
The output of models formatted to `DataSample` will be passed to `visualizer`. You can set the `img_keys` fields of `visualizer` to determine which values of images to visualize.

```python
vis_backends = [dict(type='LocalVisBackend')] # The type of visual backend
visualizer = dict( # Config to build visualizer
    type='ConcatImageVisualizer', # Type of visualizer
    vis_backends=vis_backends, # Type of visual backend
    fn_key='gt_path', # The key used to determine file name for saving image
    img_keys=['gt_img', 'input', 'pred_img'], # The key in img_keys will be visualized
    bgr2rgb=True) # Whether convert channels of image
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)] # Config of visualization hook
```

Then visual results will be saved in `work_dirs` where configured in the config file.

### TensorboardVisBackend

Here are configs using `TensorboardVisBackend` to save visual results as an example.

```python
vis_backends = [dict(type='TensorboardVisBackend')] # The type of visual backend
visualizer = dict( # Config to build visualizer
    type='ConcatImageVisualizer', # Type of visualizer
    vis_backends=vis_backends, # Type of visual backend
    fn_key='gt_path', # The key used to determine file name for saving image
    img_keys=['gt_img', 'input', 'pred_img'], # The key in img_keys will be visualized
    bgr2rgb=True) # Whether convert channels of image
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)] # Config of visualization hook
```

You can start TensorBoard server and see visual results by visiting TensorBoard in the browser.
