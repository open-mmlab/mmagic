# Config System for MMEditing

- [Config System for MMEditing](#Config-System-for-MMEditing)
  - [Modify config](#Modify-config-through-script-arguments)
  - [Config File Structure](#Config-File-Structure)
  - [Config Name Style](#Config-Name-Style)
  - [An Example of EDSR](#An-Example-of-EDSR)
    - [Model config](#Model-config)
    - [Data config](#Data-config)
    - [Evaluation config](#Evaluation-config)
    - [Training and testing config](#Training-and-testing-config)
    - [Optimization config](#Optimization-config)
    - [Hook config](#Hook-config)
    - [Runtime-config](#Runtime-config)
  - [Other examples](#Other-examples)

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments. You can learn about the usage of our config system according to following tutorials.

## Modify config through script arguments

When submitting jobs using "tools/train.py" or "tools/test.py", you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options train_dataloader.batch_size=8` changes the batch size of train dataloader.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `train_pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadMask'` in the pipeline,
  you may specify `--cfg-options train_pipeline.0.type=LoadMask`.

## Config File Structure

There are 3 basic component types under `config/_base_`: datasets, models and default_runtime.
Many methods could be easily constructed with one of each like AOT-GAN, EDVR, GLEAN.
Configs consisting of components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from existing methods.
For example, if some modification is made base on BasicVSR, user may first inherit the basic BasicVSR structure by specifying `_base_ = ../basicvsr/basicvsr_reds4.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder `xxx` under `configs`,

Please refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/config.md) for detailed documentation.

## Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{model_settings}_{module_setting}_{training_setting}_{datasets_info}
```

- `{model_settings}`: model type like `basicvsr`, `dim`, etc. Settings referred in the original paper are included in this field as well. E.g. `edvrm` of `edvrm_8xb4-600k_reds`.
- `[module_setting]`: specific setting for some modules, including Encoder, Decoder, Generator, Discriminator, Normalization, loss, Activation, etc. E.g. `c64n7` of `basicvsr-pp_c64n7_8xb1-600k_reds4`.
- `[train_setting]`: specific setting for training, including batch_size, schedule, etc. E.g. `8xb1` of `basicvsr-pp_c64n7_8xb1-600k_reds4`.
- `{datasets_info}`: dataset name and data size info like `celeba-256x256` of `deepfillv1_4xb4_celeba-256x256`, `reds4` of `basicvsr_2xb4_reds4`.

## An Example of EDSR

To help the users have a basic idea of a complete config, we make a brief comments on the [config of the EDSR model](/configs/edsr/edsr_x2c64b16_g1_300k_div2k.py) we implemented as the following. For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation and the [tutorial in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/config.md).

### Model config

In MMEditing's config, we use model fields to setup a model.

```python
model = dict(
    type='BaseEditModel',  # Name of the model
    generator=dict(  # Config of the generator
        type='EDSRNet',  # Type of the generator
        in_channels=3,  # Channel number of inputs
        out_channels=3,  # Channel number of outputs
        mid_channels=64,  # Channel number of intermediate features
        num_blocks=16,  # Block number in the trunk network
        upscale_factor=scale, # Upsampling factor
        res_scale=1,  # Used to scale the residual in residual block
        rgb_mean=(0.4488, 0.4371, 0.4040),  # Image mean in RGB orders
        rgb_std=(1.0, 1.0, 1.0)),  # Image std in RGB orders
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean')  # Config for pixel loss
    train_cfg=dict(),  # Config of training model.
    test_cfg=dict(),  # Config of testing model.
    data_preprocessor=dict(  # The Config to build data preprocessor
        type='EditDataPreprocessor', mean=[0., 0., 0.], std=[255., 255.,
                                                             255.]))
```

### Data config

[Dataloaders](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader) are required for the training, validation, and testing of the [runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html).
Dataset and data pipeline need to be set to build the dataloader. Due to the complexity of this part, we use intermediate variables to simplify the writing of dataloader configs.

#### Data pipeline

```python
train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile',  # Load images from files
        key='img',  # Keys in results to find corresponding path
        color_type='color',  # Color type of image
        channel_order='rgb',  # Channel order of image
        imdecode_backend='cv2'),  # decode backend
    dict(type='LoadImageFromFile',  # Load images from files
        key='gt',  # Keys in results to find corresponding path
        color_type='color',  # Color type of image
        channel_order='rgb',  # Channel order of image
        imdecode_backend='cv2'),  # decode backend
    dict(type='SetValues', dictionary=dict(scale=scale)),  # Set value to destination keys
    dict(type='PairedRandomCrop', gt_patch_size=96),  # Paired random crop
    dict(type='Flip',  # Flip images
        keys=['lq', 'gt'],  # Images to be flipped
        flip_ratio=0.5,  # Flip ratio
        direction='horizontal'),  # Flip direction
    dict(type='Flip',  # Flip images
        keys=['lq', 'gt'],  # Images to be flipped
        flip_ratio=0.5,  # Flip ratio
        direction='vertical'),  # Flip direction
    dict(type='RandomTransposeHW',  # Random transpose h and w for images
        keys=['lq', 'gt'],  # Images to be transposed
        transpose_ratio=0.5  # Transpose ratio
        ),
    dict(type='ToTensor', keys=['img', 'gt']),  # Convert images to tensor
    dict(type='PackEditInputs')  # The config of collecting data from current pipeline
]
test_pipeline = [  # Test pipeline
    dict(type='LoadImageFromFile',  # Load images from files
        key='img',  # Keys in results to find corresponding path
        color_type='color',  # Color type of image
        channel_order='rgb',  # Channel order of image
        imdecode_backend='cv2'),  # decode backend
    dict(type='LoadImageFromFile',  # Load images from files
        key='gt',  # Keys in results to find corresponding path
        color_type='color',  # Color type of image
        channel_order='rgb',  # Channel order of image
        imdecode_backend='cv2'),  # decode backend
    dict(type='ToTensor', keys=['img', 'gt']),  # Convert images to tensor
    dict(type='PackEditInputs')  # The config of collecting data from current pipeline
]
```

#### Dataloader

```python
dataset_type = 'BasicImageDataset'  # The type of dataset
data_root = 'data'  # Root path of data
train_dataloader = dict(
    num_workers=4,  # The number of workers to pre-fetch data for each single GPU
    persistent_workers=False,  # Whether maintain the workers Dataset instances alive
    sampler=dict(type='InfiniteSampler', shuffle=True),  # The type of data sampler
    dataset=dict(  # Train dataset config
        type=dataset_type,  # Type of dataset
        ann_file='meta_info_DIV2K800sub_GT.txt',  # Path of annotation file
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DIV2K',  # Root path of data
        data_prefix=dict(  # Prefix of image path
            img='DIV2K_train_LR_bicubic/X2_sub', gt='DIV2K_train_HR_sub'),
        filename_tmpl=dict(img='{}', gt='{}'),  # Filename template
        pipeline=train_pipeline))
val_dataloader = dict(
    num_workers=4,  # The number of workers to pre-fetch data for each single GPU
    persistent_workers=False,  # Whether maintain the workers Dataset instances alive
    drop_last=False,  # Whether drop the last incomplete batch
    sampler=dict(type='DefaultSampler', shuffle=False),  # The type of data sampler
    dataset=dict(  # Validation dataset config
        type=dataset_type,  # Type of dataset
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root=data_root + '/Set5',  # Root path of data
        data_prefix=dict(img='LRbicx2', gt='GTmod12'),  # Prefix of image path
        pipeline=test_pipeline))
test_dataloader = val_dataloader
```

### Evaluation config

[Evaluators](https://mmengine.readthedocs.io/en/latest/tutorials/metric_and_evaluator.html) are used to compute the metrics of the trained model on the validation and testing datasets.
The config of evaluators consists of one or a list of metric configs:

```python
val_evaluator = [
    dict(type='MAE'),  # The name of metrics to evaluate
    dict(type='PSNR', crop_border=scale),  # The name of metrics to evaluate
    dict(type='SSIM', crop_border=scale),  # The name of metrics to evaluate
]
test_evaluator = val_evaluator # The config for testing evaluator
```

### Training and testing config

MMEngine's runner uses Loop to control the training, validation, and testing processes.
Users can set the maximum training iteration and validation intervals with these fields.

```python
train_cfg = dict(
    type='IterBasedTrainLoop',  # The name of train loop type
    max_iters=300000,  # The number of total iterations
    val_interval=5000,  # The number of validation interval iterations
)
val_cfg = dict(type='ValLoop')  # The name of validation loop type
test_cfg = dict(type='TestLoop')  # The name of test loop type
```

### Optimization config

`optim_wrapper` is the field to configure optimization related settings.
The optimizer wrapper not only provides the functions of the optimizer, but also supports functions such as gradient clipping, mixed precision training, etc. Find more in [optimizer wrapper tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html).

```python
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.00001),
    )
)  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch.
```

`param_scheduler` is a field that configures methods of adjusting optimization hyper-parameters such as learning rate and momentum.
Users can combine multiple schedulers to create a desired parameter adjustment strategy.
Find more in [parameter scheduler tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html).

```python
param_scheduler = dict(  # Config of learning policy
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5)
```

### Hook config

Users can attach hooks to training, validation, and testing loops to insert some operations during running. There are two different hook fields, one is `default_hooks` and the other is `custom_hooks`.

`default_hooks` is a dict of hook configs. `default_hooks` are the hooks must required at runtime. They have default priority which should not be modified. If not set, runner will use the default values. To disable a default hook, users can set its config to `None`.

```python
default_hooks = dict(  # Used to build default hooks
    checkpoint=dict(  # Config to set the checkpoint hook
        type='CheckpointHook',
        interval=5000,  # The save interval is 5000 iterations
        save_optimizer=True,
        by_epoch=False,  # Count by iterations
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),  # Config to register logger hook
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
```

`custom_hooks` is a list of hook configs. Users can develop there own hooks and insert them in this field.

```python
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)] # Config of visualization hook
```

### Runtime config

```python
default_scope = 'mmedit' # Used to set registries location
env_cfg = dict(  # Parameters to setup distributed training, the port can also be set
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'  # The level of logging
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)  # Used to build log processor
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume = False  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
```

## Other examples

More details of config system are shown in following tutorials.

- [inpainting config](./config/config_inpainting.md)
- [matting config](./config/config_matting.md)
- [restoration config](./config/config_restoration.md)
