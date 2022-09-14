# Migration from MMEdit 0.x

- [Migration from MMEdit 0.x](#migration-from-mmedit-0x)
  - [New dependencies](#new-dependencies)
  - [Overall Structures](#overall-structures)
  - [Config](#config)
    - [File name](#file-name)
    - [Model settings](#model-settings)
    - [Data settings](#data-settings)
    - [Evaluation settings](#evaluation-settings)
    - [Schedule settings](#schedule-settings)
    - [Runtime settings](#runtime-settings)
    - [Other config settings](#other-config-settings)
  - [Model](#model)

We introduce some modifications in MMEdit 1.x. This document will help users of the 0.x version to quickly migrate projects to the newest version.

## New dependencies

MMEdit 1.x depends on some new packages, you can prepare a new clean environment and install again according to the [install tutorial](./get_started.md). Or install the below packages manually.

- MMEngine: MMEngine is the core of the OpenMMLab 2.0 architecture, and we split many components unrelated to computer vision from MMCV to MMEngine.
- MMCV: The computer vision package of OpenMMLab. This is not a new dependency, but you need to upgrade it to above 2.0.0rc0 version.

## Overall Structures

We refactor overall structures in MMEdit 1.x as following.

- The  `core` in the old versions of MMEdit is split into `engine`, `evaluation`, `structures`, and `visualization`
- The `pipelines` of `datasets` in the old versions of MMEdit is refactored to `transforms`
- The `models` in MMedit 1.x is refactored to five parts: `base_models`, `data_preprocessors`, `editors`, `layers` and `losses`.

## Config

### File name

We rename config file to new template: `{model_settings}_{module_setting}_{training_setting}_{datasets_info}`.

### Model settings

We update model settings in MMEdit 1.x. Important modifications are as following.

- Remove `pretrained` fields.
- Add `train_cfg` and `test_cfg` fields in model settings.
- Add `data_preprocessor` fields. Normalization and color space transforms operations are moved from datasets transforms pipelines to data_preprocessor. We will introduce data_preprocessor later.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
model = dict(
    type='BasicRestorer',  # Name of the model
    generator=dict(  # Config of the generator
        type='EDSR',  # Type of the generator
        in_channels=3,  # Channel number of inputs
        out_channels=3,  # Channel number of outputs
        mid_channels=64,  # Channel number of intermediate features
        num_blocks=16,  # Block number in the trunk network
        upscale_factor=scale, # Upsampling factor
        res_scale=1,  # Used to scale the residual in residual block
        rgb_mean=(0.4488, 0.4371, 0.4040),  # Image mean in RGB orders
        rgb_std=(1.0, 1.0, 1.0)),  # Image std in RGB orders
    pretrained=None,
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))  # Config for pixel loss model training and testing settings
```

</td>

<td valign="top">

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

</td>

</tr>
</thead>
</table>

### Data settings

#### Data pipelines

We update data pipelines settings in MMEdit 1.x. Important modifications are as following.

- Remove normalization and color space transforms operations. They are moved from datasets transforms pipelines to data_preprocessor.
- The original formatting transforms pipelines `Collect` are combined as `PackEditInputs`.
  More details of data pipelines are shown in [transform guides](/docs/en/advanced_guides/transforms.md).

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile',  # Load images from files
        io_backend='disk',  # io backend
        key='lq',  # Keys in results to find corresponding path
        flag='unchanged'),  # flag for reading images
    dict(type='LoadImageFromFile',  # Load images from files
        io_backend='disk',  # io backend
        key='gt',  # Keys in results to find corresponding path
        flag='unchanged'),  # flag for reading images
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),  # Rescale images from [0, 255] to [0, 1]
    dict(type='Normalize',  # Augmentation pipeline that normalize the input images
        keys=['lq', 'gt'],  # Images to be normalized
        mean=[0, 0, 0],  # Mean values
        std=[1, 1, 1],  # Standard variance
        to_rgb=True),  # Change to RGB channel
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
    dict(type='Collect',  # Pipeline that decides which keys in the data should be passed to the model
        keys=['lq', 'gt'],  # Keys to pass to the model
        meta_keys=['lq_path', 'gt_path']), # Meta information keys. In training, meta information is not needed
    dict(type='ToTensor',  # Convert images to tensor
        keys=['lq', 'gt'])  # Images to be converted to Tensor
]
test_pipeline = [  # Test pipeline
    dict(
        type='LoadImageFromFile',  # Load images from files
        io_backend='disk',  # io backend
        key='lq',  # Keys in results to find corresponding path
        flag='unchanged'),  # flag for reading images
    dict(
        type='LoadImageFromFile',  # Load images from files
        io_backend='disk',  # io backend
        key='gt',  # Keys in results to find corresponding path
        flag='unchanged'),  # flag for reading images
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),  # Rescale images from [0, 255] to [0, 1]
    dict(
        type='Normalize',  # Augmentation pipeline that normalize the input images
        keys=['lq', 'gt'],  # Images to be normalized
        mean=[0, 0, 0],  # Mean values
        std=[1, 1, 1],  # Standard variance
        to_rgb=True),  # Change to RGB channel
    dict(type='Collect',  # Pipeline that decides which keys in the data should be passed to the model
        keys=['lq', 'gt'],  # Keys to pass to the model
        meta_keys=['lq_path', 'gt_path']),  # Meta information keys
    dict(type='ToTensor',  # Convert images to tensor
        keys=['lq', 'gt'])  # Images to be converted to Tensor
]
```

</td>

<td valign="top">

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
    dict(type='PackEditInputs')  # The config of collecting data from current pipeline
]
```

</td>

</tr>
</thead>
</table>

#### Dataloader

We update dataloader settings in MMEdit 1.x. Important modifications are as following.

- The original `data` field is split to `train_dataloader`, `val_dataloader` and `test_dataloader`. This allows us to configure them in fine-grained. For example, you can specify different sampler and batch size during training and test.
- The samples_per_gpu is renamed to batch_size.
- The workers_per_gpu is renamed to num_workers.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
data = dict(
    # train
    samples_per_gpu=16,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    drop_last=True,  # Use drop_last in data_loader
    train=dict(  # Train dataset config
        type='RepeatDataset',  # Repeated dataset for iter-based model
        times=1000,  # Repeated times for RepeatDataset
        dataset=dict(
            type=train_dataset_type,  # Type of dataset
            lq_folder='data/DIV2K/DIV2K_train_LR_bicubic/X2_sub',  # Path for lq folder
            gt_folder='data/DIV2K/DIV2K_train_HR_sub',  # Path for gt folder
            ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',  # Path for annotation file
            pipeline=train_pipeline,  # See above for train_pipeline
            scale=scale)),  # Scale factor for upsampling
    # val
    val_samples_per_gpu=1,  # Batch size of a single GPU for validation
    val_workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU for validation
    val=dict(
        type=val_dataset_type,  # Type of dataset
        lq_folder='data/val_set5/Set5_bicLRx2',  # Path for lq folder
        gt_folder='data/val_set5/Set5_mod12',  # Path for gt folder
        pipeline=test_pipeline,  # See above for test_pipeline
        scale=scale,  # Scale factor for upsampling
        filename_tmpl='{}'),  # filename template
    # test
    test=dict(
        type=val_dataset_type,  # Type of dataset
        lq_folder='data/val_set5/Set5_bicLRx2',  # Path for lq folder
        gt_folder='data/val_set5/Set5_mod12',  # Path for gt folder
        pipeline=test_pipeline,  # See above for test_pipeline
        scale=scale,  # Scale factor for upsampling
        filename_tmpl='{}'))  # filename template
```

</td>

<td valign="top">

```python
dataset_type = 'BasicImageDataset'  # The type of dataset
data_root = 'data'  # Root path of data
train_dataloader = dict(
    batch_size=16,
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
    batch_size=1,
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

</td>

</tr>
</thead>
</table>

### Evaluation settings

We update evaluation settings in MMEdit 1.x. Important modifications are as following.

- The evaluation field is split to `val_evaluator` and `test_evaluator`. The `interval` is moved to `train_cfg.val_interval`.
- The metrics to evaluation are moved from `test_cfg` to `val_evaluator` and `test_evaluator`.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
train_cfg = None  # Training config
test_cfg = dict(  # Test config
    metrics=['PSNR'],  # Metrics used during testing
    crop_border=scale)  # Crop border during evaluation

evaluation = dict(  # The config to build the evaluation hook
    interval=5000,  # Evaluation interval
    save_image=True,  # Save images during evaluation
    gpu_collect=True)  # Use gpu collect
```

</td>

<td valign="top">

```python
val_evaluator = [
    dict(type='PSNR', crop_border=scale),  # The name of metrics to evaluate
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=5000)  # Config of train loop type
val_cfg = dict(type='ValLoop')  # The name of validation loop type
test_cfg = dict(type='TestLoop')  # The name of test loop type
```

</td>

</tr>
</thead>
</table>

### Schedule settings

We update schedule settings in MMEdit 1.x. Important modifications are as following.

- Now we use `optim_wrapper` field to specify all configuration about the optimization process. And the `optimizer` is a sub field of `optim_wrapper` now.
- The `lr_config` field is removed and we use new `param_scheduler` to replace it.
- The `total_iters` field is moved to `train_cfg` as `max_iters`, `val_cfg` and `test_cfg`, which configure the loop in training, validation and test.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
total_iters = 300000 # Total training iters
lr_config = dict( # Learning rate scheduler config used to register LrUpdater hook
    policy='Step', by_epoch=False, step=[200000], gamma=0.5)  # The policy of scheduler
```

</td>

<td valign="top">

```python
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4),
    )
)  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch.
param_scheduler = dict(  # Config of learning policy
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5)  # The policy of scheduler
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=5000)  # Config of train loop type
val_cfg = dict(type='ValLoop')  # The name of validation loop type
test_cfg = dict(type='TestLoop')  # The name of test loop type
```

</td>

</tr>
</thead>
</table>

> More details of schedule settings are shown in [MMEngine Documents](https://github.com/open-mmlab/mmengine/blob/main/docs/en/migration/migrate_param_scheduler_from_mmcv.md).

### Runtime settings

We update runtime settings in MMEdit 1.x. Important modifications are as following.

- The `checkpoint_config` is moved to `default_hooks.checkpoint` and the `log_config` is moved to `default_hooks.logger`. And we move many hooks settings from the script code to the `default_hooks` field in the runtime configuration.
- The `resume_from` is removed. And we use `resume` to replace it.
  - If resume=True and load_from is not None, resume training from the checkpoint in load_from.
  - If resume=True and load_from is None, try to resume from the latest checkpoint in the work directory.
  - If resume=False and load_from is not None, only load the checkpoint, not resume training.
  - If resume=False and load_from is None, do not load nor resume.
- The `dist_params` field is a sub field of `env_cfg` now. And there are some new configurations in the `env_cfg`.
- The `workflow` related functionalities are removed.
- New field `visualizer`: The visualizer is a new design. We use a visualizer instance in the runner to handle results & log visualization and save to different backends,  like Local, TensorBoard and Wandb.
- New field `default_scope`: The start point to search module for all registries.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=5000,  # The save interval is 5000 iterations
    save_optimizer=True,  # Also save optimizers
    by_epoch=False)  # Count by iterations
log_config = dict(  # Config to register logger hook
    interval=100,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),  # The logger used to record the training process
        dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
    ])
visual_config = None  # Visual config, we do not use it.
# runtime settings
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
log_level = 'INFO'  # The level of logging
load_from = None # load models as a pre-trained model from a given path. This will not resume training
resume_from = None # Resume checkpoints from a given path, the training will be resumed from the iteration when the checkpoint's is saved
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. Keep this unchanged when training current matting models
```

</td>

<td valign="top">

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

</td>

</tr>
</thead>
</table>

### Other config settings

More details of config are shown in [config guides](/docs/en/user_guides/config).

## Model

We refactor models in MMEdit 1.x. Important modifications are as following.

- The `models` in MMedit 1.x is refactored to five parts: `base_models`, `data_preprocessors`, `editors`, `layers` and `losses`.
- Add `data_preprocessor` module in `models`. Normalization and color space transforms operations are moved from datasets transforms pipelines to data_preprocessor. The data out from the data pipeline is transformed by this module and then fed into the model.

More details of models are shown in [model guides](/docs/en/advanced_guides/models).
