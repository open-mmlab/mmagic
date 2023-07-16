# Tutorial 1: Learn about Configs in MMagic

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

You can learn about the usage of our config system according to the following tutorials.

- [Tutorial 1: Learn about Configs in MMagic](#tutorial-1-learn-about-configs-in-mmagic)
  - [Modify config through script arguments](#modify-config-through-script-arguments)
  - [Config file structure](#config-file-structure)
  - [Config name style](#config-name-style)
  - [An example of EDSR](#an-example-of-edsr)
    - [Model config](#model-config)
    - [Data config](#data-config)
      - [Data pipeline](#data-pipeline)
      - [Dataloader](#dataloader)
    - [Evaluation config](#evaluation-config)
    - [Training and testing config](#training-and-testing-config)
    - [Optimization config](#optimization-config)
    - [Hook config](#hook-config)
    - [Runtime config](#runtime-config)
  - [An example of StyleGAN2](#an-example-of-stylegan2)
    - [Model config](#model-config-1)
    - [Dataset and evaluator config](#dataset-and-evaluator-config)
    - [Training and testing config](#training-and-testing-config-1)
    - [Optimization config](#optimization-config-1)
    - [Hook config](#hook-config-1)
    - [Runtime config](#runtime-config-1)
  - [Other examples](#other-examples)
    - [An example of config system for inpainting](#an-example-of-config-system-for-inpainting)
    - [An example of config system for matting](#an-example-of-config-system-for-matting)
    - [An example of config system for restoration](#an-example-of-config-system-for-restoration)

## Modify config through script arguments

When submitting jobs using `tools/train.py` or `tools/test.py`, you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options test_cfg.use_ema=False` changes the default sampling model to the original generator,
  and  `--cfg-options train_dataloader.batch_size=8` changes the batch size of train dataloader.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config.
  For example, the training pipeline `train_dataloader.dataset.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromWebcam'` in the pipeline,
  you may specify `--cfg-options train_dataloader.dataset.pipeline.0.type=LoadImageFromWebcam`.
  The training pipeline `train_pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadMask'` in the pipeline,
  you may specify `--cfg-options train_pipeline.0.type=LoadMask`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. You can set `--cfg-options key="[a,b]"` or `--cfg-options key=a,b`. It also allows nested list/tuple values, e.g., `--cfg-options key="[(a,b),(c,d)]"`. Note that the quotation mark " is necessary to support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Config file structure

There are 3 basic component types under `config/_base_`: datasets, models and default_runtime.
Many methods could be easily constructed with one of each like AOT-GAN, EDVR, GLEAN, StyleGAN2, CycleGAN, SinGAN, etc.
Configs consisting of components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from existing methods.
For example, if some modification is made base on BasicVSR,
user may first inherit the basic BasicVSR structure by specifying `_base_ = ../basicvsr/basicvsr_reds4.py`,
then modify the necessary fields in the config files.
If some modification is made base on StyleGAN2,
user may first inherit the basic StyleGAN2 structure by specifying `_base_ = ../styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py`,
then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods,
you may create a folder `xxx` under `configs`,

Please refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/config.md) for detailed documentation.

## Config name style

```
{model}_[module setting]_{training schedule}_{dataset}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type like `stylegan`, `dcgan`, `basicvsr`, `dim`, etc.
  Settings referred in the original paper are included in this field as well (e.g., `Stylegan2-config-f`, `edvrm` of `edvrm_8xb4-600k_reds`.)
- `[module setting]`: specific setting for some modules, including Encoder, Decoder, Generator, Discriminator, Normalization, loss, Activation, etc. E.g. `c64n7` of `basicvsr-pp_c64n7_8xb1-600k_reds4`, learning rate `Glr4e-4_Dlr1e-4` for dcgan, `gamma32.8` for stylegan3, `woReLUInplace` in sagan. In this section, information from different submodules (e.g., generator and discriminator) are connected with `_`.
- `{training_scheduler}`: specific setting for training, including batch_size, schedule, etc. For example, learning rate (e.g., `lr1e-3`), number of gpu and batch size is used (e.g., `8xb32`), and total iterations (e.g., `160kiter`) or number of images shown in the discriminator (e.g., `12Mimgs`).
- `{dataset}`: dataset name and data size info like `celeba-256x256` of `deepfillv1_4xb4_celeba-256x256`, `reds4` of `basicvsr_2xb4_reds4`, `ffhq`, `lsun-car`, `celeba-hq`.

## An example of EDSR

To help the users have a basic idea of a complete config,
we make a brief comments on the [config of the EDSR model](https://github.com/open-mmlab/mmagic/blob/main/configs/edsr/edsr_x2c64b16_g1_300k_div2k.py) we implemented as the following.
For more detailed usage and the corresponding alternative for each modules,
please refer to the API documentation and the [tutorial in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/config.md).

### Model config

In MMagic's config, we use model fields to set up a model.

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
        type='DataPreprocessor', mean=[0., 0., 0.], std=[255., 255.,
                                                             255.]))
```

### Data config

[Dataloaders](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader) are required for the training, validation, and testing of the [runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html).
Dataset and data pipeline need to be set to build the dataloader. Due to the complexity of this part, we use intermediate variables to simplify the writing of dataloader configs.

#### Data pipeline

```python
train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile',  # Load images from files
        key='img',  # Keys in results to find the corresponding path
        color_type='color',  # Color type of image
        channel_order='rgb',  # Channel order of image
        imdecode_backend='cv2'),  # decode backend
    dict(type='LoadImageFromFile',  # Load images from files
        key='gt',  # Keys in results to find the corresponding path
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
    dict(type='PackInputs')  # The config of collecting data from the current pipeline
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
    dict(type='PackInputs')  # The config of collecting data from the current pipeline
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

[Evaluators](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html) are used to compute the metrics of the trained model on the validation and testing datasets.
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
The optimizer wrapper not only provides the functions of the optimizer, but also supports functions such as gradient clipping, mixed precision training, etc. Find more in [optimizer wrapper tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html).

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
default_scope = 'mmagic' # Used to set registries location
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

## An example of StyleGAN2

Taking [Stylegan2 at 1024x1024 scale](https://github.com/open-mmlab/mmagic/blob/main/configs//styleganv2/stylegan2_c2_8xb4-fp16-global-800kiters_quicktest-ffhq-256x256.py) as an example,
we introduce each field in the config according to different function modules.

### Model config

In addition to neural network components such as generator, discriminator etc, it also requires `data_preprocessor`, `loss_config`, and some of them contains `ema_config`.
`data_preprocessor` is responsible for processing a batch of data output by dataloader.
`loss_config` is responsible for weight of loss terms.
`ema_config` is responsible for exponential moving average (EMA) operation for generator.

```python
model = dict(
    type='StyleGAN2',  # The name of the model
    data_preprocessor=dict(type='DataPreprocessor'),  # The config of data preprocessor, usually includs image normalization and padding
    generator=dict(  # The config for generator
        type='StyleGANv2Generator',  # The name of the generator
        out_size=1024,  # The output resolution of the generator
        style_channels=512),  # The number of style channels of the generator
    discriminator=dict(  # The config for discriminator
        type='StyleGAN2Discriminator',  # The name of the discriminator
        in_size=1024),  # The input resolution of the discriminator
    ema_config=dict(  # The config for EMA
        type='ExponentialMovingAverage',  # Specific the type of Average model
        interval=1,  # The interval of EMA operation
        momentum=0.9977843871238888),  # The momentum of EMA operation
    loss_config=dict(  # The config for loss terms
        r1_loss_weight=80.0,  # The weight for r1 gradient penalty
        r1_interval=16,  # The interval of r1 gradient penalty
        norm_mode='HWC',  # The normalization mode for r1 gradient penalty
        g_reg_interval=4,  # The interval for generator's regularization
        g_reg_weight=8.0,  # The weight for generator's regularization
        pl_batch_shrink=2))  # The factor of shrinking the batch size in path length regularization
```

### Dataset and evaluator config

[Dataloaders](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader) are required for the training, validation, and testing of the [runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html).
Dataset and data pipeline need to be set to build the dataloader. Due to the complexity of this part, we use intermediate variables to simplify the writing of dataloader configs.

```python
dataset_type = 'BasicImageDataset'  # Dataset type, this will be used to define the dataset
data_root = './data/ffhq/'  # Root path of data

train_pipeline = [  # Training data process pipeline
    dict(type='LoadImageFromFile', key='img'),  # First pipeline to load images from file path
    dict(type='Flip', keys=['img'], direction='horizontal'),  # Argumentation pipeline that flip the images
    dict(type='PackInputs', keys=['img'])  # The last pipeline that formats the annotation data (if have) and decides which keys in the data should be packed into data_samples
]
val_pipeline = [
    dict(type='LoadImageFromFile', key='img'),  # First pipeline to load images from file path
    dict(type='PackInputs', keys=['img'])  # The last pipeline that formats the annotation data (if have) and decides which keys in the data should be packed into data_samples
]
train_dataloader = dict(  # The config of train dataloader
    batch_size=4,  # Batch size of a single GPU
    num_workers=8,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shutdown the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(  # The config of training data sampler
        type='InfiniteSampler',  # InfiniteSampler for iteratiion-based training. Refers to https://github.com/open-mmlab/mmengine/blob/fe0eb0a5bbc8bf816d5649bfdd34908c258eb245/mmengine/dataset/sampler.py#L107
        shuffle=True),  # Whether randomly shuffle the training data
    dataset=dict(  # The config of the training dataset
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline))
val_dataloader = dict(  # The config of validation dataloader
    batch_size=4,  # Batch size of a single GPU
    num_workers=8,  # Worker to pre-fetch data for each single GPU
    dataset=dict(  # The config of the validation dataset
        type=dataset_type,
        data_root=data_root,
        pipeline=val_pipeline),
    sampler=dict(  # The config of validatioin data sampler
        type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/fe0eb0a5bbc8bf816d5649bfdd34908c258eb245/mmengine/dataset/sampler.py#L14
        shuffle=False),  # Whether randomly shuffle the validation data
    persistent_workers=True)
test_dataloader = val_dataloader  # The config of the testing dataloader
```

[Evaluators](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html) are used to compute the metrics of the trained model on the validation and testing datasets.
The config of evaluators consists of one or a list of metric configs:

```python
val_evaluator = dict(  # The config for validation evaluator
    type='Evaluator',  # The type of evaluation
    metrics=[  # The config for metrics
        dict(
            type='FrechetInceptionDistance',
            prefix='FID-Full-50k',
            fake_nums=50000,
            inception_style='StyleGAN',
            sample_model='ema'),
        dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K'),
        dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
    ])
test_evaluator = val_evaluator  # The config for testing evaluator
```

### Training and testing config

MMEngine's runner uses Loop to control the training, validation, and testing processes.
Users can set the maximum training iteration and validation intervals with these fields.

```python
train_cfg = dict(  # The config for training
    by_epoch=False,  # Set `by_epoch` as False to use iteration-based training
    val_begin=1,  # Which iteration to start the validation
    val_interval=10000,  # Validation intervals
    max_iters=800002)  # Maximum training iterations
val_cfg = dict(type='MultiValLoop')  # The validation loop type
test_cfg = dict(type='MultiTestLoop')  # The testing loop type
```

### Optimization config

`optim_wrapper` is the field to configure optimization related settings.
The optimizer wrapper not only provides the functions of the optimizer, but also supports functions such as gradient clipping, mixed precision training, etc. Find more in [optimizer wrapper tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html).

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

`param_scheduler` is a field that configures methods of adjusting optimization hyperparameters such as learning rate and momentum.
Users can combine multiple schedulers to create a desired parameter adjustment strategy.
Find more in [parameter scheduler tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html).
Since StyleGAN2 do not use parameter scheduler, we use config in [CycleGAN](https://github.com/open-mmlab/mmagic/blob/main/configs/cyclegan/cyclegan_lsgan-id0-resnet-in_1xb1-250kiters_summer2winter.py) as an example:

```python
# parameter scheduler in CycleGAN config
param_scheduler = dict(
    type='LinearLrInterval',  # The type of scheduler
    interval=400,  # The interval to update the learning rate
    by_epoch=False,  # The scheduler is called by iteration
    start_factor=0.0002,  # The number we multiply parameter value in the first iteration
    end_factor=0,  # The number we multiply parameter value at the end of linear changing process.
    begin=40000,  # The start iteration of the scheduler
    end=80000)  # The end iteration of the scheduler
```

### Hook config

Users can attach hooks to training, validation, and testing loops to insert some operations during running. There are two different hook fields, one is `default_hooks` and the other is `custom_hooks`.

`default_hooks` is a dict of hook configs. `default_hooks` are the hooks must required at runtime. They have default priority which should not be modified. If not set, runner will use the default values. To disable a default hook, users can set its config to `None`.

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

`custom_hooks` is a list of hook configs. Users can develop there own hooks and insert them in this field.

```python
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]
```

### Runtime config

```python
default_scope = 'mmagic'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

# config for environment
env_cfg = dict(
    cudnn_benchmark=True,  # whether to enable cudnn benchmark.
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # set multi process parameters.
    dist_cfg=dict(backend='nccl'),  # set distributed parameters.
)

log_level = 'INFO'  # The level of logging
log_processor = dict(
    type='LogProcessor',  # log processor to process runtime logs
    by_epoch=False)  # print log by iteration
load_from = None  # load model checkpoint as a pre-trained model for a given path
resume = False  # Whether to resume from the checkpoint define in `load_from`. If `load_from` is `None`, it will resume the latest checkpoint in `work_dir`
```

## Other examples

### An example of config system for inpainting

To help the users have a basic idea of a complete config and the modules in a inpainting system,
we make brief comments on the config of Global&Local as the following.
For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation.

```python
model = dict(
    type='GLInpaintor', # The name of inpaintor
    data_preprocessor=dict(
        type='DataPreprocessor', # The name of data preprocessor
        mean=[127.5], # Mean value used in data normalization
        std=[127.5], # Std value used in data normalization
    ),
    encdec=dict(
        type='GLEncoderDecoder', # The name of encoder-decoder
        encoder=dict(type='GLEncoder', norm_cfg=dict(type='SyncBN')), # The config of encoder
        decoder=dict(type='GLDecoder', norm_cfg=dict(type='SyncBN')), # The config of decoder
        dilation_neck=dict(
            type='GLDilationNeck', norm_cfg=dict(type='SyncBN'))), # The config of dilation neck
    disc=dict(
        type='GLDiscs', # The name of discriminator
        global_disc_cfg=dict(
            in_channels=3, # The input channel of discriminator
            max_channels=512, # The maximum middle channel in discriminator
            fc_in_channels=512 * 4 * 4, # The input channel of last fc layer
            fc_out_channels=1024, # The output channel of last fc channel
            num_convs=6, # The number of convs used in discriminator
            norm_cfg=dict(type='SyncBN') # The config of norm layer
        ),
        local_disc_cfg=dict(
            in_channels=3, # The input channel of discriminator
            max_channels=512, # The maximum middle channel in discriminator
            fc_in_channels=512 * 4 * 4, # The input channel of last fc layer
            fc_out_channels=1024, # The output channel of last fc channel
            num_convs=5, # The number of convs used in discriminator
            norm_cfg=dict(type='SyncBN') # The config of norm layer
        ),
    ),
    loss_gan=dict(
        type='GANLoss', # The name of GAN loss
        gan_type='vanilla', # The type of GAN loss
        loss_weight=0.001 # The weight of GAN loss
    ),
    loss_l1_hole=dict(
        type='L1Loss', # The type of l1 loss
        loss_weight=1.0 # The weight of l1 loss
    ))

train_cfg = dict(
    type='IterBasedTrainLoop',# The name of train loop type
    max_iters=500002, # The number of total iterations
    val_interval=50000, # The number of validation interval iterations
)
val_cfg = dict(type='ValLoop') # The name of validation loop type
test_cfg = dict(type='TestLoop') # The name of test loop type

val_evaluator = [
    dict(type='MAE', mask_key='mask', scaling=100), # The name of metrics to evaluate
    dict(type='PSNR'), # The name of metrics to evaluate
    dict(type='SSIM'), # The name of metrics to evaluate
]
test_evaluator = val_evaluator

input_shape = (256, 256) # The shape of input image

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt'), # The config of loading image
    dict(
        type='LoadMask', # The type of loading mask pipeline
        mask_mode='bbox', # The type of mask
        mask_config=dict(
            max_bbox_shape=(128, 128), # The shape of bbox
            max_bbox_delta=40, # The changing delta of bbox height and width
            min_margin=20,  # The minimum margin from bbox to the image border
            img_shape=input_shape)),  # The input image shape
    dict(
        type='Crop', # The type of crop pipeline
        keys=['gt'],  # The keys of images to be cropped
        crop_size=(384, 384),  # The size of cropped patch
        random_crop=True,  # Whether to use random crop
    ),
    dict(
        type='Resize',  # The type of resizing pipeline
        keys=['gt'],  # They keys of images to be resized
        scale=input_shape,  # The scale of resizing function
        keep_ratio=False,  # Whether to keep ratio during resizing
    ),
    dict(
        type='Normalize',  # The type of normalizing pipeline
        keys=['gt_img'],  # The keys of images to be normed
        mean=[127.5] * 3,  # Mean value used in normalization
        std=[127.5] * 3,  # Std value used in normalization
        to_rgb=False),  # Whether to transfer image channels to rgb
    dict(type='GetMaskedImage'), # The config of getting masked image pipeline
    dict(type='PackInputs'), # The config of collecting data from the current pipeline
]

test_pipeline = train_pipeline  # Constructing testing/validation pipeline

dataset_type = 'BasicImageDataset' # The type of dataset
data_root = 'data/places'  # Root path of data

train_dataloader = dict(
    batch_size=12, # Batch size of a single GPU
    num_workers=4, # The number of workers to pre-fetch data for each single GPU
    persistent_workers=False, # Whether maintain the workers Dataset instances alive
    sampler=dict(type='InfiniteSampler', shuffle=False), # The type of data sampler
    dataset=dict(  # Train dataset config
        type=dataset_type, # Type of dataset
        data_root=data_root, # Root path of data
        data_prefix=dict(gt='data_large'), # Prefix of image path
        ann_file='meta/places365_train_challenge.txt', # Path of annotation file
        test_mode=False,
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1, # Batch size of a single GPU
    num_workers=4, # The number of workers to pre-fetch data for each single GPU
    persistent_workers=False, # Whether maintain the workers Dataset instances alive
    drop_last=False, # Whether drop the last incomplete batch
    sampler=dict(type='DefaultSampler', shuffle=False), # The type of data sampler
    dataset=dict( # Validation dataset config
        type=dataset_type, # Type of dataset
        data_root=data_root, # Root path of data
        data_prefix=dict(gt='val_large'), # Prefix of image path
        ann_file='meta/places365_val.txt', # Path of annotation file
        test_mode=True,
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader

model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel') # The name of model wrapper

optim_wrapper = dict( # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper', optimizer=dict(type='Adam', lr=0.0004)),
    disc=dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=0.0004)))

default_scope = 'mmagic' # Used to set registries location
save_dir = './work_dirs' # Directory to save the model checkpoints and logs for the current experiments
exp_name = 'gl_places'  # The experiment name

default_hooks = dict( # Used to build default hooks
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100), # Config to register logger hook
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict( # Config to set the checkpoint hook
        type='CheckpointHook',
        interval=50000,
        by_epoch=False,
        out_dir=save_dir),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict( # Parameters to setup distributed training, the port can also be set
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')] # The name of visualization backend
visualizer = dict( # Config used to build visualizer
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)] # Used to build custom hooks

log_level = 'INFO' # The level of logging
log_processor = dict(type='LogProcessor', by_epoch=False) # Used to build log processor

load_from = None # load models as a pre-trained model from a given path. This will not resume training.
resume = False # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.

find_unused_parameters = False  # Whether to set find unused parameters in ddp
```

### An example of config system for matting

To help the users have a basic idea of a complete config, we make a brief comments on the config of the original DIM model we implemented as the following. For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation.

```python
# model settings
model = dict(
    type='DIM',  # The name of model (we call mattor).
    data_preprocessor=dict(  # The Config to build data preprocessor
        type='DataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        proc_inputs='normalize',
        proc_trimap='rescale_to_zero_one',
        proc_gt='rescale_to_zero_one',
    ),
    backbone=dict(  # The config of the backbone.
        type='SimpleEncoderDecoder',  # The type of the backbone.
        encoder=dict(  # The config of the encoder.
            type='VGG16'),  # The type of the encoder.
        decoder=dict(  # The config of the decoder.
            type='PlainDecoder')),  # The type of the decoder.
    pretrained='./weights/vgg_state_dict.pth',  # The pretrained weight of the encoder to be loaded.
    loss_alpha=dict(  # The config of the alpha loss.
        type='CharbonnierLoss',  # The type of the loss for predicted alpha matte.
        loss_weight=0.5),  # The weight of the alpha loss.
    loss_comp=dict(  # The config of the composition loss.
        type='CharbonnierCompLoss',  # The type of the composition loss.
        loss_weight=0.5), # The weight of the composition loss.
    train_cfg=dict(  # Config of training DIM model.
        train_backbone=True,  # In DIM stage1, backbone is trained.
        train_refiner=False),  # In DIM stage1, refiner is not trained.
    test_cfg=dict(  # Config of testing DIM model.
        refine=False,  # Whether use refiner output as output, in stage1, we don't use it.
        resize_method='pad',
        resize_mode='reflect',
        size_divisor=32,
    ),
)

# data settings
dataset_type = 'AdobeComp1kDataset'  # Dataset type, this will be used to define the dataset.
data_root = 'data/adobe_composition-1k'  # Root path of data.

train_pipeline = [  # Training data processing pipeline.
    dict(
        type='LoadImageFromFile',  # Load alpha matte from file.
        key='alpha',  # Key of alpha matte in annotation file. The pipeline will read alpha matte from path `alpha_path`.
        color_type='grayscale'),  # Load as grayscale image which has shape (height, width).
    dict(
        type='LoadImageFromFile',  # Load image from file.
        key='fg'),  # Key of image to load. The pipeline will read fg from path `fg_path`.
    dict(
        type='LoadImageFromFile',  # Load image from file.
        key='bg'),  # Key of image to load. The pipeline will read bg from path `bg_path`.
    dict(
        type='LoadImageFromFile',  # Load image from file.
        key='merged'),  # Key of image to load. The pipeline will read merged from path `merged_path`.
    dict(
        type='CropAroundUnknown',  # Crop images around unknown area (semi-transparent area).
        keys=['alpha', 'merged', 'fg', 'bg'],  # Images to crop.
        crop_sizes=[320, 480, 640]),  # Candidate crop size.
    dict(
        type='Flip',  # Augmentation pipeline that flips the images.
        keys=['alpha', 'merged', 'fg', 'bg']),  # Images to be flipped.
    dict(
        type='Resize',  # Augmentation pipeline that resizes the images.
        keys=['alpha', 'merged', 'fg', 'bg'],  # Images to be resized.
        scale=(320, 320),  # Target size.
        keep_ratio=False),  # Whether to keep the ratio between height and width.
    dict(
        type='GenerateTrimap',  # Generate trimap from alpha matte.
        kernel_size=(1, 30)),  # Kernel size range of the erode/dilate kernel.
    dict(type='PackInputs'),  # The config of collecting data from the current pipeline
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',  # Load alpha matte.
        key='alpha',  # Key of alpha matte in annotation file. The pipeline will read alpha matte from path `alpha_path`.
        color_type='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',  # Load image from file
        key='trimap',  # Key of image to load. The pipeline will read trimap from path `trimap_path`.
        color_type='grayscale',  # Load as grayscale image which has shape (height, width).
        save_original_img=True),  # Save a copy of trimap for calculating metrics. It will be saved with key `ori_trimap`
    dict(
        type='LoadImageFromFile',  # Load image from file
        key='merged'),  # Key of image to load. The pipeline will read merged from path `merged_path`.
    dict(type='PackInputs'),  # The config of collecting data from the current pipeline
]

train_dataloader = dict(
    batch_size=1,  # Batch size of a single GPU
    num_workers=4,  # The number of workers to pre-fetch data for each single GPU
    persistent_workers=False,  # Whether maintain the workers Dataset instances alive
    sampler=dict(type='InfiniteSampler', shuffle=True),  # The type of data sampler
    dataset=dict(  # Train dataset config
        type=dataset_type,  # Type of dataset
        data_root=data_root,  # Root path of data
        ann_file='training_list.json',  # Path of annotation file
        test_mode=False,
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,  # Batch size of a single GPU
    num_workers=4,  # The number of workers to pre-fetch data for each single GPU
    persistent_workers=False,  # Whether maintain the workers Dataset instances alive
    drop_last=False,  # Whether drop the last incomplete batch
    sampler=dict(type='DefaultSampler', shuffle=False),  # The type of data sampler
    dataset=dict(  # Validation dataset config
        type=dataset_type,  # Type of dataset
        data_root=data_root,  # Root path of data
        ann_file='test_list.json',  # Path of annotation file
        test_mode=True,
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='SAD'),  # The name of metrics to evaluate
    dict(type='MattingMSE'),  # The name of metrics to evaluate
    dict(type='GradientError'),  # The name of metrics to evaluate
    dict(type='ConnectivityError'),  # The name of metrics to evaluate
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop',  # The name of train loop type
    max_iters=1_000_000,  # The number of total iterations
    val_interval=40000,  # The number of validation interval iterations
)
val_cfg = dict(type='ValLoop')  # The name of validation loop type
test_cfg = dict(type='TestLoop')  # The name of test loop type

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.00001),
    )
)  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch.

default_scope = 'mmagic'  # Used to set registries location
save_dir = './work_dirs'  # Directory to save the model checkpoints and logs for the current experiments.

default_hooks = dict(  # Used to build default hooks
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),  # Config to register logger hook
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(  # Config to set the checkpoint hook
        type='CheckpointHook',
        interval=40000,  # The save interval is 40000 iterations.
        by_epoch=False,  # Count by iterations.
        out_dir=save_dir),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(  # Parameters to setup distributed training, the port can also be set
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'  # The level of logging
log_processor = dict(type='LogProcessor', by_epoch=False)  # Used to build log processor

load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume = False  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
```

### An example of config system for restoration

To help the users have a basic idea of a complete config, we make a brief comments on the config of the EDSR model we implemented as the following. For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation.

```python
exp_name = 'edsr_x2c64b16_1x16_300k_div2k'  # The experiment name
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

load_from = None  # based on pre-trained x2 model

scale = 2  # Scale factor for upsampling
# model settings
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
        type='DataPreprocessor', mean=[0., 0., 0.], std=[255., 255.,
                                                             255.]))

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
    dict(type='PackInputs')  # The config of collecting data from the current pipeline
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
    dict(type='PackInputs')  # The config of collecting data from the current pipeline
]

# dataset settings
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

val_evaluator = [
    dict(type='MAE'),  # The name of metrics to evaluate
    dict(type='PSNR', crop_border=scale),  # The name of metrics to evaluate
    dict(type='SSIM', crop_border=scale),  # The name of metrics to evaluate
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=5000)  # Config of train loop type
val_cfg = dict(type='ValLoop')  # The name of validation loop type
test_cfg = dict(type='TestLoop')  # The name of test loop type

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.00001),
    )
)  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch.

param_scheduler = dict(  # Config of learning policy
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5)

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

default_scope = 'mmagic'  # Used to set registries location
save_dir = './work_dirs'  # Directory to save the model checkpoints and logs for the current experiments.

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
