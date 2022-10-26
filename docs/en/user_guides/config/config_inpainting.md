# Config System for Inpainting

## Config Name Style

Same as [MMDetection](https://github.com/open-mmlab/mmdetection), we incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.

## Config Field Description

To help the users have a basic idea of a complete config and the modules in a inpainting system,
we make brief comments on the config of Global&Local as the following.
For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation.

```python
model = dict(
    type='GLInpaintor', # The name of inpaintor
    data_preprocessor=dict(
        type='EditDataPreprocessor', # The name of data preprocessor
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
    dict(type='PackEditInputs'), # The config of collecting data from current pipeline
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

default_scope = 'mmedit' # Used to set registries location
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
