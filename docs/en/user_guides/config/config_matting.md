# Config System for Matting

Same as [MMDetection](https://github.com/open-mmlab/mmdetection), we incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.

## An Example - Deep Image Matting Model

To help the users have a basic idea of a complete config, we make a brief comments on the config of the original DIM model we implemented as the following. For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation.

```python
# model settings
model = dict(
    type='DIM',  # The name of model (we call mattor).
    data_preprocessor=dict(  # The Config to build data preprocessor
        type='MattorPreprocessor',
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
    dict(type='PackEditInputs'),  # The config of collecting data from current pipeline
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
    dict(type='PackEditInputs'),  # The config of collecting data from current pipeline
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

default_scope = 'mmedit'  # Used to set registries location
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
