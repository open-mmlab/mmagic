# Config System for Matting

Same as [MMDetection](https://github.com/open-mmlab/mmdetection), we incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.

## An Example - Deep Image Matting Model

To help the users have a basic idea of a complete config, we make a brief comments on the config of the original DIM model we implemented as the following. For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation.

```python
# model settings
model = dict(
    type='DIM',  # The name of model (we call mattor).
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
        loss_weight=0.5))  # The weight of the composition loss.
train_cfg = dict(  # Config of training DIM model.
    train_backbone=True,  # In DIM stage1, backbone is trained.
    train_refiner=False)  # In DIM stage1, refiner is not trained.
test_cfg = dict(  # Config of testing DIM model.
    refine=False,  # Whether use refiner output as output, in stage1, we don't use it.
    metrics=['SAD', 'MSE', 'GRAD', 'CONN'])  # The metrics used when testing.

# data settings
dataset_type = 'AdobeComp1kDataset'  # Dataset type, this will be used to define the dataset.
data_root = 'data/adobe_composition-1k'  # Root path of data.
img_norm_cfg = dict(  # Image normalization config to normalize the input images.
    mean=[0.485, 0.456, 0.406],  # Mean values used to pre-training the pre-trained backbone models.
    std=[0.229, 0.224, 0.225],  # Standard variance used to pre-training the pre-trained backbone models.
    to_rgb=True)  # The channel orders of image used to pre-training the pre-trained backbone models.
train_pipeline = [  # Training data processing pipeline.
    dict(
        type='LoadImageFromFile',  # Load alpha matte from file.
        key='alpha',  # Key of alpha matte in annotation file. The pipeline will read alpha matte from path `alpha_path`.
        flag='grayscale'),  # Load as grayscale image which has shape (height, width).
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
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],  # Images to crop.
        crop_sizes=[320, 480, 640]),  # Candidate crop size.
    dict(
        type='Flip',  # Augmentation pipeline that flips the images.
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg']),  # Images to be flipped.
    dict(
        type='Resize',  # Augmentation pipeline that resizes the images.
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],  # Images to be resized.
        scale=(320, 320),  # Target size.
        keep_ratio=False),  # Whether to keep the ratio between height and width.
    dict(
        type='GenerateTrimap',  # Generate trimap from alpha matte.
        kernel_size=(1, 30)),  # Kernel size range of the erode/dilate kernel.
    dict(
        type='RescaleToZeroOne',  # Rescale images from [0, 255] to [0, 1].
        keys=['merged', 'alpha', 'ori_merged', 'fg', 'bg']),  # Images to be rescaled.
    dict(
        type='Normalize',  # Augmentation pipeline that normalize the input images.
        keys=['merged'],  # Images to be normalized.
        **img_norm_cfg),  # Normalization config. See above for definition of `img_norm_cfg`
    dict(
        type='Collect',  # Pipeline that decides which keys in the data should be passed to the model
        keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg'],  # Keys to pass to the model
        meta_keys=[]),  # Meta information keys. In training, meta information is not needed.
    dict(
        type='ImageToTensor',  # Convert images to tensor.
        keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg']),  # Images to be converted to Tensor.
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',  # Load alpha matte.
        key='alpha',  # Key of alpha matte in annotation file. The pipeline will read alpha matte from path `alpha_path`.
        flag='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',  # Load image from file
        key='trimap',  # Key of image to load. The pipeline will read trimap from path `trimap_path`.
        flag='grayscale',  # Load as grayscale image which has shape (height, width).
        save_original_img=True),  # Save a copy of trimap for calculating metrics. It will be saved with key `ori_trimap`
    dict(
        type='LoadImageFromFile',  # Load image from file
        key='merged'),  # Key of image to load. The pipeline will read merged from path `merged_path`.
    dict(
        type='Pad',  # Pipeline to pad images to align with the downsample factor of the model.
        keys=['trimap', 'merged'],  # Images to be padded.
        mode='reflect'),  # Mode of the padding.
    dict(
        type='RescaleToZeroOne',  # Same as it in train_pipeline.
        keys=['merged', 'ori_alpha']),  # Images to be rescaled.
    dict(
        type='Normalize',  # Same as it in train_pipeline.
        keys=['merged'],
        **img_norm_cfg),
    dict(
        type='Collect',  # Same as it in train_pipeline.
        keys=['merged', 'trimap'],
        meta_keys=[
            'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha',
            'ori_trimap'
        ]),
    dict(
        type='ImageToTensor',  # Same as it in train_pipeline.
        keys=['merged', 'trimap']),
]
data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU.
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU.
    drop_last=True,  # Use drop_last in data_loader.
    train=dict(  # Train dataset config.
        type=dataset_type,  # Type of dataset.
        ann_file=f'{data_root}/training_list.json',  # Path of annotation file
        data_prefix=data_root,  # Prefix of image path.
        pipeline=train_pipeline),  # See above for train_pipeline
    val=dict(  # Validation dataset config.
        type=dataset_type,
        ann_file=f'{data_root}/test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline),  # See above for test_pipeline
    test=dict(  # Test dataset config.
        type=dataset_type,
        ann_file=f'{data_root}/test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline))  # See above for test_pipeline

# optimizer
optimizers = dict(type='Adam', lr=0.00001)  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch.
# learning policy
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='Fixed')  # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.

# checkpoint saving
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=40000,  # The save interval is 40000 iterations.
    by_epoch=False)  # Count by iterations.
evaluation = dict(  # The config to build the evaluation hook.
    interval=40000)  # Evaluation interval.
log_config = dict(  # Config to register logger hook.
    interval=10,  # Interval to print the log.
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),  # The logger used to record the training process.
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported.
    ])

# runtime settings
total_iters = 1000000  # Total iterations to train the model.
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The level of logging.
work_dir = './work_dirs/dim_stage1'  # Directory to save the model checkpoints and logs for the current experiments.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. Keep this unchanged when training current matting models.
```
