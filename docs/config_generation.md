# Config System for Generation

Same as [MMDetection](https://github.com/open-mmlab/mmdetection), we incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.

## An Example - pix2pix

To help the users have a basic idea of a complete config and the modules in a generation system, we make brief comments on the config of pix2pix as the following.
For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation.

```python
# model settings
model = dict(
    type='Pix2Pix',  # The name of synthesizer
    generator=dict(
        type='UnetGenerator',  # The name of generator
        in_channels=3,  # The input channels of generator
        out_channels=3,  # The output channels of generator
        num_down=8,  # The umber of downsamplings in the generator
        base_channels=64,  # The number of channels at the last conv layer of generator
        norm_cfg=dict(type='BN'),  # The config of norm layer
        use_dropout=True,  # Whether to use dropout layers in the generator
        init_cfg=dict(type='normal', gain=0.02)),  # The config of initialization
    discriminator=dict(
        type='PatchDiscriminator',  # The name of discriminator
        in_channels=6,  # The input channels of discriminator
        base_channels=64,  # The number of channels at the first conv layer of discriminator
        num_conv=3,  # The number of stacked intermediate conv layers (excluding input and output conv layer) in the discriminator
        norm_cfg=dict(type='BN'),  # The config of norm layer
        init_cfg=dict(type='normal', gain=0.02)),  # The config of initialization
    gan_loss=dict(
        type='GANLoss',  # The name of GAN loss
        gan_type='vanilla',  # The type of GAN loss
        real_label_val=1.0,  # The value for real label of GAN loss
        fake_label_val=0.0,  # The value for fake label of GAN loss
        loss_weight=1.0),  # The weight of GAN loss
    pixel_loss=dict(type='L1Loss', loss_weight=100.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(
    direction='b2a')  # Image-to-image translation direction (the model training direction, same as testing direction) for pix2pix. Model default: a2b
test_cfg = dict(
    direction='b2a',   # Image-to-image translation direction (the model training direction, same as testing direction) for pix2pix. Model default: a2b
    show_input=True)  # Whether to show input real images when saving testing images for pix2pix

# dataset settings
train_dataset_type = 'GenerationPairedDataset'  # The type of dataset for training
val_dataset_type = 'GenerationPairedDataset'  # The type of dataset for validation/testing
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Image normalization config to normalize the input images
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',  # Load a pair of images from file path pipeline
        io_backend='disk',  # IO backend where images are store
        key='pair',  # Keys to find corresponding path
        flag='color'),  # Loading flag for images
    dict(
        type='Resize',  # Resize pipeline
        keys=['img_a', 'img_b'],  # The keys of images to be resized
        scale=(286, 286),  # The scale to resize images
        interpolation='bicubic'),  # Algorithm used for interpolation when resizing images
    dict(
        type='FixedCrop',  # Fixed crop pipeline, cropping paired images to a specific size at a specific position for pix2pix training
        keys=['img_a', 'img_b'],  # The keys of images to be cropped
        crop_size=(256, 256)),  # The size to crop images
    dict(
        type='Flip',  # Flip pipeline
        keys=['img_a', 'img_b'],  # The keys of images to be flipped
        direction='horizontal'),  # Flip images horizontally or vertically
    dict(
        type='RescaleToZeroOne',  # Rescale images from [0, 255] to [0, 1]
        keys=['img_a', 'img_b']),  # The keys of images to be rescaled
    dict(
        type='Normalize',  # Image normalization pipeline
        keys=['img_a', 'img_b'],  # The keys of images to be normalized
        to_rgb=True,  # Whether to convert image channels from BGR to RGB
        **img_norm_cfg),  # Image normalization config (see above for the definition of `img_norm_cfg`)
    dict(
       type='ImageToTensor',  # Image to tensor pipeline
       keys=['img_a', 'img_b']),  # The keys of images to be converted from image to tensor
    dict(
        type='Collect',  # Pipeline that decides which keys in the data should be passed to the synthesizer
        keys=['img_a', 'img_b'],  # The keys of images
        meta_keys=['img_a_path', 'img_b_path'])  # The meta keys of images
]
test_pipeline = [
    dict(
        type='LoadPairedImageFromFile',  # Load a pair of images from file path pipeline
        io_backend='disk',  # IO backend where images are store
        key='pair',  # Keys to find corresponding path
        flag='color'),  # Loading flag for images
    dict(
        type='Resize',  # Resize pipeline
        keys=['img_a', 'img_b'],  # The keys of images to be resized
        scale=(256, 256),  # The scale to resize images
        interpolation='bicubic'),  # Algorithm used for interpolation when resizing images
    dict(
        type='RescaleToZeroOne',  # Rescale images from [0, 255] to [0, 1]
        keys=['img_a', 'img_b']),  # The keys of images to be rescaled
    dict(
        type='Normalize',  # Image normalization pipeline
        keys=['img_a', 'img_b'],  # The keys of images to be normalized
        to_rgb=True,  # Whether to convert image channels from BGR to RGB
        **img_norm_cfg),  # Image normalization config (see above for the definition of `img_norm_cfg`)
    dict(
       type='ImageToTensor',  # Image to tensor pipeline
       keys=['img_a', 'img_b']),  # The keys of images to be converted from image to tensor
    dict(
        type='Collect',  # Pipeline that decides which keys in the data should be passed to the synthesizer
        keys=['img_a', 'img_b'],  # The keys of images
        meta_keys=['img_a_path', 'img_b_path'])  # The meta keys of images
]
data_root = 'data/pix2pix/facades'  # The root path of data
data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    drop_last=True,  # Whether to drop out the last batch of data in training
    val_samples_per_gpu=1,  # Batch size of a single GPU in validation
    val_workers_per_gpu=0,  # Worker to pre-fetch data for each single GPU in validation
    train=dict(  # Training dataset config
        type=train_dataset_type,
        dataroot=data_root,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(  # Validation dataset config
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(  # Testing dataset config
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True))

# optimizer
optimizers = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)  # Learning rate scheduler config used to register LrUpdater hook

# checkpoint saving
checkpoint_config = dict(interval=4000, save_optimizer=True, by_epoch=False)  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
evaluation = dict(  # The config to build the evaluation hook
    interval=4000,  # Evaluation interval
    save_image=True)  # Whether to save images
log_config = dict(  # config to register logger hook
    interval=100,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),   # The logger used to record the training process
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
    ])
visual_config = None  # The config to build the visualization hook

# runtime settings
total_iters = 80000  # Total iterations to train the model
cudnn_benchmark = True  # Set cudnn_benchmark
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
log_level = 'INFO'  # The level of logging
load_from = None  # Load models as a pre-trained model from a given path. This will not resume training
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. Keep this unchanged when training current generation models
exp_name = 'pix2pix_facades'  # The experiment name
work_dir = f'./work_dirs/{exp_name}'  # Directory to save the model checkpoints and logs for the current experiments.
```
