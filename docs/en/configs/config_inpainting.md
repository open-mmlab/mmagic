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
    ),
    pretrained=None) # The path of pretrained weight

train_cfg = dict(
    disc_step=1, # The steps of training discriminator before training generator
    iter_tc=90000, # Iterations of warming up generator
    iter_td=100000, # Iterations of warming up discriminator
    start_iter=0, # Starting iteration
    local_size=(128, 128)) # The size of local patches
test_cfg = dict(metrics=['l1']) # The config of testing scheme

dataset_type = 'ImgInpaintingDataset' # The type of dataset
input_shape = (256, 256) # The shape of input image

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img'), # The config of loading image
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
        keys=['gt_img'],  # The keys of images to be cropped
        crop_size=(384, 384),  # The size of cropped patch
        random_crop=True,  # Whether to use random crop
    ),
    dict(
        type='Resize',  # The type of resizing pipeline
        keys=['gt_img'],  # They keys of images to be resized
        scale=input_shape,  # The scale of resizing function
        keep_ratio=False,  # Whether to keep ratio during resizing
    ),
    dict(
        type='Normalize',  # The type of normalizing pipeline
        keys=['gt_img'],  # The keys of images to be normed
        mean=[127.5] * 3,  # Mean value used in normalization
        std=[127.5] * 3,  # Std value used in normalization
        to_rgb=False),  # Whether to transfer image channels to rgb
    dict(type='GetMaskedImage'),  # The config of getting masked image pipeline
    dict(
        type='Collect',  # The type of collecting data from current pipeline
        keys=['gt_img', 'masked_img', 'mask', 'mask_bbox'],  # The keys of data to be collected
        meta_keys=['gt_img_path']),  # The meta keys of data to be collected
    dict(type='ImageToTensor', keys=['gt_img', 'masked_img', 'mask']),  # The config dict of image to tensor pipeline
    dict(type='ToTensor', keys=['mask_bbox'])  # The config dict of ToTensor pipeline
]

test_pipeline = train_pipeline  # Constructing testing/validation pipeline

data_root = 'data/places365'  # Set data root

data = dict(
    samples_per_gpu=12,  # Batch size of a single GPU
    workers_per_gpu=8,  # Worker to pre-fetch data for each single GPU
    val_samples_per_gpu=1,  # Batch size of a single GPU in validation
    val_workers_per_gpu=8,  # Worker to pre-fetch data for each single GPU in validation
    drop_last=True,  # Whether to drop out the last batch of data
    train=dict(  # Train dataset config
        type=dataset_type,
        ann_file=f'{data_root}/train_places_img_list_total.txt',
        data_prefix=data_root,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(  # Validation dataset config
        type=dataset_type,
        ann_file=f'{data_root}/val_places_img_list.txt',
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True))

optimizers = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    generator=dict(type='Adam', lr=0.0004), disc=dict(type='Adam', lr=0.0004))

lr_config = dict(policy='Fixed', by_epoch=False)  # Learning rate scheduler config used to register LrUpdater hook

checkpoint_config = dict(by_epoch=False, interval=50000)  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
log_config = dict(  # config to register logger hook
    interval=100,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])  # The logger used to record the training process.

visual_config = dict(  # config to register visualization hook
    type='MMEditVisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=[
        'gt_img', 'masked_img', 'fake_res', 'fake_img', 'fake_gt_local'
    ],
)  # The logger used to visualize the training process.

evaluation = dict(interval=50000)  # The config to build the evaluation hook

total_iters = 500002
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The level of logging.
work_dir = None  # Directory to save the model checkpoints and logs for the current experiments.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 10000)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.
exp_name = 'gl_places'  # The experiment name
find_unused_parameters = False  # Whether to set find unused parameters in ddp
```
