# Migration of Data Settings

This section introduces the migration of data settings:

- [Migration of Data Settings](#migration-of-data-settings)
  - [Data pipelines](#data-pipelines)
  - [Dataloader](#dataloader)

## Data pipelines

We update data pipelines settings in MMagic 1.x. Important modifications are as following.

- Remove normalization and color space transforms operations. They are moved from datasets transforms pipelines to data_preprocessor.
- The original formatting transforms pipelines `Collect` and `ToTensor` are combined as `PackInputs`.
  More details of data pipelines are shown in [transform guides](../howto/transforms.md).

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
    dict(type='PackInputs')  # The config of collecting data from current pipeline
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
    dict(type='PackInputs')  # The config of collecting data from current pipeline
]
```

</td>

</tr>
</thead>
</table>

## Dataloader

We update dataloader settings in MMagic 1.x. Important modifications are as following.

- The original `data` field is split to `train_dataloader`, `val_dataloader` and `test_dataloader`. This allows us to configure them in fine-grained. For example, you can specify different sampler and batch size during training and test.
- The `samples_per_gpu` is renamed to `batch_size`.
- The `workers_per_gpu` is renamed to `num_workers`.

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
