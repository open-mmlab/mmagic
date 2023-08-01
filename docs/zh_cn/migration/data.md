# Data Settings 的迁移

本篇文档负责介绍 data settings 的迁移方式：

- \[Data Settings 的迁移\](#Data Settings 的迁移)
  - [Data Pipelines](#data-pipelines)
  - [Dataloader](#dataloader)

## Data Pipelines

在 MMagic 1.x 中我们更新了 data pipeline 的设置，有以下几个重要的修改：

- 去除了 `normalization` 和 `color space` 两种数据变换操作，并将它们移动到了 `data_preprocessor` 部分。
- 原版本中格式化数据变换 pipeline 的 `Collect` 和 `ToTensor` 在新版本中被整合为 `PackInputs`。更多的细节可以在 [数据变换文档](../howto/transforms.md) 中查看。

<table class="docutils">
<thead>
  <tr>
    <th> 原版本 </th>
    <th> 新版本 </th>
<tbody>
<tr>
<td valign="top">

```python
train_pipeline = [  # Train pipeline
    dict(type='LoadImageFromFile',  # 从文件读取图片
        io_backend='disk',  # io backend
        key='lq',  # 找到结果对应路径的 keys
        flag='unchanged'),  # 读取图片的 flag
    dict(type='LoadImageFromFile',  # 从文件读取图片
        io_backend='disk',  # io backend
        key='gt',  # 找到结果对应路径的 keys
        flag='unchanged'),  # 读取图片的 flag
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),  # 将图片从 [0, 255] 缩放到 [0, 1]
    dict(type='Normalize',  # normalize 图片的 augmentation pipeline
        keys=['lq', 'gt'],  # 需要 normalized 的图片
        mean=[0, 0, 0],  # 平均值
        std=[1, 1, 1],  # 标准差
        to_rgb=True),  # 是否转换到 rgb 通道
    dict(type='PairedRandomCrop', gt_patch_size=96),  # PairedRandomCrop
    dict(type='Flip',  # 翻转图片
        keys=['lq', 'gt'],  # 需要翻转的图片
        flip_ratio=0.5,  # 翻转概率
        direction='horizontal'),  # Flip 方向
    dict(type='Flip',  # Flip 图片
        keys=['lq', 'gt'],  # 需要翻转的图片
        flip_ratio=0.5,  # Flip ratio
        direction='vertical'),  # Flip 方向
    dict(type='RandomTransposeHW',  # 随即对图片的高和宽转置
        keys=['lq', 'gt'],  # 需要 transpose 的图片
        transpose_ratio=0.5  # Transpose ratio
        ),
    dict(type='Collect',  # Pipeline that decides which keys in the data should be passed to the model
        keys=['lq', 'gt'],  # Keys to pass to the model
        meta_keys=['lq_path', 'gt_path']), # Meta information keys. 训练时 meta information 不是必须的
    dict(type='ToTensor',  # 图片转为 tensor
        keys=['lq', 'gt'])  # 需要转换为 tensor 的图片
]
test_pipeline = [  # Test pipeline
    dict(
        type='LoadImageFromFile',   # 从文件读取图片
        io_backend='disk',  # io backend
        key='lq', # 找到结果对应路径的 keys
        flag='unchanged'),  # flag for reading images
    dict(
        type='LoadImageFromFile',   # 从文件读取图片
        io_backend='disk',  # io backend
        key='gt', # 找到结果对应路径的 keys
        flag='unchanged'),  # flag for reading images
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),  # 将图片从 [0, 255] 缩放到 [0, 1]
    dict(
        type='Normalize',  # 对输入图片执行 normalization 的数据增强 pipeline
        keys=['lq', 'gt'],  # 需要 normalized 图片
        mean=[0, 0, 0],  # Mean values
        std=[1, 1, 1],  # Standard variance
        to_rgb=True),  # 是否转为 rgb 格式
    dict(type='Collect',  # Pipeline that decides which keys in the data should be passed to the model
        keys=['lq', 'gt'],  # Keys to pass to the model
        meta_keys=['lq_path', 'gt_path']),  # Meta information keys
    dict(type='ToTensor',  # 图片转为 tensor
        keys=['lq', 'gt'])  # 需要转换为 tensor 的图片
]
```

</td>

<td valign="top">

```python
train_pipeline = [  # train pipeline
    dict(type='LoadImageFromFile',  # 从文件读取图片
        key='img',   # 找到结果对应路径的 keys
        color_type='color',  # 图片的 color type
        channel_order='rgb',  # 图片的 channel 顺序
        imdecode_backend='cv2'),  # decode backend
    dict(type='LoadImageFromFile',  # 从文件读取图片
        key='gt',   # 找到结果对应路径的 keys
        color_type='color',  # 图片的 color type
        channel_order='rgb',  # 图片的 channel 顺序
        imdecode_backend='cv2'),  # decode backend
    dict(type='SetValues', dictionary=dict(scale=scale)),  # 设置 destination keys
    dict(type='PairedRandomCrop', gt_patch_size=96),  # PairedRandomCrop
    dict(type='Flip',  # 翻转图片
        keys=['lq', 'gt'],  # 需要翻转的图片
        flip_ratio=0.5,  # Flip ratio
        direction='horizontal'),  # Flip 方向
    dict(type='Flip',  # Flip images
        keys=['lq', 'gt'],  # 需要翻转的图片
        flip_ratio=0.5,  # Flip ratio
        direction='vertical'),  # Flip 方向
    dict(type='RandomTransposeHW',  # 随即对图片的高和宽进行转置
        keys=['lq', 'gt'], # 需要转置的图片
        transpose_ratio=0.5  # Transpose ratio
        ),
    dict(type='PackInputs')  # 在当前 pipeline 中收集数据的设置
]
test_pipeline = [  # Test pipeline
    dict(type='LoadImageFromFile',   # 从文件读取图片
        key='img',   # 找到结果对应路径的 keys
        color_type='color',  # 图片的 color type
        channel_order='rgb',  # 图片的 channel order
        imdecode_backend='cv2'),  # decode backend
    dict(type='LoadImageFromFile',   # 从文件读取图片
        key='gt',  # 找到结果对应路径的 keys
        color_type='color', # 图片的 color type
        channel_order='rgb',  # 图片的 channel order
        imdecode_backend='cv2'),  # decode backend
    dict(type='PackInputs')  # 在当前 pipeline 中收集数据的设置
]
```

</td>

</tr>
</thead>
</table>

## Dataloader

在 MMagic 1.x 中我们更新了 dataloader 的设置方式，有以下几个重要的修改：

- 原版本中的 `data` 字段分为了 `train_dataloader` ， `val_dataloader` 和 `test_dataloader` 三个独立的部分。这样我们就可以细粒度的对各部分进行配置。例如用户就可以针对训练和测试制定不同的 sampler 和 batch size 。
- `samples_per_gpu` 更名为 `batch_size` 。
- `workers_per_gpu` 更名为 `num_workers` 。

<table class="docutils">
<thead>
  <tr>
    <th> 原版本 </th>
    <th> 新版本 </th>
<tbody>
<tr>
<td valign="top">

```python
data = dict(
    # train
    samples_per_gpu=16,  # 每个 GPU 上的 batch_size
    workers_per_gpu=4,  # 每个 GPU 上做 pre-fetch 的 worker 数
    drop_last=True,  # 在 data_loader 中使用 drop_last
    train=dict(  # Train dataset 配置
        type='RepeatDataset',  # 对 iter-based 模型设置为 RepeatDataset
        times=1000,  # RepeatDataset 的 repeated times 参数
        dataset=dict(
            type=train_dataset_type,  # 数据集类型
            lq_folder='data/DIV2K/DIV2K_train_LR_bicubic/X2_sub',  # lq 的文件路径
            gt_folder='data/DIV2K/DIV2K_train_HR_sub',  # ground truth 的文件路径
            ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',  # 标注文件的路径
            pipeline=train_pipeline,  # 参照 train_pipeline
            scale=scale)),  # Upsampling 的 scale factor
    # validation
    val_samples_per_gpu=1,  # validation 时每个 GPU 上的 batch_size
    val_workers_per_gpu=4,  # validation 是每个 GPU 上做 pre-fetch 的 worker 数
    val=dict(
        type=val_dataset_type,  # 数据集类型
        lq_folder='data/val_set5/Set5_bicLRx2',  # lq 的文件路径
        gt_folder='data/val_set5/Set5_mod12',  # ground truth 的文件路径
        pipeline=test_pipeline,  # 参照 test_pipeline
        scale=scale,  # Upsampling 的 scale factor
        filename_tmpl='{}'),  # filename 模板
    # test
    test=dict(
        type=val_dataset_type,  # 数据集类型
        lq_folder='data/val_set5/Set5_bicLRx2', # lq 的文件路径
        gt_folder='data/val_set5/Set5_mod12',  # ground truth 的文件路径
        pipeline=test_pipeline,  # 参照 test_pipeline
        scale=scale,  # Upsampling 的 scale factor
        filename_tmpl='{}'),  # filename 模板
)
```

</td>

<td valign="top">

```python
dataset_type = 'BasicImageDataset'  # 数据集类型
data_root = 'data'  # 数据集根目录
train_dataloader = dict(
    batch_size=16,
    num_workers=4,  # 每个 GPU 上做 pre-fetch 的 worker 数
    persistent_workers=False,  # 是否保持 workers instance 存活
    sampler=dict(type='InfiniteSampler', shuffle=True),  # data sampler 类型
    dataset=dict(  # 训练数据集 config
        type=dataset_type,  # 数据集类型
        ann_file='meta_info_DIV2K800sub_GT.txt',  # 标注文件路径
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DIV2K',  # 数据根目录
        data_prefix=dict(  # 图像文件前缀
            img='DIV2K_train_LR_bicubic/X2_sub', gt='DIV2K_train_HR_sub'),
        filename_tmpl=dict(img='{}', gt='{}'),  # Filename 模板
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,  # 每个 GPU 上做 pre-fetch 的 worker 数
    persistent_workers=False,  # 是否保持 workers instance 存活
    drop_last=False,  # 是否丢弃最后未完成的 batch
    sampler=dict(type='DefaultSampler', shuffle=False),  # data sampler 类型
    dataset=dict(  # Validation 数据集设置
        type=dataset_type,  # 数据集类型
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root=data_root + '/Set5',  # 数据根目录
        data_prefix=dict(img='LRbicx2', gt='GTmod12'),  # 图像文件前缀
        pipeline=test_pipeline))
test_dataloader = val_dataloader
```

</td>

</tr>
</thead>
</table>
