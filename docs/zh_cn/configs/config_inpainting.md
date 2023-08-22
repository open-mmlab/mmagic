# 配置文件 - 补全

## 配置名称样式

与 [MMDetection](https://github.com/open-mmlab/mmdetection) 一样，我们将模块化和继承设计融入我们的配置系统，以方便进行各种实验。

## 配置字段说明

为了帮助用户对完整的配置和修复系统中的模块有一个基本的了解，我们对 Global&Local 的配置作如下简要说明。更详细的用法和各个模块对应的替代方案，请参考 API 文档。

```python
model = dict(
    type='GLInpaintor', # 补全器的名称
    encdec=dict(
        type='GLEncoderDecoder', # 编码器-解码器的名称
        encoder=dict(type='GLEncoder', norm_cfg=dict(type='SyncBN')), # 编码器的配置
        decoder=dict(type='GLDecoder', norm_cfg=dict(type='SyncBN')), # 解码器的配置
        dilation_neck=dict(
            type='GLDilationNeck', norm_cfg=dict(type='SyncBN'))), # 扩颈的配置
    disc=dict(
        type='GLDiscs', # 判别器的名称
        global_disc_cfg=dict(
            in_channels=3, # 判别器的输入通道数
            max_channels=512, # 判别器中的最大通道数
            fc_in_channels=512 * 4 * 4, # 最后一个全连接层的输入通道
            fc_out_channels=1024, # 最后一个全连接层的输出通道
            num_convs=6, # 判别器中使用的卷积数量
            norm_cfg=dict(type='SyncBN') # 归一化层的配置
        ),
        local_disc_cfg=dict(
            in_channels=3, # 判别器的输入通道数
            max_channels=512, # 判别器中的最大通道数
            fc_in_channels=512 * 4 * 4, # 最后一个全连接层的输入通道
            fc_out_channels=1024, # 最后一个全连接层的输出通道
            num_convs=5, # 判别器中使用的卷积数量
            norm_cfg=dict(type='SyncBN') # 归一化层的配置
        ),
    ),
    loss_gan=dict(
        type='GANLoss', # GAN 损失的名称
        gan_type='vanilla', # GAN 损失的类型
        loss_weight=0.001 # GAN 损失函数的权重
    ),
    loss_l1_hole=dict(
        type='L1Loss', # L1 损失的类型
        loss_weight=1.0 # L1 损失函数的权重
    ),
    pretrained=None) # 预训练权重的路径

train_cfg = dict(
    disc_step=1, # 训练生成器之前训练判别器的迭代次数
    iter_tc=90000, # 预热生成器的迭代次数
    iter_td=100000, # 预热判别器的迭代次数
    start_iter=0, # 开始的迭代
    local_size=(128, 128)) # 图像块的大小
test_cfg = dict(metrics=['l1']) # 测试的配置

dataset_type = 'ImgInpaintingDataset' # 数据集类型
input_shape = (256, 256) # 输入图像的形状

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img'), # 加载图片的配置
    dict(
        type='LoadMask', # 加载掩码
        mask_mode='bbox', # 掩码的类型
        mask_config=dict(
            max_bbox_shape=(128, 128), # 检测框的形状
            max_bbox_delta=40, # 检测框高宽的变化
            min_margin=20,  # 检测框到图片边界的最小距离
            img_shape=input_shape)),  # 输入图像的形状
    dict(
        type='Crop', # 裁剪
        keys=['gt_img'],  # 要裁剪的图像的关键词
        crop_size=(384, 384),  # 裁剪图像块的大小
        random_crop=True,  # 是否使用随机裁剪
    ),
    dict(
        type='Resize',  # 图像大小调整
        keys=['gt_img'],  # 要调整大小的图像的关键词
        scale=input_shape,  # 调整图像大小的比例
        keep_ratio=False,  # 调整大小时是否保持比例
    ),
    dict(
        type='Normalize',  # 图像归一化
        keys=['gt_img'],  # 要归一化的图像的关键词
        mean=[127.5] * 3,  # 归一化中使用的均值
        std=[127.5] * 3,  # 归一化中使用的标准差
        to_rgb=False),  # 是否将图像通道从 BGR 转换为 RGB
    dict(type='GetMaskedImage'),  # 获取被掩盖的图像
    dict(
        type='Collect',  # 决定数据中哪些键应该传递给合成器
        keys=['gt_img', 'masked_img', 'mask', 'mask_bbox'],  # 要收集的数据的关键词
        meta_keys=['gt_img_path']),  # 要收集的数据的元关键词
    dict(type='ImageToTensor', keys=['gt_img', 'masked_img', 'mask']),  # 将图像转化为 Tensor
    dict(type='ToTensor', keys=['mask_bbox'])  # 转化为 Tensor
]

test_pipeline = train_pipeline  # 构建测试/验证流程

data_root = 'data/places365'  # 数据根目录

data = dict(
    samples_per_gpu=12,  # 单个 GPU 的批量大小
    workers_per_gpu=8,  # 为每个 GPU 预取数据的 Worker 数
    val_samples_per_gpu=1,  # 验证中单个 GPU 的批量大小
    val_workers_per_gpu=8,  # 在验证中为每个 GPU 预取数据的 Worker 数
    drop_last=True,  # 是否丢弃训练中的最后一批数据
    train=dict(  # 训练数据集配置
        type=dataset_type,
        ann_file=f'{data_root}/train_places_img_list_total.txt',
        data_prefix=data_root,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(  # 验证数据集配置
        type=dataset_type,
        ann_file=f'{data_root}/val_places_img_list.txt',
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True))

optimizers = dict(  # 用于构建优化器的配置，支持 PyTorch 中所有优化器，且参数与 PyTorch 中对应优化器相同
    generator=dict(type='Adam', lr=0.0004), disc=dict(type='Adam', lr=0.0004))

lr_config = dict(policy='Fixed', by_epoch=False)  # 用于注册 LrUpdater 钩子的学习率调度程序配置

checkpoint_config = dict(by_epoch=False, interval=50000)  # 配置检查点钩子，实现参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
log_config = dict(  # 配置注册记录器钩子
    interval=100,  # 打印日志的时间间隔
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),  # 支持 Tensorboard 记录器
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])  # 用于记录训练过程的记录器

visual_config = dict(  # 构建可视化钩子的配置
    type='MMEditVisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=[
        'gt_img', 'masked_img', 'fake_res', 'fake_img', 'fake_gt_local'
    ],
)  # 用于可视化训练过程的记录器。

evaluation = dict(interval=50000)  # 构建验证钩子的配置

total_iters = 500002
dist_params = dict(backend='nccl')  # 设置分布式训练的参数，端口也可以设置
log_level = 'INFO'  # 日志级别
work_dir = None  # 保存当前实验的模型检查点和日志的目录
load_from = None  # 从给定路径加载模型作为预训练模型。 这不会恢复训练
resume_from = None  # 从给定路径恢复检查点，当检查点被保存时，训练将从该 epoch 恢复
workflow = [('train', 10000)]  # runner 的工作流程。 [('train', 1)] 表示只有一个工作流程，名为 'train' 的工作流程执行一次。 训练当前生成模型时保持不变
exp_name = 'gl_places'  # 实验名称
find_unused_parameters = False  # 是否在分布式训练中查找未使用的参数
```
