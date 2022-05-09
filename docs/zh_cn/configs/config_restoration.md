# 配置文件 - 复原

## 示例-EDSR

为了帮助用户理解 mmediting 的配置文件结构，这里以 EDSR 为例，给出其配置文件的注释。对于每个模块的详细用法以及对应参数的选择，请参照 API 文档。

```python
exp_name = 'edsr_x2c64b16_1x16_300k_div2k'  # 实验名称

scale = 2  # 上采样放大因子

# 模型设置
model = dict(
    type='BasicRestorer',  # 图像恢复模型类型
    generator=dict(  # 生成器配置
        type='EDSR',  # 生成器类型
        in_channels=3,  # 输入通道数
        out_channels=3,  # 输出通道数
        mid_channels=64,  # 中间特征通道数
        num_blocks=16,  # 残差块数目
        upscale_factor=scale, # 上采样因子
        res_scale=1,  # 残差缩放因子
        rgb_mean=(0.4488, 0.4371, 0.4040),  # 输入图像 RGB 通道的平均值
        rgb_std=(1.0, 1.0, 1.0)),  # 输入图像 RGB 通道的方差
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))  # 像素损失函数的配置

# 模型训练和测试设置
train_cfg = None  # 训练的配置
test_cfg = dict(  # 测试的配置
    metrics=['PSNR'],  # 测试时使用的评价指标
    crop_border=scale)  # 测试时裁剪的边界尺寸

# 数据集设置
train_dataset_type = 'SRAnnotationDataset'  # 用于训练的数据集类型
val_dataset_type = 'SRFolderDataset'  #  用于验证的数据集类型
train_pipeline = [  # 训练数据前处理流水线步骤组成的列表
    dict(type='LoadImageFromFile',  # 从文件加载图像
        io_backend='disk',  # 读取图像时使用的 io 类型
        key='lq',  # 设置LR图像的键来找到相应的路径
        flag='unchanged'),  # 读取图像的标识
    dict(type='LoadImageFromFile',  # 从文件加载图像
        io_backend='disk',  # 读取图像时使用的io类型
        key='gt',  # 设置HR图像的键来找到相应的路径
        flag='unchanged'),  # 读取图像的标识
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),  # 将图像从[0，255]重缩放到[0，1]
    dict(type='Normalize',  # 正则化图像
        keys=['lq', 'gt'],  # 执行正则化图像的键
        mean=[0, 0, 0],  # 平均值
        std=[1, 1, 1],  # 标准差
        to_rgb=True),  # 更改为 RGB 通道
    dict(type='PairedRandomCrop', gt_patch_size=96),  # LR 和 HR 成对随机裁剪
    dict(type='Flip',  # 图像翻转
        keys=['lq', 'gt'],  # 执行翻转图像的键
        flip_ratio=0.5,  # 执行翻转的几率
        direction='horizontal'),  # 翻转方向
    dict(type='Flip',  # 图像翻转
        keys=['lq', 'gt'],  # 执行翻转图像的键
        flip_ratio=0.5,  # 执行翻转几率
        direction='vertical'),  # 翻转方向
    dict(type='RandomTransposeHW',  # 图像的随机的转置
        keys=['lq', 'gt'],  # 执行转置图像的键
        transpose_ratio=0.5  # 执行转置的几率
        ),
    dict(type='Collect',  # Collect 类决定哪些键会被传递到生成器中
        keys=['lq', 'gt'],  # 传入模型的键
        meta_keys=['lq_path', 'gt_path']), # 元信息键。在训练中，不需要元信息
    dict(type='ImageToTensor',  # 将图像转换为张量
        keys=['lq', 'gt'])  # 执行图像转换为张量的键
]
test_pipeline = [  # 测试数据前处理流水线步骤组成的列表
    dict(
        type='LoadImageFromFile',  # 从文件加载图像
        io_backend='disk',  # 读取图像时使用的io类型
        key='lq',  # 设置LR图像的键来找到相应的路径
        flag='unchanged'),  # 读取图像的标识
    dict(
        type='LoadImageFromFile',  # 从文件加载图像
        io_backend='disk',  # 读取图像时使用的io类型
        key='gt',  # 设置HR图像的键来找到相应的路径
        flag='unchanged'),  # 读取图像的标识
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),  # 将图像从[0，255]重缩放到[0，1]
    dict(
        type='Normalize',  # 正则化图像
        keys=['lq', 'gt'],  # 执行正则化图像的键
        mean=[0, 0, 0],  # 平均值
        std=[1, 1, 1],  # 标准差
        to_rgb=True),  # 更改为RGB通道
    dict(type='Collect',  # Collect类决定哪些键会被传递到生成器中
        keys=['lq', 'gt'],  # 传入模型的键
        meta_keys=['lq_path', 'gt_path']),  # 元信息键
    dict(type='ImageToTensor',  # 将图像转换为张量
        keys=['lq', 'gt'])  # 执行图像转换为张量的键
]

data = dict(
    # 训练
    samples_per_gpu=16,  # 单个 GPU 的批大小
    workers_per_gpu=6,  # 单个 GPU 的 dataloader 的进程
    drop_last=True,  # 在训练过程中丢弃最后一个批次
    train=dict(  # 训练数据集的设置
        type='RepeatDataset',  # 基于迭代的重复数据集
        times=1000,  # 重复数据集的重复次数
        dataset=dict(
            type=train_dataset_type,  # 数据集类型
            lq_folder='data/DIV2K/DIV2K_train_LR_bicubic/X2_sub',  # lq文件夹的路径
            gt_folder='data/DIV2K/DIV2K_train_HR_sub',  # gt文件夹的路径
            ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',  # 批注文件的路径
            pipeline=train_pipeline,  # 训练流水线，如上所示
            scale=scale)),  # 上采样放大因子

    # 验证
    val_samples_per_gpu=1,  # 验证时单个 GPU 的批大小
    val_workers_per_gpu=1,  # 验证时单个 GPU 的 dataloader 的进程
    val=dict(
        type=val_dataset_type,  # 数据集类型
        lq_folder='data/val_set5/Set5_bicLRx2',  # lq 文件夹的路径
        gt_folder='data/val_set5/Set5_mod12',  # gt 文件夹的路径
        pipeline=test_pipeline,  # 测试流水线，如上所示
        scale=scale,  # 上采样放大因子
        filename_tmpl='{}'),  # 文件名模板

    # 测试
    test=dict(
        type=val_dataset_type,  # 数据集类型
        lq_folder='data/val_set5/Set5_bicLRx2',  # lq 文件夹的路径
        gt_folder='data/val_set5/Set5_mod12',  # gt 文件夹的路径
        pipeline=test_pipeline,  # 测试流水线，如上所示
        scale=scale,  # 上采样放大因子
        filename_tmpl='{}'))  # 文件名模板

# 优化器设置
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))  # 用于构建优化器的设置，支持PyTorch中所有参数与PyTorch中参数相同的优化器

# 学习策略
total_iters = 300000  # 训练模型的总迭代数
lr_config = dict( # 用于注册LrUpdater钩子的学习率调度程序配置
    policy='Step', by_epoch=False, step=[200000], gamma=0.5)  # 调度器的策略，还支持余弦、循环等

checkpoint_config = dict(  # 模型权重钩子设置，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
    interval=5000,  # 模型权重文件保存间隔为5000次迭代
    save_optimizer=True,  # 保存优化器
    by_epoch=False)  # 按迭代次数计数
evaluation = dict(  # 构建验证钩子的配置
    interval=5000,  # 执行验证的间隔为5000次迭代
    save_image=True,  # 验证期间保存图像
    gpu_collect=True)  # 使用gpu收集
log_config = dict(  # 注册日志钩子的设置
    interval=100,  # 打印日志间隔
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),  # 记录训练过程信息的日志
        dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
    ])
visual_config = None  # 可视化的设置

# 运行设置
dist_params = dict(backend='nccl')  # 建立分布式训练的设置，其中端口号也可以设置
log_level = 'INFO'  # 日志等级
work_dir = f'./work_dirs/{exp_name}'  # 记录当前实验日志和模型权重文件的文件夹
load_from = None # 从给定路径加载模型作为预训练模型. 这个选项不会用于断点恢复训练
resume_from = None # 加载给定路径的模型权重文件作为断点续连的模型, 训练将从该时间点保存的周期点继续进行
workflow = [('train', 1)]  # runner 的执行流. [('train', 1)] 代表只有一个执行流，并且这个名为 train 的执行流只执行一次
```
