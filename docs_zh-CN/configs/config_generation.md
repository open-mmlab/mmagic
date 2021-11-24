# 配置文件 - 生成

与 [MMDetection](https://github.com/open-mmlab/mmdetection) 一样，我们将模块化和继承设计融入我们的配置系统，以方便进行各种实验。

## 示例 - pix2pix

为了帮助用户对完整的配置和生成系统中的模块有一个基本的了解，我们对 pix2pix 的配置做如下简要说明。
更详细的用法和各个模块对应的替代方案，请参考 API 文档。

```python
# 模型设置
model = dict(
    type='Pix2Pix',  # 合成器名称
    generator=dict(
        type='UnetGenerator',  # 生成器名称
        in_channels=3,  # 生成器的输入通道数
        out_channels=3,  # 生成器的输出通道数
        num_down=8,  # # 生成器中下采样的次数
        base_channels=64,  # 生成器最后卷积层的通道数
        norm_cfg=dict(type='BN'),  # 归一化层的配置
        use_dropout=True,  # 是否在生成器中使用 dropout
        init_cfg=dict(type='normal', gain=0.02)),  # 初始化配置
    discriminator=dict(
        type='PatchDiscriminator',  # 判别器的名称
        in_channels=6,  # 判别器的输入通道数
        base_channels=64,  # 判别器第一卷积层的通道数
        num_conv=3,  # 判别器中堆叠的中间卷积层（不包括输入和输出卷积层）的数量
        norm_cfg=dict(type='BN'),  # 归一化层的配置
        init_cfg=dict(type='normal', gain=0.02)),  # 初始化配置
    gan_loss=dict(
        type='GANLoss',  # GAN 损失的名称
        gan_type='vanilla',  # GAN 损失的类型
        real_label_val=1.0,  # GAN 损失函数中真实标签的值
        fake_label_val=0.0,  # GAN 损失函数中伪造标签的值
        loss_weight=1.0),  # GAN 损失函数的权重
    pixel_loss=dict(type='L1Loss', loss_weight=100.0, reduction='mean'))
# 模型训练和测试设置
train_cfg = dict(
    direction='b2a')  # pix2pix 的图像到图像的转换方向 (模型训练的方向，和测试方向一致)。模型默认: a2b
test_cfg = dict(
    direction='b2a',   # pix2pix 的图像到图像的转换方向 (模型测试的方向，和训练方向一致)。模型默认: a2b
    show_input=True)  # 保存 pix2pix 的测试图像时是否显示输入的真实图像

# 数据设置
train_dataset_type = 'GenerationPairedDataset'  # 训练数据集的类型
val_dataset_type = 'GenerationPairedDataset'  # 验证/测试数据集类型
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 输入图像归一化配置
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',  # 从文件路径加载图像对
        io_backend='disk',  # 存储图像的 IO 后端
        key='pair',  # 查找对应路径的关键词
        flag='color'),  # 加载图像标志
    dict(
        type='Resize',  # 图像大小调整
        keys=['img_a', 'img_b'],  # 要调整大小的图像的关键词
        scale=(286, 286),  # 调整图像大小的比例
        interpolation='bicubic'),  # 调整图像大小时用于插值的算法
    dict(
        type='FixedCrop',  # 固定裁剪，在特定位置将配对图像裁剪为特定大小以训练 pix2pix
        keys=['img_a', 'img_b'],  # 要裁剪的图像的关键词
        crop_size=(256, 256)),  # 裁剪图像的大小
    dict(
        type='Flip',  # 翻转图像
        keys=['img_a', 'img_b'],  # 要翻转的图像的关键词
        direction='horizontal'),  # 水平或垂直翻转图像
    dict(
        type='RescaleToZeroOne',  # 将图像从 [0, 255] 缩放到 [0, 1]
        keys=['img_a', 'img_b']),  # 要重新缩放的图像的关键词
    dict(
        type='Normalize',  # 图像归一化
        keys=['img_a', 'img_b'],  # 要归一化的图像的关键词
        to_rgb=True,  # 是否将图像通道从 BGR 转换为 RGB
        **img_norm_cfg),  # 图像归一化配置（`img_norm_cfg` 的定义见上文）
    dict(
       type='ImageToTensor',  # 将图像转化为 Tensor
       keys=['img_a', 'img_b']),  # 要从图像转换为 Tensor 的图像的关键词
    dict(
        type='Collect',  # 决定数据中哪些键应该传递给合成器
        keys=['img_a', 'img_b'],  # 图像的关键词
        meta_keys=['img_a_path', 'img_b_path'])  # 图片的元关键词
]
test_pipeline = [
    dict(
        type='LoadPairedImageFromFile',  # 从文件路径加载图像对
        io_backend='disk',  # 存储图像的 IO 后端
        key='pair',  # 查找对应路径的关键词
        flag='color'),  # 加载图像标志
    dict(
        type='Resize',  # 图像大小调整
        keys=['img_a', 'img_b'],  # 要调整大小的图像的关键词
        scale=(256, 256),  # 调整图像大小的比例
        interpolation='bicubic'),  # 调整图像大小时用于插值的算法
    dict(
        type='RescaleToZeroOne',  # 将图像从 [0, 255] 缩放到 [0, 1]
        keys=['img_a', 'img_b']),  # 要重新缩放的图像的关键词
    dict(
        type='Normalize',  # 图像归一化
        keys=['img_a', 'img_b'],  # 要归一化的图像的关键词
        to_rgb=True,  # 是否将图像通道从 BGR 转换为 RGB
        **img_norm_cfg),  # 图像归一化配置（`img_norm_cfg` 的定义见上文）
    dict(
       type='ImageToTensor',  # 将图像转化为 Tensor
       keys=['img_a', 'img_b']),  # 要从图像转换为 Tensor 的图像的关键词
    dict(
        type='Collect',  # 决定数据中哪些键应该传递给合成器
        keys=['img_a', 'img_b'],  # 图像的关键词
        meta_keys=['img_a_path', 'img_b_path'])  # 图片的元关键词
]
data_root = 'data/pix2pix/facades'  # 数据的根路径
data = dict(
    samples_per_gpu=1,  # 单个 GPU 的批量大小
    workers_per_gpu=4,  # 为每个 GPU 预取数据的 Worker 数
    drop_last=True,  # 是否丢弃训练中的最后一批数据
    val_samples_per_gpu=1,  # 验证中单个 GPU 的批量大小
    val_workers_per_gpu=0,  # 在验证中为每个 GPU 预取数据的 Worker 数
    train=dict(  # 训练数据集配置
        type=train_dataset_type,
        dataroot=data_root,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(  # 验证数据集配置
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(  # 测试数据集配置
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True))

# 优化器
optimizers = dict(  # 用于构建优化器的配置，支持 PyTorch 中所有优化器，且参数与 PyTorch 中对应优化器相同
    generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# 学习策略
lr_config = dict(policy='Fixed', by_epoch=False)  # 用于注册 LrUpdater 钩子的学习率调度程序配置

# 检查点保存
checkpoint_config = dict(interval=4000, save_optimizer=True, by_epoch=False)  # 配置检查点钩子，实现参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
evaluation = dict(  # 构建验证钩子的配置
    interval=4000,  # 验证区间
    save_image=True)  # 是否保存图片
log_config = dict(  # 配置注册记录器钩子
    interval=100,  # 打印日志的时间间隔
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),  # 用于记录训练过程的记录器
        # dict(type='TensorboardLoggerHook')  # 还支持 Tensorboard 记录器
    ])
visual_config = None  # 构建可视化钩子的配置

# 运行设置
total_iters = 80000  # 训练模型的总迭代次数
cudnn_benchmark = True  # 设置 cudnn_benchmark
dist_params = dict(backend='nccl')  # 设置分布式训练的参数，端口也可以设置
log_level = 'INFO'  # 日志级别
load_from = None  # 从给定路径加载模型作为预训练模型。 这不会恢复训练
resume_from = None  # 从给定路径恢复检查点，当检查点被保存时，训练将从该 epoch 恢复
workflow = [('train', 1)]  # runner 的工作流程。 [('train', 1)] 表示只有一个工作流程，名为 'train' 的工作流程执行一次。 训练当前生成模型时保持不变
exp_name = 'pix2pix_facades'  # 实验名称
work_dir = f'./work_dirs/{exp_name}'  # 保存当前实验的模型检查点和日志的目录
```
