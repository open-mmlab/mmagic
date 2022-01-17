# 配置文件 - 抠图

与 [MMDetection](https://github.com/open-mmlab/mmdetection) 一样，我们将模块化和继承设计融入我们的配置系统，以方便进行各种实验。

## 例子 - Deep Image Matting Model

为了帮助用户对一个完整的配置有一个基本的了解，我们对我们实现的原始 DIM 模型的配置做一个简短的评论，如下所示。 更详细的用法和各个模块对应的替代方案，请参考 API 文档。

```python
# 模型配置
model = dict(
    type='DIM',  # 模型的名称（我们称之为抠图器）
    backbone=dict(  # 主干网络的配置
        type='SimpleEncoderDecoder',  # 主干网络的类型
        encoder=dict(  # 编码器的配置
            type='VGG16'),  # 编码器的类型
        decoder=dict(  # 解码器的配置
            type='PlainDecoder')),  # 解码器的类型
    pretrained='./weights/vgg_state_dict.pth',  # 编码器的预训练权重
    loss_alpha=dict(  # alpha 损失的配置
        type='CharbonnierLoss',  # 预测的 alpha 遮罩的损失类型
        loss_weight=0.5),  # alpha 损失的权重
    loss_comp=dict(  # 组合损失的配置
        type='CharbonnierCompLoss',  # 组合损失的类型
        loss_weight=0.5))  # 组合损失的权重
train_cfg = dict(  # 训练 DIM 模型的配置
    train_backbone=True,  # 在 DIM stage 1 中，会对主干网络进行训练
    train_refiner=False)  # 在 DIM stage 1 中，不会对精炼器进行训练
test_cfg = dict(  # 测试 DIM 模型的配置
    refine=False,  # 是否使用精炼器输出作为输出，在 stage 1 中，我们不使用它
    metrics=['SAD', 'MSE', 'GRAD', 'CONN'])  # 测试时使用的指标

# 数据配置
dataset_type = 'AdobeComp1kDataset'  # 数据集类型，这将用于定义数据集
data_root = 'data/adobe_composition-1k'  # 数据的根目录
img_norm_cfg = dict(  # 归一化输入图像的配置
    mean=[0.485, 0.456, 0.406],  # 归一化中使用的均值
    std=[0.229, 0.224, 0.225],  # 归一化中使用的标准差
    to_rgb=True)  # 是否将图像通道从 BGR 转换为 RGB
train_pipeline = [  # 训练数据处理流程
    dict(
        type='LoadImageFromFile',  # 从文件加载 alpha 遮罩
        key='alpha',  # 注释文件中 alpha 遮罩的键关键词。流程将从路径 “alpha_path” 中读取 alpha 遮罩
        flag='grayscale'),  # 加载灰度图像，形状为（高度、宽度）
    dict(
        type='LoadImageFromFile',  # 从文件中加载图像
        key='fg'),  # 要加载的图像的关键词。流程将从路径 “fg_path” 读取 fg
    dict(
        type='LoadImageFromFile',  # 从文件中加载图像
        key='bg'),  # 要加载的图像的关键词。流程将从路径 “bg_path” 读取 bg
    dict(
        type='LoadImageFromFile',  # 从文件中加载图像
        key='merged'),  # 要加载的图像的关键词。流程将从路径 “merged_path” 读取并合并
    dict(
        type='CropAroundUnknown',  # 在未知区域（半透明区域）周围裁剪图像
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],  # 要裁剪的图像
        crop_sizes=[320, 480, 640]),  # 裁剪大小
    dict(
        type='Flip',  # 翻转图像
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg']),  # 要翻转的图像
    dict(
        type='Resize',  # 图像大小调整
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],  # 图像调整大小的图像
        scale=(320, 320),  # 目标大小
        keep_ratio=False),  # 是否保持高宽比例
    dict(
        type='GenerateTrimap',  # 从 alpha 遮罩生成三元图。
        kernel_size=(1, 30)),  # 腐蚀/扩张内核大小的范围
    dict(
        type='RescaleToZeroOne', # 将图像从 [0, 255] 缩放到 [0, 1]
        keys=['merged', 'alpha', 'ori_merged', 'fg', 'bg']),  # 要重新缩放的图像
    dict(
        type='Normalize',  # 图像归一化
        keys=['merged'],  # 要归一化的图像
        **img_norm_cfg),  # 图像归一化配置（`img_norm_cfg` 的定义见上文）
    dict(
        type='Collect',  # 决定数据中哪些键应该传递给合成器
        keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg'],  # 图像的关键词
        meta_keys=[]),  # 图片的元关键词，这里不需要元信息。
    dict(
        type='ImageToTensor',  # 将图像转化为 Tensor
        keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg']),  # 要转换为 Tensor 的图像
]
test_pipeline = [
    dict(
        type='LoadImageFromFile', # 从文件加载 alpha 遮罩
        key='alpha',  # 注释文件中 alpha 遮罩的键关键词。流程将从路径 “alpha_path” 中读取 alpha 遮罩
        flag='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',  # 从文件中加载图像
        key='trimap',  # 要加载的图像的关键词。流程将从路径 “trimap_path” 读取三元图
        flag='grayscale', # 加载灰度图像，形状为（高度、宽度）
        save_original_img=True), # 保存三元图用于计算指标。 它将与 “ori_trimap” 一起保存
    dict(
        type='LoadImageFromFile',  # 从文件中加载图像
        key='merged'),  # 要加载的图像的关键词。流程将从路径 “merged_path” 读取并合并
    dict(
        type='Pad',  # 填充图像以与模型的下采样因子对齐
        keys=['trimap', 'merged'],  # 要填充的图像
        mode='reflect'),  # 填充模式
    dict(
        type='RescaleToZeroOne',  # 与 train_pipeline 相同
        keys=['merged', 'ori_alpha']),  # 要缩放的图像
    dict(
        type='Normalize',  # 与 train_pipeline 相同
        keys=['merged'],
        **img_norm_cfg),
    dict(
        type='Collect',  # 与 train_pipeline 相同
        keys=['merged', 'trimap'],
        meta_keys=[
            'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha',
            'ori_trimap'
        ]),
    dict(
        type='ImageToTensor',  # 与 train_pipeline 相同
        keys=['merged', 'trimap']),
]
data = dict(
    samples_per_gpu=1,  #单个 GPU 的批量大小
    workers_per_gpu=4,  # 为每个 GPU 预取数据的 Worker 数
    drop_last=True,  # 是否丢弃训练中的最后一批数据
    train=dict(  # 训练数据集配置
        type=dataset_type,  # 数据集的类型
        ann_file=f'{data_root}/training_list.json',  # 注解文件路径
        data_prefix=data_root,  # 图像路径的前缀
        pipeline=train_pipeline),  # 见上文 train_pipeline
    val=dict(  # 验证数据集配置
        type=dataset_type,
        ann_file=f'{data_root}/test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline),  # 见上文 test_pipeline
    test=dict(  # 测试数据集配置
        type=dataset_type,
        ann_file=f'{data_root}/test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline))  # 见上文 test_pipeline

# 优化器
optimizers = dict(type='Adam', lr=0.00001)  # 用于构建优化器的配置，支持 PyTorch 中所有优化器，且参数与 PyTorch 中对应优化器相同
# 学习策略
lr_config = dict(  # 用于注册 LrUpdater 钩子的学习率调度程序配置
    policy='Fixed')  # 调度器的策略，支持 CosineAnnealing、Cyclic 等。支持的 LrUpdater 详情请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9。

# 检查点保存
checkpoint_config = dict(  # 配置检查点钩子，实现参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
    interval=40000,  # 保存间隔为 40000 次迭代
    by_epoch=False)  # 按迭代计数
evaluation = dict(  # # 构建验证钩子的配置
    interval=40000)  # 验证区间
log_config = dict(  # 配置注册记录器钩子
    interval=10,  # 打印日志的时间间隔
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),  # 用于记录训练过程的记录器
        # dict(type='TensorboardLoggerHook')  # 支持 Tensorboard 记录器
    ])

# runtime settings
total_iters = 1000000  # 训练模型的总迭代次数
dist_params = dict(backend='nccl')  # 设置分布式训练的参数，端口也可以设置
log_level = 'INFO'  # 日志级别
work_dir = './work_dirs/dim_stage1'  # 保存当前实验的模型检查点和日志的目录
load_from = None  # 从给定路径加载模型作为预训练模型。 这不会恢复训练
resume_from = None  # 从给定路径恢复检查点，当检查点被保存时，训练将从该 epoch 恢复
workflow = [('train', 1)]  # runner 的工作流程。 [('train', 1)] 表示只有一个工作流程，名为 'train' 的工作流程执行一次。 训练当前抠图模型时保持不变
```
