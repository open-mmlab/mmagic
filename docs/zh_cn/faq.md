## 常见问题解答

### 在配置中使用中间变量

配置文件中使用了一些中间变量，如数据集中的 `train_pipeline` 和 `test_pipeline`。

例如，我们通常先定义 `train_pipeline` 和 `test_pipeline`，再将它们传递到 `data` 中。因此， `train_pipeline` 和 `test_pipeline` 是中间变量。

```python
...
train_dataset_type = 'SRAnnotationDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    ...
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    ...
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    # 训练
    train_dataloader = dict(
        samples_per_gpu=16,
        workers_per_gpu=6,
        drop_last=True),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/DIV2K/DIV2K_train_LR_bicubic/X2_sub',
            gt_folder='data/DIV2K/DIV2K_train_HR_sub',
            ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',
            pipeline=train_pipeline,
            scale=scale)),
    # 验证
    val_dataloader = dict(samples_per_gpu=1, workers_per_gpu=1),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/val_set5/Set5_bicLRx2',
        gt_folder='data/val_set5/Set5_mod12',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}')

empty_cache = True  # 在每次迭代后清空缓存。
```
