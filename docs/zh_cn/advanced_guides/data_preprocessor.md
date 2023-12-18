# 数据预处理器

## 数据preprocessor在训练流程中的位置

在模型训练过程中，图片数据先通过mmcv中的transform进行数据增强，并加载为dataloader，而后通过preprocessor将数据从cpu搬运到cuda上，并进行padding和归一化

mmcv中的transform来自各下游算法库中transform的迁移，防止各下游算法库中transform的冗余，以`configs/_base_/datasets/unpaired_imgs_256x256.py`为例，其完整config中的`train_pipeline`如下所示

```python
...
train_pipeline = [
    dict(color_type='color', key='img_A', type='LoadImageFromFile'),
    dict(color_type='color', key='img_B', type='LoadImageFromFile'),
    dict(auto_remap=True, mapping=dict(img=['img_A', 'img_B',]),
        share_random_params=True,
        transforms=[dict(interpolation='bicubic', scale=(286, 286,), type='Resize'),
                    dict(crop_size=(256, 256,), keys=['img',], random_crop=True, type='Crop'),],
        type='TransformBroadcaster'),
    dict(direction='horizontal', keys=['img_A', ], type='Flip'),
    dict(direction='horizontal', keys=['img_B', ], type='Flip'),
    dict(mapping=dict(img_mask='img_B', img_photo='img_A'),
        remapping=dict(img_mask='img_mask', img_photo='img_photo'),
        type='KeyMapper'),
    dict(data_keys=['img_photo', 'img_mask',],
        keys=['img_photo', 'img_mask',], type='PackInputs'),
]
...
```

data_preprocessor会对transform后的数据进行数据搬移，拼接和归一化，而后输入到网络中，以`mmagic/models/editors/cyclegan/cyclegan.py`中的`train_step`函数为例，代码中的引用逻辑如下

```python
...
message_hub = MessageHub.get_current_instance()
curr_iter = message_hub.get_info('iter')
data = self.data_preprocessor(data, True)
disc_optimizer_wrapper = optim_wrapper['discriminators']

inputs_dict = data['inputs']
outputs, log_vars = dict(), dict()
...
```

在mmagic中的data_processor，其代码实现路径为`mmagic/models/data_preprocessors/data_preprocessor.py`，其数据处理流程如下图
![image](https://github.com/jinxianwei/CloudImg/assets/81373517/f52a92ab-f86d-486d-86ac-a2f388a83ced)
