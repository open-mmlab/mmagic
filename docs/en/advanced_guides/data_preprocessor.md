# Data pre-processor

## The position of the data preprocessor in the training pipeline.

During the model training process, image data undergoes data augmentation using the transforms provided by mmcv. The augmented data is then loaded into a dataloader. Subsequently, a preprocessor is used to move the data from the CPU to CUDA (GPU), perform padding, and normalize the data.

Below is an example of the `train_pipeline` in the complete configuration file using `configs/_base_/datasets/unpaired_imgs_256x256.py`. The train_pipeline typically defines a sequence of transformations applied to training images using the mmcv library. This pipeline is designed to prevent redundancy in the transformation functions across different downstream algorithm libraries.

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

In the `train_step` function in the `mmagic/models/editors/cyclegan/cyclegan.py` script, the data preprocessing steps involve moving, concatenating, and normalizing the transformed data before feeding it into the neural network. Below is an example of the relevant code logic:

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

In mmagic, the code implementation for the data processor is located at `mmagic/models/data_preprocessors/data_preprocessor.py`. The data processing workflow is as follows:
![image](https://github.com/jinxianwei/CloudImg/assets/81373517/f52a92ab-f86d-486d-86ac-a2f388a83ced)
