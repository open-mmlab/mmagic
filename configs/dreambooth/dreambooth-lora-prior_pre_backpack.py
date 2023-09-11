_base_ = 'dreambooth-lora-prior_pre.py'

val_prompts = [
    'a sks backpack in Grand Canyon',
    'a sks backpack on the mountain',
    'a sks backpack in the city of Versailles',
    'a sks backpack in water',
]
model = dict(val_prompts=val_prompts, class_prior_prompt='a backpack')

pipeline = [
    dict(
        type='LoadImageFromHuggingFaceDataset', key='img',
        channel_order='rgb'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='PackInputs')
]
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='HuggingFaceDreamBoothDataset',
        dataset='google/dreambooth',
        dataset_sub_dir='backpack',
        prompt='a sks backpack',
        pipeline=pipeline))
